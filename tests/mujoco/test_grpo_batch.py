"""Batched rollout ordering, EOS handling, and full-parameter checkpoint tests."""

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task.grpo import collect_episodes, restore_checkpoint, save_checkpoint, validate_config
from mujoco_task.grpo_policy import ActionSample, batch_prompts, configure_training


def test_multimodal_batch_padding():
    prompts = [
        {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "pixel_values": torch.ones(4, 3),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "input_ids": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "pixel_values": torch.full((8, 3), 2),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        },
    ]
    batched = batch_prompts(prompts, 99)
    assert batched["input_ids"].tolist() == [[99, 1, 2], [3, 4, 5]]
    assert batched["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batched["pixel_values"].shape == (12, 3)
    assert batched["pixel_values"][4:].eq(2).all()
    assert prompts[0]["input_ids"].tolist() == [[1, 2]]


def test_batched_eos_keeps_original_prompt_and_correct_score(monkeypatch):
    from test_grpo import toy_actor

    actor = toy_actor()
    actor.policy.config = OmegaConf.create({"framework": {"use_state_prompt": False, "action_model": {"action_dim": 7}}})
    actor.policy.action_horizon = 8
    actor.policy.action_model = SimpleNamespace(fast_tokenizer=None)
    actor.interface.processor.tokenizer.pad_token_id = 0
    actor.interface.build_qwenvl_inputs = lambda **kw: {
        "input_ids": torch.tensor([[1] * len(kw["instructions"][0])]),
        "attention_mask": torch.ones(1, len(kw["instructions"][0]), dtype=torch.long),
    }
    scores = [torch.arange(16).reshape(2, 8).float() * (i + 1) for i in range(3)]

    def generate(**kw):
        assert kw["input_ids"].tolist() == [[0, 1], [1, 1]]
        return SimpleNamespace(sequences=torch.tensor([[0, 1, 3, 7, 0], [1, 1, 4, 5, 7]]), scores=scores)

    actor.model.generate = generate
    monkeypatch.setattr("mujoco_task.grpo_policy.strict_decode", lambda *a: np.zeros((1, 8, 7)))
    samples = actor.sample_batch([{"image": [], "lang": "a"}, {"image": [], "lang": "bb"}], 3)
    assert samples[0].tokens.tolist() == [3, 7]
    assert samples[1].tokens.tolist() == [4, 5, 7]
    assert samples[0].inputs["input_ids"].tolist() == [[1]]
    for row, sample in enumerate(samples):
        expected = torch.stack([scores[i][row].log_softmax(0)[token] for i, token in enumerate(sample.tokens)])
        torch.testing.assert_close(sample.old_logprobs, expected)


def test_collect_preserves_groups_across_uneven_environment_waves():
    class Env:
        def __init__(self, index):
            self.index = index
            self.task = SimpleNamespace(info=lambda: {"success": False})

        def reset(self, seed):
            self.steps = 0
            return {"image": np.zeros((32, 32, 3), dtype=np.uint8), "proprio": np.full(21, self.index)}

        def step(self, action, observe=True):
            self.steps += 1
            return {"image": np.zeros((32, 32, 3), dtype=np.uint8), "proprio": np.full(21, self.index)}, {
                "success": self.steps >= self.index + 1
            }

    batch_sizes = []

    def sample_batch(examples, max_new_tokens):
        batch_sizes.append(len(examples))
        return [
            ActionSample(
                {},
                torch.tensor([3, 7]),
                torch.zeros(2),
                np.zeros((1, 7)),
                "invalid" if int(example["state"][0]) == 2 else None,
            )
            for example in examples
        ]

    cfg = OmegaConf.create(
        dict(
            seed=10,
            groups_per_iteration=2,
            group_size=2,
            max_steps=3,
            execute=1,
            max_new_tokens=3,
            kl_beta=0,
            reward_mode="sparse",
        )
    )
    actor = SimpleNamespace(sample_batch=sample_batch)
    norm = SimpleNamespace(metadata={"contract": {"instruction": "test"}}, unnormalize=lambda a: a)
    episodes = collect_episodes([Env(i) for i in range(3)], actor, norm, 1, cfg)
    assert [e.seed for e in episodes] == [12, 12, 13, 13]
    assert [e.steps for e in episodes] == [1, 2, 0, 1]
    assert [e.reward for e in episodes] == [1, 1, -1, 1]
    assert batch_sizes == [3, 1, 1]


def test_chunk_rollout_only_renders_observations_used_by_policy():
    from mujoco_task.grpo import rollout_batch

    class Env:
        def __init__(self):
            self.task = SimpleNamespace(info=lambda: {"success": False})
            self.flags = []
            self.steps = 0

        def observation(self):
            return {"image": np.zeros((32, 32, 3), np.uint8), "proprio": np.full(21, self.steps)}

        def reset(self, seed):
            return self.observation()

        def step(self, action, observe=True):
            self.steps += 1
            self.flags.append(observe)
            return self.observation() if observe else {}, {"success": False}

    observed_steps = []

    def sample_batch(examples, max_new_tokens):
        observed_steps.append(int(examples[0]["state"][0]))
        return [ActionSample({}, torch.tensor([3, 7]), torch.zeros(2), np.zeros((8, 7)), None)]

    env = Env()
    cfg = OmegaConf.create(dict(execute=8, max_steps=10, max_new_tokens=256, kl_beta=0, reward_mode="sparse"))
    norm = SimpleNamespace(metadata={"contract": {"instruction": "test"}}, unnormalize=lambda a: a)
    episode = rollout_batch([env], SimpleNamespace(sample_batch=sample_batch), norm, [42], cfg)[0]
    assert observed_steps == [0, 8]
    assert env.flags == [False] * 7 + [True, False, True]
    assert episode.steps == 10


def test_full_weights_reference_and_checkpoint(tmp_path):
    from test_grpo import ToyBackbone

    class Backbone(ToyBackbone):
        def gradient_checkpointing_enable(self, **kwargs):
            self.checkpointing = True

        def gradient_checkpointing_disable(self):
            self.checkpointing = False

    policy = torch.nn.Module()
    policy.qwen_vl_interface = torch.nn.Module()
    policy.qwen_vl_interface.model = Backbone()
    policy.qwen_vl_interface._ACTION_TOKEN_MIN = 3
    policy.qwen_vl_interface._ACTION_TOKEN_MAX = 5
    policy.qwen_vl_interface.processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7))
    source = tmp_path / "sft"
    source.mkdir()
    (source / "normalization.json").write_text("{}")
    (source / "config.full.yaml").write_text("framework: {}")
    torch.save(policy.state_dict(), source / "base.pt")
    cfg = OmegaConf.create(
        dict(train_mode="full", kl_beta=0.02, device="cpu", temperature=0.7, checkpoint=str(source / "base.pt"))
    )
    actor = configure_training(policy, cfg)
    assert actor.reference is not None
    assert actor.model.weight.dtype == torch.float32
    assert all(p.requires_grad for p in actor.model.parameters())
    assert not any(p.requires_grad for p in actor.reference.model.parameters())
    assert actor.model.weight.data_ptr() != actor.reference.model.weight.data_ptr()
    sample = ActionSample(
        {"input_ids": torch.tensor([[1, 2]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)},
        torch.tensor([3, 7]),
        torch.zeros(2),
        None,
        None,
    )
    reference = actor.reference_logprobs(sample).clone()
    optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.01, foreach=False)
    (-actor.logprobs(sample).mean()).backward()
    optimizer.step()
    torch.testing.assert_close(reference, actor.reference_logprobs(sample))
    assert not torch.equal(reference, actor.logprobs(sample).detach())
    saved = save_checkpoint(tmp_path, 1, actor, optimizer, cfg, {})
    assert (saved / "pytorch_model.pt").is_file() and not (saved / "adapter").exists()
    restored = copy.deepcopy(policy)
    restored.load_state_dict(torch.load(source / "base.pt", weights_only=True), strict=True)
    restored_actor = configure_training(restored, cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.01, foreach=False)
    assert restore_checkpoint(restored_actor, restored_optimizer, saved) == 1
    torch.testing.assert_close(restored.qwen_vl_interface.model.weight, actor.model.weight)
    torch.testing.assert_close(restored_actor.reference_logprobs(sample), reference)
    torch.testing.assert_close(
        restored_optimizer.state[restored.qwen_vl_interface.model.weight]["exp_avg"],
        optimizer.state[actor.model.weight]["exp_avg"],
    )


@pytest.mark.parametrize(
    "key,value", [("num_envs", 0), ("num_envs", 9), ("torch_dtype", "bfloat16"), ("train_mode", "unknown")]
)
def test_a100_config_validation(key, value):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/mujoco/qwenfast_grpo_a100.yaml")
    validate_config(cfg)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_real_tiny_qwen_batch_full_update_and_reload(monkeypatch, tmp_path):
    """Exercise actual multimodal left padding, full gradients and portable loading."""
    metadata = Path("playground/processor_check")
    fast = Path("playground/Pretrained_models/fast")
    if not (metadata / "tokenizer.json").is_file() or not (fast / "tokenizer.json").is_file():
        pytest.skip("Local processor/FAST assets required; no automatic downloads")
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

    from mujoco_task.evaluate import load_policy
    from mujoco_task.grpo_policy import FastGRPOPolicy
    from starVLA.model.framework.VLM4A.QwenFast import Qwenvl_Fast
    from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model

    config = Qwen2_5_VLConfig(
        vocab_size=153713,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "fullatt_block_indexes": [0],
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    backbone = Qwen2_5_VLForConditionalGeneration(config).eval()
    processor = AutoProcessor.from_pretrained(metadata, min_pixels=50176, max_pixels=50176)
    sft_cfg = OmegaConf.load("configs/mujoco/qwenfast.yaml")
    codec = get_action_model(sft_cfg)
    backbone.save_pretrained(tmp_path / "Qwen2.5-VL-Action")
    processor.save_pretrained(tmp_path / "Qwen2.5-VL-Action")
    codec.fast_tokenizer.save_pretrained(tmp_path / "fast")
    sft_cfg.framework.qwenvl.base_vlm = str(tmp_path / "Qwen2.5-VL-Action")
    sft_cfg.framework.action_model.fast_tokenizer = str(tmp_path / "fast")
    with monkeypatch.context() as patch:
        patch.setattr(
            "starVLA.model.modules.vlm.QWen2_5.Qwen2_5_VLForConditionalGeneration.from_pretrained",
            lambda *a, **kw: backbone,
        )
        patch.setattr("starVLA.model.modules.vlm.QWen2_5.AutoProcessor.from_pretrained", lambda *a, **kw: processor)
        patch.setattr("starVLA.model.framework.VLM4A.QwenFast.get_action_model", lambda config: codec)
        policy = Qwenvl_Fast(sft_cfg).eval()
    source = tmp_path / "sft"
    source.mkdir()
    OmegaConf.save(sft_cfg, source / "config.full.yaml")
    (source / "normalization.json").write_text(
        '{"fit_split":"train","min":[-1,-1,-1,-1,-1,-1,-1],' '"max":[1,1,1,1,1,1,1]}'
    )
    torch.save(policy.state_dict(), source / "base.pt")
    cfg = OmegaConf.create(
        dict(train_mode="full", kl_beta=0.02, device="cpu", temperature=0.7, checkpoint=str(source / "base.pt"))
    )
    actor = configure_training(policy, cfg)
    examples = [
        {"image": [Image.new("RGB", (128, 128), color)], "lang": instruction, "state": np.zeros(21)}
        for color, instruction in (("green", "move green"), ("red", "move the green block to the right side"))
    ]
    samples = actor.sample_batch(examples, 4)
    for sample in samples:
        with torch.no_grad():
            torch.testing.assert_close(actor.logprobs(sample).cpu(), sample.old_logprobs, atol=1e-5, rtol=1e-5)
    reference = actor.reference_logprobs(samples[0]).clone()
    actor.model.train()
    optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.001, foreach=False)
    loss = -actor.logprobs(samples[0]).mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in actor.model.named_parameters() if "visual" in n)
    optimizer.step()
    actor.model.eval()
    torch.testing.assert_close(reference, actor.reference_logprobs(samples[0]))
    with torch.no_grad():
        expected = actor.logprobs(samples[0]).clone()
    saved = save_checkpoint(tmp_path, 1, actor, optimizer, cfg, {})
    (source / "base.pt").unlink()  # full export must not depend on the original SFT weights
    restored, _, _ = load_policy(saved, "cpu")
    with torch.no_grad():
        torch.testing.assert_close(FastGRPOPolicy(restored, cfg.temperature).logprobs(samples[0]), expected)
