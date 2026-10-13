"""GRPO grouping, probability alignment, reward and optimizer regression tests."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task.grpo import (
    Episode,
    group_advantages,
    load_rlinf,
    rollout,
    terminal_reward,
    update_actor,
    validate_config,
)
from mujoco_task.grpo_policy import ActionSample, ActionVocabulary, FastGRPOPolicy


@pytest.fixture(scope="module")
def algorithms():
    root = Path(__file__).resolve().parents[3] / "Rlinf4/Rlinf4"
    if not root.exists():
        pytest.skip("This integration test needs the configured RLinf checkout")
    return load_rlinf(root)[:2]


def test_group_relative_advantages(algorithms):
    compute, _ = algorithms
    actual = group_advantages([0, 1, 10, 10], 2, compute)
    torch.testing.assert_close(actual, torch.tensor([-(2**-0.5), 2**-0.5, 0, 0]), atol=2e-6, rtol=0)
    with pytest.raises(ValueError, match="complete groups"):
        group_advantages([1], 1, compute)


class ToyBackbone(torch.nn.Module):
    """Causal logits that vary by prefix position, to detect off-by-one scoring."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.device = torch.device("cpu")

    def forward(self, input_ids, **kwargs):
        positions = torch.arange(input_ids.shape[1], dtype=torch.float32)
        logits = positions[:, None] * torch.arange(8, dtype=torch.float32)[None, :] * self.weight
        return SimpleNamespace(logits=logits.unsqueeze(0))


def toy_actor():
    model = ToyBackbone()
    interface = SimpleNamespace(
        model=model,
        _ACTION_TOKEN_MIN=3,
        _ACTION_TOKEN_MAX=5,
        processor=SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda _: 7)),
    )
    return FastGRPOPolicy(SimpleNamespace(qwen_vl_interface=interface), temperature=0.7)


def test_logprobs_shift_temperature_eos_and_gradient():
    actor = toy_actor()
    sample = ActionSample(
        {"input_ids": torch.tensor([[1, 2, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
        torch.tensor([3, 5, 7]),
        torch.zeros(3),
        None,
        None,
    )
    actual = actor.logprobs(sample)
    logits = torch.arange(2, 5)[:, None] * torch.tensor([3, 4, 5, 7])[None, :] * 0.2 / 0.7
    expected = logits.log_softmax(-1)[torch.arange(3), torch.tensor([0, 2, 3])]
    torch.testing.assert_close(actual, expected)
    (-actual.sum()).backward()
    assert actor.model.weight.grad.abs() > 0
    assert sample.inputs["input_ids"].shape[1] == 3  # replay buffer is immutable


def test_action_vocabulary_matches_scoring():
    actor = toy_actor()
    logits = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    masked = ActionVocabulary(actor.allowed)(torch.tensor([[1]]), logits)
    assert torch.isneginf(masked[0, [0, 1, 2, 6]]).all()
    torch.testing.assert_close(masked[0, actor.allowed], logits[0, actor.allowed])


def test_minibatch_shuffle_steps_once_per_group_without_touching_global_rng(algorithms):
    _, loss_fn = algorithms
    actor = toy_actor()
    episodes = []
    for index, tokens in enumerate(([3, 7], [5, 7], [3, 5, 7], [4, 7])):
        sample = ActionSample(
            {"input_ids": torch.tensor([[1, 2, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
            torch.tensor(tokens),
            torch.zeros(len(tokens)),
            None,
            None,
        )
        sample.old_logprobs = actor.logprobs(sample).detach()
        episodes.append(Episode([sample], float(index), index, 1, {}, None))
    cfg = OmegaConf.create(
        dict(update_epochs=1, clip_ratio=0.2, kl_beta=0.0, max_grad_norm=1.0, minibatch_size=2, seed=0)
    )
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.001)
    rng_before = torch.get_rng_state()
    metrics = update_actor(actor, episodes, torch.tensor([1.0, -1.0, 0.5, -0.5]), optimizer, loss_fn, cfg, iteration=3)
    assert torch.equal(rng_before, torch.get_rng_state())
    assert metrics["updated"] and metrics["optimizer_steps"] == 2


def test_update_changes_policy_and_zero_variance_skips(algorithms):
    _, loss_fn = algorithms
    actor = toy_actor()
    samples = []
    for tokens in ([3, 7], [5, 7]):
        sample = ActionSample(
            {"input_ids": torch.tensor([[1, 2, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
            torch.tensor(tokens),
            torch.zeros(2),
            None,
            None,
        )
        sample.old_logprobs = actor.logprobs(sample).detach()
        sample.ref_logprobs = sample.old_logprobs.clone()
        samples.append(sample)
    episodes = [Episode([s], float(i), 42, 1, {}, None) for i, s in enumerate(samples)]
    cfg = OmegaConf.create(dict(update_epochs=2, clip_ratio=0.2, kl_beta=0.02, max_grad_norm=1.0))
    optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.001)
    initial = actor.model.weight.detach().clone()
    skipped = update_actor(actor, episodes, torch.zeros(2), optimizer, loss_fn, cfg)
    assert not skipped["updated"]
    assert not optimizer.state
    metrics = update_actor(actor, episodes, torch.tensor([-1.0, 1.0]), optimizer, loss_fn, cfg)
    assert metrics["updated"] and metrics["grad_norm"] > 0
    assert not torch.equal(initial, actor.model.weight.detach())
    assert all(not sample.old_logprobs.requires_grad for sample in samples)


def task_info():
    return dict(
        success=False,
        bilateral_grasp_seen=False,
        lifted_by_black=False,
        green_position=[0, 0, 0],
        goal_position=[0.2, 0, 0],
        unexpected_contacts=[],
        max_red_drift_m=0,
    )


def test_rewards_preserve_success_and_penalize_invalid():
    info = task_info()
    assert terminal_reward(info, False, "sparse") == 0
    assert terminal_reward(info, True, "staged") == -1
    info.update(bilateral_grasp_seen=True, lifted_by_black=True)
    assert 0 < terminal_reward(info, False, "staged") < 1
    assert not info["success"]
    info.update(success=True)
    assert terminal_reward(info, False, "sparse") == 1


def test_rollout_same_seed_keeps_invalid_tokens_and_stops_on_success():
    class Env:
        def __init__(self):
            self.seeds = []
            self.info = task_info()
            self.task = SimpleNamespace(info=lambda: self.info)
            self.obs = {"image": np.zeros((32, 32, 3), np.uint8), "proprio": np.zeros(21)}

        def reset(self, seed):
            self.seeds.append(seed)
            self.info = task_info()
            return self.obs

        def step(self, action):
            self.info["success"] = True
            return self.obs, self.info

    env = Env()
    cfg = OmegaConf.create(dict(max_steps=10, execute=8, max_new_tokens=256, kl_beta=0, reward_mode="sparse"))
    norm = SimpleNamespace(metadata={"contract": {"instruction": "move green right"}}, unnormalize=lambda a: a)
    sample = ActionSample({}, torch.tensor([3, 7]), torch.zeros(2), np.zeros((8, 7)), None)
    actor = SimpleNamespace(sample=lambda *a: sample)
    successful = rollout(env, actor, norm, 42, cfg)
    assert successful.steps == 1 and successful.reward == 1
    sample.failure = "invalid chunk"
    invalid = rollout(env, actor, norm, 42, cfg)
    assert env.seeds == [42, 42]
    assert invalid.steps == 0 and invalid.reward == -1
    assert invalid.samples[0].tokens.tolist() == [3, 7]


def test_sampling_eos_and_truncation(monkeypatch):
    actor = toy_actor()
    actor.policy.config = OmegaConf.create({"framework": {"use_state_prompt": False, "action_model": {"action_dim": 7}}})
    actor.policy.action_horizon = 8
    actor.policy.action_model = SimpleNamespace(fast_tokenizer=None)
    actor.interface.processor.tokenizer.pad_token_id = 0
    actor.interface.build_qwenvl_inputs = lambda **kw: {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }

    def generate(**kwargs):
        assert kwargs["generation_config"].top_k == 0
        assert kwargs["generation_config"].top_p == 1
        assert kwargs["use_model_defaults"] is False
        assert kwargs["repetition_penalty"] == 1
        return SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 7]]), scores=(torch.zeros(1, 8),) * 2)

    actor.model.generate = generate
    monkeypatch.setattr("mujoco_task.grpo_policy.strict_decode", lambda *a: np.zeros((1, 8, 7)))
    sample = actor.sample({"image": [], "lang": "test"}, 2)
    assert sample.tokens.tolist() == [3, 7] and sample.failure is None
    assert sample.old_logprobs.shape == (2,)
    actor.model.generate = lambda **kw: SimpleNamespace(
        sequences=torch.tensor([[1, 2, 3, 4]]), scores=(torch.zeros(1, 8),) * 2
    )
    truncated = actor.sample({"image": [], "lang": "test"}, 2)
    assert "without EOS" in truncated.failure
    assert truncated.tokens.tolist() == [3, 4]


@pytest.mark.parametrize(
    "key,value",
    [("group_size", 1), ("temperature", 0), ("execute", 0), ("kl_beta", -1), ("learning_rate", float("nan"))],
)
def test_bad_configuration(key, value):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/mujoco/qwenfast_grpo.yaml")
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
