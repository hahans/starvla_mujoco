"""Opt-in real SFT/GPU wiring test, using synthetic advantages (not an RL result)."""

import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


@pytest.mark.skipif(os.environ.get("RUN_GRPO_GPU_TEST") != "1", reason="Opt-in 3B checkpoint/CUDA test")
def test_real_checkpoint_sampling_backward_and_reload(tmp_path):
    from mujoco_task.env import TransferEnv
    from mujoco_task.evaluate import load_policy
    from mujoco_task.grpo import Episode, load_rlinf, save_checkpoint, update_actor
    from mujoco_task.grpo_policy import FastGRPOPolicy, configure_training

    cfg = OmegaConf.load(os.environ.get("GRPO_GPU_CONFIG", "configs/mujoco/qwenfast_grpo.yaml"))
    if cfg.get("train_mode", "lora") == "full" and torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        pytest.skip("Full 3.8B training validation requires an 80GB-class GPU")
    _, loss_fn, fingerprints = load_rlinf(Path(cfg.rlinf_root))
    torch.manual_seed(42)
    policy, _, norm = load_policy(
        Path(cfg.checkpoint), "cuda", cfg.base_vlm, cfg.fast_tokenizer, torch_dtype=cfg.torch_dtype
    )
    actor = configure_training(policy, cfg)
    env = TransferEnv(image_size=norm.metadata["contract"]["image_size"])
    try:
        examples = []
        for index in range(cfg.get("num_envs", 1)):
            obs = env.reset(seed=cfg.seed + index)
            examples.append(
                {
                    "image": [Image.fromarray(obs["image"])],
                    "state": obs["proprio"],
                    "lang": norm.metadata["contract"]["instruction"] + (" Keep the red block still." if index else ""),
                }
            )
        samples = actor.sample_batch(examples, cfg.max_new_tokens)
        sample = samples[0]
        with torch.no_grad():
            for replay in samples:
                torch.testing.assert_close(actor.logprobs(replay).cpu(), replay.old_logprobs, atol=0.15, rtol=0.05)
            before = actor.logprobs(sample)
            # Cached bf16 generation and parallel teacher forcing differ by rounding.
            torch.testing.assert_close(before.cpu(), sample.old_logprobs, atol=0.15, rtol=0.05)
            reference = actor.reference_logprobs(sample)
            torch.testing.assert_close(before.cpu(), reference, atol=0.15, rtol=0.05)
        sample.ref_logprobs = reference.cpu()
        optimizer = torch.optim.AdamW(
            [p for p in actor.model.parameters() if p.requires_grad], lr=cfg.learning_rate, foreach=False
        )
        full_head_before = None
        lora_before = None
        if cfg.get("train_mode", "lora") == "full":
            full_head_before = actor.model.get_output_embeddings().weight[-2048:].detach().cpu().clone()
        else:
            lora_before = {
                name: p.detach().cpu().clone() for name, p in actor.model.named_parameters() if p.requires_grad
            }
        episode = Episode([sample], 1, cfg.seed, 0, env.task.info(), sample.failure)
        metrics = update_actor(actor, [episode], torch.ones(1), optimizer, loss_fn, cfg)
        assert metrics["grad_norm"] > 0
        with torch.no_grad():
            after = actor.logprobs(sample).cpu()
            frozen = actor.reference_logprobs(sample)
        torch.testing.assert_close(frozen, reference.cpu())
        if full_head_before is not None:
            assert not torch.equal(full_head_before, actor.model.get_output_embeddings().weight[-2048:].detach().cpu())
        else:
            # Near-certain constrained tokens can keep identical rounded bf16
            # log-probs after one small step; verify the actual FP32 adapter update.
            assert any(
                not torch.equal(lora_before[name], p.detach().cpu())
                for name, p in actor.model.named_parameters()
                if p.requires_grad
            )
        saved = save_checkpoint(tmp_path, 1, actor, optimizer, cfg, fingerprints)
        assert (saved / "COMPLETE").is_file()
        assert (
            torch.load(saved / "training_state.pt", map_location="cpu", mmap=True, weights_only=True)["iteration"] == 1
        )
        policy.cpu()
        if actor.reference is not None:
            actor.reference.policy.cpu()
        del optimizer
        torch.cuda.empty_cache()
        restored, _, _ = load_policy(saved, "cuda", cfg.base_vlm, cfg.fast_tokenizer, torch_dtype=cfg.torch_dtype)
        restored_actor = FastGRPOPolicy(
            restored,
            cfg.temperature,
            strict_fast=cfg.get("strict_fast", False),
            repetition_penalty=cfg.get("repetition_penalty", 1.0),
        )
        with torch.no_grad():
            torch.testing.assert_close(restored_actor.logprobs(sample).cpu(), after)
        print(
            {
                "sample_tokens": len(sample.tokens),
                "train_mode": cfg.get("train_mode", "lora"),
                "sample_batch_size": len(samples),
                "valid_action": sample.failure is None,
                "failure": sample.failure,
                "grad_norm": metrics["grad_norm"],
                "max_logprob_difference": float((before.cpu() - sample.old_logprobs).abs().max()),
                "ppo_clip_fraction": metrics["clip_fraction"],
                "importance_sampling_fix": cfg.get("importance_sampling_fix", False),
                "importance_sampling_clip": cfg.get("importance_sampling_clip", 1.0),
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
        )
    finally:
        env.close()


@pytest.mark.skipif(
    os.environ.get("RUN_GRPO_GPU_TEST") != "1" or not os.environ.get("GRPO_AUDIT_CHECKPOINT"),
    reason="Opt-in existing adapter audit requires GRPO_AUDIT_CHECKPOINT",
)
def test_existing_adapter_mixed_observations_match_behavior_probabilities():
    """Check a nonzero trained adapter with distinct images and prompt lengths."""
    from mujoco_task.data import read_json
    from mujoco_task.env import TransferEnv
    from mujoco_task.evaluate import load_policy
    from mujoco_task.grpo_policy import FastGRPOPolicy

    cfg = OmegaConf.load("configs/mujoco/qwenfast_grpo_v2.yaml")
    policy, _, norm = load_policy(
        Path(os.environ["GRPO_AUDIT_CHECKPOINT"]), "cuda", cfg.base_vlm, cfg.fast_tokenizer, "bfloat16"
    )
    actor = FastGRPOPolicy(policy, cfg.temperature, strict_fast=True, repetition_penalty=cfg.repetition_penalty)
    env = TransferEnv(image_size=norm.metadata["contract"]["image_size"])
    try:
        examples = []
        replay_steps = []
        for index, seed in enumerate([0, 1, 4, 6]):
            obs = env.reset(seed=seed)
            trajectory_root = os.environ.get("GRPO_AUDIT_TRAJECTORIES")
            count = 0
            if trajectory_root:
                recorded = read_json(Path(trajectory_root) / f"seed_{seed}.json")
                actions = [action for prediction in recorded["predictions"] for action in prediction["applied"]]
                count = min(index * 80, len(actions))
                for action in actions[:count]:
                    env.step(action, observe=False)
                obs = env.observe()
            replay_steps.append(count)
            examples.append(
                {
                    "image": [Image.fromarray(obs["image"])],
                    "state": obs["proprio"],
                    "lang": norm.metadata["contract"]["instruction"] + " Keep the red block still." * index,
                }
            )
        torch.manual_seed(42)
        samples = actor.sample_batch(examples, cfg.max_new_tokens)
        assert len({s.inputs["input_ids"].shape[1] for s in samples}) > 1
        errors = []
        ratios = []
        with torch.no_grad():
            for sample in samples:
                assert sample.failure is None
                difference = actor.logprobs(sample).cpu() - sample.old_logprobs
                errors.append(difference.abs())
                ratios.append(difference.exp())
        errors = torch.cat(errors)
        ratios = torch.cat(ratios)
        assert errors.max() < 0.5 and errors.mean() < 0.01
        print(
            {
                "mixed_batch_replay_error_max": errors.max().item(),
                "mixed_batch_replay_error_mean": errors.mean().item(),
                "observation_control_steps": replay_steps,
                "behavior_replay_outside_clip_fraction": float(((ratios < 0.9) | (ratios > 1.1)).float().mean()),
            }
        )
    finally:
        env.close()
