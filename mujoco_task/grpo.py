"""Single-GPU Qwen FAST reinforcement learning using the selected RLinf algorithms."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from .data import ActionNormalizer, read_json, write_json
from .evaluate import checkpoint_run_dir, load_policy
from .grpo_policy import ActionSample, FastGRPOPolicy, configure_training

if TYPE_CHECKING:
    from .env import TransferEnv

LOGGER = logging.getLogger(__name__)
IMPLEMENTATION_VERSION = 2


def load_rlinf(root: Path) -> tuple[Callable, Callable, dict]:
    """Import the requested checkout, refusing to silently use another RLinf tree."""
    root = root.resolve()
    if not (root / "rlinf/algorithms/advantages.py").is_file():
        raise ValueError(f"rlinf_root must contain rlinf/algorithms: {root}")
    sys.path.insert(0, str(root))
    package = importlib.import_module("rlinf")
    if Path(package.__file__).resolve().parent != root / "rlinf":
        raise RuntimeError(f"Another RLinf is already imported: {package.__file__}; start a fresh process")
    advantages = importlib.import_module("rlinf.algorithms.advantages")
    losses = importlib.import_module("rlinf.algorithms.losses")
    fingerprints = {}
    for module in (advantages, losses):
        source = Path(module.__file__)
        fingerprints[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    return advantages.compute_grpo_advantages, losses.compute_grpo_actor_loss_fn, fingerprints


def group_advantages(rewards: list[float], group_size: int, compute: Callable) -> torch.Tensor:
    """Adapt RLinf's task/subtask return signature to scalar episode advantages."""
    if group_size < 2 or not rewards or len(rewards) % group_size:
        raise ValueError("Rewards must contain complete groups of at least two trajectories")
    scores = torch.tensor(rewards, dtype=torch.float32)
    if not torch.isfinite(scores).all():
        raise ValueError("Episode rewards must be finite")
    result = compute(rewards=scores, loss_mask=torch.ones(1, len(scores)), group_size=group_size)
    return result[0].reshape(-1)


def terminal_reward(info: dict, invalid: bool, mode: str) -> float:
    """Reward task evidence once per episode; object truth never enters the policy."""
    if invalid:
        return -1.0
    if mode == "sparse":
        return float(info["success"])
    if mode != "staged":
        raise ValueError(f"Unknown reward mode: {mode}")
    # These bonuses guide exploration but never replace the original success flag.
    reward = float(info["success"])
    reward += 0.1 * bool(info["bilateral_grasp_seen"])
    reward += 0.2 * bool(info["lifted_by_black"])
    if info["lifted_by_black"]:
        distance = np.linalg.norm(np.asarray(info["green_position"])[:2] - np.asarray(info["goal_position"])[:2])
        reward += 0.2 * float(np.exp(-distance / 0.05))
    reward -= 0.2 * bool(info["unexpected_contacts"])
    reward -= min(float(info["max_red_drift_m"]) / 0.05, 1.0) * 0.2
    return float(reward)


@dataclass
class Episode:
    """One same-seed group member and its on-policy action tokens."""

    samples: list[ActionSample]
    reward: float
    seed: int
    steps: int
    info: dict
    failure: str | None


def rollout(env: TransferEnv, actor: FastGRPOPolicy, norm: ActionNormalizer, seed: int, cfg: DictConfig) -> Episode:
    """Run learned actions only, terminating on success, invalid FAST, or time limit."""
    obs = env.reset(seed=seed)
    info = env.task.info()
    samples = []
    steps = 0
    failure = None
    while steps < cfg.max_steps and not info["success"]:
        example = {
            "image": [Image.fromarray(obs["image"])],
            "lang": norm.metadata["contract"]["instruction"],
            "state": obs["proprio"],
        }
        sample = actor.sample(example, cfg.max_new_tokens)
        if cfg.kl_beta > 0:
            sample.ref_logprobs = actor.reference_logprobs(sample)
        samples.append(sample)
        if sample.failure:
            failure = sample.failure
            break
        actions = norm.unnormalize(sample.actions)
        for action in actions[: min(cfg.execute, cfg.max_steps - steps)]:
            obs, info = env.step(action)
            steps += 1
            if info["success"]:
                break
    return Episode(samples, terminal_reward(info, failure is not None, cfg.reward_mode), seed, steps, info, failure)


def rollout_batch(
    envs: list[TransferEnv],
    actor: FastGRPOPolicy,
    norm: ActionNormalizer,
    seeds: list[int],
    cfg: DictConfig,
    deterministic: bool = False,
    store_samples: bool = True,
) -> list[Episode]:
    """Collect independent environments with batched VLA inference and stable ordering.

    MuJoCo physics/rendering advances sequentially on CPU. Finished environments
    leave the inference batch; they are never reset into the current trajectories.
    """
    if len(envs) != len(seeds) or not seeds:
        raise ValueError("Each rollout seed needs one independent environment")
    observations = [env.reset(seed=seed) for env, seed in zip(envs, seeds, strict=True)]
    episodes = [Episode([], 0.0, seed, 0, env.task.info(), None) for env, seed in zip(envs, seeds, strict=True)]
    active = list(range(len(envs)))
    rounds = 0
    while active:
        examples = [
            {
                "image": [Image.fromarray(observations[i]["image"])],
                "lang": norm.metadata["contract"]["instruction"],
                "state": observations[i]["proprio"],
            }
            for i in active
        ]
        samples = (
            actor.sample_batch(examples, cfg.max_new_tokens, do_sample=False)
            if deterministic
            else actor.sample_batch(examples, cfg.max_new_tokens)
        )
        remaining = []
        for i, sample in zip(active, samples, strict=True):
            episode = episodes[i]
            if store_samples:
                episode.samples.append(sample)
            if cfg.kl_beta > 0:
                sample.ref_logprobs = actor.reference_logprobs(sample)
            if sample.failure:
                episode.failure = sample.failure
                continue
            actions = norm.unnormalize(sample.actions)[: min(cfg.execute, cfg.max_steps - episode.steps)]
            for index, action in enumerate(actions):
                # Intermediate RGB frames are unused while executing a chunk.
                # Every physics tick and task-success check still runs.
                observations[i], episode.info = envs[i].step(action, observe=index + 1 == len(actions))
                episode.steps += 1
                if episode.info["success"]:
                    break
            if episode.steps < cfg.max_steps and not episode.info["success"]:
                remaining.append(i)
        active = remaining
        rounds += 1
        if deterministic and rounds % 10 == 0:
            LOGGER.info(
                "Validation progress: max control step=%d/%d active=%d successes=%d",
                max(e.steps for e in episodes),
                cfg.max_steps,
                len(active),
                sum(e.info["success"] for e in episodes),
            )
    for episode in episodes:
        episode.reward = terminal_reward(episode.info, episode.failure is not None, cfg.reward_mode)
    return episodes


def training_seed_range(cfg: DictConfig) -> range:
    """Return all possible training reset seeds, including both configured bounds."""
    if cfg.get("train_seed_min") is not None:
        return range(cfg.train_seed_min, cfg.train_seed_max + 1)
    epochs = int(cfg.get("rollout_epoch", 1))
    return range(cfg.seed, cfg.seed + cfg.iterations * epochs * cfg.groups_per_iteration)


def load_validation_seeds(cfg: DictConfig, metadata: dict) -> list[int]:
    """Load fixed evaluation resets from the selected split in manifest order."""
    split = cfg.get("validation_split", "val")
    source = read_json(Path(cfg.validation_data_root) / f"block_{split}/meta/source.json")
    if source["manifest_sha256"] != metadata["manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from SFT normalization source")
    if cfg.validation_episodes > len(source["episodes"]):
        raise ValueError(f"Requested more evaluation episodes than available in {split}")
    seeds = [episode["seed"] for episode in source["episodes"][: cfg.validation_episodes]]
    # Train-split monitoring intentionally shares resets with training.
    if split != "train" and any(seed in training_seed_range(cfg) for seed in seeds):
        raise ValueError("Training reset seeds overlap validation seeds")
    return seeds


def group_reset_seeds(iteration: int, cfg: DictConfig, epoch: int = 0) -> list[int]:
    """Choose one reset seed per group for this rollout pass.

    Pass 0 keeps the historical iteration-keyed draw. Later passes draw again, matching
    RLinf refreshing reset ids at the end of each ``rollout_epoch``. Members of a group
    still share one seed. The RNG is keyed by configuration seed, iteration and pass, so
    resumes do not depend on how many earlier passes were consumed.
    """
    if cfg.get("train_seed_min") is None:
        groups = cfg.groups_per_iteration
        if epoch == 0:
            offset = iteration * groups
        else:
            epochs = int(cfg.get("rollout_epoch", 1))
            offset = cfg.iterations * groups + (iteration * (epochs - 1) + (epoch - 1)) * groups
        return [cfg.seed + offset + group for group in range(groups)]
    entropy = [cfg.seed, iteration] if epoch == 0 else [cfg.seed, iteration, epoch]
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return rng.integers(cfg.train_seed_min, cfg.train_seed_max, size=cfg.groups_per_iteration, endpoint=True).tolist()


def collect_episodes(
    envs: list[TransferEnv],
    actor: FastGRPOPolicy,
    norm: ActionNormalizer,
    iteration: int,
    cfg: DictConfig,
    epoch: int = 0,
) -> list[Episode]:
    """Collect all same-seed groups even when the environment count cuts a group."""
    seeds = [seed for seed in group_reset_seeds(iteration, cfg, epoch) for _ in range(cfg.group_size)]
    episodes = []
    for offset in range(0, len(seeds), len(envs)):
        chunk = seeds[offset : offset + len(envs)]
        episodes.extend(rollout_batch(envs[: len(chunk)], actor, norm, chunk, cfg))
    return episodes


def collect_rollout(
    envs: list[TransferEnv],
    actor: FastGRPOPolicy,
    norm: ActionNormalizer,
    iteration: int,
    cfg: DictConfig,
) -> list[Episode]:
    """Collect ``rollout_epoch`` passes with a fresh scene draw each pass, then return the buffer.

    Each pass is its own GRPO groups of ``group_size``. One actor update consumes the buffer.
    """
    epochs = int(cfg.get("rollout_epoch", 1))
    episodes: list[Episode] = []
    for epoch in range(epochs):
        LOGGER.info(
            "iteration=%d rollout_epoch=%d/%d seeds=%s",
            iteration + 1,
            epoch + 1,
            epochs,
            group_reset_seeds(iteration, cfg, epoch),
        )
        batch = collect_episodes(envs, actor, norm, iteration, cfg, epoch)
        for index, episode in enumerate(batch):
            LOGGER.info(
                "iteration=%d rollout_epoch=%d seed=%d member=%d steps=%d reward=%.4f success=%s failure=%s",
                iteration + 1,
                epoch + 1,
                episode.seed,
                index % cfg.group_size,
                episode.steps,
                episode.reward,
                episode.info["success"],
                episode.failure,
            )
        episodes.extend(batch)
    return episodes


def iter_minibatches(
    episodes: list[Episode],
    advantages: torch.Tensor,
    cfg: DictConfig,
    iteration: int,
) -> list[tuple[list[Episode], torch.Tensor]]:
    """Shuffle trajectories once, then split them into Adam-step sized groups.

    RLinf permutes the rollout with a private generator and takes one optimizer step per
    global batch. A minibatch at least as large as the buffer keeps the single step.
    """
    size = min(int(cfg.get("minibatch_size", len(episodes))), len(episodes))
    if len(episodes) % size != 0:
        raise ValueError("Trajectory count must be divisible by minibatch_size")
    order = list(range(len(episodes)))
    if size < len(episodes):
        generator = torch.Generator()
        generator.manual_seed(int(cfg.get("seed", 0)) + int(iteration))
        order = torch.randperm(len(episodes), generator=generator).tolist()
    batches = []
    for start in range(0, len(episodes), size):
        index = order[start : start + size]
        batches.append(([episodes[i] for i in index], advantages[index]))
    return batches


def update_actor(
    actor: FastGRPOPolicy,
    episodes: list[Episode],
    advantages: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    cfg: DictConfig,
    iteration: int = 0,
) -> dict:
    """Accumulate a per-trajectory token mean; never backpropagate through rollout."""
    has_advantage = bool(torch.any(advantages != 0))
    if not has_advantage and cfg.kl_beta == 0:
        return {"updated": False, "loss": 0.0, "kl": 0.0, "grad_norm": 0.0, "optimizer_steps": 0}
    metrics = {
        "updated": False,
        "loss": 0.0,
        "kl": 0.0,
        "grad_norm": 0.0,
        "kl_only": not has_advantage,
        "policy_loss": 0.0,
        "ratio": 0.0,
        "clip_fraction": 0.0,
        "replay_logprob_abs_max": 0.0,
        "replay_logprob_abs_mean": 0.0,
        "behavior_replay_outside_clip_fraction": 0.0,
        "importance_weight_mean": 0.0,
        "optimizer_steps": 0,
    }
    batches = iter_minibatches(episodes, advantages, cfg, iteration)
    scale = 1 / (len(batches) * cfg.update_epochs)
    replay_scale = 1 / len(batches)
    actor.model.train()
    grad_norms = []
    for epoch in range(cfg.update_epochs):
        for batch_episodes, batch_advantages in batches:
            optimizer.zero_grad(set_to_none=True)
            for episode, advantage in zip(batch_episodes, batch_advantages, strict=True):
                token_count = sum(len(sample.tokens) for sample in episode.samples)
                for sample in episode.samples:
                    logprobs = actor.logprobs(sample)
                    behavior = sample.old_logprobs.to(logprobs.device)
                    weight = len(sample.tokens) / token_count / len(batch_episodes)
                    if epoch == 0:
                        differences = logprobs.detach() - behavior
                        error = float(differences.abs().max())
                        metrics["replay_logprob_abs_max"] = max(metrics["replay_logprob_abs_max"], error)
                        metrics["replay_logprob_abs_mean"] += float(differences.abs().mean()) * weight * replay_scale
                        numerical_ratio = differences.exp()
                        outside_clip = (numerical_ratio < 1 - cfg.clip_ratio) | (numerical_ratio > 1 + cfg.clip_ratio)
                        metrics["behavior_replay_outside_clip_fraction"] += (
                            float(outside_clip.float().mean()) * weight * replay_scale
                        )
                        # Freeze this minibatch's pre-step forward. Later epochs must not treat
                        # parameter changes as rollout/recompute mismatch.
                        sample.replay_logprobs = logprobs.detach().cpu()
                    if cfg.get("importance_sampling_fix", False):
                        correction = importance_weights(
                            sample.replay_logprobs.to(logprobs.device),
                            behavior,
                            cfg.get("importance_sampling_clip", 1.0),
                        )
                    else:
                        correction = torch.ones_like(behavior)
                    loss, loss_metrics = loss_fn(
                        logprobs=logprobs,
                        old_logprobs=behavior,
                        advantages=correction * advantage.to(logprobs.device),
                        clip_ratio_low=cfg.clip_ratio,
                        clip_ratio_high=cfg.clip_ratio,
                        loss_mask=torch.ones_like(logprobs, dtype=torch.bool),
                    )
                    kl = torch.zeros((), device=logprobs.device)
                    if cfg.kl_beta > 0:
                        if sample.ref_logprobs is None:
                            raise ValueError("KL enabled but rollout has no frozen SFT reference probabilities")
                        kl = bounded_reference_kl(logprobs, sample.ref_logprobs.to(logprobs.device)).mean()
                    combined = (loss + cfg.kl_beta * kl) * weight
                    if not torch.isfinite(combined):
                        raise FloatingPointError("Non-finite GRPO loss; no optimizer step applied")
                    combined.backward()
                    metrics["loss"] += float(combined.detach()) * scale
                    metrics["kl"] += float(kl.detach()) * weight * scale
                    metrics["policy_loss"] += float(loss.detach()) * weight * scale
                    metrics["importance_weight_mean"] += float(correction.mean()) * weight * scale
                    for name in ("ratio", "clip_fraction"):
                        metrics[name] += float(loss_metrics[f"actor/{name}"]) * weight * scale
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in actor.model.parameters() if p.requires_grad], cfg.max_grad_norm, error_if_nonfinite=True
            )
            # A zero-advantage batch can still have a useful KL gradient. Conversely,
            # do not let Adam momentum move the policy when every gradient is zero.
            grad_norms.append(float(grad_norm))
            if grad_norm > 0:
                optimizer.step()
                metrics["updated"] = True
                metrics["optimizer_steps"] += 1
    metrics["grad_norm"] = float(np.mean(grad_norms)) if grad_norms else 0.0
    optimizer.zero_grad(set_to_none=True)
    actor.model.eval()
    return metrics


def bounded_reference_kl(logprobs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Sampled k3 KL with RLinf's finite bounds and stable small-delta arithmetic."""
    delta = (reference.float() - logprobs.float()).clamp(-20, 20)
    return (torch.expm1(delta) - delta).clamp(0, 10)


def importance_weights(recompute: torch.Tensor, rollout: torch.Tensor, clip: float) -> torch.Tensor:
    """Match RLinf's importance-sampling fix.

    PPO keeps the rollout log-probabilities as ``old_logprobs``. Advantages are
    multiplied by ``clamp(exp(recompute - rollout), min=clip)``. The weight is
    detached, has no upper cap, and never aborts the update.
    """
    return (recompute.float() - rollout.float()).detach().exp().clamp(min=float(clip))


def save_checkpoint(
    output: Path,
    iteration: int,
    actor: FastGRPOPolicy,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    fingerprints: dict,
    validation_state: dict | None = None,
) -> Path:
    """Atomically save LoRA/full weights, optimizer and RNG without overwriting SFT."""
    target = output / "checkpoints" / f"iteration_{iteration:06d}"
    partial = target.with_suffix(".partial")
    if target.exists() or partial.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    mode = cfg.get("train_mode", "lora")
    if mode == "full":
        size = sum(t.numel() * t.element_size() for t in actor.policy.state_dict().values())
        size += sum(
            t.numel() * t.element_size()
            for state in optimizer.state.values()
            for t in state.values()
            if torch.is_tensor(t)
        )
        if shutil.disk_usage(output).free < size + 64 * 1024**2:
            raise OSError(f"Full checkpoint requires at least {size / 1024**3:.1f} GiB free disk space")
    partial.mkdir(parents=True)
    if mode == "lora":
        actor.model.save_pretrained(partial / "adapter")
    else:
        torch.save(actor.policy.state_dict(), partial / "pytorch_model.pt")
    write_json(
        partial / "grpo_metadata.json",
        {
            "sft_checkpoint": str(Path(cfg.checkpoint).resolve()),
            "train_mode": mode,
            "iteration": iteration,
            "implementation_version": IMPLEMENTATION_VERSION,
            "rlinf_sources": fingerprints,
        },
    )
    OmegaConf.save(cfg, partial / "grpo_config.yaml")
    if validation_state is not None:
        write_json(partial / "validation_state.json", validation_state)
    root = checkpoint_run_dir(Path(cfg.checkpoint))
    for name in ("normalization.json", "config.full.yaml"):
        shutil.copy2(root / name, partial / name)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        partial / "training_state.pt",
    )
    (partial / "COMPLETE").write_text(f"GRPO {mode} weights and training state complete\n")
    partial.rename(target)
    return target


def restore_checkpoint(
    actor: FastGRPOPolicy,
    optimizer: torch.optim.Optimizer,
    checkpoint: Path,
) -> int:
    """Restore only the actor/optimizer/RNG; retain the original frozen SFT reference."""
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("Cannot resume an incomplete checkpoint")
    metadata = read_json(checkpoint / "grpo_metadata.json")
    if metadata.get("implementation_version") != IMPLEMENTATION_VERSION:
        raise ValueError("Cannot exactly resume an older GRPO implementation; start the corrected run from SFT")
    if metadata.get("train_mode", "lora") == "lora":
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(actor.model, load_file(str(checkpoint / "adapter/adapter_model.safetensors")))
    else:
        actor.policy.load_state_dict(
            torch.load(checkpoint / "pytorch_model.pt", map_location="cpu", weights_only=True), strict=True
        )
    state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
    if int(state["iteration"]) != int(metadata["iteration"]):
        raise ValueError("Checkpoint iteration metadata and training state disagree")
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if actor.model.device.type == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return int(state["iteration"])


def validate_config(cfg: DictConfig) -> None:
    """Fail early on invalid group sizes, rollout lengths and optimizer settings."""
    for key in (
        "iterations",
        "groups_per_iteration",
        "max_steps",
        "max_new_tokens",
        "execute",
        "update_epochs",
        "save_interval",
        "lora_rank",
        "lora_alpha",
    ):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not isinstance(cfg.group_size, int) or cfg.group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    rollout_epoch = cfg.get("rollout_epoch", 1)
    if not isinstance(rollout_epoch, int) or rollout_epoch < 1:
        raise ValueError("rollout_epoch must be a positive integer")
    minibatch_size = cfg.get("minibatch_size", None)
    if minibatch_size is not None:
        if not isinstance(minibatch_size, int) or minibatch_size < 1:
            raise ValueError("minibatch_size must be a positive integer")
        buffer = cfg.groups_per_iteration * cfg.group_size * rollout_epoch
        if buffer > minibatch_size and buffer % minibatch_size != 0:
            raise ValueError("Trajectory buffer must be divisible by minibatch_size")
    for key in ("temperature", "learning_rate", "max_grad_norm"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    penalty = cfg.get("repetition_penalty", 1.0)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    if cfg.get("importance_sampling_fix", False):
        importance_clip = cfg.get("importance_sampling_clip", 1.0)
        if not np.isfinite(importance_clip) or importance_clip < 0:
            raise ValueError("importance_sampling_clip must be finite and nonnegative")
    if not np.isfinite(cfg.kl_beta) or cfg.kl_beta < 0 or not 0 < cfg.clip_ratio < 1:
        raise ValueError("Require finite kl_beta >= 0 and 0 < clip_ratio < 1")
    for key in ("validation_interval", "early_stop_patience"):
        value = cfg.get(key, 0)
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if cfg.get("validation_interval", 0):
        count = cfg.get("validation_episodes", 0)
        if not isinstance(count, int) or count < 1:
            raise ValueError("Enabled validation requires integer validation_episodes >= 1")
    if cfg.get("validation_split", "val") not in ("train", "val"):
        raise ValueError("validation_split must be train or val")
    if cfg.reward_mode not in ("sparse", "staged"):
        raise ValueError("reward_mode must be sparse or staged")
    if cfg.device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if cfg.torch_dtype not in ("float32", "bfloat16"):
        raise ValueError("torch_dtype must be float32 or bfloat16")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    lower, upper = cfg.get("train_seed_min"), cfg.get("train_seed_max")
    if lower is not None or upper is not None:
        if any(type(value) is not int or not 0 <= value <= np.iinfo(np.int64).max for value in (lower, upper)):
            raise ValueError("train_seed_min and train_seed_max must both be nonnegative int64 integers")
        if lower > upper:
            raise ValueError("train_seed_min must be <= train_seed_max")
    if cfg.get("train_mode", "lora") not in ("lora", "full"):
        raise ValueError("train_mode must be lora or full")
    if cfg.get("train_mode", "lora") == "full" and cfg.torch_dtype != "float32":
        raise ValueError("Full training requires torch_dtype=float32 master weights; forward uses bf16 autocast")
    count = cfg.get("num_envs", 1)
    if not isinstance(count, int) or not 1 <= count <= cfg.groups_per_iteration * cfg.group_size:
        raise ValueError("num_envs must be between 1 and the number of trajectories per iteration")


def main() -> None:
    """Train or check assets with YAML plus explicit key=value overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/mujoco/qwenfast_grpo.yaml"))
    parser.add_argument("--check", action="store_true", help="Validate local assets/scene without loading the VLA")
    parser.add_argument("--resume", type=Path, help="Resume an iteration checkpoint into a NEW output directory")
    parser.add_argument("overrides", nargs="*", help="Static config overrides: key=value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base_cfg = OmegaConf.load(args.config)
    OmegaConf.set_struct(base_cfg, True)
    cfg = OmegaConf.merge(base_cfg, OmegaConf.from_dotlist(args.overrides))
    validate_config(cfg)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This GRPO entrypoint is single-process; launch it without torchrun/accelerate")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from .env import TransferEnv

    advantage_fn, loss_fn, fingerprints = load_rlinf(Path(cfg.rlinf_root))
    checkpoint = Path(cfg.checkpoint).resolve()
    root = checkpoint_run_dir(checkpoint)
    if not checkpoint.is_file() or checkpoint.suffix not in (".pt", ".safetensors"):
        raise ValueError("Start GRPO from a native SFT .pt/.safetensors checkpoint")
    if checkpoint.suffix == ".pt" and not zipfile.is_zipfile(checkpoint):
        raise ValueError(f"Truncated or invalid PyTorch checkpoint: {checkpoint}")
    sft_cfg = OmegaConf.load(root / "config.full.yaml")
    metadata = read_json(root / "normalization.json")
    if not 1 <= cfg.execute <= sft_cfg.framework.action_model.action_horizon:
        raise ValueError("execute must be within the SFT action_horizon")
    if cfg.get("strict_fast", False):
        action_cfg = sft_cfg.framework.action_model
        minimum_tokens = 4 * action_cfg.action_horizon * action_cfg.action_dim + 1
        if cfg.max_new_tokens < minimum_tokens:
            raise ValueError(f"Strict FAST needs max_new_tokens >= {minimum_tokens}")
    validation_seeds = []
    if cfg.get("validation_interval", 0):
        validation_seeds = load_validation_seeds(cfg, metadata)
        LOGGER.info("Fixed evaluation split=%s seeds=%s", cfg.get("validation_split", "val"), validation_seeds)
    if not cfg.base_vlm or not (Path(cfg.base_vlm) / "config.json").is_file():
        raise ValueError("base_vlm must be a local Qwen Action model directory")
    if not cfg.fast_tokenizer or not (Path(cfg.fast_tokenizer) / "tokenizer.json").is_file():
        raise ValueError("fast_tokenizer must be a local FAST tokenizer directory")
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fix the runtime or explicitly select device=cpu")
    if cfg.get("train_mode", "lora") == "lora":
        import peft  # noqa: F401 -- check dependency before allocating the model

    env = TransferEnv(image_size=metadata["contract"]["image_size"])
    envs = [env]
    try:
        if env.contract() != metadata["contract"]:
            raise ValueError("MuJoCo scene/version/camera/controller differs from the SFT contract")
        if args.check:
            env.reset(seed=group_reset_seeds(0, cfg)[0])
            LOGGER.info("Assets, rendering, simulation contract and RLinf imports OK: %s", fingerprints)
            return
        output = Path(cfg.output).resolve()
        if output.exists():
            raise FileExistsError(f"Use a new output directory: {output}")
        start = 0
        if args.resume:
            if not (args.resume / "COMPLETE").is_file():
                raise ValueError("Cannot resume an incomplete checkpoint")
            previous = OmegaConf.load(args.resume / "grpo_config.yaml")
            # Changing rollout/optimization semantics is a new experiment, not an exact resume.
            for key in cfg:
                if key not in ("iterations", "output", "save_interval", "early_stop_patience") and cfg[
                    key
                ] != previous.get(key, {"train_mode": "lora", "num_envs": 1, "validation_split": "val"}.get(key)):
                    raise ValueError(f"Resume config differs at {key}")
            saved_meta = read_json(args.resume / "grpo_metadata.json")
            if saved_meta.get("implementation_version") != IMPLEMENTATION_VERSION:
                raise ValueError("Cannot exactly resume an older GRPO implementation; start the corrected run from SFT")
            if saved_meta["rlinf_sources"] != fingerprints:
                raise ValueError("RLinf sources changed since the saved checkpoint")
            start = int(saved_meta["iteration"])
            if cfg.iterations <= start:
                raise ValueError("iterations must exceed the resumed iteration")
        torch.manual_seed(cfg.seed)
        policy, _, norm = load_policy(
            checkpoint, cfg.device, cfg.base_vlm, cfg.fast_tokenizer, torch_dtype=cfg.torch_dtype
        )
        actor = configure_training(policy, cfg)
        optimizer = torch.optim.AdamW(
            [p for p in actor.model.parameters() if p.requires_grad],
            lr=cfg.learning_rate,
            weight_decay=0.0,
            foreach=False,
        )
        if args.resume:
            restored_iteration = restore_checkpoint(actor, optimizer, args.resume)
            if restored_iteration != start:
                raise ValueError("Checkpoint iteration metadata and training state disagree")
        output.mkdir(parents=True)
        OmegaConf.save(cfg, output / "grpo_config.yaml")
        write_json(output / "rlinf_sources.json", fingerprints)
        for _ in range(cfg.get("num_envs", 1) - 1):
            envs.append(TransferEnv(image_size=metadata["contract"]["image_size"]))
        validation_state = None
        if cfg.get("validation_interval", 0):
            from .grpo_validation import evaluate_actor, improves

            if args.resume:
                validation_state = read_json(args.resume / "validation_state.json")
            else:
                LOGGER.info("Evaluating frozen initial SFT baseline on fixed validation seeds")
                baseline = evaluate_actor(
                    actor,
                    envs,
                    norm,
                    validation_seeds,
                    cfg.execute,
                    cfg.max_steps,
                    cfg.max_new_tokens,
                    split=cfg.get("validation_split", "val"),
                )
                validation_state = {
                    "best_checkpoint": str(checkpoint),
                    "best_iteration": 0,
                    "best_metrics": baseline,
                    "baseline_metrics": baseline,
                    "stale_checks": 0,
                }
            write_json(output / "best_checkpoint.json", validation_state)
        trainable = sum(p.numel() for p in actor.model.parameters() if p.requires_grad)
        LOGGER.info(
            "train_mode=%s trainable_parameters=%d num_envs=%d", cfg.get("train_mode", "lora"), trainable, len(envs)
        )
        for iteration in range(start, cfg.iterations):
            started = time.monotonic()
            if cfg.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            episodes = collect_rollout(envs, actor, norm, iteration, cfg)
            rollout_seconds = time.monotonic() - started
            rewards = [episode.reward for episode in episodes]
            advantages = group_advantages(rewards, cfg.group_size, advantage_fn)
            metrics = update_actor(actor, episodes, advantages, optimizer, loss_fn, cfg, iteration)
            metrics.update(
                {
                    "iteration": iteration + 1,
                    "rollout_epoch": int(cfg.get("rollout_epoch", 1)),
                    "trajectories": len(episodes),
                    "train_mode": cfg.get("train_mode", "lora"),
                    "num_envs": len(envs),
                    "trainable_parameters": trainable,
                    "rollout_seconds": rollout_seconds,
                    "env_steps_per_second": sum(e.steps for e in episodes) / max(rollout_seconds, 1e-9),
                    "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3 if cfg.device == "cuda" else 0,
                    "peak_gpu_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3 if cfg.device == "cuda" else 0,
                    "reward_mean": float(np.mean(rewards)),
                    "success_rate": float(np.mean([e.info["success"] for e in episodes])),
                    "invalid_rate": float(np.mean([e.failure is not None for e in episodes])),
                    "nonzero_advantage_fraction": float((advantages != 0).float().mean()),
                    "wall_seconds": time.monotonic() - started,
                    "episodes": [
                        {
                            "rollout_epoch": index // (cfg.groups_per_iteration * cfg.group_size) + 1,
                            "seed": episode.seed,
                            "reward": episode.reward,
                            "steps": episode.steps,
                            "failure": episode.failure,
                            "info": episode.info,
                        }
                        for index, episode in enumerate(episodes)
                    ],
                }
            )
            checked = False
            if validation_state is not None and (
                (iteration + 1) % cfg.validation_interval == 0 or iteration + 1 == cfg.iterations
            ):
                checked = True
                validation = evaluate_actor(
                    actor,
                    envs,
                    norm,
                    validation_seeds,
                    cfg.execute,
                    cfg.max_steps,
                    cfg.max_new_tokens,
                    split=cfg.get("validation_split", "val"),
                )
                improved = improves(validation, validation_state["best_metrics"])
                if improved:
                    validation_state.update(
                        best_checkpoint=str(output / "checkpoints" / f"iteration_{iteration + 1:06d}"),
                        best_iteration=iteration + 1,
                        best_metrics=validation,
                        stale_checks=0,
                    )
                else:
                    validation_state["stale_checks"] += 1
                metrics["validation"] = validation
                metrics["validation_improved"] = improved
                LOGGER.info(
                    "Validation success=%.3f; best iteration=%d (0 means SFT); stale checks=%d",
                    validation["success_rate"],
                    validation_state["best_iteration"],
                    validation_state["stale_checks"],
                )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            LOGGER.info(
                "iteration=%d success=%.3f invalid=%.3f updated=%s loss=%.6f seconds=%.1f",
                iteration + 1,
                metrics["success_rate"],
                metrics["invalid_rate"],
                metrics["updated"],
                metrics["loss"],
                metrics["wall_seconds"],
            )
            if checked or (iteration + 1) % cfg.save_interval == 0 or iteration + 1 == cfg.iterations:
                save_checkpoint(output, iteration + 1, actor, optimizer, cfg, fingerprints, validation_state)
                if validation_state is not None:
                    write_json(output / "best_checkpoint.json", validation_state)
            if (
                checked
                and cfg.get("early_stop_patience", 0) > 0
                and validation_state["stale_checks"] >= cfg.early_stop_patience
            ):
                LOGGER.info(
                    "Early stopping: no validation improvement; selected %s", validation_state["best_checkpoint"]
                )
                break
    finally:
        for instance in envs:
            instance.close()


if __name__ == "__main__":
    main()
