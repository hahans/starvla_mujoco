"""Fixed-seed evaluation and conservative checkpoint selection against the SFT base."""

from __future__ import annotations

import logging

import torch
from omegaconf import OmegaConf

from .grpo import rollout_batch

LOGGER = logging.getLogger(__name__)


def evaluate_actor(
    actor,
    envs: list,
    norm,
    seeds: list[int],
    execute: int,
    max_steps: int,
    max_new_tokens: int,
    split: str | None = None,
) -> dict:
    """Evaluate greedily without consuming training RNG or recording gradient data."""
    if not envs or not seeds:
        raise ValueError("Validation needs at least one environment and seed")
    cfg = OmegaConf.create(
        dict(execute=execute, max_steps=max_steps, max_new_tokens=max_new_tokens, kl_beta=0.0, reward_mode="sparse")
    )
    device = actor.model.device
    devices = (
        [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    )
    was_training = actor.model.training
    episodes = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            for offset in range(0, len(seeds), len(envs)):
                chunk = seeds[offset : offset + len(envs)]
                episodes.extend(
                    rollout_batch(envs[: len(chunk)], actor, norm, chunk, cfg, deterministic=True, store_samples=False)
                )
                LOGGER.info("Validation completed %d/%d episodes", len(episodes), len(seeds))
    finally:
        actor.model.train(was_training)
    successes = sum(bool(e.info["success"]) for e in episodes)
    invalid = sum(e.failure is not None for e in episodes)
    return {
        "split": split,
        "seeds": seeds,
        "execute": execute,
        "max_steps": max_steps,
        "decoding": "constrained" if actor.grammar is not None else "aligned",
        "repetition_penalty": actor.repetition_penalty,
        "torch_dtype": str(next(actor.model.parameters()).dtype),
        "successes": successes,
        "invalid": invalid,
        "success_rate": successes / len(seeds),
        "invalid_rate": invalid / len(seeds),
        "episodes": [
            {
                "seed": e.seed,
                "success": bool(e.info["success"]),
                "steps": e.steps,
                "failure": e.failure,
                "final_info": e.info,
            }
            for e in episodes
        ],
    }


def improves(candidate: dict, incumbent: dict) -> bool:
    """Prefer more successes, then fewer invalid episodes; retain incumbent on ties."""
    # Pre-split-metadata checkpoints used the val split during training.
    if candidate.get("split", "val") != incumbent.get("split", "val"):
        raise ValueError("Cannot select checkpoints across different evaluation split")
    for key in ("seeds", "execute", "max_steps", "decoding", "torch_dtype", "repetition_penalty"):
        if candidate[key] != incumbent[key]:
            raise ValueError(f"Cannot select checkpoints across different evaluation {key}")
    return (candidate["successes"], -candidate["invalid"]) > (incumbent["successes"], -incumbent["invalid"])
