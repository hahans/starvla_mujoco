"""Audit FAST quantization and replay recorded actions without training a VLA."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .data import TaskDataset, read_arrays, write_json


def replay(dataset: TaskDataset, episode: int, actions: np.ndarray) -> dict:
    """Replay recorded/coded actuator targets; no collection expert is invoked."""
    from .env import TransferEnv

    env = TransferEnv(image_size=dataset.source["contract"]["image_size"], render=False)
    if env.contract() != dataset.source["contract"]:
        raise ValueError("Simulator contract differs from collected data")
    entry = dataset.source["episodes"][episode]
    state, _ = read_arrays(dataset.root, episode)
    observation = env.reset(entry["seed"])
    if not np.allclose(observation["proprio"], state[0], atol=1e-6, rtol=0):
        raise ValueError("Reset state differs from demonstration")
    try:
        for action in actions:
            _, info = env.step(action, observe=False)
        return {"seed": entry["seed"], "success": bool(info["success"]), "final_info": info}
    finally:
        env.close()


def main() -> None:
    """Write actionable quantization diagnostics and optional simulation results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mujoco/qwenfast.yaml")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/codec_check.json"))
    args, overrides = parser.parse_known_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(overrides))
    horizon = int(cfg.framework.action_model.action_horizon)
    ds = TaskDataset(
        args.data_root / f"block_{args.split}",
        args.data_root / "normalization.json",
        horizon,
        max_episodes=args.episodes,
    )
    from starVLA.model.framework.VLM4A.fast_utils import strict_decode
    from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model

    codec = get_action_model(cfg)
    errors, counts, results = [], [], []
    for episode in ds.episodes:
        ep = episode["episode_index"]
        _, raw = read_arrays(ds.root, ep)
        # Audit every frame's predicted chunk, matching training's repeat-last padding.
        starts = np.arange(len(raw))[:, None]
        indices = np.minimum(starts + np.arange(horizon), len(raw) - 1)
        normalized = ds.normalizer.normalize(raw[indices])
        tokens = codec.encoder_action2fastoken(normalized)
        reconstructed = ds.normalizer.unnormalize(strict_decode(codec.fast_tokenizer, tokens, horizon, 7))
        errors.append((reconstructed - raw[indices]).reshape(-1, 7))
        counts.extend(map(len, tokens))
        # Reject missing BPE byte characters, not just DCT rounding error.
        from scipy.fft import dct

        coefficients = np.rint(dct(normalized, axis=1, norm="ortho") * codec.fast_tokenizer.scale)
        for sequence, expected in zip(tokens, coefficients):
            decoded = codec.fast_tokenizer.bpe_tokenizer.decode(sequence)
            actual = np.asarray([ord(c) + codec.fast_tokenizer.min_token for c in decoded])
            if not np.array_equal(actual, expected.flatten()):
                raise ValueError("FAST BPE failed to preserve quantized DCT coefficients")
        result = {"episode": ep, "seed": ds.source["episodes"][ep]["seed"]}
        if args.replay:
            result["raw_replay"] = replay(ds, ep, raw)
            # execute=1: first action from each ground-truth observation's coded chunk.
            result["codec_replay_execute1"] = replay(ds, ep, reconstructed[:, 0])
            # execute=horizon: reconstruct consecutive non-overlapping chunks.
            chunked = reconstructed[::horizon].reshape(-1, 7)[: len(raw)]
            result["codec_replay_execute_horizon"] = replay(ds, ep, chunked)
        results.append(result)
        print(f"codec checked episode {ep}", flush=True)
    error = np.concatenate(errors)
    report = {
        "split": args.split,
        "episodes": results,
        "horizon": horizon,
        "fast_scale": codec.fast_tokenizer.scale,
        "fast_min_token": codec.fast_tokenizer.min_token,
        "max_abs_error_rad_m": np.abs(error).max(0).tolist(),
        "rmse_rad_m": np.sqrt((error**2).mean(0)).tolist(),
        "max_action_tokens": max(counts),
        "mean_action_tokens": float(np.mean(counts)),
        "manifest_sha256": ds.source["manifest_sha256"],
        "not_a_policy_evaluation": True,
    }
    write_json(args.output, report)
    print({k: v for k, v in report.items() if k != "episodes"})
    if max(counts) + 1 > cfg.framework.action_model.max_new_tokens:
        raise SystemExit("max_new_tokens is too small, including EOS; inspect report")
    if args.replay and any(
        not row[key]["success"]
        for row in results
        for key in ("raw_replay", "codec_replay_execute1", "codec_replay_execute_horizon")
    ):
        raise SystemExit("Replay failed: inspect codec report before training")


if __name__ == "__main__":
    main()
