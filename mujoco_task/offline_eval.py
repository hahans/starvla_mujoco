"""Teacher-forced token CE and generated-action error on recorded observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import TaskDataset, write_json
from .evaluate import checkpoint_run_dir, load_policy


def main() -> None:
    """Diagnose imitation fit separately from local closed-loop success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", help="Local Qwen2.5-VL model directory for native checkpoints")
    parser.add_argument("--fast-tokenizer", help="Local FAST tokenizer directory")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output", type=Path, default=Path("outputs/offline_eval.json"))
    args = parser.parse_args()
    if args.stride < 1 or args.episodes < 1:
        parser.error("stride and episodes must be positive")
    policy, cfg, norm = load_policy(args.checkpoint, args.device, args.base_vlm, args.fast_tokenizer)
    ds = TaskDataset(
        args.data_root / f"block_{args.split}",
        checkpoint_run_dir(args.checkpoint) / "normalization.json",
        cfg.framework.action_model.action_horizon,
        max_episodes=args.episodes,
    )
    total_loss, token_count, errors, failed = 0.0, 0, [], []
    indices = list(range(0, len(ds), args.stride))
    with torch.inference_mode():
        for index in indices:
            example = ds[index]
            result = policy(examples=[example])
            n = int(result["num_action_tokens"])
            total_loss += float(result["action_loss"]) * n
            token_count += n
            try:
                predicted = policy.predict_action([example])["normalized_actions"][0]
                errors.append(norm.unnormalize(predicted) - norm.unnormalize(example["action"]))
            except ValueError as exc:
                failed.append({"frame": index, "error": str(exc)})
            print(f"offline frame {index}/{len(ds)}", flush=True)
    report = {
        "split": args.split,
        "samples": len(indices),
        "token_ce": total_loss / token_count,
        "invalid_generations": failed,
        "invalid_rate": len(failed) / len(indices),
        "closed_loop_success": "not measured by this script",
    }
    if errors:
        err = np.stack(errors)
        report["rmse_rad_m_valid_only"] = np.sqrt((err**2).mean(axis=(0, 1))).tolist()
        report["first_action_rmse_rad_m_valid_only"] = np.sqrt((err[:, 0] ** 2).mean(axis=0)).tolist()
    write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
