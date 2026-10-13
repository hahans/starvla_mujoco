"""Paired greedy SFT/LoRA evaluation with identical seeds, precision and decoding."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from .data import read_json, write_json
from .evaluate import load_policy
from .grpo_policy import FastGRPOPolicy
from .grpo_validation import evaluate_actor, improves


def main() -> None:
    """Compare existing LoRA checkpoints; select a winner only on the val split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--fast-tokenizer", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/block_lerobot"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--execute", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--decoding", choices=["aligned", "constrained"], default="constrained")
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.episodes, args.num_envs, args.max_steps, args.execute) < 1:
        parser.error("Episode, environment and step counts must be positive")
    if args.output.exists():
        raise FileExistsError(f"Use a new comparison output directory: {args.output}")
    metadata = [read_json(path / "grpo_metadata.json") for path in args.checkpoints]
    source = Path(metadata[0]["sft_checkpoint"]).resolve()
    if any(Path(m["sft_checkpoint"]).resolve() != source or m.get("train_mode", "lora") != "lora" for m in metadata):
        raise ValueError("All comparison checkpoints must be LoRA adapters of the same SFT")
    if any(not (path / "COMPLETE").is_file() for path in args.checkpoints):
        raise ValueError("Cannot compare incomplete checkpoints")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger(__name__)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from peft import PeftModel

    from .env import TransferEnv

    policy, cfg, norm = load_policy(source, "cuda", args.base_vlm, args.fast_tokenizer, torch_dtype="bfloat16")
    if args.execute > policy.action_horizon:
        raise ValueError("execute exceeds SFT action horizon")
    data = read_json(args.data_root / f"block_{args.split}/meta/source.json")
    if data["manifest_sha256"] != norm.metadata["manifest_sha256"]:
        raise ValueError("Evaluation dataset differs from SFT dataset")
    if args.episodes > len(data["episodes"]):
        raise ValueError("Requested more episodes than available")
    seeds = [e["seed"] for e in data["episodes"][: args.episodes]]
    envs = []
    results = []
    try:
        for _ in range(min(args.num_envs, args.episodes)):
            envs.append(TransferEnv(image_size=norm.metadata["contract"]["image_size"]))
        if envs[0].contract() != norm.metadata["contract"]:
            raise ValueError("Evaluation scene differs from SFT contract")
        args.output.mkdir(parents=True)
        winner = None
        for index in range(len(args.checkpoints) + 1):
            checkpoint = source if index == 0 else args.checkpoints[index - 1].resolve()
            if index == 1:
                policy.qwen_vl_interface.model = PeftModel.from_pretrained(
                    policy.qwen_vl_interface.model, checkpoint / "adapter", adapter_name="candidate_1"
                ).eval()
            elif index > 1:
                name = f"candidate_{index}"
                policy.qwen_vl_interface.model.load_adapter(checkpoint / "adapter", adapter_name=name)
                policy.qwen_vl_interface.model.set_adapter(name)
                policy.eval()
            torch.manual_seed(42)
            actor = FastGRPOPolicy(
                policy,
                temperature=1.0,
                strict_fast=args.decoding == "constrained",
                repetition_penalty=args.repetition_penalty,
            )
            logger.info("Evaluating %s on %s seeds %s", checkpoint, args.split, seeds)
            result = evaluate_actor(
                actor,
                envs,
                norm,
                seeds,
                args.execute,
                args.max_steps,
                int(cfg.framework.action_model.max_new_tokens),
                split=args.split,
            )
            result["checkpoint"] = str(checkpoint)
            results.append(result)
            if winner is None or (args.split == "val" and improves(result, winner)):
                winner = result
            write_json(args.output / f"candidate_{index}.json", result)
            summary = {
                "split": args.split,
                "results": results,
                "selected_checkpoint": winner["checkpoint"] if args.split == "val" else None,
            }
            write_json(args.output / "summary.json", summary)
            logger.info("Completed: successes=%d/%d invalid=%d", result["successes"], len(seeds), result["invalid"])
    finally:
        for env in envs:
            env.close()


if __name__ == "__main__":
    main()
