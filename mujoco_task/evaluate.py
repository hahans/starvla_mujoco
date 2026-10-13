"""Local MuJoCo closed-loop evaluation of exported QwenFast checkpoints."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from .data import ActionNormalizer, read_json, write_json


def checkpoint_run_dir(checkpoint: Path) -> Path:
    """Locate native run metadata, or the root of a legacy exported checkpoint."""
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.is_dir():
        return checkpoint
    for parent in (checkpoint.parent, checkpoint.parent.parent):
        if (parent / "config.full.yaml").is_file():
            return parent
    raise ValueError("Native checkpoint requires its run's config.full.yaml and normalization.json")


def use_robot_state(cfg, no_state: bool) -> bool:
    """Keep the checkpoint prompt, or drop robot state for an ablation."""
    enabled = False if no_state else bool(cfg.framework.get("use_state_prompt", False))
    cfg.framework.use_state_prompt = enabled
    return enabled


def decoding_options(checkpoint: Path, decoding: str, repetition_penalty: float | None) -> tuple[str, float | None]:
    """Use saved v2 decoding; preserve historical evaluation for SFT and old LoRA."""
    root = checkpoint_run_dir(checkpoint)
    saved = OmegaConf.load(root / "grpo_config.yaml") if (root / "grpo_config.yaml").is_file() else {}
    if decoding == "auto":
        if "repetition_penalty" in saved:
            decoding = "constrained" if saved.get("strict_fast", False) else "aligned"
        else:
            decoding = "legacy"
    if decoding == "legacy":
        if repetition_penalty is not None:
            raise ValueError("Use --decoding aligned/constrained to explicitly override repetition penalty")
        return decoding, None
    penalty = repetition_penalty if repetition_penalty is not None else saved.get("repetition_penalty", 1.0)
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("repetition_penalty must be finite and positive")
    return decoding, float(penalty)


def load_policy(
    checkpoint: Path,
    device: str,
    base_vlm: str | None = None,
    fast_tokenizer: str | None = None,
    torch_dtype: str | None = None,
    sft_checkpoint: Path | None = None,
):
    """Load native SFT weights, a GRPO LoRA adapter, or a legacy HF task export."""
    checkpoint = Path(checkpoint).resolve()
    if (checkpoint / "grpo_metadata.json").is_file():
        if not (checkpoint / "COMPLETE").is_file():
            raise ValueError("Incomplete GRPO checkpoint")
        metadata = read_json(checkpoint / "grpo_metadata.json")
        if metadata.get("train_mode", "lora") == "full":
            return load_policy(checkpoint / "pytorch_model.pt", device, base_vlm, fast_tokenizer, torch_dtype)
        from peft import PeftModel

        source = sft_checkpoint or Path(metadata["sft_checkpoint"])
        policy, cfg, norm = load_policy(source, device, base_vlm, fast_tokenizer, torch_dtype)
        policy.qwen_vl_interface.model = PeftModel.from_pretrained(
            policy.qwen_vl_interface.model, checkpoint / "adapter"
        ).eval()
        return policy, cfg, norm
    root = checkpoint_run_dir(checkpoint)
    native = checkpoint.is_file()
    if native:
        cfg = OmegaConf.load(root / "config.full.yaml")
    else:
        if not (root / "COMPLETE").is_file():
            raise ValueError("Incomplete/missing legacy task checkpoint")
        cfg = OmegaConf.load(root / "task_config.yaml")
        cfg.framework.qwenvl.base_vlm = str(root / "Qwen2.5-VL-Action")
        cfg.framework.action_model.fast_tokenizer = str(root / "fast")
    if base_vlm is not None:
        cfg.framework.qwenvl.base_vlm = base_vlm
    if fast_tokenizer is not None:
        cfg.framework.action_model.fast_tokenizer = fast_tokenizer
    if cfg.framework.name != "QwenFast":
        raise ValueError("Expected a QwenFast checkpoint")
    cfg.framework.qwenvl.attn_implementation = "sdpa"
    if device == "cpu":
        cfg.framework.qwenvl.torch_dtype = "float32"
    elif torch_dtype is not None:
        cfg.framework.qwenvl.torch_dtype = torch_dtype
    from starVLA.model.framework.VLM4A.QwenFast import Qwenvl_Fast

    policy = Qwenvl_Fast(cfg)
    if native:
        if checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint), device="cpu")
        elif checkpoint.suffix == ".pt":
            try:
                state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except RuntimeError as exc:
                message = str(exc).lower()
                if "central directory" in message or "zip archive" in message:
                    size = checkpoint.stat().st_size
                    raise RuntimeError(
                        f"Checkpoint is truncated or corrupt ({checkpoint}, {size} bytes). "
                        "Re-copy the complete .pt from the training machine and compare "
                        "byte sizes (or sha256) before evaluating."
                    ) from exc
                raise
        else:
            raise ValueError("Native checkpoint must be .pt or .safetensors")
        policy.load_state_dict(state, strict=True)
        del state
    policy = policy.to(device).eval()
    norm = ActionNormalizer(read_json(root / "normalization.json"))
    return policy, cfg, norm


def run_group(
    envs: list,
    seeds: list[int],
    predict,
    unnormalize,
    instruction: str,
    execute: int,
    max_steps: int,
    video: bool,
    on_finish=None,
) -> list[tuple[dict, list, list]]:
    """Evaluate one left-padded group, dropping finished episodes from later generates.

    This matches training validation: every active environment is generated together,
    and an episode that succeeds or fails leaves the next batch.
    """
    if len(envs) != len(seeds) or not seeds:
        raise ValueError("Each grouped seed needs one environment")
    observations = [env.reset(seed) for env, seed in zip(envs, seeds, strict=True)]
    states = []
    for env, seed, obs in zip(envs, seeds, observations, strict=True):
        states.append(
            {
                "seed": seed,
                "frames": [obs["image"].copy()] if video else [],
                "records": [],
                "failure": None,
                "info": env.task.info(),
                "step": 0,
                "start": time.monotonic(),
                "result": None,
            }
        )
    active = [i for i, state in enumerate(states) if state["step"] < max_steps and not state["info"]["success"]]
    while active:
        examples = [
            {
                "image": [Image.fromarray(observations[i]["image"])],
                "lang": instruction,
                "state": observations[i]["proprio"],
            }
            for i in active
        ]
        predictions = predict(examples)
        if len(predictions) != len(active):
            raise ValueError("Batched prediction count does not match active environments")
        remaining = []
        for index, prediction in zip(active, predictions, strict=True):
            state = states[index]
            if prediction["failure"]:
                state["failure"] = prediction["failure"]
                _finish_episode(state, on_finish)
                continue
            actions = unnormalize(prediction["actions"])
            limit = min(execute, max_steps - state["step"])
            chunk = actions[:limit]
            record = {
                "step": state["step"],
                "fast_token_ids": prediction["fast_token_ids"],
                "generated_token_ids": prediction["generated_token_ids"],
                "applied": [],
            }
            for action_index, action in enumerate(chunk):
                observations[index], state["info"] = envs[index].step(
                    action, observe=video or action_index + 1 == len(chunk)
                )
                state["step"] += 1
                record["applied"].append(state["info"]["applied_action"])
                if video:
                    state["frames"].append(observations[index]["image"].copy())
                if state["info"]["success"]:
                    break
            state["records"].append(record)
            if state["step"] < max_steps and not state["info"]["success"]:
                remaining.append(index)
            else:
                _finish_episode(state, on_finish)
        active = remaining
    missing = [state["seed"] for state in states if state["result"] is None]
    if missing:
        raise RuntimeError(f"Grouped evaluation dropped seeds {missing}")
    return [(state["result"], state["frames"], state["records"]) for state in states]


def _finish_episode(state: dict, on_finish=None) -> None:
    state["result"] = {
        "seed": state["seed"],
        "success": bool(state["info"]["success"]),
        "steps": state["step"],
        "wall_seconds": time.monotonic() - state["start"],
        "failure": state["failure"],
        "final_info": state["info"],
    }
    if on_finish is not None:
        on_finish(state["result"], state["frames"], state["records"])


def _greedy_rows(predictor, examples: list[dict], max_new_tokens: int) -> list[dict]:
    """Greedy rows from the training left-padded sampler, or a legacy batched predict."""
    if hasattr(predictor, "sample_batch"):
        samples = predictor.sample_batch(examples, max_new_tokens, do_sample=False)
        rows = []
        for sample in samples:
            if sample.failure or sample.actions is None:
                rows.append(
                    {
                        "failure": sample.failure or "invalid_policy_output",
                        "actions": None,
                        "fast_token_ids": None,
                        "generated_token_ids": None,
                    }
                )
                continue
            rows.append(
                {
                    "failure": None,
                    "actions": sample.actions,
                    "fast_token_ids": [int(token) - predictor.action_min for token in sample.tokens[:-1].tolist()],
                    "generated_token_ids": sample.tokens.tolist(),
                }
            )
        return rows
    try:
        prediction = predictor.predict_action(examples)
    except (ValueError, FloatingPointError) as exc:
        failure = f"invalid_policy_output: {exc}"
        return [
            {"failure": failure, "actions": None, "fast_token_ids": None, "generated_token_ids": None} for _ in examples
        ]
    return [
        {
            "failure": None,
            "actions": prediction["normalized_actions"][index],
            "fast_token_ids": prediction["fast_token_ids"][index],
            "generated_token_ids": prediction["generated_token_ids"][index],
        }
        for index in range(len(examples))
    ]


def main() -> None:
    """Run only learned actions with the original simulation clock and success test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-vlm", help="Local Qwen2.5-VL model directory when evaluating native .pt weights")
    parser.add_argument("--fast-tokenizer", help="Local FAST tokenizer directory")
    parser.add_argument("--sft-checkpoint", type=Path, help="Relocated base SFT weights for a GRPO adapter")
    parser.add_argument("--torch-dtype", choices=["float32", "bfloat16"], help="Override model loading precision")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Environments generated together with left padding. 4 matches the fixed seed 0-3 validation batch.",
    )
    parser.add_argument("--seed-start", type=int, default=3000000)
    parser.add_argument("--data-root", type=Path, help="Use recorded split reset seeds")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--execute", type=int, default=1, help="Actions executed before observing again")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=Path("outputs/local_eval"))
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--decoding",
        choices=["auto", "legacy", "aligned", "constrained"],
        default="auto",
        help="auto uses saved v2 decoding, or historical decoding for SFT/old adapters",
    )
    parser.add_argument("--repetition-penalty", type=float, help="Explicit penalty for aligned/constrained decoding")
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not append the 21-D robot state to the prompt, even if the checkpoint was trained with it",
    )
    args = parser.parse_args()
    args.decoding, args.repetition_penalty = decoding_options(args.checkpoint, args.decoding, args.repetition_penalty)
    if args.episodes < 1 or args.max_steps < 1 or args.num_envs < 1:
        parser.error("episodes, max-steps, and num-envs must be positive")
    if args.output.exists():
        raise FileExistsError(f"Use a new evaluation output directory: {args.output}")
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    from .convert import encode_video
    from .env import TransferEnv

    policy, cfg, norm = load_policy(
        args.checkpoint, args.device, args.base_vlm, args.fast_tokenizer, args.torch_dtype, args.sft_checkpoint
    )
    state_prompt = use_robot_state(cfg, args.no_state)
    print(f"robot_state_prompt={state_prompt}", flush=True)
    horizon = int(cfg.framework.action_model.action_horizon)
    predictor = policy
    if args.decoding != "legacy":
        from .grpo_policy import FastGRPOPolicy

        predictor = FastGRPOPolicy(
            policy,
            temperature=1.0,
            strict_fast=args.decoding == "constrained",
            repetition_penalty=args.repetition_penalty,
        )
    if not 1 <= args.execute <= horizon:
        parser.error("execute must be between 1 and action_horizon")
    contract = norm.metadata["contract"]
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if args.data_root:
        source = read_json(args.data_root / f"block_{args.split}/meta/source.json")
        if source["manifest_sha256"] != norm.metadata["manifest_sha256"]:
            raise ValueError("Evaluation dataset differs from checkpoint dataset")
        seeds = [e["seed"] for e in source["episodes"][: args.episodes]]
    env = TransferEnv(image_size=contract["image_size"])
    if env.contract() != contract:
        raise ValueError("MuJoCo version, scene, camera, or controller differs from training data")
    args.output.mkdir(parents=True)
    results = []

    def publish(result: dict, frames: list, records: list) -> None:
        results.append(result)
        write_json(args.output / f"seed_{result['seed']}.json", {"result": result, "predictions": records})
        if frames:
            encode_video(args.output / f"seed_{result['seed']}.mp4", np.stack(frames), 20)
        write_json(
            args.output / "results.json",
            {
                "checkpoint": str(args.checkpoint),
                "decoding": args.decoding,
                "repetition_penalty": (
                    policy.qwen_vl_interface.model.generation_config.repetition_penalty
                    if args.decoding == "legacy"
                    else args.repetition_penalty
                ),
                "torch_dtype": str(next(policy.parameters()).dtype),
                "max_steps": args.max_steps,
                "base_vlm": str(cfg.framework.qwenvl.base_vlm),
                "sft_checkpoint_override": str(args.sft_checkpoint) if args.sft_checkpoint else None,
                "execute": args.execute,
                "use_state_prompt": state_prompt,
                "num_envs": args.num_envs,
                "split": args.split if args.data_root else "new_seeds",
                "episodes": results,
                "success_rate": float(np.mean([item["success"] for item in results])),
            },
        )
        print(result, flush=True)

    try:
        if args.num_envs > 1:
            max_new_tokens = int(cfg.framework.action_model.get("max_new_tokens", 256))
            if (
                (args.checkpoint if args.checkpoint.is_dir() else checkpoint_run_dir(args.checkpoint))
                .joinpath("grpo_config.yaml")
                .is_file()
            ):
                saved_cfg = OmegaConf.load(checkpoint_run_dir(args.checkpoint) / "grpo_config.yaml")
                if "max_new_tokens" in saved_cfg:
                    max_new_tokens = int(saved_cfg.max_new_tokens)
            for offset in range(0, len(seeds), args.num_envs):
                group_seeds = seeds[offset : offset + args.num_envs]
                group_envs = [TransferEnv(image_size=contract["image_size"]) for _ in group_seeds]
                try:
                    run_group(
                        group_envs,
                        group_seeds,
                        lambda examples: _greedy_rows(predictor, examples, max_new_tokens),
                        norm.unnormalize,
                        contract["instruction"],
                        args.execute,
                        args.max_steps,
                        args.video,
                        on_finish=publish,
                    )
                finally:
                    for group_env in group_envs:
                        group_env.close()
            return
        for seed in seeds:
            obs = env.reset(seed)
            frames, records, failure = ([obs["image"].copy()] if args.video else []), [], None
            info = env.task.info()
            step = 0
            start = time.monotonic()
            while step < args.max_steps and not info["success"]:
                example = {
                    "image": [Image.fromarray(obs["image"])],
                    "lang": contract["instruction"],
                    "state": obs["proprio"],
                }
                try:
                    with torch.inference_mode():
                        prediction = predictor.predict_action([example])
                    actions = norm.unnormalize(prediction["normalized_actions"][0])
                except (ValueError, FloatingPointError) as exc:
                    failure = f"invalid_policy_output: {exc}"
                    break
                record = {
                    "step": step,
                    "fast_token_ids": prediction["fast_token_ids"][0],
                    "generated_token_ids": prediction["generated_token_ids"][0],
                    "applied": [],
                }
                for action in actions[: min(args.execute, args.max_steps - step)]:
                    obs, info = env.step(action)
                    step += 1
                    record["applied"].append(info["applied_action"])
                    if args.video:
                        frames.append(obs["image"].copy())
                    if info["success"]:
                        break
                records.append(record)
            result = {
                "seed": seed,
                "success": bool(info["success"]),
                "steps": step,
                "wall_seconds": time.monotonic() - start,
                "failure": failure,
                "final_info": info,
            }
            results.append(result)
            write_json(args.output / f"seed_{seed}.json", {"result": result, "predictions": records})
            if frames:
                encode_video(args.output / f"seed_{seed}.mp4", np.stack(frames), 20)
            write_json(
                args.output / "results.json",
                {
                    "checkpoint": str(args.checkpoint),
                    "decoding": args.decoding,
                    "repetition_penalty": (
                        policy.qwen_vl_interface.model.generation_config.repetition_penalty
                        if args.decoding == "legacy"
                        else args.repetition_penalty
                    ),
                    "torch_dtype": str(next(policy.parameters()).dtype),
                    "max_steps": args.max_steps,
                    "base_vlm": str(cfg.framework.qwenvl.base_vlm),
                    "sft_checkpoint_override": str(args.sft_checkpoint) if args.sft_checkpoint else None,
                    "execute": args.execute,
                    "use_state_prompt": state_prompt,
                    "split": args.split if args.data_root else "new_seeds",
                    "episodes": results,
                    "success_rate": float(np.mean([r["success"] for r in results])),
                },
            )
            print(result, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
