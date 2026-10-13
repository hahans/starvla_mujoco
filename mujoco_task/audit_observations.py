"""Compare recorded observations with a replay in the evaluation environment.

No VLA is loaded. Set MUJOCO_GL before starting Python, exactly as for evaluate.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .data import episode_path, read_arrays, read_json, read_video, write_json


def compare_observation(recorded_image: np.ndarray, recorded_state: np.ndarray, observation: dict) -> dict:
    """Measure visual and robot-state differences without declaring a tolerance safe."""
    rendered = observation["image"]
    state = observation["proprio"]
    if rendered.shape != recorded_image.shape or state.shape != recorded_state.shape:
        raise ValueError("Recorded and replayed observation shapes differ")
    delta = np.abs(rendered.astype(np.float64) - recorded_image.astype(np.float64))
    state_delta = np.abs(state.astype(np.float64) - recorded_state.astype(np.float64))
    return {
        "image_exact": bool(np.array_equal(rendered, recorded_image)),
        "image_mae_8bit": float(delta.mean()),
        "image_max_abs_8bit": float(delta.max()),
        "image_changed_pixel_fraction": float(np.any(delta > 0, axis=-1).mean()),
        "recorded_image_sha256": hashlib.sha256(recorded_image.tobytes()).hexdigest(),
        "rendered_image_sha256": hashlib.sha256(rendered.tobytes()).hexdigest(),
        "proprio_max_abs": float(state_delta.max()),
        "proprio_within_1e_6": bool(np.allclose(state, recorded_state, atol=1e-6, rtol=0)),
    }


def main() -> None:
    """Replay recorded expert actions and compare pre-action observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 8, 48], help="Pre-action frames; last always added")
    parser.add_argument("--output", type=Path, required=True, help="New directory for report and comparison images")
    args = parser.parse_args()
    if args.episodes < 1 or min(args.frames) < 0:
        parser.error("episodes must be positive and frames nonnegative")
    if args.output.exists():
        parser.error("Output already exists; use a new directory")

    # Import here so help and numeric unit tests need no OpenGL context.
    import os

    os.environ.setdefault("MUJOCO_GL", "osmesa")  # Match evaluate.py's fallback.
    import mujoco

    from .env import TransferEnv

    root = args.data_root / f"block_{args.split}"
    source = read_json(root / "meta/source.json")
    normalization = read_json(args.data_root / "normalization.json")
    if source["manifest_sha256"] != normalization["manifest_sha256"]:
        raise ValueError("Dataset and normalization manifests differ")
    if source["contract"] != normalization["contract"]:
        raise ValueError("Dataset and normalization contracts differ")
    entries = source["episodes"][: args.episodes]
    if not entries:
        raise ValueError("Empty dataset split")
    env = TransferEnv(image_size=source["contract"]["image_size"])
    try:
        if env.contract() != source["contract"]:
            raise ValueError("Current evaluation environment contract differs from collection")
        from OpenGL import GL

        report = {
            "data_root": str(args.data_root.resolve()),
            "split": args.split,
            "mujoco_gl_env": os.environ.get("MUJOCO_GL"),
            "mujoco_version": mujoco.__version__,
            "manifest_sha256": source["manifest_sha256"],
            "gl_vendor": GL.glGetString(GL.GL_VENDOR).decode(),
            "gl_renderer": GL.glGetString(GL.GL_RENDERER).decode(),
            "gl_version": GL.glGetString(GL.GL_VERSION).decode(),
            "requested_samples": int(env.model.vis.quality.offsamples),
            "actual_samples": int(env.renderer._mjr_context.offSamples),
            "episodes": [],
        }
        args.output.mkdir(parents=True)
        for episode_index, entry in enumerate(entries):
            states, actions = read_arrays(root, episode_index)
            images = read_video(episode_path(root, episode_index, video=True))
            if not len(actions) or len(images) != len(actions) or len(states) != len(actions):
                raise ValueError(f"Invalid recorded frame counts for episode {episode_index}")
            selected = {f for f in args.frames if f < len(actions)} | {len(actions) - 1}
            observations = []
            env.reset(entry["seed"])
            for frame, action in enumerate(actions):
                if frame in selected:
                    observation = env.observe()
                    measurements = compare_observation(images[frame], states[frame], observation)
                    observations.append({"frame": frame, **measurements})
                    comparison = np.concatenate([images[frame], observation["image"]], axis=1)
                    Image.fromarray(comparison).save(args.output / f"seed_{entry['seed']}_frame_{frame}.png")
                _, info = env.step(action, observe=False)
            result = {"seed": entry["seed"], "frames": observations, "expert_replay_success": bool(info["success"])}
            report["episodes"].append(result)
            write_json(args.output / "report.json", report)
            print(result, flush=True)
        comparisons = [frame for episode in report["episodes"] for frame in episode["frames"]]
        report["summary"] = {
            "checked_frames": len(comparisons),
            "all_checked_images_exact": all(frame["image_exact"] for frame in comparisons),
            "all_checked_states_within_1e_6": all(frame["proprio_within_1e_6"] for frame in comparisons),
            "worst_frame_image_mae_8bit": max(frame["image_mae_8bit"] for frame in comparisons),
            "all_expert_replays_successful": all(episode["expert_replay_success"] for episode in report["episodes"]),
            "scope": "Selected frames of recorded expert replay; not a VLA success-rate test",
        }
        write_json(args.output / "report.json", report)
        print(report["summary"], flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
