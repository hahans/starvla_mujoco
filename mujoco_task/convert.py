"""Convert collected HDF5 episodes to portable LeRobot v2.1 repositories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .data import DATA_PATH, VIDEO_KEY, VIDEO_PATH, episode_path, read_json, write_json


def stats(array: np.ndarray) -> dict:
    """Compute LeRobot scalar/vector statistics."""
    array = np.asarray(array, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [len(array)],
    }


def encode_video(path: Path, images: np.ndarray, fps: int) -> None:
    """Write lossless RGB H.264, preserving the original 128px observations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width, stream.height = images.shape[2], images.shape[1]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "fast", "g": "20"}
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_lines(path: Path, rows: list[dict]) -> None:
    """Write JSONL without non-finite values."""
    import json

    path.write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))


def read_subtasks(h5: h5py.File, n: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Read and validate frame-aligned, confirmed planner labels from HDF5."""
    group = h5["planner"]
    indices = group["subtask_index"][:]
    boundaries = group["subtask_boundary"][:]
    review = group["needs_review"][:]
    ids, goals = group["subtask_id"].asstr()[:], group["subgoal"].asstr()[:]
    if any(array.shape != (n,) for array in (indices, boundaries, review, ids, goals)):
        raise ValueError("Planner labels must align one-to-one with action frames")
    if indices.dtype.kind not in "iu" or boundaries.dtype.kind != "b" or review.dtype.kind != "b":
        raise ValueError("Invalid planner index/boundary/review types")
    if review.any():
        raise ValueError("Unreviewed planner labels cannot enter confirmed LeRobot training data")
    verification = json.loads(group.attrs["verification"])
    if verification.get("quality_status") != "verified_by_physics_replay" or verification.get("frames_checked") != n:
        raise ValueError("Missing full-episode planner verification")
    unique = np.unique(indices)
    if len(unique) not in (3, 4) or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("Planner indices must cover three or four consecutive subtask IDs")
    if indices[0] != 0 or not np.isin(np.diff(indices.astype(np.int64)), (0, 1)).all():
        raise ValueError("Planner subtask order is invalid")
    expected_boundaries = np.r_[True, indices[1:] != indices[:-1]]
    if not np.array_equal(boundaries, expected_boundaries):
        raise ValueError("Planner boundary flags disagree with frame labels")
    tasks = []
    for index in unique:
        selected = indices == index
        task_ids, texts = set(ids[selected]), set(goals[selected])
        if len(task_ids) != 1 or len(texts) != 1 or not next(iter(task_ids)) or not next(iter(texts)):
            raise ValueError("Planner subtask text or ID changes within an index")
        tasks.append({"task_index": int(index), "subtask_id": next(iter(task_ids)), "task": next(iter(texts))})
    if len({t["task"] for t in tasks}) != len(tasks) or len({t["subtask_id"] for t in tasks}) != len(tasks):
        raise ValueError("Duplicate planner subtask text or ID")
    return indices.astype(np.int64), boundaries, tasks


def convert(source: Path, destination: Path, limit: int | None = None, task_mode: str = "auto") -> dict:
    """Validate source hashes and export all declared splits atomically.

    Args:
        source: Collection directory containing manifest.json and episode HDF5s.
        destination: New directory; an existing directory is never overwritten.
        limit: Optional smoke-test maximum episodes per split.
        task_mode: auto detects planner labels; subtask requires them; global uses the full task.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    manifest = read_json(source / "manifest.json")
    digest = hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
    contract = manifest["contract"]
    if (contract["action_dim"], contract["proprio_dim"], contract["control_hz"]) != (7, 21, 20):
        raise ValueError("This adapter requires the collected 7-action/21-state/20-Hz contract")
    entries = manifest["episodes"]
    if not entries or not any(e["split"] == "train" for e in entries):
        raise ValueError("At least one train episode is required to fit normalization")
    if task_mode not in ("auto", "global", "subtask"):
        raise ValueError("task_mode must be auto, global or subtask")
    auto_detect = task_mode == "auto"
    if auto_detect:
        first_path = (source / entries[0]["path"]).resolve()
        if not first_path.is_relative_to(source):
            raise ValueError("Episode path escapes source directory")
        with h5py.File(first_path, "r") as first:
            task_mode = "subtask" if "planner" in first else "global"
    if any(e["split"] not in ("train", "val", "test") for e in entries):
        raise ValueError("Unknown split in manifest")
    fingerprints = [e["sha256"] for e in entries]
    seeds = [e["seed"] for e in entries]
    if len(set(fingerprints)) != len(entries) or len(set(seeds)) != len(entries):
        raise ValueError("Duplicate episode hash or reset seed: possible split leakage")
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"Unfinished conversion exists: inspect/remove {staging}")
    staging.mkdir(parents=True)
    summary, train_actions = {}, []
    subtask_definitions = None
    try:
        for split in ("train", "val", "test"):
            selected = [e for e in entries if e["split"] == split]
            selected = selected[:limit] if limit is not None else selected
            if not selected:
                continue
            root = staging / f"block_{split}"
            (root / "meta").mkdir(parents=True)
            episodes, episode_stats, source_episodes = [], [], []
            global_index = 0
            for episode, entry in enumerate(selected):
                path = (source / entry["path"]).resolve()
                if not path.is_relative_to(source):
                    raise ValueError("Episode path escapes source directory")
                if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Source checksum mismatch: {path}")
                with h5py.File(path) as h5:
                    if auto_detect and ("planner" in h5) != (task_mode == "subtask"):
                        raise ValueError("Cannot auto-convert a mixture of labeled and unlabeled episodes")
                    images, state, action, times = (h5[k][:] for k in ("image", "proprio", "action", "time"))
                    n = len(action)
                    size = contract["image_size"]
                    if (
                        images.shape != (n, size, size, 3)
                        or images.dtype != np.uint8
                        or state.shape != (n, 21)
                        or action.shape != (n, 7)
                        or n != entry["length"]
                        or n < 1
                    ):
                        raise ValueError(f"Invalid shapes/dtypes: {path}")
                    if not np.isfinite(state).all() or not np.isfinite(action).all():
                        raise ValueError(f"Non-finite state/action: {path}")
                    if not np.allclose(times, np.arange(n) / 20, atol=1e-7, rtol=0):
                        raise ValueError(f"Invalid observation-before-action timestamps: {path}")
                    if int(h5.attrs["seed"]) != entry["seed"] or h5.attrs["instruction"] != contract["instruction"]:
                        raise ValueError("Episode instruction/seed differs from manifest")
                    if not entry["success"]:
                        raise ValueError("Dataset contains an unsuccessful demonstration")
                    if np.any(action[:, 6] < -1e-7) or np.any(action[:, 6] > 0.035001):
                        raise ValueError("Unexpected gripper units")
                    task_indices = np.zeros(n, dtype=np.int64)
                    if task_mode == "subtask":
                        if "planner" not in h5:
                            raise ValueError(f"Missing planner labels: {path}")
                        task_indices, subtask_boundaries, definitions = read_subtasks(h5, n)
                        if subtask_definitions is None:
                            subtask_definitions = definitions
                        elif subtask_definitions != definitions:
                            raise ValueError("Subtask vocabulary differs between episodes or splits")
                columns = {
                    "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 21)),
                    "action": pa.array(action.tolist(), type=pa.list_(pa.float32(), 7)),
                    "timestamp": pa.array(times, type=pa.float32()),
                    "frame_index": pa.array(np.arange(n), type=pa.int64()),
                    "episode_index": pa.array(np.full(n, episode), type=pa.int64()),
                    "index": pa.array(np.arange(global_index, global_index + n), type=pa.int64()),
                    "task_index": pa.array(task_indices),
                }
                if task_mode == "subtask":
                    columns["subtask_boundary"] = pa.array(subtask_boundaries)
                parquet = episode_path(root, episode)
                parquet.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.table(columns), parquet, compression="zstd")
                encode_video(episode_path(root, episode, True), images, 20)
                ep_stats = {"observation.state": stats(state), "action": stats(action)}
                pixels = images.astype(np.float32) / 255
                ep_stats[VIDEO_KEY] = {
                    op: getattr(pixels, op)(axis=(0, 1, 2)).reshape(3, 1, 1).tolist()
                    for op in ("min", "max", "mean", "std")
                }
                ep_stats[VIDEO_KEY]["count"] = [n]
                for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                    ep_stats[key] = stats(np.asarray(columns[key]).reshape(n, 1))
                if task_mode == "subtask":
                    ep_stats["subtask_boundary"] = stats(subtask_boundaries.reshape(n, 1))
                episode_stats.append({"episode_index": episode, "stats": ep_stats})
                episode_tasks = (
                    [definition["task"] for definition in subtask_definitions]
                    if task_mode == "subtask"
                    else [contract["instruction"]]
                )
                episodes.append({"episode_index": episode, "tasks": episode_tasks, "length": n})
                source_episodes.append({**entry, "episode_index": episode})
                global_index += n
                if split == "train":
                    train_actions.append(action)
                print(f"{split}: {episode + 1}/{len(selected)}", flush=True)
            features = {
                "observation.state": {"dtype": "float32", "shape": [21], "names": None},
                "action": {"dtype": "float32", "shape": [7], "names": None},
                VIDEO_KEY: {
                    "dtype": "video",
                    "shape": [size, size, 3],
                    "names": ["height", "width", "channels"],
                    "info": {
                        "video.height": size,
                        "video.width": size,
                        "video.codec": "h264",
                        "video.pix_fmt": "gbrp",
                        "video.fps": 20,
                        "video.channels": 3,
                        "video.is_depth_map": False,
                        "has_audio": False,
                    },
                },
            }
            for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
                features[key] = {"dtype": "float32" if key == "timestamp" else "int64", "shape": [1], "names": None}
            if task_mode == "subtask":
                features["subtask_boundary"] = {"dtype": "bool", "shape": [1], "names": None}
            write_json(
                root / "meta/info.json",
                {
                    "codebase_version": "v2.1",
                    "robot_type": "dual_ur5e_black_gripper",
                    "fps": 20,
                    "total_episodes": len(episodes),
                    "total_frames": global_index,
                    "total_tasks": len(subtask_definitions) if task_mode == "subtask" else 1,
                    "total_videos": len(episodes),
                    "chunks_size": 1000,
                    "total_chunks": math.ceil(len(episodes) / 1000),
                    "splits": {split: f"0:{len(episodes)}"},
                    "data_path": DATA_PATH,
                    "video_path": VIDEO_PATH,
                    "features": features,
                },
            )
            write_lines(root / "meta/episodes.jsonl", episodes)
            write_lines(root / "meta/episodes_stats.jsonl", episode_stats)
            task_rows = (
                [{"task_index": t["task_index"], "task": t["task"]} for t in subtask_definitions]
                if task_mode == "subtask"
                else [{"task_index": 0, "task": contract["instruction"]}]
            )
            write_lines(root / "meta/tasks.jsonl", task_rows)
            if task_mode == "subtask":
                write_json(
                    root / "meta/subtasks.json",
                    {
                        "subtasks": subtask_definitions,
                        "global_task": contract["instruction"],
                        "planner_verification_id": manifest.get("planner_verification_id"),
                        "label_alignment": "subtask to execute from observation before action at this frame",
                    },
                )
            write_json(
                root / "meta/modality.json",
                {
                    "state": {"robot": {"original_key": "observation.state", "start": 0, "end": 21}},
                    "action": {
                        "joint_targets": {"original_key": "action", "start": 0, "end": 6},
                        "gripper": {"original_key": "action", "start": 6, "end": 7},
                    },
                    "video": {"primary": {"original_key": VIDEO_KEY}},
                    "annotation": {"human.task_description": {"original_key": "task_index"}},
                },
            )
            write_json(
                root / "meta/source.json",
                {
                    "manifest_sha256": digest,
                    "contract": contract,
                    "split": split,
                    "episodes": source_episodes,
                    "terminal_image": "omitted: has no corresponding action",
                    "task_mode": task_mode,
                    "planner_verification_id": manifest.get("planner_verification_id"),
                },
            )
            summary[split] = {"episodes": len(episodes), "frames": global_index}
        actions = np.concatenate(train_actions)
        low, high = actions.min(axis=0), actions.max(axis=0)
        # Constant joints still need an invertible normalization.
        high = np.maximum(high, low + 1e-4)
        low[6], high[6] = 0, 0.035
        write_json(
            staging / "normalization.json",
            {
                "fit_split": "train",
                "mode": "min_max_absolute",
                "manifest_sha256": digest,
                "min": low.tolist(),
                "max": high.tolist(),
                "action_units": ["rad"] * 6 + ["m"],
                "contract": contract,
            },
        )
        write_json(
            staging / "conversion.json",
            {
                "manifest_sha256": digest,
                "splits": summary,
                "format": "LeRobot v2.1",
                "rgb_video": "lossless libx264rgb",
                "smoke_limit": limit,
                "task_mode": task_mode,
            },
        )
        staging.rename(destination)
    except Exception:
        # Keep the partial directory for diagnosis; never overwrite a previous result.
        raise
    return summary


def main() -> None:
    """Command-line converter; no models or training dependencies are loaded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--task-mode", choices=("auto", "global", "subtask"), default="auto")
    args = parser.parse_args()
    print(convert(args.source, args.output, args.limit_per_split, args.task_mode))


if __name__ == "__main__":
    main()
