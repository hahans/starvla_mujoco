"""LeRobot v2.1 reader with explicit absolute-action normalization.

The task launcher uses this small reader rather than the upstream multi-robot
mixture loader. Episodes never cross splits; action chunks repeat the final
target at episode or subtask boundaries. Robot state is raw, not action-normalized.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

VIDEO_KEY = "observation.images.primary"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


def read_json(path: Path) -> dict:
    """Read UTF-8 metadata."""
    return json.loads(Path(path).read_text())


def write_json(path: Path, value: object) -> None:
    """Write finite, human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def json_lines(path: Path) -> list[dict]:
    """Read a JSONL metadata table."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def episode_path(root: Path, episode: int, video: bool = False) -> Path:
    """Resolve a portable LeRobot episode path."""
    return root / (VIDEO_PATH if video else DATA_PATH).format(
        episode_chunk=episode // 1000, episode_index=episode, video_key=VIDEO_KEY
    )


def read_arrays(root: Path, episode: int) -> tuple[np.ndarray, np.ndarray]:
    """Load raw state and actuator targets from Parquet."""
    table = pq.read_table(episode_path(root, episode))
    return tuple(np.asarray(table[key].to_pylist(), dtype=np.float32) for key in ("observation.state", "action"))


def read_video(path: Path) -> np.ndarray:
    """Decode RGB frames with PyAV (also supports lossless H.264 RGB)."""
    with av.open(str(path)) as container:
        return np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])


class ActionNormalizer:
    """Train-only min/max scaling, with a fixed physical finger range."""

    def __init__(self, metadata: dict):
        if metadata.get("fit_split") != "train":
            raise ValueError("Action normalization must be fitted on train only")
        self.metadata = metadata
        self.low = np.asarray(metadata["min"], dtype=np.float32)
        self.high = np.asarray(metadata["max"], dtype=np.float32)
        self.width = self.high - self.low
        if self.low.shape != (7,) or np.any(self.width <= 0):
            raise ValueError("Expected seven nonzero action ranges")

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        """Scale without silently clipping out-of-range held-out actions."""
        return (np.asarray(actions) - self.low) * 2 / self.width - 1

    def unnormalize(self, actions: np.ndarray) -> np.ndarray:
        """Return physical joint targets in radians and finger opening in metres."""
        return (np.asarray(actions) + 1) * self.width / 2 + self.low


class TaskDataset:
    """Random-access frames with a bounded, per-worker episode video cache."""

    def __init__(
        self,
        root: str | Path,
        normalization: str | Path,
        horizon: int = 8,
        max_episodes: int | None = None,
        cache_episodes: int = 2,
    ):
        self.root = Path(root)
        self.info = read_json(self.root / "meta/info.json")
        self.source = read_json(self.root / "meta/source.json")
        self.episodes = json_lines(self.root / "meta/episodes.jsonl")
        if max_episodes is not None:
            if max_episodes < 1:
                raise ValueError("max_episodes must be positive")
            self.episodes = self.episodes[:max_episodes]
        self.ends = np.cumsum([e["length"] for e in self.episodes])
        self.normalizer = ActionNormalizer(read_json(Path(normalization)))
        if self.normalizer.metadata["manifest_sha256"] != self.source["manifest_sha256"]:
            raise ValueError("Normalization and dataset came from different manifests")
        self.horizon = int(horizon)
        if self.horizon < 1 or not len(self.episodes):
            raise ValueError("Empty dataset or invalid horizon")
        self.cache_episodes = max(1, cache_episodes)
        self.cache = OrderedDict()
        task_rows = json_lines(self.root / "meta/tasks.jsonl")
        self.tasks = {row["task_index"]: row["task"] for row in task_rows}
        if not self.tasks or len(self.tasks) != len(task_rows):
            raise ValueError("Empty or duplicate task registry")
        self.lang = task_rows[0]["task"]  # Legacy alias; samples use their frame's task_index.

    def __len__(self) -> int:
        return int(self.ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        """Map a global sample to its episode and frame."""
        if not 0 <= index < len(self):
            raise IndexError(index)
        ep = int(np.searchsorted(self.ends, index, side="right"))
        return ep, index - (int(self.ends[ep - 1]) if ep else 0)

    def __getitem__(self, index: int) -> dict:
        ep, frame = self.locate(index)
        episode = self.episodes[ep]["episode_index"]
        if episode not in self.cache:
            state, action = read_arrays(self.root, episode)
            images = read_video(episode_path(self.root, episode, video=True))
            if len(images) != len(action):
                raise ValueError("Video/action frame count mismatch")
            task_indices = np.asarray(
                pq.read_table(episode_path(self.root, episode), columns=["task_index"])["task_index"].to_pylist(),
                dtype=np.int64,
            )
            if len(task_indices) != len(action) or not set(task_indices).issubset(self.tasks):
                raise ValueError("Frame task indices do not match the task registry")
            self.cache[episode] = (state, action, images, task_indices)
            while len(self.cache) > self.cache_episodes:
                self.cache.popitem(last=False)
        self.cache.move_to_end(episode)
        state, action, images, task_indices = self.cache[episode]
        next_subtask = np.flatnonzero(task_indices[frame:] != task_indices[frame])
        last = frame + int(next_subtask[0]) - 1 if len(next_subtask) else len(action) - 1
        indices = np.minimum(np.arange(frame, frame + self.horizon), last)
        return {
            "image": [Image.fromarray(images[frame])],
            "lang": self.tasks[int(task_indices[frame])],
            "state": state[frame].copy(),
            "action": self.normalizer.normalize(action[indices]).astype(np.float32),
        }


def collate_examples(examples: list[dict]) -> list[dict]:
    """Keep StarVLA's list-of-examples interface, including PIL images."""
    return examples
