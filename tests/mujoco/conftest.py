"""Small synthetic LeRobot fixtures shared by task integration checks."""

import hashlib
import json

import h5py
import numpy as np
import pytest

from mujoco_task.convert import convert


@pytest.fixture
def converted(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rng = np.random.default_rng(123)
    contract = {
        "action_dim": 7,
        "proprio_dim": 21,
        "control_hz": 20,
        "image_size": 32,
        "instruction": "move green right",
    }
    entries, originals = [], {}
    for split_index, split in enumerate(("train", "val", "test")):
        for episode in range(2 if split == "train" else 1):
            path = source / f"{split}_{episode}.h5"
            image = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
            state = rng.normal(size=(5, 21)).astype(np.float32)
            action = rng.uniform(-1, 1, (5, 7)).astype(np.float32)
            action[:, :6] += split_index * 10  # held-out extrema must not enter training stats
            action[:, 6] = np.linspace(0, 0.035, 5)
            with h5py.File(path, "w") as h5:
                for key, value in {
                    "image": image,
                    "proprio": state,
                    "action": action,
                    "time": np.arange(5) / 20,
                    "terminal_image": image[-1],
                }.items():
                    h5[key] = value
                h5.attrs["instruction"] = contract["instruction"]
                h5.attrs["seed"] = split_index * 100 + episode
            entries.append(
                {
                    "path": path.name,
                    "split": split,
                    "seed": split_index * 100 + episode,
                    "length": 5,
                    "success": True,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            originals[split, episode] = (image, state, action)
    (source / "manifest.json").write_text(json.dumps({"contract": contract, "episodes": entries}))
    target = tmp_path / "lerobot"
    convert(source, target)
    return source, target, originals
