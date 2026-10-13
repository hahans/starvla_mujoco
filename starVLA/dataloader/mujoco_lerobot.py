"""Task data adapter for StarVLA's native VLATrainer.

Only dataset reading/sampling and contract export live here. Optimizer,
backpropagation, scheduling and checkpointing belong to train_starvla.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from mujoco_task.data import TaskDataset, collate_examples, read_json, write_json


class EpisodeSampler(Sampler):
    """Shuffle episodes and their frames while keeping a useful decoder cache."""

    def __init__(self, dataset: TaskDataset, seed: int):
        self.dataset, self.seed, self.epoch = dataset, seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        starts = np.r_[0, self.dataset.ends[:-1]]
        for ep in rng.permutation(len(starts)):
            yield from rng.permutation(np.arange(starts[ep], self.dataset.ends[ep])).tolist()


def make_mujoco_dataloader(cfg, *, save_metadata: bool = True) -> DataLoader:
    """Read only the training repository and export its exact normalization."""
    data_cfg = cfg.datasets.vla_data
    root = Path(data_cfg.data_root_dir)
    if data_cfg.data_mix != "mujoco_block_train":
        raise ValueError("This adapter trains only on data_mix=mujoco_block_train")
    if data_cfg.get("action_type", "absolute") != "absolute":
        raise ValueError("MuJoCo actions are absolute joint targets, not deltas")
    dataset = TaskDataset(
        root / "block_train",
        root / "normalization.json",
        horizon=int(cfg.framework.action_model.action_horizon),
        max_episodes=data_cfg.get("max_train_episodes", None),
    )
    if dataset.source["split"] != "train":
        raise ValueError("Refusing to train on held-out episodes")
    if dataset.source["contract"] != dataset.normalizer.metadata["contract"]:
        raise ValueError("Dataset and normalization simulator contracts differ")
    workers = int(data_cfg.get("num_workers", 4))
    loader_kwargs = dict(
        batch_size=int(data_cfg.per_device_batch_size),
        num_workers=workers,
        collate_fn=collate_examples,
        pin_memory=False,
        sampler=EpisodeSampler(dataset, int(cfg.get("seed", 42))),
    )
    if workers:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)
    if save_metadata and (not dist.is_initialized() or dist.get_rank() == 0):
        output = Path(cfg.output_dir)
        norm_path = output / "normalization.json"
        if norm_path.exists() and read_json(norm_path) != dataset.normalizer.metadata:
            raise ValueError("Run directory contains normalization from a different dataset")
        write_json(norm_path, dataset.normalizer.metadata)
        write_json(
            output / "dataset_statistics.json",
            {
                "mujoco_block_train": {
                    "action": {
                        "min": dataset.normalizer.low.tolist(),
                        "max": dataset.normalizer.high.tolist(),
                        "mask": [True] * 7,
                    },
                    "normalization_mode": "min_max_absolute",
                    "num_trajectories": len(dataset.episodes),
                    "num_transitions": len(dataset),
                }
            },
        )
    return loader
