"""Validate the native StarVLA config/data route without loading a VLM."""

from __future__ import annotations

import argparse

from .cli import load_config


def main() -> None:
    """Inspect one sample through the same task factory used by the native loader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_yaml", default="configs/mujoco/qwenfast.yaml")
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.config_yaml, overrides)
    from starVLA.dataloader.mujoco_lerobot import make_mujoco_dataloader

    loader = make_mujoco_dataloader(cfg, save_metadata=False)
    sample = loader.dataset[0]
    print(
        {
            "trainer": "starVLA/training/train_starvla.py",
            "dataset_py": cfg.datasets.vla_data.dataset_py,
            "frames": len(loader.dataset),
            "episodes": len(loader.dataset.episodes),
            "image": sample["image"][0].size,
            "state": sample["state"].shape,
            "action": sample["action"].shape,
            "max_train_steps": cfg.trainer.max_train_steps,
            "gradient_accumulation_steps": cfg.trainer.gradient_accumulation_steps,
        }
    )


if __name__ == "__main__":
    main()
