"""Blend recorded real frames with the simulator's policy camera at the same joint angles.

Checks the sim/real alignment (camera pose, arm, gripper, table) on data recorded by
``ur5_record.py``; the cubes stay at the simulator's nominal positions::

    python -m mujoco_task.real_overlay --episode data/real_raw/episode_0000 --frames 0 300 600 --out logs/overlay.png

Each row is: real centre crop | simulator | 50 % blend. Run with ``MUJOCO_GL=egl`` on a headless machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from .scene.block_task import ARM_JOINTS
from .scene.run_scene import load_scene


def render_rows(episode: Path, frames: list[int], size: int) -> np.ndarray:
    model, data = load_scene()
    renderer = mujoco.Renderer(model, size, size)
    arm = [model.joint("left_" + name).qposadr[0] for name in ARM_JOINTS]
    fingers = [model.joint("left_finger_" + side + "_joint").qposadr[0] for side in ("left", "right")]
    record = np.load(episode / "data.npz")
    rows = []
    try:
        for frame in frames:
            image = np.asarray(Image.open(episode / "rgb" / f"{frame:06d}.png").convert("RGB"))
            h, w = image.shape[:2]
            side = min(h, w)
            crop = image[(h - side) // 2 : (h + side) // 2, (w - side) // 2 : (w + side) // 2]
            real = np.asarray(Image.fromarray(crop).resize((size, size), Image.BILINEAR))
            mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
            data.qpos[arm] = record["q"][frame]
            if "gripper_opening_m" in record:
                data.qpos[fingers] = np.clip(record["gripper_opening_m"][frame] / 2, 0, 0.035)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera="policy_cam")
            sim = renderer.render()
            blend = (0.5 * real.astype(np.float32) + 0.5 * sim).astype(np.uint8)
            rows.append(np.concatenate([real, sim, blend], 1))
    finally:
        if hasattr(renderer, "close"):
            renderer.close()
    return np.concatenate(rows, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episode", type=Path, required=True, help="episode_XXXX folder from ur5_record.py")
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--size", type=int, default=480, help="Rendered square size in pixels")
    parser.add_argument("--out", type=Path, default=Path("logs/overlay.png"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_rows(args.episode, args.frames, args.size)).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
