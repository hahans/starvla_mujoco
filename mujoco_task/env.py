"""Shared observation/action contract for collection and policy evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from .contract import INSTRUCTION
from .scene.block_task import ARM_JOINTS, BlockTransferTask
from .scene.run_scene import load_scene

SCENE = Path(__file__).resolve().parent / "scene"
# Cube reset spread; covers the three real_raw2 episodes (green x spans 0.1 m).
SHIFT_X, SHIFT_Y, JITTER = 0.06, 0.02, 0.005


def scene_digest() -> str:
    """Fingerprint all runtime scene assets, independent of absolute location."""
    h = hashlib.sha256()
    paths = [SCENE / "scene.xml", SCENE / "layout.json"]
    paths += sorted((SCENE / "assets").rglob("*"))
    for path in paths:
        if path.is_file():
            h.update(str(path.relative_to(SCENE)).encode())
            h.update(path.read_bytes())
    return h.hexdigest()


class TransferEnv:
    """20 Hz actuator-target environment; physics and success checks run at 500 Hz.

    Policy inputs contain RGB and robot state only. Cube truth is used solely by
    reset, the collection expert, and evaluation metrics. The white arm is held.
    """

    def __init__(self, image_size: int = 128, render: bool = True) -> None:
        if image_size < 32 or image_size % 8:
            raise ValueError("image_size must be >=32 and divisible by 8")
        self.model, self.data = load_scene()
        self.task = BlockTransferTask(self.model, self.data)
        self.control_dt = 0.05
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.image_size = image_size
        self.arm_q = self.task.arm_q
        self.arm_dof = [self.model.joint("left_" + j).dofadr[0] for j in ARM_JOINTS]
        self.finger_q = [self.model.joint("left_finger_" + s + "_joint").qposadr[0] for s in ("left", "right")]
        self.limits = self.model.jnt_range[[self.model.joint("left_" + j).id for j in ARM_JOINTS]]
        self.max_delta = 0.12  # rad per control step, from previous commanded target
        self.model.vis.quality.shadowsize = 1024
        self.model.vis.quality.offsamples = 2
        self.camera = "policy_cam"
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size) if render else None
        self.last_action = self._from_controls(self.data.ctrl)
        self.steps = 0

    def contract(self) -> dict:
        """Return the exact portable policy interface stored in data/checkpoints."""
        cam = self.model.camera(self.camera)
        return dict(
            version=1,
            scene_sha256=scene_digest(),
            mujoco_version=mujoco.__version__,
            image_size=self.image_size,
            control_hz=20,
            physics_dt=0.002,
            action_dim=7,
            proprio_dim=21,
            instruction=INSTRUCTION,
            action="six absolute left joint targets (rad), common finger target (m)",
            proprio="left q[6], left dq[6], finger q[2], previous applied action[7]",
            joint_names=list(ARM_JOINTS),
            max_joint_delta_per_step=self.max_delta,
            camera=dict(
                name=self.camera,
                pos=cam.pos.tolist(),
                quat_wxyz=cam.quat.tolist(),
                fovy=float(cam.fovy[0]),
                real="RealSense D435I 640x480 colour, centre 480x480 crop resized to image_size",
            ),
            reset_distribution=(
                f"cubes common x +/-{SHIFT_X}m y +/-{SHIFT_Y}m, each cube xy +/-{JITTER}m; green always left"
            ),
        )

    def _from_controls(self, ctrl: np.ndarray) -> np.ndarray:
        return np.r_[ctrl[self.task.arm_u], np.mean(ctrl[self.task.grip_u])].astype(np.float32)

    def reset(self, seed: int = 0, randomize: bool = True) -> dict:
        """Reset an episode; seed 0 with randomize=False reproduces the photo task."""
        self.task.reset()
        rng = np.random.default_rng(seed)
        shift = rng.uniform([-SHIFT_X, -SHIFT_Y], [SHIFT_X, SHIFT_Y]) if randomize else np.zeros(2)
        for color in ("green", "red"):
            addr = self.model.joint(color + "_block_joint").qposadr[0]
            self.data.qpos[addr : addr + 2] += shift
            if randomize:
                self.data.qpos[addr : addr + 2] += rng.uniform(-JITTER, JITTER, 2)
        mujoco.mj_forward(self.model, self.data)
        self.task.green_start = self.task.position("green").copy()
        self.task.red_start = self.task.position("red").copy()
        self.task.goal = self.task.red_start + np.asarray([self.task.cfg["goal_offset_x"], 0, 0])
        self.last_action = self._from_controls(self.data.ctrl)
        self.steps = 0
        return self.observe()

    def observe(self) -> dict:
        """Read only observable robot state and a fixed camera frame."""
        proprio = np.r_[
            self.data.qpos[self.arm_q],
            self.data.qvel[self.arm_dof],
            self.data.qpos[self.finger_q],
            self.last_action,
        ].astype(np.float32)
        result = {"proprio": proprio}
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera=self.camera)
            result["image"] = self.renderer.render().copy()
        return result

    def constrain(self, action: np.ndarray) -> np.ndarray:
        """Apply identical joint-limit, slew-rate and gripper limits in all modes."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Action must be seven finite values")
        q = np.clip(
            action[:6],
            self.last_action[:6] - self.max_delta,
            self.last_action[:6] + self.max_delta,
        )
        q = np.clip(q, self.limits[:, 0], self.limits[:, 1])
        return np.r_[q, np.clip(action[6], 0, 0.035)].astype(np.float32)

    def step(self, action: np.ndarray, observe: bool = True) -> tuple[dict, dict]:
        """Hold a model/expert command for exactly one control period."""
        applied = self.constrain(action)
        controls = self.task.home.copy()
        controls[self.task.arm_u] = applied[:6]
        controls[self.task.grip_u] = applied[6]
        for _ in range(self.substeps):
            self.task.step_control(controls)
        self.last_action = applied
        self.steps += 1
        info = self.task.info()
        info["applied_action"] = applied.tolist()
        info["action_clipped"] = bool(np.max(np.abs(applied - action)) > 1e-6)
        return self.observe() if observe else {}, info

    def start_expert(self) -> None:
        """Plan from this randomized reset; never called by learned evaluation."""
        self.task.start(reset=False)

    def expert_action(self) -> tuple[np.ndarray, str, bool]:
        """Sample the expert trajectory at the next control tick, before stepping."""
        if not self.task.plan:
            raise RuntimeError("Call start_expert first")
        elapsed = self.data.time - self.task.plan_start + self.control_dt
        cursor = 0.0
        for name, duration, start, end in self.task.plan:
            if elapsed < cursor + duration - 1e-9:
                t = np.clip((elapsed - cursor) / duration, 0, 1)
                smooth = t**3 * (10 - 15 * t + 6 * t * t)
                self.task.phase = name
                return (
                    self.constrain(self._from_controls(start + (end - start) * smooth)),
                    name,
                    False,
                )
            cursor += duration
        return self._from_controls(self.task.plan[-1][3]), "settle", True

    def close(self) -> None:
        """Release the rendering context, including older MuJoCo releases."""
        if self.renderer is not None:
            if hasattr(self.renderer, "close"):
                self.renderer.close()
            self.renderer = None
