"""Contact-driven transfer of the green cube by the left (black) gripper."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from .run_scene import step

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class BlockTransferTask:
    """A resettable task with observations, success checks and a scripted policy.

    Cubes are free bodies moved exclusively by MuJoCo contacts. IK works on a
    separate MjData; the controller only sends position targets to actuators.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Bind a compiled scene and initialize a green-left/red-right episode."""
        self.model, self.data = model, data
        cfg = json.loads((ROOT / "layout.json").read_text())
        self.cfg = cfg["block_task"]
        self.half = float(model.geom_size[model.geom("green_cube").id, 0])
        table = model.geom("optical_table").id
        self.table_top = float(data.geom_xpos[table, 2] + model.geom_size[table, 2])
        self.arm_q = [model.joint("left_" + name).qposadr[0] for name in ARM_JOINTS]
        self.arm_u = [
            model.actuator("left_" + name + "_position").id for name in ARM_JOINTS
        ]
        self.grip_u = [
            model.actuator("left_finger_" + side + "_position").id
            for side in ("left", "right")
        ]
        self.block_geoms = {
            color: model.geom(color + "_cube").id for color in ("green", "red")
        }
        self.finger_geoms = [
            {
                model.geom("left_finger_" + side + suffix).id
                for suffix in ("_link", "_tip")
            }
            for side in ("left", "right")
        ]
        self.white_q = [
            model.jnt_qposadr[j]
            for j in range(model.njnt)
            if model.joint(j).name.startswith("right_")
        ]
        self.allowed_contacts = {
            frozenset((table, g)) for g in self.block_geoms.values()
        }
        self.allowed_contacts.update(
            frozenset((self.block_geoms["green"], g))
            for side in self.finger_geoms
            for g in side
        )
        self.scratch = mujoco.MjData(model)
        self.reset()

    def reset(self) -> dict:
        """Restore both cubes and robots, always with green left of red."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("home").id)
        mujoco.mj_forward(self.model, self.data)
        self.home = self.data.ctrl.copy()
        # Tool orientation of the real go_home pose; the expert keeps it for every waypoint.
        self.home_rotation = self.data.site_xmat[self.model.site("left_tcp").id].reshape(3, 3).copy()
        self.green_start = self.position("green").copy()
        self.red_start = self.position("red").copy()
        self.goal = self.red_start + np.array([self.cfg["goal_offset_x"], 0, 0])
        self.running = False
        self.success = False
        self.failed = False
        self.phase = "ready"
        self.phase_log = []
        self.max_lift = 0.0
        self.max_red_drift = 0.0
        self.max_white_motion = 0.0
        self.white_start = self.data.qpos[self.white_q].copy()
        self.unexpected_contacts = set()
        self.max_penetration = 0.0
        self.bilateral_grasp_seen = False
        self.lifted_by_black = False
        self.stable_time = 0.0
        self.plan = []
        self.plan_start = 0.0
        return self.observation()

    def position(self, color: str) -> np.ndarray:
        """Return a cube's current world-space center."""
        return self.data.xpos[self.model.body(color + "_block").id]

    def observation(self) -> dict:
        """Return state observations suitable for an external controller."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "green_position": self.position("green").copy(),
            "red_position": self.position("red").copy(),
            "goal_position": self.goal.copy(),
            "black_tcp": self.data.site_xpos[self.model.site("left_tcp").id].copy(),
        }

    def contacts(self) -> tuple[list[bool], bool]:
        """Detect each black finger's cube contact and cube/table support."""
        fingers = [False, False]
        grounded = False
        green = self.block_geoms["green"]
        table = self.model.geom("optical_table").id
        for contact in self.data.contact[: self.data.ncon]:
            pair = {contact.geom1, contact.geom2}
            if green not in pair:
                continue
            grounded |= table in pair
            for i, geoms in enumerate(self.finger_geoms):
                fingers[i] |= bool(pair & geoms)
        return fingers, grounded

    def update_success(self) -> None:
        """Require a real grasp/lift, release, stable placement and unmoved red."""
        green, red = self.position("green"), self.position("red")
        fingers, grounded = self.contacts()
        self.bilateral_grasp_seen |= all(fingers)
        lift = float(green[2] - self.green_start[2])
        self.max_lift = max(self.max_lift, lift)
        self.lifted_by_black |= all(fingers) and lift > 0.05
        self.max_red_drift = max(
            self.max_red_drift, float(np.linalg.norm(red - self.red_start))
        )
        self.max_white_motion = max(
            self.max_white_motion,
            float(np.abs(self.data.qpos[self.white_q] - self.white_start).max()),
        )
        for contact in self.data.contact[: self.data.ncon]:
            self.max_penetration = max(self.max_penetration, float(-contact.dist))
            if frozenset((contact.geom1, contact.geom2)) not in self.allowed_contacts:
                self.unexpected_contacts.add(
                    tuple(
                        sorted(
                            (
                                self.model.geom(contact.geom1).name,
                                self.model.geom(contact.geom2).name,
                            )
                        )
                    )
                )
        jid = self.model.joint("green_block_joint").id
        dof = self.model.jnt_dofadr[jid]
        velocity = self.data.qvel[dof : dof + 6]
        stable = (
            self.lifted_by_black
            and not any(fingers)
            and grounded
            and green[0] - red[0] > 2 * self.half + 0.015
            and abs(green[1] - red[1]) < self.cfg["placement_tolerance"]
            and abs(green[2] - self.green_start[2]) < 0.008
            and self.max_red_drift < 0.02
            and not self.unexpected_contacts
            and self.max_white_motion < 0.005
            and np.linalg.norm(velocity[:3]) < 0.03
            and np.linalg.norm(velocity[3:]) < 0.3
        )
        self.stable_time = self.stable_time + self.model.opt.timestep if stable else 0.0
        if self.stable_time >= self.cfg["settle_seconds"]:
            self.success = True

    def info(self) -> dict:
        """Return serializable task progress and physical grasp evidence."""
        return {
            "success": self.success,
            "failed": self.failed,
            "phase": self.phase,
            "time": float(self.data.time),
            "green_position": self.position("green").tolist(),
            "red_position": self.position("red").tolist(),
            "goal_position": self.goal.tolist(),
            "bilateral_grasp_seen": bool(self.bilateral_grasp_seen),
            "lifted_by_black": bool(self.lifted_by_black),
            "max_lift_m": self.max_lift,
            "max_red_drift_m": self.max_red_drift,
            "stable_seconds": self.stable_time,
            "max_white_joint_motion": self.max_white_motion,
            "unexpected_contacts": sorted(self.unexpected_contacts),
            "max_penetration_m": self.max_penetration,
            "warnings": [int(w.number) for w in self.data.warning],
        }

    def solve_ik(self, target: np.ndarray, seed: np.ndarray) -> np.ndarray:
        """Solve a downward grasp orientation without altering simulation state."""
        self.scratch.qpos[:] = self.data.qpos
        site = self.model.site("left_tcp").id
        desired = self.home_rotation

        def residual(q: np.ndarray) -> np.ndarray:
            self.scratch.qpos[self.arm_q] = q
            mujoco.mj_forward(self.model, self.scratch)
            rotation = self.scratch.site_xmat[site].reshape(3, 3)
            return np.concatenate(
                (
                    8 * (self.scratch.site_xpos[site] - target),
                    (rotation - desired).ravel(),
                    0.001 * (q - seed),
                )
            )

        joint_ids = [self.model.joint("left_" + name).id for name in ARM_JOINTS]
        bounds = self.model.jnt_range[joint_ids].T
        solution = least_squares(
            residual,
            seed,
            bounds=(bounds[0], bounds[1]),
            max_nfev=400,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        residual(solution.x)
        error = np.linalg.norm(self.scratch.site_xpos[site] - target)
        if error > 0.004:
            raise RuntimeError(
                f"Unreachable task waypoint {target}: error {error:.4f} m"
            )
        return solution.x

    def start(self, reset: bool = True) -> None:
        """Reset and prepare the contact-driven pick/lift/place demonstration."""
        if reset:
            self.reset()
        grasp = self.green_start + np.array([0, 0, 0.004])
        place = self.goal + np.array([0, 0, 0.006])
        lift = np.array([0, 0, self.cfg["lift_height"]])
        q = self.data.qpos[self.arm_q].copy()
        previous = self.home.copy()
        previous[self.grip_u] = 0.035
        self.plan = [("prepare", 0.5, self.home.copy(), previous.copy())]
        waypoints = [
            ("approach", 3.0, grasp + lift, 0.035),
            ("descend", 1.8, grasp, 0.035),
            ("grasp", 1.0, grasp, 0.0),
            ("lift", 1.8, grasp + lift, 0.0),
            ("transfer", 2.2, place + lift, 0.0),
            ("place", 1.8, place, 0.0),
            ("release", 1.0, place, 0.035),
            ("retreat", 1.8, place + lift, 0.035),
        ]
        for name, duration, target, opening in waypoints:
            q = self.solve_ik(target, q)
            controls = self.home.copy()
            controls[self.arm_u] = q
            controls[self.grip_u] = opening
            self.plan.append((name, duration, previous.copy(), controls.copy()))
            previous = controls
        self.plan.append(("settle", 1.0, previous.copy(), previous.copy()))
        self.plan_start = self.data.time
        self.running = True
        self.phase = "prepare"
        LOGGER.info("Task started: black gripper moves green to the right of red")

    def step_control(self, controls: np.ndarray) -> dict:
        """Advance physics using actuator targets and update the task outcome."""
        step(self.model, self.data, controls)
        self.update_success()
        if not np.isfinite(self.data.qpos).all() or any(
            w.number for w in self.data.warning
        ):
            raise RuntimeError("MuJoCo reported invalid dynamics during the task")
        return self.info()

    def tick(self) -> dict:
        """Advance one simulation step of the scripted policy or hold still."""
        if not self.running:
            return self.step_control(self.data.ctrl.copy())
        elapsed = self.data.time - self.plan_start
        cursor = 0.0
        for name, duration, start, end in self.plan:
            if elapsed < cursor + duration:
                t = np.clip((elapsed - cursor) / duration, 0, 1)
                smooth = t * t * t * (10 + t * (-15 + 6 * t))
                result = self.step_control(start + (end - start) * smooth)
                if self.phase != name:
                    self.phase = name
                    self.phase_log.append(self.info())
                    LOGGER.info(
                        "Phase %s: green=%s, lift=%.3f m, contacts=%s",
                        name,
                        self.position("green").round(3),
                        self.max_lift,
                        self.contacts()[0],
                    )
                return result
            cursor += duration
        self.running = False
        self.success = self.success and not self.unexpected_contacts
        self.failed = not self.success
        self.phase = "success" if self.success else "failed"
        LOGGER.info("Task result: %s", json.dumps(self.info()))
        return self.info()

    def run(self) -> dict:
        """Execute one full headless demonstration and return its outcome."""
        self.start()
        while self.running:
            self.tick()
        return self.info()


if __name__ == "__main__":
    from .run_scene import load_scene

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model, data = load_scene()
    task = BlockTransferTask(model, data)
    result = task.run()
    if not result["success"]:
        raise SystemExit(1)
