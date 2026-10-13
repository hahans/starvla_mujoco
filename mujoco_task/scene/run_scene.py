"""View, render, or validate the photo-inspired MuJoCo laboratory."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)


def load_scene():
    """Load the scene and its reproducible home pose."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(ROOT / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    return model, data


def step(model, data, home: np.ndarray, demo: bool = False) -> None:
    """Step position servos with gravity compensation and optional small motions."""
    import mujoco

    data.ctrl[:] = home
    if demo:
        for prefix, phase in (("left_", 0), ("right_", np.pi)):
            aid = model.actuator(prefix + "wrist_3_joint_position").id
            data.ctrl[aid] += 0.16 * np.sin(1.3 * data.time + phase)
            for side in ("left", "right"):
                aid = model.actuator(prefix + "finger_" + side + "_position").id
                data.ctrl[aid] = 0.018 + 0.013 * np.sin(1.3 * data.time + phase)
    # Compensate only actuated robot DoFs, without applying forces to fixed scene geometry.
    data.qfrc_applied[:] = 0
    for joint in model.actuator_trnid[:, 0]:
        dof = model.jnt_dofadr[joint]
        data.qfrc_applied[dof] = data.qfrc_bias[dof]
    mujoco.mj_step(model, data)


def render(model, data, path: Path, camera: str, width: int, height: int) -> None:
    """Save a true MuJoCo rendering with compatibility for MuJoCo 2.3.7."""
    import mujoco
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    Image.fromarray(renderer.render()).save(path)
    # MuJoCo 2.3.7 releases renderer resources via its destructor.
    if hasattr(renderer, "close"):
        renderer.close()
    LOGGER.info("Saved %s", path)


def validate(model, data, seconds: float, demo: bool) -> dict:
    """Check numerical stability, servo tracking, contacts and joint bounds."""
    import mujoco

    home, initial = data.ctrl.copy(), data.qpos.copy()
    min_contact_distance = 0.0
    max_speed = 0.0
    for _ in range(round(seconds / model.opt.timestep)):
        step(model, data, home, demo)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Simulation produced a non-finite state.")
        max_speed = max(max_speed, float(np.abs(data.qvel).max()))
        if data.ncon:
            min_contact_distance = min(
                min_contact_distance, min(c.dist for c in data.contact[: data.ncon])
            )
    if any(w.number for w in data.warning):
        raise RuntimeError("MuJoCo reported a numerical or capacity warning.")
    max_error = max(
        abs(data.qpos[model.jnt_qposadr[j]] - data.ctrl[i])
        for i, j in enumerate(model.actuator_trnid[:, 0])
    )
    if max_error > 0.08:
        raise RuntimeError(f"Servo tracking error is too large: {max_error:.3f}")
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            q = data.qpos[model.jnt_qposadr[j]]
            low, high = model.jnt_range[j]
            if q < low - 0.03 or q > high + 0.03:
                raise RuntimeError(f"Joint {j} exceeded its limits: {q}")
    report = {
        "mujoco_version": mujoco.__version__,
        "simulated_seconds": seconds,
        "demo": demo,
        "nq": model.nq,
        "nu": model.nu,
        "ngeom": model.ngeom,
        "max_servo_error": float(max_error),
        "max_speed": max_speed,
        "deepest_contact_m": min_contact_distance,
        "max_pose_change": float(np.abs(data.qpos - initial).max()),
        "warnings": [int(w.number) for w in data.warning],
    }
    LOGGER.info("Validation: %s", json.dumps(report, indent=2))
    return report


def configure_viewer_camera(camera, cfg: dict) -> None:
    """Match the reference camera while allowing interactive orbit controls."""
    import mujoco

    target = np.array(cfg["camera_target"])
    delta = np.array(cfg["camera_position"]) - target
    distance = np.linalg.norm(delta)
    if distance <= 0:
        raise ValueError("Camera position must differ from its target.")
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = target
    camera.distance = distance
    # MuJoCo azimuth describes the viewing direction, not target-to-camera.
    camera.azimuth = np.degrees(np.arctan2(delta[1], delta[0])) + 180
    camera.elevation = -np.degrees(np.arcsin(delta[2] / distance))


def view(
    model, data, demo: bool, duration: float | None = None, task_mode: bool = False
) -> None:
    """View the scene: T starts block transfer, Space pauses, and R resets."""
    import mujoco
    import mujoco.viewer

    from .block_task import BlockTransferTask

    paused = False
    reset = False
    start_task = task_mode
    task = BlockTransferTask(model, data)

    def key_callback(key: int) -> None:
        nonlocal paused, reset, start_task
        if key == 32:
            paused = not paused
        elif key in (ord("R"), ord("r")):
            reset = True
        elif key in (ord("T"), ord("t")):
            start_task = True

    home = data.ctrl.copy()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        cfg = json.loads((ROOT / "layout.json").read_text())
        with viewer.lock():
            configure_viewer_camera(viewer.cam, cfg)
        viewer.sync()
        started = time.monotonic()
        while viewer.is_running():
            if duration is not None and time.monotonic() - started >= duration:
                break
            tick = time.monotonic()
            with viewer.lock():
                if reset:
                    task.reset()
                    configure_viewer_camera(viewer.cam, cfg)
                    reset = False
                    start_task = False
                if start_task:
                    task.start()
                    task_mode = True
                    paused = False
                    start_task = False
                if not paused:
                    for _ in range(10):
                        if task_mode:
                            task.tick()
                        else:
                            step(model, data, home, demo)
            viewer.sync()
            time.sleep(max(0, 10 * model.opt.timestep - (time.monotonic() - tick)))


def main() -> None:
    """Parse mode and select an OpenGL backend before importing MuJoCo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render", type=Path, help="Save a PNG instead of opening a window"
    )
    parser.add_argument(
        "--camera",
        default="reference",
        choices=("reference", "overview", "mount_closeup", "policy_cam"),
    )
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--validate", action="store_true", help="Run headless dynamics checks"
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--demo", action="store_true", help="Gently move wrists and grippers"
    )
    parser.add_argument(
        "--task", action="store_true", help="Run black-gripper block transfer"
    )
    parser.add_argument(
        "--validate-task",
        action="store_true",
        help="Verify a complete transfer headlessly",
    )
    parser.add_argument("--report", type=Path, help="Write validation metrics as JSON")
    parser.add_argument(
        "--duration", type=float, help="Auto-close viewer after this many seconds"
    )
    args = parser.parse_args()
    if args.seconds <= 0 or args.width <= 0 or args.height <= 0:
        parser.error("seconds, width, and height must be positive")
    if args.width > 1600 or args.height > 1200:
        parser.error("render dimensions must not exceed the model's 1600 x 1200 buffer")
    if (args.task or args.validate_task) and (args.demo or args.validate):
        parser.error("block transfer cannot be combined with --demo or --validate")
    os.environ.setdefault(
        "MUJOCO_GL",
        "osmesa" if args.render or args.validate or args.validate_task else "glfw",
    )
    model, data = load_scene()
    if args.validate_task or (args.task and args.render):
        from .block_task import BlockTransferTask

        report = BlockTransferTask(model, data).run()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
        if not report["success"]:
            raise RuntimeError("Block transfer failed; inspect the task report")
    if args.validate:
        report = validate(model, data, args.seconds, args.demo)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.render:
        render(model, data, args.render, args.camera, args.width, args.height)
    elif not args.validate and not args.validate_task:
        view(model, data, args.demo, args.duration, args.task)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
