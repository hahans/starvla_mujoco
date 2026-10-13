"""Keyboard teleoperation recorder for the left UR5: RGB, aligned depth and robot state at 20 Hz.

Recordings are raw hardware data in UR controller conventions (no ``joint_offset``).
They are used to calibrate the MuJoCo scene against the real cell and later as
real demonstrations.

    python -m mujoco_task.ur5_record --ip 192.168.1.101 --gripper dm --gripper-stroke-m 0.07 --out data/real_raw

Teleop follows ``ur5_control/keyboard_ur5_teleop.py``: keys are read globally with pynput
(any focused window, needs an X11 session), the arm moves while a key is held and stops
on release. Each tick the servoL target is the measured TCP plus an EMA-smoothed step,
so the target never runs away from the arm. Translations use the world frame of
``go_home.py`` (base turned by ``--world-tilt-deg`` about Y), rotations the TCP's own axes.

    W / S   world X+ / X-        A / D   world Y+ / Y-        Q / E   world Z+ / Z-
    U / O   TCP roll + / -       I / K   TCP pitch + / -      J / L   TCP yaw + / -
    SPACE   gripper open <-> closed           [ / ]   gripper 1 % more closed / open
    R       start / stop-and-save an episode  X       discard the current episode
    P       save one calibration frame (hold the arm still: image, depth, joints, TCP)
    ESC     quit (a running episode is saved)

Near a singularity (Jacobian condition number >= ``--jacobian-stop``) only moves that
improve the conditioning are sent, so the arm can always be driven back out.
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .ur5_robot import make_camera, make_gripper

# UR5 (CB series) DH parameters; they reproduce the controller's TCP readings to ~1 mm.
UR5_DH_D = (0.089159, 0.0, 0.0, 0.10915, 0.09465, 0.0823)
UR5_DH_A = (0.0, -0.425, -0.39225, 0.0, 0.0, 0.0)
UR5_DH_ALPHA = (np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0)
WORLD_TILT_DEG = {"192.168.1.101": -45.0, "192.168.1.102": 45.0}

MOVE_KEYS = {"w": (0, 1), "s": (0, -1), "a": (1, 1), "d": (1, -1), "q": (2, 1), "e": (2, -1)}
TURN_KEYS = {"i": (3, 1), "k": (3, -1), "u": (4, 1), "o": (4, -1), "j": (5, 1), "l": (5, -1)}
EVENT_KEYS = {"space", "[", "]", "r", "x", "p", "esc"}


def ur5_jacobian(q) -> np.ndarray:
    """Geometric Jacobian of the flange in the controller base frame, rows [linear; angular]."""
    transform = np.eye(4)
    origins, axes = [], []
    for theta, d, a, alpha in zip(q, UR5_DH_D, UR5_DH_A, UR5_DH_ALPHA):
        origins.append(transform[:3, 3].copy())
        axes.append(transform[:3, 2].copy())
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array(
            [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]]
        )
    axes, origins = np.asarray(axes), np.asarray(origins)
    return np.vstack([np.cross(axes, transform[:3, 3] - origins).T, axes.T])


def singularity_gate(q, twist, stop: float, damping: float = 0.01) -> tuple[float, bool]:
    """Condition number at ``q`` and whether a base-frame ``twist`` [dx, drot] may be sent."""
    jac = ur5_jacobian(q)
    cond = float(np.linalg.cond(jac))
    if cond < stop:
        return cond, True
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(6), np.asarray(twist, dtype=np.float64))
    return cond, bool(np.linalg.cond(ur5_jacobian(np.asarray(q) + dq)) < cond)


def servo_target(tcp: np.ndarray, world_from_base, delta) -> np.ndarray:
    """Measured TCP shifted by ``delta`` = [dx, dy, dz, dpitch, droll, dyaw].

    Translation is in the world frame; rotation is about the TCP's own axes in zxy order,
    the convention of ``keyboard_ur5_teleop.py``.
    """
    from scipy.spatial.transform import Rotation

    delta = np.asarray(delta, dtype=np.float64)
    position = tcp[:3] + world_from_base.inv().apply(delta[:3])
    orientation = Rotation.from_rotvec(tcp[3:]) * Rotation.from_euler("zxy", [delta[5], delta[3], delta[4]])
    return np.r_[position, orientation.as_rotvec()]


def pose_twist(tcp: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    turn = Rotation.from_rotvec(target[3:]) * Rotation.from_rotvec(tcp[3:]).inv()
    return np.r_[target[:3] - tcp[:3], turn.as_rotvec()]


class KeyboardState:
    """Held keys for continuous motion plus edge-triggered events (key repeat is ignored)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._events: list[str] = []
        self._listener = None

    @staticmethod
    def _name(key) -> str | None:
        char = getattr(key, "char", None)
        if char == " ":
            return "space"
        return char.lower() if char else getattr(key, "name", None)

    def press(self, key) -> None:
        name = self._name(key)
        with self._lock:
            if name in EVENT_KEYS and name not in self._pressed:
                self._events.append(name)
            if name is not None:
                self._pressed.add(name)

    def release(self, key) -> None:
        with self._lock:
            self._pressed.discard(self._name(key))

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise ImportError("Install pynput (pip install pynput); it needs an X11 desktop session") from exc
        self._listener = keyboard.Listener(on_press=self.press, on_release=self.release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def take_events(self) -> list[str]:
        with self._lock:
            events, self._events = self._events, []
        return events

    def delta(self, step_m: float, step_rad: float) -> np.ndarray:
        """[dx, dy, dz, dpitch, droll, dyaw] for the keys held right now."""
        with self._lock:
            pressed = set(self._pressed)
        delta = np.zeros(6)
        for keys, step in ((MOVE_KEYS, step_m), (TURN_KEYS, step_rad)):
            for name, (axis, sign) in keys.items():
                if name in pressed:
                    delta[axis] += sign * step
        return delta


class FrameWriter:
    """Write PNGs on a background thread so disk I/O never stalls the control loop."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, array = item
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(path)

    def put(self, path: Path, array: np.ndarray) -> None:
        self._queue.put((path, array))

    @property
    def backlog(self) -> int:
        return self._queue.qsize()

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()


class Episode:
    """One teleoperated demonstration: per-step arrays plus frames on disk."""

    def __init__(self, root: Path, meta: dict) -> None:
        self.root, self.meta = root, meta
        self.rows: dict[str, list] = {}
        self.start = time.monotonic()

    def add(self, **values) -> int:
        for key, value in values.items():
            self.rows.setdefault(key, []).append(value)
        return len(self.rows["t"]) - 1

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        np.savez(self.root / "data.npz", **{k: np.asarray(v) for k, v in self.rows.items()})
        self.meta["steps"] = len(self.rows.get("t", []))
        (self.root / "meta.json").write_text(json.dumps(self.meta, indent=2) + "\n")


def depth_to_color(depth_raw: np.ndarray, scale_m: float, max_m: float = 2.0) -> np.ndarray:
    """False-colour RGB view of raw depth for humans (near = red, far = blue, no data = black)."""
    import cv2

    depth_m = depth_raw.astype(np.float32) * scale_m
    gray = np.clip(255.0 * (1.0 - depth_m / max_m), 0, 255).astype(np.uint8)
    color = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    color[depth_raw == 0] = 0
    return color


def next_index(root: Path, prefix: str) -> int:
    existing = [int(p.name[len(prefix) :]) for p in root.glob(prefix + "*") if p.name[len(prefix) :].isdigit()]
    return max(existing, default=-1) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ip", default="192.168.1.101", help="Left UR5 (black gripper)")
    parser.add_argument("--rtde-frequency", type=float, default=200.0)
    parser.add_argument("--ur-cap-port", type=int, default=50002)
    parser.add_argument("--camera", default="realsense", help="realsense[:serial], OpenCV index or device path")
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--no-depth", action="store_true", help="Record RGB only")
    parser.add_argument("--depth-view-max-m", type=float, default=2.0, help="Far end of the depth colour scale")
    parser.add_argument("--gripper", choices=["none", "dm", "robotiq"], default="dm")
    parser.add_argument("--gripper-serial", default="/dev/ttyACM0", help="DM4310 USB-CAN adapter")
    parser.add_argument("--gripper-port", type=int, default=63352, help="Robotiq URCap socket port")
    parser.add_argument("--gripper-stroke-m", type=float, default=0.085, help="Measured fingertip distance, open")
    parser.add_argument("--gripper-max-close", type=float, default=0.99, help="Closing limit, fraction of travel")
    parser.add_argument("--gripper-step", type=float, default=0.01, help="Closing change per [ or ] press")
    parser.add_argument("--hz", type=float, default=20.0, help="Control and recording rate (training uses 20)")
    parser.add_argument("--speed", type=float, default=0.15, help="Translation speed while a key is held (m/s)")
    parser.add_argument("--turn-speed", type=float, default=0.24, help="Rotation speed while a key is held (rad/s)")
    parser.add_argument("--smoothing-n", type=int, default=4, help="EMA window of the key input")
    parser.add_argument("--world-tilt-deg", type=float, help="go_home.py rotation_y_deg; default from --ip")
    parser.add_argument("--lookahead-time", type=float, default=0.1)
    parser.add_argument("--gain", type=float, default=100.0)
    parser.add_argument("--jacobian-warn", type=float, default=25.0)
    parser.add_argument("--jacobian-stop", type=float, default=35.0)
    parser.add_argument("--out", type=Path, default=Path("data/real_raw"))
    args = parser.parse_args()
    if args.world_tilt_deg is None:
        args.world_tilt_deg = WORLD_TILT_DEG.get(args.ip, 0.0)

    import cv2
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    from scipy.spatial.transform import Rotation

    camera_device = int(args.camera) if args.camera.isdigit() else args.camera
    depth = not args.no_depth
    world_from_base = Rotation.from_euler("y", args.world_tilt_deg, degrees=True)
    period = 1.0 / args.hz
    alpha = 2.0 / (max(args.smoothing_n, 1) + 1.0)
    keys = KeyboardState()
    rtde_r = RTDEReceive(args.ip, args.rtde_frequency)
    rtde_c = RTDEControl(
        args.ip, args.rtde_frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT, args.ur_cap_port
    )
    gripper = camera = None
    writer = FrameWriter()
    episode = None
    try:
        gripper = make_gripper(args.gripper, args.ip, args.gripper_port, args.gripper_stroke_m, args.gripper_serial)
        camera = make_camera(camera_device, args.camera_width, args.camera_height, depth=depth)
        depth_scale = float(getattr(camera, "info", {}).get("depth_scale_m", 0.001))
        meta = {
            "robot_ip": args.ip,
            "robot_model": "UR5 (CB), measured from TCP readings",
            "conventions": "UR controller joints (rad) and base-frame TCP pose [x,y,z,rx,ry,rz], no joint_offset",
            "world_tilt_deg": args.world_tilt_deg,
            "tcp_offset": list(rtde_c.getTCPOffset()),
            "hz": args.hz,
            "gripper": args.gripper,
            "gripper_stroke_m": args.gripper_stroke_m,
            "camera": getattr(camera, "info", {"device": args.camera}),
            "fields": {
                "t": "s since episode start",
                "q": "actual joints (rad)",
                "qd": "actual joint velocity (rad/s)",
                "q_target": "controller joint target (rad)",
                "tcp": "actual TCP pose, base frame",
                "tcp_command": "servoL target (equals tcp when no command was sent)",
                "servo_sent": "whether a servoL command was sent this tick",
                "teleop_delta": "smoothed key step [dx, dy, dz (world), dpitch, droll, dyaw (TCP)]",
                "jacobian_cond": "Jacobian condition number at q",
                "gripper_close_cmd": "commanded closing, 0 open .. 1 closed",
                "gripper_command_m": "commanded fingertip distance",
                "gripper_opening_m": "measured fingertip distance (linear in motor angle for dm)",
                "gripper_torque": "DM4310 joint torque (Nm), NaN for other grippers",
            },
        }
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "calib").mkdir(exist_ok=True)
        keys.start()
        close_cmd, grip_cmd_m = 0.0, args.gripper_stroke_m
        gripper.move(grip_cmd_m)
        ema = np.zeros(6)
        last_warning = 0.0
        window = "ur5_record"
        cv2.namedWindow(window)
        print(__doc__[__doc__.index("    W / S") : __doc__.index("Near a")], flush=True)
        deadline = time.monotonic()
        quit_requested = False
        while not quit_requested:
            if rtde_r.isEmergencyStopped() or rtde_r.isProtectiveStopped():
                raise RuntimeError("UR5 is emergency or protective stopped")
            if not rtde_c.isProgramRunning():
                raise RuntimeError("RTDE control script is no longer running on the UR5")
            tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64)
            q = np.asarray(rtde_r.getActualQ(), dtype=np.float64)
            ema = alpha * keys.delta(args.speed * period, args.turn_speed * period) + (1.0 - alpha) * ema

            calib_requested = False
            for event in keys.take_events():
                if event == "esc":
                    quit_requested = True
                elif event == "space":
                    close_cmd = args.gripper_max_close if close_cmd <= 0.5 * args.gripper_max_close else 0.0
                    print(f"Gripper {'close' if close_cmd else 'open'} ({close_cmd * 100:.0f} %)", flush=True)
                elif event in ("[", "]"):
                    step = args.gripper_step if event == "[" else -args.gripper_step
                    close_cmd = float(np.clip(close_cmd + step, 0.0, args.gripper_max_close))
                    print(f"Gripper {close_cmd * 100:.0f} %", flush=True)
                elif event == "r" and episode is None:
                    index = next_index(args.out, "episode_")
                    episode = Episode(args.out / f"episode_{index:04d}", dict(meta, start_time=time.time()))
                    print(f"Recording {episode.root}", flush=True)
                elif event == "r":
                    episode.save()
                    print(f"Saved {episode.root} ({episode.meta['steps']} steps)", flush=True)
                    episode = None
                elif event == "x" and episode is not None:
                    print(f"Discarded {episode.root}", flush=True)
                    episode = None
                elif event == "p":
                    calib_requested = True

            target, sent, cond, warning = tcp.copy(), False, float("nan"), None
            if np.linalg.norm(tcp[:3]) <= 0.01:
                warning = "TCP reading invalid, motion skipped"
            else:
                command = servo_target(tcp, world_from_base, ema)
                cond, allowed = singularity_gate(q, pose_twist(tcp, command), args.jacobian_stop)
                if allowed:
                    if not rtde_c.servoL(command.tolist(), 0.2, 0.2, period, args.lookahead_time, args.gain):
                        raise RuntimeError("servoL was rejected by the controller")
                    target, sent = command, True
                else:
                    warning = f"near singularity (cond {cond:.1f}), only moves out of it are sent"
                if cond >= args.jacobian_warn and warning is None:
                    warning = f"approaching singularity (cond {cond:.1f})"
            if warning and time.monotonic() - last_warning > 1.0:
                last_warning = time.monotonic()
                print("WARNING:", warning, flush=True)
            if (1.0 - close_cmd) * args.gripper_stroke_m != grip_cmd_m:
                grip_cmd_m = (1.0 - close_cmd) * args.gripper_stroke_m
                gripper.move(grip_cmd_m)

            if depth:
                rgb, depth_raw = camera.read_rgbd()
            else:
                rgb, depth_raw = camera.read(), None
            opening = gripper.opening()
            torque = gripper.torque() if hasattr(gripper, "torque") else float("nan")
            state = {
                "q": q,
                "qd": np.asarray(rtde_r.getActualQd()),
                "q_target": np.asarray(rtde_r.getTargetQ()),
                "tcp": tcp,
                "tcp_command": target,
                "servo_sent": sent,
                "teleop_delta": ema.copy(),
                "jacobian_cond": cond,
                "gripper_close_cmd": close_cmd,
                "gripper_command_m": grip_cmd_m,
                "gripper_opening_m": opening,
                "gripper_torque": torque,
            }
            if calib_requested:
                frame = args.out / "calib" / f"frame_{next_index(args.out / 'calib', 'frame_'):04d}"
                frame.mkdir(parents=True)
                writer.put(frame / "rgb.png", rgb)
                if depth_raw is not None:
                    writer.put(frame / "depth.png", depth_raw)
                    writer.put(frame / "depth_vis.png", depth_to_color(depth_raw, depth_scale, args.depth_view_max_m))
                record = {k: np.asarray(v).tolist() for k, v in state.items()}
                (frame / "state.json").write_text(json.dumps(dict(record, meta=meta), indent=2))
                print(f"Calibration frame {frame.name} saved", flush=True)
            if episode is not None:
                step = episode.add(t=time.monotonic() - episode.start, **state)
                writer.put(episode.root / "rgb" / f"{step:06d}.png", rgb)
                if depth_raw is not None:
                    writer.put(episode.root / "depth" / f"{step:06d}.png", depth_raw)

            preview = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            status = "REC %s %.1fs" % (episode.root.name, time.monotonic() - episode.start) if episode else "idle"
            lines = [
                status,
                "tcp world [m]: %.3f %.3f %.3f" % tuple(world_from_base.apply(tcp[:3])),
                "jacobian cond %.1f" % cond,
                "gripper %.1f mm (cmd %.0f %%)  backlog %d" % (opening * 1000, close_cmd * 100, writer.backlog),
            ]
            for i, line in enumerate(lines):
                color = (0, 0, 255) if i == 0 and episode else (0, 255, 0)
                cv2.putText(preview, line, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if depth_raw is not None:
                depth_view = depth_to_color(depth_raw, depth_scale, args.depth_view_max_m)
                if depth_view.shape[:2] != preview.shape[:2]:
                    depth_view = cv2.resize(depth_view, (preview.shape[1], preview.shape[0]))
                preview = np.hstack([preview, cv2.cvtColor(depth_view, cv2.COLOR_RGB2BGR)])
            cv2.imshow(window, preview)
            cv2.waitKey(1)
            deadline += period
            time.sleep(max(0.0, deadline - time.monotonic()))
            if time.monotonic() - deadline > 5 * period:
                deadline = time.monotonic()
    finally:
        keys.stop()
        if episode is not None:
            episode.save()
            print(f"Saved {episode.root} ({episode.meta['steps']} steps)", flush=True)
        try:
            rtde_c.servoStop()
            rtde_c.stopScript()
            rtde_c.disconnect()
        except Exception:
            pass
        rtde_r.disconnect()
        writer.close()
        for resource in (gripper, camera):
            if resource is not None:
                resource.close()
        try:
            import cv2

            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
