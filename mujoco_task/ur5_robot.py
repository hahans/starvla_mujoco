"""UR5 hardware adapter for ``python -m mujoco_task.real_robot run --robot ur5``.

The arm is driven over RTDE (``ur_rtde``) in joint space with ``servoJ``, the same
connection used by ``go_home.py``. The policy controls the left arm of the simulator
(black gripper, base tilted -45 deg about world Y), i.e. the left UR5 at
``192.168.1.101``. The simulator uses UR5 CB kinematics and the UR controller base
frame (calibrated from ``real_raw2``), so joint angles map one-to-one and the default
``joint_offset`` is zero.

Standalone tools (they never load the policy)::

    python -m mujoco_task.ur5_robot status   --ip 192.168.1.101
    python -m mujoco_task.ur5_robot go-start --ip 192.168.1.101 [--gripper dm]
    python -m mujoco_task.ur5_robot snapshot --camera realsense --out logs/camera.png

``gripper="dm"`` drives the DM4310 CAN gripper through ``ur5_control`` (the folder next
to this repository, needs ``pyserial``).
"""

from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Left-arm joints of the MuJoCo keyframe "home" (= go_home.py pose); every training episode starts here.
START_Q = (-1.0004, -1.4391, -2.3295, -1.441, 0.932, 0.738)
START_FINGER_M = 0.035
FINGER_MAX_M = 0.035


def parse_joint_offset(offset=None) -> np.ndarray:
    """Real joint angle = simulator joint angle + offset (rad).

    ``[pi, 0, 0, 0, 0, 0]`` when the real base is turned 180 deg about its own axis
    relative to the simulated one, e.g. after confusing the MuJoCo base frame with the
    UR controller base frame (they differ by exactly that rotation).
    """
    offset = np.zeros(6) if offset is None else np.asarray(offset, dtype=np.float64).reshape(-1)
    if offset.shape != (6,) or not np.isfinite(offset).all():
        raise ValueError(f"joint_offset must be six finite values in rad, got {offset}")
    return offset


class NoGripper:
    """Arm-only operation: the finger reading echoes the last command."""

    def __init__(self) -> None:
        self._opening = 2 * START_FINGER_M

    def move(self, opening_m: float) -> None:
        self._opening = float(opening_m)

    def opening(self) -> float:
        return self._opening

    def close(self) -> None:
        pass


class RobotiqGripper:
    """Robotiq 2F gripper through the Robotiq URCap socket (port 63352 on the UR controller).

    Opening is the distance between the fingertips; position 0 is fully open
    (``stroke_m``) and 255 fully closed, assumed linear in between.
    """

    def __init__(
        self, host: str, port: int = 63352, stroke_m: float = 0.085, speed: int = 255, force: int = 100
    ) -> None:
        self.stroke_m = float(stroke_m)
        try:
            self._sock = socket.create_connection((host, port), timeout=2.0)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot reach the Robotiq URCap socket at {host}:{port} ({exc}); install/enable the Robotiq "
                "URCap on the UR controller, or use gripper='none' to run the arm only"
            ) from exc
        self._lock = threading.Lock()
        self._target = None
        if self._get("STA") != 3:
            self._set(ACT=1)
            deadline = time.monotonic() + 10.0
            while self._get("STA") != 3:
                if time.monotonic() > deadline:
                    raise RuntimeError("Robotiq gripper did not finish activation")
                time.sleep(0.1)
        self._set(GTO=1, SPE=int(speed), FOR=int(force))

    def _send(self, line: str) -> str:
        with self._lock:
            self._sock.sendall(line.encode("ascii"))
            return self._sock.recv(1024).decode("ascii").strip()

    def _set(self, **values: int) -> None:
        command = "SET " + " ".join(f"{k} {v}" for k, v in values.items()) + "\n"
        reply = self._send(command)
        if reply != "ack":
            raise RuntimeError(f"Robotiq rejected {command.strip()!r}: {reply!r}")

    def _get(self, name: str) -> int:
        reply = self._send(f"GET {name}\n").split()
        if len(reply) != 2 or reply[0] != name:
            raise RuntimeError(f"Unexpected Robotiq reply to GET {name}: {reply}")
        return int(reply[1])

    def move(self, opening_m: float) -> None:
        target = int(round(np.clip(1.0 - opening_m / self.stroke_m, 0.0, 1.0) * 255))
        if target != self._target:
            self._set(POS=target)
            self._target = target

    def opening(self) -> float:
        return (1.0 - self._get("POS") / 255.0) * self.stroke_m

    def close(self) -> None:
        self._sock.close()


class _LatestFrame:
    """Background thread keeping only the newest RGB frame, so ``read`` never returns a stale one."""

    def _start(self, name: str, max_age_s: float = 0.5) -> None:
        self.max_age_s = float(max_age_s)
        self._frame, self._stamp = None, 0.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while self._frame is None:
            if time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"Camera {name} delivered no frame")
            time.sleep(0.02)

    def _grab(self) -> np.ndarray | None:
        raise NotImplementedError

    def _release(self) -> None:
        raise NotImplementedError

    def _loop(self) -> None:
        while self._running:
            frame = self._grab()
            if frame is not None:
                with self._lock:
                    self._frame, self._stamp = frame, time.monotonic()
            else:
                time.sleep(0.005)

    def _latest(self):
        with self._lock:
            frame, stamp = self._frame, self._stamp
        age = time.monotonic() - stamp
        if age > self.max_age_s:
            raise RuntimeError(f"Camera frame is {age:.2f}s old")
        return frame

    def read(self) -> np.ndarray:
        """Return the newest frame as RGB uint8; fail rather than reuse a stale one."""
        return self._latest().copy()

    def close(self) -> None:
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release()


class Camera(_LatestFrame):
    """OpenCV / V4L2 capture by index or device path."""

    def __init__(self, device: int | str = 0, width: int | None = 640, height: int | None = 480) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device!r}; for an Intel RealSense use camera='realsense'")
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._start(repr(device))

    def _grab(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return np.ascontiguousarray(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)) if ok else None

    def _release(self) -> None:
        self._cap.release()


def _closest_profile(sensors, stream, formats, width: int, height: int, fps: int):
    """Pick the supported profile nearest to the request instead of failing with 'Couldn't resolve requests'."""
    profiles = [
        p.as_video_stream_profile()
        for sensor in sensors
        for p in sensor.get_stream_profiles()
        if p.stream_type() == stream and p.format() in formats
    ]
    if not profiles:
        return None
    return min(
        profiles,
        key=lambda p: (
            abs(p.width() - width) + abs(p.height() - height),
            abs(p.fps() - fps),
            formats.index(p.format()),
        ),
    )


class RealSenseCamera(_LatestFrame):
    """Color stream of an Intel RealSense through ``pyrealsense2``.

    With ``depth=True`` a depth stream aligned to the color image is grabbed as well
    (``read_rgbd``); the policy never sees it, it is recorded for calibration.
    """

    def __init__(
        self,
        serial: str | None = None,
        width: int | None = 640,
        height: int | None = 480,
        fps: int = 30,
        depth: bool = False,
    ) -> None:
        import pyrealsense2 as rs

        devices = [
            d for d in rs.context().query_devices() if not serial or d.get_info(rs.camera_info.serial_number) == serial
        ]
        if not devices:
            raise RuntimeError(f"No RealSense {serial or ''} detected; check rs-enumerate-devices")
        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        usb_info = rs.camera_info.usb_type_descriptor
        usb = device.get_info(usb_info) if device.supports(usb_info) else "?"
        sensors = device.query_sensors()
        color_formats = [rs.format.rgb8, rs.format.bgr8]
        color = _closest_profile(sensors, rs.stream.color, color_formats, width or 640, height or 480, fps)
        if color is None:
            raise RuntimeError(f"RealSense {serial} (USB {usb}) offers no RGB8/BGR8 color stream")
        self._bgr = color.format() == rs.format.bgr8
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, color.width(), color.height(), color.format(), color.fps())
        self.depth = bool(depth)
        if self.depth:
            depth_profile = _closest_profile(
                sensors, rs.stream.depth, [rs.format.z16], color.width(), color.height(), fps
            )
            if depth_profile is None:
                raise RuntimeError(f"RealSense {serial} offers no z16 depth stream")
            config.enable_stream(
                rs.stream.depth, depth_profile.width(), depth_profile.height(), rs.format.z16, depth_profile.fps()
            )
            self._align = rs.align(rs.stream.color)
        print(
            f"RealSense {device.get_info(rs.camera_info.name)} {serial} USB {usb}: color "
            f"{color.width()}x{color.height()} {color.format()} {color.fps()} fps, "
            f"depth={'on' if self.depth else 'off'}",
            flush=True,
        )
        if usb.startswith("2"):
            print("Warning: RealSense is on USB 2; use a USB 3 port and cable", flush=True)
        self._pipeline = rs.pipeline()
        try:
            profile = self._pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(f"Cannot start RealSense {serial}: {exc}; close realsense-viewer first") from exc
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.info = {
            "name": device.get_info(rs.camera_info.name),
            "serial": serial,
            "usb": usb,
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "distortion_model": str(intr.model),
            "distortion_coeffs": list(intr.coeffs),
        }
        if self.depth:
            self.info["depth_scale_m"] = profile.get_device().first_depth_sensor().get_depth_scale()
        try:
            for _ in range(30):  # let auto exposure settle
                self._pipeline.wait_for_frames(5000)
        except BaseException:
            self._pipeline.stop()
            raise
        self._start(f"realsense:{serial}")

    def _grab(self):
        try:
            frames = self._pipeline.wait_for_frames(1000)
            if self.depth:
                frames = self._align.process(frames)
        except RuntimeError:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        rgb = np.asanyarray(color.get_data())
        rgb = np.ascontiguousarray(rgb[..., ::-1] if self._bgr else rgb)
        if not self.depth:
            return rgb
        depth = frames.get_depth_frame()
        return (rgb, np.asanyarray(depth.get_data()).copy()) if depth else None

    def read(self) -> np.ndarray:
        frame = self._latest()
        return (frame[0] if self.depth else frame).copy()

    def read_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        """Newest RGB uint8 frame and aligned raw depth (uint16, multiply by ``info['depth_scale_m']``)."""
        if not self.depth:
            raise RuntimeError("RealSenseCamera was opened without depth")
        rgb, depth = self._latest()
        return rgb.copy(), depth.copy()

    def _release(self) -> None:
        self._pipeline.stop()


def make_camera(device: int | str, width: int | None, height: int | None, depth: bool = False) -> _LatestFrame:
    """``realsense`` / ``realsense:<serial>`` opens a RealSense; anything else goes to OpenCV."""
    if isinstance(device, str) and device.split(":")[0] == "realsense":
        return RealSenseCamera(device.partition(":")[2] or None, width, height, depth=depth)
    if depth:
        raise ValueError("Depth recording needs camera='realsense'")
    return Camera(device, width, height)


def _import_dm_adapter():
    """``ur5_control`` (the DM4310 CAN driver) sits in the folder that contains this repository."""
    import importlib
    import sys

    try:
        return importlib.import_module("ur5_control.lerobot_adapter")
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        return importlib.import_module("ur5_control.lerobot_adapter")


class DMGripper:
    """DM4310 CAN gripper from ``ur5_control`` (0 % open .. 100 % closed on the motor scale).

    MIT position commands must be refreshed continuously, so a background thread sends
    them at ``rate_hz``; ``move`` only changes the target. Fingertip distance is assumed
    linear in the closing percentage, from ``stroke_m`` at 0 % to 0 at 100 %.
    """

    def __init__(
        self,
        serial_port: str = "/dev/ttyACM0",
        stroke_m: float = 0.085,
        max_close: float = 0.99,
        rate_hz: float = 50.0,
        read_only: bool = False,
    ) -> None:
        Driver = _import_dm_adapter().DMGripper
        self.stroke_m, self.max_close, self.read_only = float(stroke_m), float(max_close), bool(read_only)
        self._driver = Driver(serial_port, read_only=self.read_only)
        self._running, self._error = False, None
        if self.read_only:
            return
        self._driver.set_pos(0.0)
        self._running = True
        self._period = 1.0 / float(rate_hz)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                self._driver.ctl(non_block=True)
            except Exception as exc:
                self._error, self._running = exc, False
                return
            time.sleep(self._period)

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"DM gripper control loop stopped: {self._error}") from self._error

    def percent(self) -> float:
        """Measured closing percentage."""
        self._check()
        if self.read_only:
            return float(self._driver.sdk.get_angle())
        return float(self._driver.get_angle())

    def move(self, opening_m: float) -> None:
        self._check()
        percent = (1.0 - float(opening_m) / self.stroke_m) * 100.0
        self._driver.set_pos(float(np.clip(percent, 0.0, 100.0 * self.max_close)))

    def opening(self) -> float:
        return float(np.clip(1.0 - self.percent() / 100.0, 0.0, 1.0)) * self.stroke_m

    def torque(self) -> float:
        return float(self._driver.get_currents()["DM4310"])

    def close(self) -> None:
        if self._running:
            self._running = False
            self._thread.join(timeout=1.0)
        self._driver.close()


def make_gripper(
    kind: str,
    host: str,
    port: int,
    stroke_m: float,
    serial_port: str = "/dev/ttyACM0",
    read_only: bool = False,
):
    if kind == "none":
        return NoGripper()
    if kind == "dm":
        return DMGripper(serial_port, stroke_m, read_only=read_only)
    if kind == "robotiq":
        return RobotiqGripper(host, port, stroke_m)
    raise ValueError("gripper must be 'none', 'dm' or 'robotiq'")


class UR5Robot:
    """``Robot`` implementation: RTDE servoJ arm, DM4310/Robotiq gripper, OpenCV or RealSense camera.

    ``read_only`` connects only the RTDE receive interface and never moves anything,
    which is what ``run --no-send`` uses.
    """

    def __init__(
        self,
        robot_ip: str = "192.168.1.101",
        camera: int | str = 0,
        camera_width: int | None = 640,
        camera_height: int | None = 480,
        rgb_profile: str | None = None,
        gripper: str = "none",
        gripper_port: int = 63352,
        gripper_serial: str = "/dev/ttyACM0",
        gripper_stroke_m: float = 0.085,
        control_hz: float = 20.0,
        rtde_frequency: float = 200.0,
        ur_cap_port: int = 50002,
        lookahead_time: float = 0.1,
        gain: float = 300.0,
        max_tracking_error: float = 0.3,
        start_tolerance: float = 0.15,
        check_start: bool = True,
        joint_offset: list[float] | None = None,
        read_only: bool = False,
    ) -> None:
        from rtde_receive import RTDEReceiveInterface as RTDEReceive

        self.offset = parse_joint_offset(joint_offset)
        self.dt = 1.0 / float(control_hz)
        self.lookahead_time = float(lookahead_time)
        self.gain = float(gain)
        self.max_tracking_error = float(max_tracking_error)
        self.read_only = bool(read_only)
        self._rtde_r = self._rtde_c = self._camera = self._gripper = None
        self._profile = None
        try:
            self._rtde_r = RTDEReceive(robot_ip, rtde_frequency)
            if check_start:
                error = np.abs(self.joint_pos() - (np.asarray(START_Q) + self.offset)).max()
                if error > start_tolerance:
                    raise RuntimeError(
                        f"UR5 is {error:.3f} rad from the training start pose (joint_offset {self.offset.tolist()}); "
                        f"run 'python -m mujoco_task.ur5_robot go-start --ip {robot_ip}' with the same offset first"
                    )
            if not self.read_only:
                from rtde_control import RTDEControlInterface as RTDEControl

                self._rtde_c = RTDEControl(
                    robot_ip,
                    rtde_frequency,
                    RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT,
                    ur_cap_port,
                )
            self._gripper = make_gripper(
                gripper, robot_ip, gripper_port, gripper_stroke_m, gripper_serial, read_only=self.read_only
            )
            if not self.read_only:
                self._gripper.move(2.0 * START_FINGER_M)
            if rgb_profile:
                from .rgb_to_sim import load_profile

                self._profile = load_profile(Path(rgb_profile))
            self._camera = make_camera(camera, camera_width, camera_height)
        except BaseException:
            self.close()
            raise

    def joint_pos(self) -> np.ndarray:
        if not self._rtde_r.isConnected():
            self._rtde_r.reconnect()
        q = np.asarray(self._rtde_r.getActualQ(), dtype=np.float64)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise RuntimeError(f"Invalid getActualQ() output: {q}")
        return q

    def _check_safety(self) -> None:
        if self._rtde_r.isEmergencyStopped() or self._rtde_r.isProtectiveStopped():
            raise RuntimeError("UR5 is emergency or protective stopped")
        if self._rtde_c is not None and not self._rtde_c.isProgramRunning():
            raise RuntimeError("RTDE control script is no longer running on the UR5")

    def observe(self) -> dict:
        """Newest camera frame plus joint state in simulator convention; fingers are symmetric."""
        self._check_safety()
        image = self._camera.read()
        if self._profile is not None:
            from .rgb_to_sim import convert_rgb

            image = convert_rgb(image, self._profile)
        finger = min(self._gripper.opening() / 2.0, FINGER_MAX_M)
        return {
            "image": image,
            "joint_pos": (self.joint_pos() - self.offset).astype(np.float32),
            "joint_vel": np.asarray(self._rtde_r.getActualQd(), dtype=np.float32),
            "finger_pos": np.full(2, finger, dtype=np.float32),
        }

    def command(self, action: np.ndarray) -> None:
        """Stream one joint target with servoJ and update the gripper only when it changes."""
        if self.read_only:
            raise RuntimeError("UR5Robot was opened read-only")
        action = np.asarray(action, dtype=np.float64)
        q = action[:6] + self.offset
        self._check_safety()
        error = np.abs(q - self.joint_pos()).max()
        if error > self.max_tracking_error:
            self.stop()
            raise RuntimeError(f"Target is {error:.3f} rad from the measured pose; stopped")
        if not self._rtde_c.servoJ(q.tolist(), 0.5, 0.5, self.dt, self.lookahead_time, self.gain):
            self.stop()
            raise RuntimeError("servoJ was rejected by the controller")
        self._gripper.move(2.0 * float(action[6]))

    def stop(self) -> None:
        if self._rtde_c is not None:
            try:
                self._rtde_c.servoStop()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        for close in (
            getattr(self._rtde_c, "stopScript", None),
            getattr(self._rtde_c, "disconnect", None),
            getattr(self._rtde_r, "disconnect", None),
            getattr(self._gripper, "close", None),
            getattr(self._camera, "close", None),
        ):
            if close is not None:
                try:
                    close()
                except Exception:
                    pass


def _status(args: argparse.Namespace) -> None:
    from rtde_receive import RTDEReceiveInterface as RTDEReceive

    rtde_r = RTDEReceive(args.ip, args.rtde_frequency)
    try:
        q = np.asarray(rtde_r.getActualQ())
        start = np.asarray(START_Q) + parse_joint_offset(args.joint_offset)
        print("joint q (rad):     ", np.round(q, 4).tolist())
        print("start q (real):    ", np.round(start, 4).tolist())
        print("q - start:         ", np.round(q - start, 4).tolist())
        print("TCP pose (base):   ", np.round(rtde_r.getActualTCPPose(), 4).tolist())
        return q
    finally:
        rtde_r.disconnect()


def check_large_move(q: np.ndarray, target: np.ndarray, limit: float) -> None:
    """moveJ interpolates every joint at once; a long move can sweep the whole workspace."""
    delta = np.abs(np.asarray(target) - np.asarray(q))
    if delta.max() > limit:
        joint = int(np.argmax(delta))
        raise SystemExit(
            f"Joint {joint} would move {delta[joint]:.2f} rad (> {limit} rad). Jog the arm close to the start "
            "pose with the teach pendant first, or pass --allow-large-move after checking the path is clear."
        )


def _go_start(args: argparse.Namespace) -> None:
    from rtde_control import RTDEControlInterface as RTDEControl

    start = np.asarray(START_Q) + parse_joint_offset(args.joint_offset)
    q = _status(args)
    if not args.allow_large_move:
        check_large_move(q, start, args.max_move)
    if not args.yes and input("moveJ to the training start pose? [y/N] ").strip().lower() != "y":
        return
    rtde_c = RTDEControl(
        args.ip, args.rtde_frequency, RTDEControl.FLAG_VERBOSE | RTDEControl.FLAG_UPLOAD_SCRIPT, args.ur_cap_port
    )
    gripper = None
    try:
        if not rtde_c.moveJ(start.tolist(), args.speed, args.acc):
            raise RuntimeError("moveJ failed")
        gripper = make_gripper(args.gripper, args.ip, args.gripper_port, args.gripper_stroke_m, args.gripper_serial)
        gripper.move(2 * START_FINGER_M)
        time.sleep(1.0)
        print("Reached the training start pose.")
    finally:
        if gripper is not None:
            gripper.close()
        rtde_c.stopScript()
        rtde_c.disconnect()


def _snapshot(args: argparse.Namespace) -> None:
    from .real_robot import prepare_image

    camera = make_camera(args.camera, args.camera_width, args.camera_height)
    try:
        image = camera.read()
    finally:
        camera.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.out)
    policy_view = args.out.with_name(args.out.stem + f"_policy{args.image_size}.png")
    Image.fromarray(prepare_image(image, args.image_size)).save(policy_view)
    print(f"Saved {args.out} {image.shape} and the policy input {policy_view}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("status", "Print joints, distance to the start pose and TCP pose"),
        ("go-start", "moveJ to the training start pose"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--ip", default="192.168.1.101", help="Left UR5 (black gripper)")
        p.add_argument("--rtde-frequency", type=float, default=200.0)
        p.add_argument("--ur-cap-port", type=int, default=50002)
        p.add_argument(
            "--joint-offset",
            type=float,
            nargs=6,
            metavar="RAD",
            help="Real minus simulator joint angles, e.g. 3.141593 0 0 0 0 0 for a base turned 180 deg",
        )
    go = sub.choices["go-start"]
    go.add_argument("--max-move", type=float, default=1.5, help="Refuse moveJ if any joint moves more (rad)")
    go.add_argument("--allow-large-move", action="store_true")
    go.add_argument("--speed", type=float, default=0.2, help="rad/s")
    go.add_argument("--acc", type=float, default=0.3, help="rad/s^2")
    go.add_argument("--gripper", choices=["none", "dm", "robotiq"], default="none")
    go.add_argument("--gripper-port", type=int, default=63352, help="Robotiq URCap socket port")
    go.add_argument("--gripper-serial", default="/dev/ttyACM0", help="DM4310 USB-CAN adapter")
    go.add_argument("--gripper-stroke-m", type=float, default=0.085, help="Fingertip distance when fully open")
    go.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    snap = sub.add_parser("snapshot", help="Save one camera frame and its 128x128 policy input")
    snap.add_argument("--camera", default="0", help="OpenCV index, device path, or realsense[:serial]")
    snap.add_argument("--camera-width", type=int, default=640)
    snap.add_argument("--camera-height", type=int, default=480)
    snap.add_argument("--image-size", type=int, default=128)
    snap.add_argument("--out", type=Path, default=Path("logs/camera.png"))
    args = parser.parse_args()
    if args.command == "snapshot":
        args.camera = int(args.camera) if args.camera.isdigit() else args.camera
        _snapshot(args)
    elif args.command == "status":
        _status(args)
    else:
        _go_start(args)


if __name__ == "__main__":
    main()
