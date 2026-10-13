"""UR5 adapter against fake ur_rtde modules and a fake camera; no hardware needed."""

import sys
import time
import types

import numpy as np
import pytest

from mujoco_task import ur5_robot
from mujoco_task.ur5_robot import START_FINGER_M, START_Q, NoGripper, RobotiqGripper, UR5Robot


class _Receive:
    q = np.array(START_Q)
    protective = False

    def __init__(self, ip, frequency) -> None:
        self.disconnected = False

    def isConnected(self):
        return True

    def getActualQ(self):
        return list(_Receive.q)

    def getActualQd(self):
        return [0.0] * 6

    def isEmergencyStopped(self):
        return False

    def isProtectiveStopped(self):
        return _Receive.protective

    def disconnect(self):
        self.disconnected = True


class _Control:
    FLAG_VERBOSE, FLAG_UPLOAD_SCRIPT = 1, 2
    instances = []

    def __init__(self, ip, frequency, flags, port) -> None:
        self.servo, self.stopped, self.script_stopped = [], 0, False
        _Control.instances.append(self)

    def isProgramRunning(self):
        return True

    def servoJ(self, q, speed, acc, t, lookahead, gain):
        self.servo.append((q, t))
        _Receive.q = np.asarray(q)
        return True

    def servoStop(self):
        self.stopped += 1

    def stopScript(self):
        self.script_stopped = True

    def disconnect(self):
        pass


class _Camera:
    def __init__(self, device, width, height) -> None:
        self.closed = False

    def read(self):
        return np.zeros((48, 64, 3), np.uint8)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fake_hardware(monkeypatch):
    monkeypatch.setitem(sys.modules, "rtde_receive", types.SimpleNamespace(RTDEReceiveInterface=_Receive))
    monkeypatch.setitem(sys.modules, "rtde_control", types.SimpleNamespace(RTDEControlInterface=_Control))
    monkeypatch.setattr(ur5_robot, "Camera", _Camera)
    _Receive.q, _Receive.protective = np.array(START_Q), False
    _Control.instances.clear()


def test_observe_and_command_follow_the_robot_protocol():
    robot = UR5Robot(control_hz=20)
    obs = robot.observe()
    assert obs["image"].shape == (48, 64, 3)
    np.testing.assert_allclose(obs["joint_pos"], START_Q, atol=1e-6)
    np.testing.assert_allclose(obs["finger_pos"], [START_FINGER_M, START_FINGER_M])
    target = np.r_[np.asarray(START_Q) + 0.05, 0.01]
    robot.command(target)
    q, t = _Control.instances[0].servo[-1]
    np.testing.assert_allclose(q, target[:6])
    assert t == pytest.approx(0.05)
    np.testing.assert_allclose(robot.observe()["finger_pos"], [0.01, 0.01])
    robot.close()
    assert _Control.instances[0].script_stopped


def test_refuses_to_start_away_from_training_pose():
    _Receive.q = np.zeros(6)
    with pytest.raises(RuntimeError, match="go-start"):
        UR5Robot()
    assert not _Control.instances


def test_joint_offset_converts_between_simulator_and_real_angles():
    offset = np.r_[np.pi, np.zeros(5)]
    _Receive.q = np.asarray(START_Q) + offset
    robot = UR5Robot(joint_offset=offset.tolist())
    np.testing.assert_allclose(robot.observe()["joint_pos"], START_Q, atol=1e-6)
    target = np.r_[np.asarray(START_Q) + 0.05, 0.02]
    robot.command(target)
    np.testing.assert_allclose(_Control.instances[0].servo[-1][0], target[:6] + offset)
    robot.close()
    with pytest.raises(RuntimeError, match="start pose"):
        UR5Robot()


def test_go_start_refuses_large_moves():
    ur5_robot.check_large_move(np.zeros(6), np.full(6, 1.0), 1.5)
    with pytest.raises(SystemExit, match="Joint 2"):
        ur5_robot.check_large_move(np.zeros(6), np.r_[0, 0, 4.9, 0, 0, 0], 1.5)


def test_large_jump_and_protective_stop_halt_the_arm():
    robot = UR5Robot()
    with pytest.raises(RuntimeError, match="measured pose"):
        robot.command(np.r_[np.asarray(START_Q) + 0.5, 0.02])
    assert _Control.instances[0].stopped and not _Control.instances[0].servo
    _Receive.protective = True
    with pytest.raises(RuntimeError, match="stopped"):
        robot.observe()


def test_read_only_never_connects_control():
    robot = UR5Robot(read_only=True)
    robot.observe()
    assert not _Control.instances
    with pytest.raises(RuntimeError, match="read-only"):
        robot.command(np.r_[START_Q, 0.02])


def test_make_camera_dispatches_realsense(monkeypatch):
    opened = []
    monkeypatch.setattr(ur5_robot, "RealSenseCamera", lambda serial, w, h, depth: opened.append(("rs", serial)))
    ur5_robot.make_camera("realsense", 640, 480)
    ur5_robot.make_camera("realsense:123", 640, 480)
    assert opened == [("rs", None), ("rs", "123")]
    assert isinstance(ur5_robot.make_camera(2, 640, 480), _Camera)


def test_robotiq_position_mapping():
    gripper = RobotiqGripper.__new__(RobotiqGripper)
    gripper.stroke_m, gripper._target = 0.085, None
    sent = []
    gripper._set = lambda **values: sent.append(values["POS"])
    gripper._get = lambda name: 255
    gripper.move(0.085)
    gripper.move(0.0)
    gripper.move(0.0)
    assert sent == [0, 255] and gripper.opening() == 0.0
    no = NoGripper()
    no.move(0.03)
    assert no.opening() == 0.03


class _DMDriver:
    def __init__(self, port, *, read_only=False) -> None:
        self.port, self.read_only = port, read_only
        self.target, self.ticks, self.closed = None, 0, False

    def set_pos(self, percent):
        self.target = percent

    def ctl(self, non_block=True):
        self.ticks += 1

    def get_angle(self):
        return 40.0

    def get_currents(self):
        return {"DM4310": -0.2}

    def close(self):
        self.closed = True


def test_dm_gripper_maps_opening_and_keeps_the_mit_loop_alive(monkeypatch):
    monkeypatch.setattr(ur5_robot, "_import_dm_adapter", lambda: types.SimpleNamespace(DMGripper=_DMDriver))
    gripper = ur5_robot.make_gripper("dm", "ip", 0, 0.08, "/dev/ttyACM1")
    assert gripper._driver.port == "/dev/ttyACM1" and gripper._driver.target == 0.0
    gripper.move(0.02)
    assert gripper._driver.target == pytest.approx(75.0)
    gripper.move(0.0)
    assert gripper._driver.target == pytest.approx(99.0)
    assert gripper.opening() == pytest.approx(0.048) and gripper.torque() == -0.2
    deadline = time.monotonic() + 1.0
    while gripper._driver.ticks < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert gripper._driver.ticks >= 3
    gripper.close()
    assert gripper._driver.closed and not gripper._thread.is_alive()
