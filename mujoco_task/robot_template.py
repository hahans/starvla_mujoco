"""Hardware adapter template for ``python -m mujoco_task.real_robot run --robot ...``.

Copy this file, fill in the four methods for your arm, camera and gripper, then run
``--robot your_module:YourRobot``. Every value must use the simulator's conventions:
see ``mujoco_task.real_robot.Robot``.
"""

from __future__ import annotations

import numpy as np


class TemplateRobot:
    """Skeleton showing the exact observation and command contract."""

    def __init__(self, robot_ip: str = "192.168.1.10", camera_id: int = 0) -> None:
        # TODO(agent): connect the arm (e.g. UR RTDE), the gripper and the fixed camera.
        # The camera must reproduce the training viewpoint: the "policy_cam" camera of
        # scene.xml, i.e. the calibrated RealSense pose (centre 480x480 crop, fovy 42.7).
        raise NotImplementedError("Implement the connection for your hardware")

    def observe(self) -> dict:
        """Return the newest synchronized camera frame and robot state."""
        image = np.zeros((480, 640, 3), np.uint8)  # RGB, not BGR; center-cropped and resized to 128
        joint_pos = np.zeros(6, np.float32)  # rad: shoulder_pan, shoulder_lift, elbow, wrist_1..3
        joint_vel = np.zeros(6, np.float32)  # rad/s
        finger_pos = np.zeros(2, np.float32)  # m per finger, 0 closed .. 0.035 open
        return {"image": image, "joint_pos": joint_pos, "joint_vel": joint_vel, "finger_pos": finger_pos}

    def command(self, action: np.ndarray) -> None:
        """Track ``action[:6]`` (absolute joint targets, rad) and ``action[6]`` (finger target, m) for 50 ms."""
        raise NotImplementedError

    def close(self) -> None:
        """Stop motion and disconnect."""
