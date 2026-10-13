"""Teleop kinematics and key handling of the UR5 recorder (no hardware)."""

from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from mujoco_task.ur5_record import KeyboardState, servo_target, singularity_gate, ur5_jacobian

UR_HOME = np.array([0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0])


def test_servo_target_moves_in_world_frame_and_turns_about_tcp():
    world_from_base = Rotation.from_euler("y", -45, degrees=True)
    tcp = np.r_[0.03, -0.35, 0.18, 0.0, -2.3562, 0.0]
    moved = servo_target(tcp, world_from_base, [0, 0, 0.01, 0, 0, 0])
    np.testing.assert_allclose(world_from_base.apply(moved[:3] - tcp[:3]), [0, 0, 0.01], atol=1e-12)
    np.testing.assert_allclose(moved[3:], tcp[3:], atol=1e-12)

    turned = servo_target(tcp, world_from_base, [0, 0, 0, 0, 0, np.pi / 2])
    np.testing.assert_allclose(turned[:3], tcp[:3])
    expected = Rotation.from_rotvec(tcp[3:]) * Rotation.from_euler("z", 90, degrees=True)
    np.testing.assert_allclose(Rotation.from_rotvec(turned[3:]).as_matrix(), expected.as_matrix(), atol=1e-9)


def test_jacobian_matches_finite_differences_of_the_flange_position():
    q = np.array([-1.0004, -1.439, -2.3294, -1.441, 0.932, 0.7382])
    eps = 1e-6
    numeric = np.stack([(_tip(q + eps * e) - _tip(q - eps * e)) / (2 * eps) for e in np.eye(6)], axis=1)
    np.testing.assert_allclose(ur5_jacobian(q)[:3], numeric, atol=1e-6)
    assert np.linalg.cond(ur5_jacobian(UR_HOME)) < 10


def test_singularity_gate_only_lets_moves_out_of_a_wrist_singularity():
    q = UR_HOME.copy()
    q[4] = 0.05
    assert singularity_gate(UR_HOME, np.zeros(6), stop=35)[1]
    jac = ur5_jacobian(q)
    away = jac @ np.r_[0, 0, 0, 0, 0.03, 0]
    cond, allowed = singularity_gate(q, away, stop=35)
    assert cond > 35 and allowed
    assert not singularity_gate(q, -away, stop=35)[1]


def test_keyboard_state_holds_moves_and_ignores_key_repeat_for_events():
    keys = KeyboardState()
    w, j, space = SimpleNamespace(char="W"), SimpleNamespace(char="j"), SimpleNamespace(char=None, name="space")
    keys.press(w)
    keys.press(j)
    keys.press(space)
    keys.press(space)
    np.testing.assert_allclose(keys.delta(0.01, 0.02), [0.01, 0, 0, 0, 0, 0.02])
    assert keys.take_events() == ["space"]
    keys.release(w)
    keys.release(space)
    keys.press(space)
    np.testing.assert_allclose(keys.delta(0.01, 0.02), [0, 0, 0, 0, 0, 0.02])
    assert keys.take_events() == ["space"]
    keys.release(space)
    keys.press(SimpleNamespace(char=" "))
    assert keys.take_events() == ["space"]


def _tip(q):
    from mujoco_task.ur5_record import UR5_DH_A, UR5_DH_ALPHA, UR5_DH_D

    transform = np.eye(4)
    for theta, d, a, alpha in zip(q, UR5_DH_D, UR5_DH_A, UR5_DH_ALPHA):
        ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
        transform = transform @ np.array(
            [[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1]]
        )
    return transform[:3, 3]
