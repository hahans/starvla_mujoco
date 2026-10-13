"""Deployment loop contract: state layout, safety limits and chunk execution."""

import contextlib
import types

import numpy as np
import pytest

from mujoco_task.real_robot import (
    LocalPolicy,
    SafetyFilter,
    build_state,
    control_loop,
    initial_command,
    prepare_image,
)


def test_state_matches_simulator_order():
    state = build_state(np.arange(6), np.arange(6, 12), [0.02, 0.03], np.arange(20, 27))
    assert state.shape == (21,) and state.dtype == np.float32
    np.testing.assert_array_equal(state, np.r_[np.arange(12), 0.02, 0.03, np.arange(20, 27)].astype(np.float32))
    with pytest.raises(ValueError):
        build_state(np.zeros(6), np.zeros(6), np.zeros(2), np.zeros(6))
    with pytest.raises(ValueError):
        build_state(np.full(6, np.nan), np.zeros(6), np.zeros(2), np.zeros(7))


def test_initial_command_uses_current_pose_and_mean_finger():
    np.testing.assert_allclose(
        initial_command({"joint_pos": np.ones(6), "finger_pos": [0.02, 0.03]}), np.r_[np.ones(6), 0.025]
    )


def test_safety_filter_limits_slew_range_and_finger():
    safety = SafetyFilter(np.r_[-np.ones(6), 0.0], np.r_[np.ones(6), 0.035], max_delta=0.1, margin=0.05)
    last = np.zeros(7)
    applied, clipped = safety(np.r_[0.05, 0.5, -0.5, 0, 0, 0, 0.01], last)
    np.testing.assert_allclose(applied, [0.05, 0.1, -0.1, 0, 0, 0, 0.01], atol=1e-6)
    assert clipped
    applied, clipped = safety(np.r_[np.full(6, 2.0), 0.1], np.full(7, 1.04))
    np.testing.assert_allclose(applied, np.r_[np.full(6, 1.05), 0.035], atol=1e-6)
    assert clipped
    _, clipped = safety(np.r_[np.zeros(6), 0.02], last)
    assert not clipped
    applied, _ = safety(np.r_[np.full(6, 2.0), 0.02], np.full(7, 1.5))
    np.testing.assert_allclose(applied[:6], 1.4, atol=1e-6)  # outside the range: no jump to the bound
    with pytest.raises(ValueError):
        safety(np.full(7, np.inf), last)


def test_prepare_image_center_crops_to_training_size():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:, 20:60] = 255  # the center square
    out = prepare_image(image, 16)
    assert out.shape == (16, 16, 3) and out.min() == 255
    with pytest.raises(ValueError):
        prepare_image(np.zeros((8, 8), np.uint8), 16)


class _Robot:
    def __init__(self, succeed_after: int | None = None) -> None:
        self.commands = []
        self.succeed_after = succeed_after

    @property
    def success(self) -> bool:
        return self.succeed_after is not None and len(self.commands) >= self.succeed_after

    def observe(self) -> dict:
        q = self.commands[-1][:6] if self.commands else np.zeros(6)
        image = np.zeros((8, 8, 3), np.uint8)
        return {"image": image, "joint_pos": q, "joint_vel": np.zeros(6), "finger_pos": [0.02] * 2}

    def command(self, action: np.ndarray) -> None:
        self.commands.append(action)


class _Policy:
    def __init__(self, failure_at: int | None = None) -> None:
        self.states = []
        self.failure_at = failure_at

    def predict_action(self, image, state, instruction=None) -> dict:
        self.states.append(state)
        if self.failure_at is not None and len(self.states) > self.failure_at:
            return {"failure": "bad FAST", "actions": None, "latency_s": 0.0}
        actions = np.tile(np.r_[np.full(6, 1.0), 0.03], (8, 1))
        return {"failure": None, "actions": actions, "latency_s": 0.0}


def _safety() -> SafetyFilter:
    return SafetyFilter(np.r_[-np.ones(6), 0.0], np.r_[np.ones(6), 0.035], max_delta=0.12, margin=0.0)


def test_loop_executes_filtered_chunks_and_feeds_back_last_command():
    robot, policy = _Robot(), _Policy()
    result = control_loop(robot, policy, _safety(), execute=4, max_steps=10, control_hz=1e6)
    assert result == {"steps": 10, "failure": None, "success": False}
    assert len(robot.commands) == 10 and len(policy.states) == 3
    np.testing.assert_allclose(robot.commands[0][:6], 0.12, atol=1e-6)
    np.testing.assert_allclose(robot.commands[3][:6], 0.48, atol=1e-6)
    np.testing.assert_allclose(policy.states[1][14:], robot.commands[3], atol=1e-6)


def test_loop_stops_on_success_and_on_invalid_output():
    robot = _Robot(succeed_after=3)
    result = control_loop(robot, _Policy(), _safety(), execute=8, max_steps=50, control_hz=1e6)
    assert result["success"] and result["steps"] == 3
    robot, policy = _Robot(), _Policy(failure_at=1)
    result = control_loop(robot, policy, _safety(), execute=2, max_steps=50, control_hz=1e6)
    assert result["failure"] == "bad FAST" and len(robot.commands) == 2


def _local_policy(use_state: bool):
    policy = LocalPolicy.__new__(LocalPolicy)
    policy.use_state, policy.center_crop, policy.max_new_tokens = use_state, True, 8
    policy.metadata = {"image_size": 8, "instruction": "move green"}
    policy.norm = types.SimpleNamespace(unnormalize=lambda a: np.asarray(a) * 2)
    policy._torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    policy.predictor, examples = None, []

    def rows(predictor, batch, max_new_tokens):
        examples.extend(batch)
        return [{"failure": None, "actions": np.ones((8, 7)), "fast_token_ids": [1]}]

    policy._greedy_rows = rows
    return policy, examples


def test_state_is_optional_only_for_no_state_checkpoints():
    image = np.zeros((8, 8, 3), np.uint8)
    policy, examples = _local_policy(use_state=False)
    result = policy.predict_action(image=image)
    assert "state" not in examples[0] and result["actions"].max() == 2
    policy, examples = _local_policy(use_state=True)
    with pytest.raises(ValueError, match="state"):
        policy.predict_action(image=image)
    policy.predict_action(image=image, state=np.zeros(21))
    assert examples[0]["state"].shape == (21,)


def test_no_send_never_commands():
    robot = _Robot()
    control_loop(robot, _Policy(), _safety(), execute=8, max_steps=8, control_hz=1e6, send=False)
    assert robot.commands == []
