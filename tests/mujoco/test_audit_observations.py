"""Detect image, physics and observation-shape mismatches independently."""

import numpy as np
import pytest

from mujoco_task.audit_observations import compare_observation


def test_distinguishes_visual_difference_from_state_difference():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.zeros(21, dtype=np.float32)
    rendered = image.copy()
    rendered[0, 0, 1] = 12
    result = compare_observation(image, state, {"image": rendered, "proprio": state.copy()})
    assert not result["image_exact"]
    assert result["image_mae_8bit"] == 1
    assert result["image_changed_pixel_fraction"] == 0.25
    assert result["proprio_within_1e_6"]
    changed = state.copy()
    changed[0] = 0.01
    result = compare_observation(image, state, {"image": image.copy(), "proprio": changed})
    assert result["image_exact"]
    assert not result["proprio_within_1e_6"]
    assert result["recorded_image_sha256"] == result["rendered_image_sha256"]


def test_no_uint8_subtraction_wraparound():
    black = np.zeros((2, 2, 3), dtype=np.uint8)
    white = np.full_like(black, 255)
    state = np.zeros(21, dtype=np.float32)
    result = compare_observation(white, state, {"image": black, "proprio": state})
    assert result["image_mae_8bit"] == 255
    assert result["image_max_abs_8bit"] == 255


def test_rejects_observation_shape_mismatch():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    state = np.zeros(21, dtype=np.float32)
    with pytest.raises(ValueError, match="shapes differ"):
        compare_observation(image, state, {"image": image, "proprio": state[:-1]})
