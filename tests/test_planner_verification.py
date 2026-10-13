"""Physics checks must not approve missing contact or inconsistent teacher labels."""

import copy

import pytest

from mujoco_task.publish_verified_planner import confirmed_record
from mujoco_task.verify_planner import check_boundary


@pytest.mark.parametrize("task", ["grasp_lift", "transfer", "place_release"])
def test_boundary_requires_physical_objective(task):
    good = {
        "tcp_green_distance": 0.004,
        "min_finger_opening": 0.035,
        "finger_contacts": [True, True],
        "lift_m": 0.17,
        "grounded": False,
        "goal_xy_error": 0.001,
    }
    check_boundary(task, good, 0.035)
    bad = {**good, "tcp_green_distance": 0.1, "finger_contacts": [False, True]}
    with pytest.raises(ValueError):
        check_boundary(task, bad, 0.035)


def test_confirmation_retains_original_uncertainty_and_rejects_conflicts():
    tasks = [{"id": str(i)} for i in range(4)]
    segments = [{"start": i * 10, "end": (i + 1) * 10, "subtask_id": str(i)} for i in range(4)]
    teacher = {
        "source_sha256": "abc",
        "source": "train/a.h5",
        "annotation": {
            "segments": [{**s, "confidence": 0.7, "needs_review": True, "reason": "uncertain"} for s in segments]
        },
    }
    replay = {
        "status": "verified",
        "entry": {"sha256": "abc", "path": "train/a.h5", "length": 40},
        "segments": segments,
        "verification_id": "id",
        "method": "full_deterministic_physics_replay",
        "frames_checked": 40,
        "annotation_semantics": "expert_subtask_to_execute_from_pre_action_observation",
    }
    result = confirmed_record(teacher, replay, tasks)
    assert result["teacher_annotation"] == teacher["annotation"]
    assert teacher["annotation"]["segments"][0]["needs_review"]
    assert not result["annotation"]["segments"][0]["needs_review"]
    conflicting = copy.deepcopy(replay)
    conflicting["segments"][0]["end"] = 11
    conflicting["segments"][1]["start"] = 11
    with pytest.raises(ValueError, match="differs"):
        confirmed_record(teacher, conflicting, tasks)
    with pytest.raises(ValueError, match="failed"):
        confirmed_record(teacher, {**replay, "status": "failed"}, tasks)
