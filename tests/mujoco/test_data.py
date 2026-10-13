"""Regression checks for split isolation, frame alignment and absolute actions."""

import json

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from mujoco_task.annotate_planner import file_hash
from mujoco_task.convert import convert
from mujoco_task.data import TaskDataset, episode_path, read_json, read_video
from mujoco_task.publish_verified_planner import add_hdf5_labels


def test_conversion_lossless_alignment_and_stats(converted):
    _, target, original = converted
    assert read_json(target / "conversion.json")["splits"]["train"]["frames"] == 10
    normalization = read_json(target / "normalization.json")
    assert max(normalization["max"][:6]) < 1.01  # excludes val/test extrema at 10/20
    for split in ("train", "val", "test"):
        root = target / f"block_{split}"
        images, state, action = original[split, 0]
        assert np.array_equal(read_video(episode_path(root, 0, True)), images)
        table = pq.read_table(episode_path(root, 0))
        np.testing.assert_array_equal(table["action"].to_pylist(), action)
        np.testing.assert_array_equal(table["observation.state"].to_pylist(), state)
        assert len(table) == len(images)  # terminal_image was not appended


def test_chunk_padding_never_crosses_episode(converted):
    _, target, originals = converted
    ds = TaskDataset(target / "block_train", target / "normalization.json", horizon=8)
    sample = ds[4]
    expected = np.repeat(originals["train", 0][2][-1:], 8, axis=0)
    np.testing.assert_allclose(ds.normalizer.unnormalize(sample["action"]), expected, atol=1e-7)
    assert ds.locate(5) == (1, 0)
    assert sample["state"].shape == (21,)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_hash_corruption_and_overwrite_fail(converted, tmp_path):
    source, target, _ = converted
    with pytest.raises(FileExistsError):
        convert(source, target)
    with (source / "train_0.h5").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        convert(source, tmp_path / "corrupt_output")


def test_cross_split_duplicate_is_rejected(converted, tmp_path):
    source, _, _ = converted
    manifest = read_json(source / "manifest.json")
    manifest["episodes"][-1]["seed"] = manifest["episodes"][0]["seed"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        convert(source, tmp_path / "leak")


def test_official_lerobot_reader_optional(converted):
    pytest.importorskip("lerobot.datasets.lerobot_dataset")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    _, target, originals = converted
    ds = LeRobotDataset("local/block_train", root=target / "block_train", video_backend="pyav")
    assert len(ds) == 10
    sample = ds[3]
    assert sample["observation.images.primary"].shape == (3, 32, 32)
    np.testing.assert_allclose(sample["action"].numpy(), originals["train", 0][2][3])


def label_source(source):
    """Add a tiny four-stage planner fixture and update its source manifest."""
    tasks = [{"id": f"subtask_{i}", "subgoal": f"do subtask {i}"} for i in range(4)]
    manifest = read_json(source / "manifest.json")
    for entry in manifest["episodes"]:
        record = {
            "annotation": {
                "segments": [
                    {"start": i, "end": i + 1 if i < 3 else 5, "subtask_id": task["id"]} for i, task in enumerate(tasks)
                ]
            },
            "verification": {"quality_status": "verified_by_physics_replay", "frames_checked": 5},
            "teacher_annotation": {"fixture": True},
            "source_sha256": entry["sha256"],
        }
        path = source / entry["path"]
        add_hdf5_labels(path, record, tasks)
        entry["sha256"] = file_hash(path)
    (source / "manifest.json").write_text(json.dumps(manifest))
    return manifest, tasks


def test_subtask_conversion_preserves_frame_labels_train_only(converted, tmp_path):
    source, _, originals = converted
    manifest, tasks = label_source(source)
    manifest["episodes"] = [e for e in manifest["episodes"] if e["split"] == "train"]
    (source / "manifest.json").write_text(json.dumps(manifest))
    target = tmp_path / "subtask_lerobot"
    summary = convert(source, target, task_mode="subtask")
    assert set(summary) == {"train"}
    root = target / "block_train"
    table = pq.read_table(episode_path(root, 0))
    assert table["task_index"].to_pylist() == [0, 1, 2, 3, 3]
    assert table["subtask_boundary"].to_pylist() == [True, True, True, True, False]
    np.testing.assert_array_equal(table["action"].to_pylist(), originals["train", 0][2])
    np.testing.assert_array_equal(read_video(episode_path(root, 0, True)), originals["train", 0][0])
    info = read_json(root / "meta/info.json")
    assert info["total_tasks"] == 4
    assert info["features"]["subtask_boundary"]["dtype"] == "bool"
    registry = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines()]
    assert [r["task"] for r in registry] == [t["subgoal"] for t in tasks]
    assert read_json(root / "meta/subtasks.json")["global_task"] == manifest["contract"]["instruction"]
    assert read_json(target / "conversion.json")["task_mode"] == "subtask"
    dataset = TaskDataset(root, target / "normalization.json", horizon=3)
    assert [dataset[i]["lang"] for i in range(5)] == [tasks[i]["subgoal"] for i in (0, 1, 2, 3, 3)]
    # A chunk for subtask 2 may not include the action of subtask 3.
    np.testing.assert_allclose(
        dataset.normalizer.unnormalize(dataset[2]["action"]),
        np.repeat(originals["train", 0][2][2:3], 3, axis=0),
        atol=1e-6,
    )


def test_auto_detects_subtask_labels(converted, tmp_path):
    source, _, _ = converted
    label_source(source)
    target = tmp_path / "auto_subtasks"
    convert(source, target)
    assert read_json(target / "conversion.json")["task_mode"] == "subtask"
    assert read_json(target / "block_val/meta/info.json")["total_tasks"] == 4


@pytest.mark.parametrize("fault", ["review", "boundary", "shape", "vocabulary"])
def test_subtask_conversion_rejects_invalid_labels(converted, tmp_path, fault):
    source, _, _ = converted
    manifest, _ = label_source(source)
    entry = manifest["episodes"][0]
    path = source / entry["path"]
    with h5py.File(path, "r+") as h5:
        group = h5["planner"]
        if fault == "review":
            group["needs_review"][0] = True
        elif fault == "boundary":
            group["subtask_boundary"][1] = False
        elif fault == "shape":
            del group["subtask_index"]
            group["subtask_index"] = np.array([0, 1], dtype=np.uint8)
        else:
            group["subgoal"][4] = "inconsistent name"
    entry["sha256"] = file_hash(path)
    (source / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        convert(source, tmp_path / f"invalid_{fault}", task_mode="subtask")


def test_subtask_mode_rejects_unlabeled_data(converted, tmp_path):
    with pytest.raises(ValueError, match="Missing planner labels"):
        convert(converted[0], tmp_path / "unlabeled", task_mode="subtask")
