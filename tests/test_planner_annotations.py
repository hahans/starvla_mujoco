"""Offline checks for planner annotation provenance, API recovery and causal export."""

import base64
import copy
import io
import json
import sys
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import requests
from PIL import Image

from mujoco_task import annotate_planner as ap


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    tasks = ap.load_taxonomy(ap.PROJECT / "configs/mujoco/planner_subtasks.json")
    contract = {"image_size": 16, "control_hz": 20, "instruction": "move green right of red"}
    entries = []
    for seed, split in enumerate(("train", "val", "test")):
        path = source / f"{split}.h5"
        with h5py.File(path, "w") as h5:
            h5["image"] = np.stack([np.full((16, 16, 3), i, dtype=np.uint8) for i in range(48)])
            h5["terminal_image"] = np.full((16, 16, 3), 255, dtype=np.uint8)
            h5["action"] = np.zeros((48, 7), dtype=np.float32)
            h5["time"] = np.arange(48) / 20
            h5["phase"] = np.asarray([t["phases"][0] for t in tasks for _ in range(12)], dtype="S16")
            h5.attrs["seed"] = seed
            h5.attrs["instruction"] = contract["instruction"]
        entries.append(
            {
                "path": path.name,
                "split": split,
                "seed": seed,
                "length": 48,
                "success": True,
                "sha256": ap.file_hash(path),
            }
        )
    ap.write_json(source / "manifest.json", {"contract": contract, "episodes": entries})
    return source, tmp_path / "output", tasks, contract, entries


def cli(monkeypatch, dataset, command, *extra):
    """Exercise the public CLI with an isolated dataset and output directory."""
    source, output, *_ = dataset
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate_planner",
            command,
            "--source",
            str(source),
            "--output",
            str(output),
            "--splits",
            "train",
            "val",
            "test",
            *extra,
        ],
    )
    return ap.main()


def api_response(payload):
    """Return a deterministic model response using candidates only in the test fixture."""
    metadata = json.loads(payload["messages"][1]["content"][0]["text"])
    segments = [
        {**s, "confidence": 0.95, "needs_review": False, "reason": "test-only visual evidence"}
        for s in metadata["candidate_segments_from_controller"]
    ]
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"segments": segments})}}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
        "model": "mock-not-a-real-annotation",
    }


def test_prepare_no_network_and_four_segments(dataset, monkeypatch):
    network = Mock(side_effect=AssertionError("No network in prepare"))
    monkeypatch.setattr(ap.requests, "post", network)
    assert cli(monkeypatch, dataset, "prepare") == 0
    report = ap.read_json(dataset[1] / "prepared.json")
    assert report["selected_episodes"] == 3
    assert report["source_frames"] == 144
    assert [s["start"] for s in report["episodes"][0]["proposed_segments"]] == [0, 12, 24, 36]
    assert not (dataset[1] / "annotations").exists()
    network.assert_not_called()


def test_vision_payload_alignment_and_no_hidden_state(dataset):
    source, _, tasks, contract, entries = dataset
    episode = ap.inspect_episode(source, entries[0], contract, tasks, 12)
    payload = ap.build_request(source, episode, contract, tasks, "deepseek-flash")
    content = payload["messages"][1]["content"]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["model"] == "deepseek-flash"
    assert not any(k in content[0]["text"] for k in ('"action"', '"success"', '"proprio"'))
    for frame, label, image in zip(episode["sampled_frames"], content[1::2], content[2::2], strict=True):
        assert label["text"].startswith(f"frame={frame},")
        data = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
        assert np.asarray(Image.open(io.BytesIO(data))).min() == frame
        assert frame < 48


def test_annotate_resume_and_causal_export(dataset, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    complete = Mock(side_effect=lambda payload: api_response(payload))
    monkeypatch.setattr(ap.APIClient, "complete", lambda self, payload: complete(payload))
    assert cli(monkeypatch, dataset, "annotate") == 0
    assert complete.call_count == 3
    assert cli(monkeypatch, dataset, "annotate") == 0
    assert complete.call_count == 3
    # Simulate a crash after persisting the response, before writing the annotation.
    ap.annotation_path(dataset[1], dataset[4][0]).unlink()
    assert cli(monkeypatch, dataset, "annotate") == 0
    assert complete.call_count == 3
    assert cli(monkeypatch, dataset, "export", "--export-stride", "1") == 0
    root = dataset[1] / "dataset"
    for split in ("train", "val", "test"):
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").read_text().splitlines()]
        assert rows
        for row in rows:
            frame = row["source"]["frame_index"]
            task_index = frame // 12
            assert row["source"]["split"] == split
            assert row["subgoal_history"] == [t["subgoal"] for t in dataset[2][:task_index]]
            assert json.loads(row["messages"][1]["content"]) == {"subgoal": dataset[2][task_index]["subgoal"]}
            assert np.asarray(Image.open(root / row["images"][0])).max() == frame
            assert all(abs(frame - boundary) > 3 for boundary in (12, 24, 36))
            assert not any(token in row["messages"][0]["content"] for token in ("confidence", "phase", "frame="))
    with pytest.raises(FileExistsError):
        cli(monkeypatch, dataset, "export")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s[1].update(start=13),
        lambda s: s[1].update(start=11),
        lambda s: s[-1].update(end=47),
        lambda s: s[0].update(subtask_id="invented"),
        lambda s: s[0].update(confidence=float("nan")),
        lambda s: s[0].update(confidence=True),
        lambda s: s[0].update(start=False),
        lambda s: s[0].update(needs_review="false"),
        lambda s: s[0].update(reason=""),
    ],
)
def test_reject_invalid_annotations(dataset, mutation):
    segments = [
        {
            "start": i * 12,
            "end": (i + 1) * 12,
            "subtask_id": t["id"],
            "confidence": 0.95,
            "needs_review": False,
            "reason": "visible",
        }
        for i, t in enumerate(dataset[2])
    ]
    mutation(segments)
    with pytest.raises(ValueError):
        ap.validate_annotation({"segments": segments}, 48, dataset[2])


def test_review_excludes_whole_episode_and_missing_report(dataset, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(ap.APIClient, "complete", lambda self, payload: api_response(payload))
    assert cli(monkeypatch, dataset, "annotate") == 0
    path = ap.annotation_path(dataset[1], dataset[4][0])
    record = ap.read_json(path)
    record["annotation"]["segments"][0]["confidence"] = 0.5
    ap.write_json(path, record)
    ap.annotation_path(dataset[1], dataset[4][-1]).unlink()
    assert cli(monkeypatch, dataset, "export") == 1
    root = dataset[1] / "dataset"
    assert (root / "train.jsonl").read_text() == ""
    assert (root / "test.jsonl").read_text() == ""
    assert (root / "val.jsonl").read_text()
    report = ap.read_json(root / "report.json")
    assert len(report["needs_review"]) == len(report["missing_annotations"]) == 1


def test_source_and_config_provenance(dataset, monkeypatch):
    assert cli(monkeypatch, dataset, "prepare") == 0
    with pytest.raises(ValueError, match="another dataset/config"):
        cli(monkeypatch, dataset, "prepare", "--model", "changed-model")
    with (dataset[0] / "train.h5").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        cli(monkeypatch, dataset, "prepare")


def test_duplicate_seed_and_path_escape(dataset, monkeypatch):
    manifest = ap.read_json(dataset[0] / "manifest.json")
    manifest["episodes"][1]["seed"] = manifest["episodes"][0]["seed"]
    ap.write_json(dataset[0] / "manifest.json", manifest)
    with pytest.raises(ValueError, match="leakage"):
        cli(monkeypatch, dataset, "prepare")
    with pytest.raises(ValueError, match="escapes"):
        ap.source_path(dataset[0], {"path": "../outside.h5"})


def test_unknown_or_reordered_phase_rejected(dataset):
    with pytest.raises(ValueError, match="Unmapped"):
        ap.phase_segments(["unknown"], dataset[2])
    with pytest.raises(ValueError, match="order"):
        ap.phase_segments(["prepare", "transfer", "grasp", "place"], dataset[2])


def test_retry_rate_limit_and_auth_stops(monkeypatch):
    monkeypatch.setattr(ap.time, "sleep", lambda seconds: None)
    ok = Mock(status_code=200)
    ok.json.return_value = {"choices": []}
    post = Mock(side_effect=[requests.Timeout(), Mock(status_code=429, headers={"Retry-After": "0"}), ok])
    monkeypatch.setattr(ap.requests, "post", post)
    client = ap.APIClient("https://api.deepseek.com", "test-secret", 30, 120, 4)
    assert client.complete({"model": "deepseek-flash"}) == {"choices": []}
    assert post.call_count == 3
    assert post.call_args.kwargs["allow_redirects"] is False
    post.reset_mock(side_effect=True)
    post.return_value = Mock(status_code=401)
    with pytest.raises(ap.FatalAPIError, match="HTTP 401"):
        client.complete({})
    assert post.call_count == 1


def test_invalid_response_is_preserved_without_fabricated_labels(dataset, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    malformed = {"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}
    complete = Mock(return_value=malformed)
    monkeypatch.setattr(ap.APIClient, "complete", lambda self, payload: complete(payload))
    assert cli(monkeypatch, dataset, "annotate") == 1
    assert not (dataset[1] / "annotations").exists()
    assert len(list((dataset[1] / "responses").rglob("*.json"))) == 3
    assert cli(monkeypatch, dataset, "annotate") == 1
    assert complete.call_count == 3


def test_large_boundary_shift_needs_review(dataset):
    segments = [
        {
            "start": i * 12,
            "end": (i + 1) * 12,
            "subtask_id": t["id"],
            "confidence": 0.99,
            "needs_review": False,
            "reason": "visible",
        }
        for i, t in enumerate(dataset[2])
    ]
    candidate = copy.deepcopy(segments)
    segments[0]["end"], segments[1]["start"] = 21, 21
    ap.validate_annotation({"segments": segments}, 48, dataset[2])
    assert ap.review_reasons(segments, candidate, 0.8, 8)


def test_llamafactory_export_preserves_labels_and_separates_review(dataset, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(ap.APIClient, "complete", lambda self, payload: api_response(payload))
    assert cli(monkeypatch, dataset, "annotate") == 0
    # Keep one accepted train split and one unreviewed validation split.
    path = ap.annotation_path(dataset[1], dataset[4][1])
    annotation = ap.read_json(path)
    annotation["annotation"]["segments"][1].update(confidence=0.6, needs_review=True)
    ap.write_json(path, annotation)
    ap.annotation_path(dataset[1], dataset[4][2]).unlink()
    monkeypatch.setattr(ap.requests, "post", Mock(side_effect=AssertionError("Export must be offline")))
    assert cli(monkeypatch, dataset, "export", "--export-format", "llamafactory", "--export-stride", "1") == 1
    root = dataset[1] / "dataset_llamafactory"
    info = ap.read_json(root / "dataset_info.json")
    assert set(info) == {"planner_train", "planner_review_val"}
    assert ap.read_json(root / "test.json") == []
    assert ap.read_json(root / "val.json") == []
    metadata = [json.loads(line) for line in (root / "samples_metadata.jsonl").read_text().splitlines()]
    for name, settings in info.items():
        assert settings["formatting"] == "sharegpt"
        assert settings["columns"] == {"messages": "conversations", "images": "images"}
        rows = ap.read_json(root / settings["file_name"])
        traces = [row for row in metadata if row["dataset"] == name]
        assert len(traces) == len(rows)
        for sample, trace in zip(rows, traces, strict=True):
            assert set(sample) == {"conversations", "images"}
            assert [m["from"] for m in sample["conversations"]] == ["human", "gpt"]
            assert sum(m["value"].count("<image>") for m in sample["conversations"]) == len(sample["images"]) == 1
            frame = trace["source"]["frame_index"]
            assert np.asarray(Image.open(root / sample["images"][0])).min() == frame
            assert json.loads(sample["conversations"][1]["value"]) == {"subgoal": trace["subgoal"]}
            assert trace["quality_status"] == ("needs_review" if "review" in name else "accepted")
            assert bool(trace["review_reasons"]) == ("review" in name)
            assert trace["subgoal_history"] == [t["subgoal"] for t in dataset[2][: frame // 12]]
    assert not (dataset[1] / "dataset").exists()


@pytest.mark.parametrize(
    "prompt,answer,images",
    [
        ("no token", "ok", ["one.png"]),
        ("<image><image>", "ok", ["one.png"]),
        ("<image>", "ok", []),
        ("question", "<image>", ["one.png"]),
    ],
)
def test_sharegpt_rejects_image_token_mismatches(prompt, answer, images):
    with pytest.raises(ValueError):
        ap.sharegpt_sample(
            {
                "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                "images": images,
            }
        )
