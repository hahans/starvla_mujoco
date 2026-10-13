"""Publish replay-confirmed HDF5 copies and LlamaFactory planner examples."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from . import annotate_planner as ap
from .verify_planner import check_boundary, check_final, require

LOGGER = logging.getLogger(__name__)


def confirmed_record(original: dict, replay: dict, tasks: list[dict]) -> dict:
    """Preserve teacher uncertainty while deriving labels from verified expert phases."""
    require(replay["status"] == "verified", "Cannot publish a failed replay")
    require(original["source_sha256"] == replay["entry"]["sha256"], "Annotation/replay source mismatch")
    require(original["source"] == replay["entry"]["path"], "Annotation/replay path mismatch")
    original_segments = ap.validate_annotation(original["annotation"], replay["entry"]["length"], tasks)
    require(
        [{key: segment[key] for key in ("start", "end", "subtask_id")} for segment in original_segments]
        == replay["segments"],
        "Teacher boundary differs from replay-confirmed expert phase; do not silently approve it",
    )
    record = copy.deepcopy(original)
    record["teacher_annotation"] = record.pop("annotation")
    record["verification"] = {
        "id": replay["verification_id"],
        "method": replay["method"],
        "quality_status": "verified_by_physics_replay",
        "frames_checked": replay["frames_checked"],
        "annotation_semantics": replay["annotation_semantics"],
    }
    record["annotation"] = {
        "segments": [
            {
                **segment,
                "confidence": 1.0,
                "confidence_kind": "deterministic_check_pass_not_model_probability",
                "needs_review": False,
                "reason": (
                    "Original expert phase, action and state matched full replay; "
                    "physical boundary and final checks passed."
                ),
            }
            for segment in replay["segments"]
        ]
    }
    ap.validate_annotation(record["annotation"], replay["entry"]["length"], tasks)
    return record


def add_hdf5_labels(path: Path, record: dict, tasks: list[dict]) -> None:
    """Add labels to an independent HDF5 copy, preserving all original observations."""
    segments = record["annotation"]["segments"]
    n = segments[-1]["end"]
    indices = np.empty(n, dtype=np.uint8)
    boundaries = np.zeros(n, dtype=np.bool_)
    for i, segment in enumerate(segments):
        indices[segment["start"] : segment["end"]] = i
        boundaries[segment["start"]] = True
    with h5py.File(path, "r+") as h5:
        require("planner" not in h5, "HDF5 already contains planner labels")
        require(len(h5["action"]) == n, "Label length differs from HDF5 action length")
        group = h5.create_group("planner")
        group["subtask_index"] = indices
        group["subtask_boundary"] = boundaries
        group["needs_review"] = np.zeros(n, dtype=np.bool_)
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset("subtask_id", data=[tasks[i]["id"] for i in indices], dtype=dtype)
        group.create_dataset("subgoal", data=[tasks[i]["subgoal"] for i in indices], dtype=dtype)
        group.attrs["verification"] = json.dumps(record["verification"])
        group.attrs["segments"] = json.dumps(segments)
        group.attrs["teacher_annotation"] = json.dumps(record["teacher_annotation"])
        group.attrs["source_sha256"] = record["source_sha256"]
        group.attrs["observation_alignment"] = "frame t is the observation before action t"


def publish(source: Path, annotations: Path, verification: Path, destination: Path) -> dict:
    """Publish only if every training trajectory passes replay and provenance checks."""
    source, annotations, verification, destination = (
        p.resolve() for p in (source, annotations, verification, destination)
    )
    staging = destination.with_name(destination.name + ".partial")
    require(not destination.is_relative_to(source), "Destination must be outside the original source dataset")
    if destination.exists() or staging.exists():
        raise FileExistsError("Confirmed destination or .partial already exists")
    manifest = ap.read_json(source / "manifest.json")
    entries = [e for e in manifest["episodes"] if e["split"] == "train"]
    require(bool(entries), "No source training episodes")
    summary = ap.read_json(verification / "verification_summary.json")
    config = ap.read_json(verification / "verification_run.json")
    run = ap.read_json(annotations / "run.json")
    tasks = run["taxonomy"]
    require(config["taxonomy"] == tasks, "Verification taxonomy mismatch")
    require(
        config["source_manifest_sha256"] == run["manifest_sha256"] == ap.file_hash(source / "manifest.json"),
        "Source manifest changed",
    )
    require(summary["verification_id"] == ap.digest(config), "Verification ID mismatch")
    require(
        summary["episodes"] == summary["verified"] == len(entries) and not summary["failures"],
        "All original training episodes must be verified before publishing",
    )
    records, episodes = [], []
    for entry in entries:
        replay_path = verification / "verification" / f"seed_{entry['seed']:07d}.json"
        replay = ap.read_json(replay_path)
        require(
            replay["entry"] == entry and replay["verification_id"] == summary["verification_id"],
            "Replay provenance mismatch",
        )
        require(replay["frames_checked"] == entry["length"], "Incomplete replay")
        require(
            replay["max_action_error"] <= config["action_atol"] and replay["max_state_error"] <= config["state_atol"],
            "Replay numerical errors exceed tolerances",
        )
        require(len(replay["boundaries"]) == len(tasks) - 1, "Missing boundary evidence")
        for boundary, segment in zip(replay["boundaries"], replay["segments"][1:], strict=True):
            require(
                boundary["frame"] == segment["start"] and boundary["next_subtask"] == segment["subtask_id"],
                "Boundary evidence is misaligned",
            )
            check_boundary(boundary["next_subtask"], boundary, 0.035)
        episode = ap.inspect_episode(source, entry, manifest["contract"], tasks, run["image_stride"])
        require(episode["proposed_segments"] == replay["segments"], "Replay phases differ from HDF5")
        with h5py.File(ap.source_path(source, entry), "r") as h5:
            check_final(replay["replay_result"], replay["final"], json.loads(h5.attrs["result"]), 0.5)
        original = ap.read_json(ap.annotation_path(annotations, entry))
        require(original["run_id"] == run["run_id"], "Teacher annotation run mismatch")
        records.append(confirmed_record(original, replay, tasks))
        episodes.append(episode)
    staging.mkdir(parents=True)
    ap.write_json(staging / "run.json", run)
    ap.write_json(staging / "verification_summary.json", summary)
    new_entries = []
    for index, (entry, record) in enumerate(zip(entries, records, strict=True)):
        ap.write_json(ap.annotation_path(staging, entry), record)
        copy_path = staging / "hdf5" / entry["path"]
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ap.source_path(source, entry), copy_path)
        require(ap.file_hash(copy_path) == entry["sha256"], "HDF5 copy checksum mismatch before adding labels")
        add_hdf5_labels(copy_path, record, tasks)
        new_entries.append({**entry, "sha256": ap.file_hash(copy_path), "source_sha256": entry["sha256"]})
        if (index + 1) % 100 == 0:
            LOGGER.info("Published labeled HDF5 copies: %d/%d", index + 1, len(entries))
    ap.write_json(
        staging / "hdf5/manifest.json",
        {
            "format_version": manifest["format_version"],
            "contract": manifest["contract"],
            "episodes": new_entries,
            "planner_verification_id": summary["verification_id"],
            "source_manifest_sha256": run["manifest_sha256"],
            "label_semantics": "expert_subtask_to_execute_from_pre_action_observation",
        },
    )
    args = SimpleNamespace(
        source=source,
        output=staging,
        export_dir=staging / "dataset_llamafactory",
        export_format="llamafactory",
        splits=["train"],
        export_stride=8,
        boundary_margin=3,
        min_confidence=0.8,
        max_boundary_shift=8,
    )
    exported = ap.export_dataset(args, episodes, manifest["contract"], tasks, run["run_id"])
    require(
        not exported["needs_review"] and not exported["missing_annotations"],
        "Confirmed export unexpectedly excluded episodes",
    )
    report = {
        "verified_episodes": len(entries),
        "verified_frames": summary["frames_checked"],
        "samples": exported["samples"],
        "verification_id": summary["verification_id"],
        "original_hdf5_modified": False,
        "method": "full_deterministic_physics_replay",
        "teacher_confidence_preserved_in": "annotations/* and HDF5 planner attributes",
        "model_visual_observability_verified": False,
        "scope": "four expert demonstration subtasks; not recovery planning or universal visual completion detection",
    }
    ap.write_json(staging / "confirmation_report.json", report)
    staging.rename(destination)
    LOGGER.info("Confirmed dataset published: %s", destination)
    return report


def main() -> None:
    """Publish a separate confirmed dataset without changing the original collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ap.PROJECT.parent / "pocketvla_mujoco/data/block_transfer2")
    parser.add_argument("--annotations", type=Path, default=ap.PROJECT / "outputs/planner_annotations")
    parser.add_argument("--verification", type=Path, default=ap.PROJECT / "outputs/planner_verified")
    parser.add_argument("--destination", type=Path, default=ap.PROJECT / "outputs/planner_verified/confirmed")
    args = parser.parse_args()
    publish(args.source, args.annotations, args.verification, args.destination)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
