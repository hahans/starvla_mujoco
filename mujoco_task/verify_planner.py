"""Verify four-stage demonstration labels by deterministic MuJoCo replay."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path

import h5py
import numpy as np

from . import annotate_planner as ap

LOGGER = logging.getLogger(__name__)
ACTION_ATOL = 1e-6
STATE_ATOL = 1e-5


def require(condition: bool, message: str) -> None:
    """Reject an episode when replay evidence fails a required condition."""
    if not condition:
        raise ValueError(message)


def check_boundary(task_id: str, evidence: dict, tolerance: float) -> None:
    """Check the previous subtask's physical objective before the next action."""
    if task_id == "grasp_lift":
        require(evidence["tcp_green_distance"] < 0.02, "Approach did not reach the green block")
        require(evidence["min_finger_opening"] > 0.025, "Approach ended without an open gripper")
    elif task_id == "transfer":
        require(all(evidence["finger_contacts"]), "Lift ended without bilateral grasp contact")
        require(evidence["lift_m"] > 0.05 and not evidence["grounded"], "Block not lifted clear before transfer")
    elif task_id == "place_release":
        require(all(evidence["finger_contacts"]), "Transfer ended without bilateral grasp contact")
        require(evidence["lift_m"] > 0.05 and not evidence["grounded"], "Transfer did not keep block lifted")
        require(evidence["goal_xy_error"] < tolerance, "Transfer did not reach above destination")


def snapshot(env) -> dict:
    """Record privileged physical evidence for review only, never student input."""
    task = env.task
    state = task.observation()
    fingers, grounded = task.contacts()
    return {
        "time": float(env.data.time),
        "tcp_green_distance": float(np.linalg.norm(state["black_tcp"] - state["green_position"])),
        "min_finger_opening": float(env.data.qpos[env.finger_q].min()),
        "finger_contacts": [bool(value) for value in fingers],
        "grounded": bool(grounded),
        "lift_m": float(state["green_position"][2] - task.green_start[2]),
        "goal_xy_error": float(np.linalg.norm(state["green_position"][:2] - task.goal[:2])),
        "green_position": state["green_position"].tolist(),
        "red_position": state["red_position"].tolist(),
    }


def check_final(info: dict, evidence: dict, recorded: dict, settle_seconds: float) -> None:
    """Verify stable release and agreement with the original collection result."""
    require(info["success"] and not info["failed"], "Replay did not succeed")
    require(info["bilateral_grasp_seen"] and info["lifted_by_black"], "No physical grasp/lift evidence")
    require(info["stable_seconds"] >= settle_seconds, "Final block placement was not stable")
    require(evidence["grounded"] and not any(evidence["finger_contacts"]), "Final block not released on table")
    require(evidence["min_finger_opening"] > 0.025, "Final gripper not open")
    require(info["max_red_drift_m"] < 0.02, "Red block moved too far")
    require(info["max_white_joint_motion"] < 0.005, "White arm moved")
    require(not info["unexpected_contacts"] and not any(info["warnings"]), "Unexpected contact or simulation warning")
    for key, value in recorded.items():
        if key not in info:
            continue
        if isinstance(value, bool) or isinstance(value, str) or key == "unexpected_contacts":
            require(value == info[key], f"Recorded result differs: {key}")
        else:
            require(np.allclose(value, info[key], atol=STATE_ATOL, rtol=0), f"Recorded result differs: {key}")


def replay_episode(job: dict) -> dict:
    """Replay all stored actions and verify phase, state and boundary alignment."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    sys.path.insert(0, job["collector"])
    from pvla_mujoco.env import TransferEnv

    entry, tasks = job["entry"], job["tasks"]
    source = Path(job["source"])
    evidence = {"entry": entry, "verification_id": job["verification_id"], "status": "failed"}
    env = None
    try:
        candidate = ap.inspect_episode(source, entry, job["contract"], tasks, 12)
        segments = candidate["proposed_segments"]
        with h5py.File(ap.source_path(source, entry), "r") as h5:
            actions, states = h5["action"][:], h5["proprio"][:]
            phases, times = h5["phase"].asstr()[:], h5["time"][:]
            recorded, initial = json.loads(h5.attrs["result"]), json.loads(h5.attrs["initial"])
        require(states.shape == (entry["length"], 21), "Invalid recorded state shape")
        require(np.isfinite(states).all() and np.isfinite(actions).all(), "Non-finite trajectory")
        env = TransferEnv(image_size=job["contract"]["image_size"], render=False)
        require(env.contract() == job["contract"], "Runtime scene/version/contract differs from collection")
        env.reset(entry["seed"])
        for color in ("green", "red"):
            require(np.allclose(env.task.position(color), initial[color], atol=1e-9, rtol=0), "Reset mismatch")
        env.start_expert()
        boundaries = {segment["start"]: segment["subtask_id"] for segment in segments[1:]}
        max_action_error = max_state_error = 0.0
        boundary_evidence = []
        for frame, action in enumerate(actions):
            require(abs(env.data.time - times[frame]) < 1e-7, f"Time mismatch at frame {frame}")
            state_error = float(np.max(np.abs(env.observe()["proprio"] - states[frame])))
            max_state_error = max(max_state_error, state_error)
            require(state_error <= STATE_ATOL, f"State mismatch at frame {frame}: {state_error}")
            expected, phase, done = env.expert_action()
            require(phase == phases[frame], f"Expert phase mismatch at frame {frame}")
            require(done == (frame == len(actions) - 1), f"Expert termination mismatch at frame {frame}")
            action_error = float(np.max(np.abs(expected - action)))
            max_action_error = max(max_action_error, action_error)
            require(action_error <= ACTION_ATOL, f"Expert action mismatch at frame {frame}: {action_error}")
            if frame in boundaries:
                point = snapshot(env)
                check_boundary(boundaries[frame], point, env.task.cfg["placement_tolerance"])
                boundary_evidence.append({"frame": frame, "next_subtask": boundaries[frame], **point})
            _, info = env.step(action, observe=False)
        final = snapshot(env)
        check_final(info, final, recorded, env.task.cfg["settle_seconds"])
        evidence.update(
            status="verified",
            method="full_deterministic_physics_replay",
            frames_checked=len(actions),
            max_action_error=max_action_error,
            max_state_error=max_state_error,
            segments=segments,
            boundaries=boundary_evidence,
            final=final,
            replay_result=info,
            annotation_semantics="expert_subtask_to_execute_from_pre_action_observation",
        )
    except (ValueError, RuntimeError, KeyError, OSError) as error:
        evidence["error"] = str(error)
    finally:
        if env is not None:
            env.close()
    return evidence


def main() -> int:
    """Audit all selected episodes, resuming only matching saved replay results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ap.PROJECT.parent / "pocketvla_mujoco/data/block_transfer2")
    parser.add_argument("--collector", type=Path, default=ap.PROJECT.parent / "pocketvla_mujoco")
    parser.add_argument("--output", type=Path, default=ap.PROJECT / "outputs/planner_verified")
    parser.add_argument("--taxonomy", type=Path, default=ap.PROJECT / "configs/mujoco/planner_subtasks.json")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Workers and optional limit must be positive")
    args.source, args.output, args.collector = args.source.resolve(), args.output.resolve(), args.collector.resolve()
    tasks = ap.load_taxonomy(args.taxonomy)
    require(
        [t["id"] for t in tasks] == ["approach", "grasp_lift", "transfer", "place_release"],
        "Requires four-stage taxonomy",
    )
    manifest = ap.read_json(args.source / "manifest.json")
    entries = [entry for entry in manifest["episodes"] if entry["split"] == "train"]
    for key in ("seed", "path", "sha256"):
        require(len({e[key] for e in manifest["episodes"]}) == len(manifest["episodes"]), f"Duplicate manifest {key}")
    entries = entries[: args.limit] if args.limit else entries
    implementation = [Path(__file__), Path(ap.__file__), *sorted((args.collector / "pvla_mujoco").rglob("*.py"))]
    config = {
        "source_manifest_sha256": ap.file_hash(args.source / "manifest.json"),
        "taxonomy": tasks,
        "implementation_sha256": {str(path.resolve()): ap.file_hash(path) for path in implementation},
        "action_atol": ACTION_ATOL,
        "state_atol": STATE_ATOL,
    }
    verification_id = ap.digest(config)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "verification_run.json"
    if config_path.exists():
        require(ap.read_json(config_path) == config, "Verification configuration changed; use a new output directory")
    ap.write_json(config_path, config)
    results, jobs = [], []
    for entry in entries:
        path = args.output / "verification" / f"seed_{entry['seed']:07d}.json"
        if path.exists():
            cached = ap.read_json(path)
            require(cached["verification_id"] == verification_id and cached["entry"] == entry, "Invalid replay cache")
            require(ap.file_hash(ap.source_path(args.source, entry)) == entry["sha256"], "Source changed since replay")
            results.append(cached)
        else:
            jobs.append(
                {
                    "source": str(args.source),
                    "collector": str(args.collector),
                    "entry": entry,
                    "contract": manifest["contract"],
                    "tasks": tasks,
                    "verification_id": verification_id,
                }
            )
    # Limit BLAS threads within each independent, CPU-only replay process.
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(replay_episode, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            ap.write_json(args.output / "verification" / f"seed_{result['entry']['seed']:07d}.json", result)
            results.append(result)
            if len(results) % 10 == 0 or result["status"] != "verified":
                LOGGER.info(
                    "Replayed %d/%d, failed=%d",
                    len(results),
                    len(entries),
                    sum(r["status"] != "verified" for r in results),
                )
    summary = {
        "verification_id": verification_id,
        "episodes": len(results),
        "verified": sum(r["status"] == "verified" for r in results),
        "failures": [r for r in results if r["status"] != "verified"],
        "frames_checked": sum(r.get("frames_checked", 0) for r in results),
        "max_action_error": max((r.get("max_action_error", 0) for r in results), default=0),
        "max_state_error": max((r.get("max_state_error", 0) for r in results), default=0),
    }
    ap.write_json(args.output / "verification_summary.json", summary)
    LOGGER.info("Replay summary: %s", summary)
    return int(bool(summary["failures"]))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
