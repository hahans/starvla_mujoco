"""Collect expert demonstrations as HDF5 episodes plus ``manifest.json`` for ``mujoco_task.convert``.

Every episode is ``TransferEnv.reset(seed)`` followed by the scripted expert, so ``codec_check``, ``evaluate`` and
``grpo`` reproduce the same initial states from the stored seeds. Frame ``t`` stores the observation *before*
action ``t``; the final observation is kept as ``terminal_image`` only. Unsuccessful seeds are skipped and listed
in the manifest. Splits use disjoint seed ranges (train from 0, val from 100000, test from 200000)::

    MUJOCO_GL=egl python -m mujoco_task.collect --output data/block_raw --workers 16 --expect-renderer NVIDIA

The renderer (``MUJOCO_GL`` and the GL driver) must be the same for collection and closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

SEED_BASE = {"train": 0, "val": 100_000, "test": 200_000}
_ENV = None


def _worker_init(image_size: int) -> None:
    global _ENV
    from .env import TransferEnv

    _ENV = TransferEnv(image_size=image_size)


def _gl_renderer() -> str:
    try:
        from OpenGL import GL

        return f"{GL.glGetString(GL.GL_VENDOR).decode()} | {GL.glGetString(GL.GL_RENDERER).decode()}"
    except Exception as exc:  # The context may not be current under every backend.
        return f"unknown ({type(exc).__name__})"


def record_episode(job: tuple[str, int, str]) -> dict:
    """Run one seed in this worker's environment and write ``<split>_<seed>.h5``; return its manifest entry."""
    import h5py

    split, seed, directory = job
    env = _ENV
    obs = env.reset(seed=seed)
    env.start_expert()
    images, proprio, actions, phases = [], [], [], []
    done, info, settle = False, {}, 40
    while True:
        if not done:
            action, phase, done = env.expert_action()
        elif info.get("success") or settle == 0:
            break
        else:
            settle, phase = settle - 1, "settle"
        images.append(obs["image"])
        proprio.append(obs["proprio"])
        obs, info = env.step(action)
        actions.append(info["applied_action"])
        phases.append(phase)
    if not info.get("success"):
        return {"split": split, "seed": seed, "success": False}
    n = len(actions)
    name = f"{split}_{seed:06d}.h5"
    path = Path(directory) / "episodes" / name
    with h5py.File(path, "w") as h5:
        h5.create_dataset("image", data=np.asarray(images, np.uint8), chunks=(1, *images[0].shape),
                          compression="gzip", compression_opts=4)
        h5["proprio"] = np.asarray(proprio, np.float32)
        h5["action"] = np.asarray(actions, np.float32)
        h5["time"] = np.arange(n) / 20
        h5["phase"] = np.asarray(phases, dtype="S16")
        h5["terminal_image"] = obs["image"]
        h5.attrs["seed"] = seed
        h5.attrs["instruction"] = env.contract()["instruction"]
        h5.attrs["result"] = json.dumps(info)
        h5.attrs["green_start"] = env.task.green_start
        h5.attrs["red_start"] = env.task.red_start
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": f"episodes/{name}", "split": split, "seed": seed, "length": n, "success": True, "sha256": digest}


def collect(output: Path, counts: dict[str, int], workers: int, image_size: int,
            expect_renderer: str | None = None) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    staging = output.with_name(output.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"Unfinished collection exists: inspect/remove {staging}")

    from .env import TransferEnv

    probe = TransferEnv(image_size=image_size)
    probe.reset(seed=0)
    contract, renderer = probe.contract(), _gl_renderer()
    probe.close()
    print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')} renderer: {renderer}", flush=True)
    if expect_renderer and expect_renderer.lower() not in renderer.lower():
        raise RuntimeError(f"Renderer {renderer!r} does not contain {expect_renderer!r}")
    (staging / "episodes").mkdir(parents=True)

    entries, skipped = [], []
    started = time.monotonic()
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=_worker_init, initargs=(image_size,)) as pool:
        for split, count in counts.items():
            done, next_seed = [], SEED_BASE[split]
            while len(done) < count:
                missing = count - len(done)
                jobs = [(split, next_seed + i, str(staging)) for i in range(missing)]
                next_seed += missing
                for result in pool.imap(record_episode, jobs):
                    if result["success"]:
                        done.append(result)
                    else:
                        skipped.append(result["seed"])
                    total = len(entries) + len(done)
                    if total % 10 == 0 or len(done) == count:
                        print(f"{split}: {len(done)}/{count}, skipped {len(skipped)}, "
                              f"{time.monotonic() - started:.0f} s", flush=True)
            entries += sorted(done, key=lambda e: e["seed"])
    manifest = {
        "format_version": 1,
        "collector": "mujoco_task.collect",
        "contract": contract,
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "gl_renderer": renderer,
        "seed_base": SEED_BASE,
        "skipped_unsuccessful_seeds": skipped,
        "episodes": entries,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    return {split: sum(e["split"] == split for e in entries) for split in counts} | {"skipped": len(skipped)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New directory (never overwritten)")
    parser.add_argument("--train", type=int, default=600)
    parser.add_argument("--val", type=int, default=20)
    parser.add_argument("--test", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--image-size", type=int, default=128, help="Policy image size; part of the contract")
    parser.add_argument("--expect-renderer", help="Abort unless the GL renderer string contains this, e.g. NVIDIA")
    args = parser.parse_args()
    os.environ.setdefault("MUJOCO_GL", "osmesa")  # Same fallback as evaluate.py.
    counts = {split: n for split, n in (("train", args.train), ("val", args.val), ("test", args.test)) if n > 0}
    if "train" not in counts:
        parser.error("at least one train episode is required for normalization")
    print(collect(args.output, counts, args.workers, args.image_size, args.expect_renderer))


if __name__ == "__main__":
    main()
