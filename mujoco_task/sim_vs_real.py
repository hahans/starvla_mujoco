"""Collect expert episodes in the simulator and compare them with real ``ur5_record.py`` episodes.

``collect`` stores simulator episodes in the ``ur5_record.py`` layout (``data.npz``, ``meta.json``, ``rgb/``) with the
same field conventions (UR base-frame ``tcp``, fingertip distance in m). With ``--real`` the green cube is placed where
the real gripper grasped it (simulator forward kinematics of the real joints at the first closing frame) and red
``--gap`` metres to its right; ``--episodes`` adds randomized resets::

    python -m mujoco_task.sim_vs_real collect --real real_raw2/episode_0000 real_raw2/episode_0001 --out data/sim_raw
    python -m mujoco_task.sim_vs_real compare --real real_raw2/episode_0000 --sim data/sim_raw/episode_0000 \
        --out logs/compare_0000

``compare`` aligns both episodes at start, grasp (gripper starts closing), release and end, then writes
``side_by_side.mp4`` (real centre crop | simulator), ``keyframes.png``, ``trajectory.png`` (TCP in the go_home world
frame and fingertip distance) and ``summary.json``. Run with ``MUJOCO_GL=egl`` on a headless machine.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from .env import TransferEnv

STROKE_M = 0.07
CLOSED = 0.1
EVENTS = ("start", "grasp", "release", "end")


def world_from_base(tilt_deg: float) -> np.ndarray:
    """``go_home.py`` world frame: base turned about world Y."""
    return Rotation.from_euler("y", tilt_deg, degrees=True).as_matrix()


def real_grasp_point(env: TransferEnv, episode: Path) -> np.ndarray:
    """Simulator TCP site at the real joints of the first closing frame = where the cube centre was."""
    record = np.load(episode / "data.npz")
    closing = np.flatnonzero(record["gripper_close_cmd"] > CLOSED)
    if not len(closing):
        raise ValueError(f"{episode}: the gripper never closes")
    scratch = mujoco.MjData(env.model)
    scratch.qpos[:] = env.data.qpos
    scratch.qpos[env.arm_q] = record["q"][closing[0]]
    mujoco.mj_forward(env.model, scratch)
    return scratch.site_xpos[env.model.site("left_tcp").id].copy()


def place_cubes(env: TransferEnv, green_xy: np.ndarray, red_xy: np.ndarray) -> None:
    for color, xy in (("green", green_xy), ("red", red_xy)):
        addr = env.model.joint(color + "_block_joint").qposadr[0]
        env.data.qpos[addr : addr + 2] = xy
    mujoco.mj_forward(env.model, env.data)
    env.task.green_start = env.task.position("green").copy()
    env.task.red_start = env.task.position("red").copy()
    env.task.goal = env.task.red_start + np.asarray([env.task.cfg["goal_offset_x"], 0, 0])


def base_tcp(env: TransferEnv) -> np.ndarray:
    """Flange pose in the UR base frame, as ``getActualTCPPose`` with a zero TCP offset."""
    m, d = env.model, env.data
    base, hand = m.body("left_base").id, m.body("left_right_hand").id
    # The MuJoCo UR model's base body is the UR controller base frame turned 180 deg about z.
    rotation = d.xmat[base].reshape(3, 3) @ np.diag([-1.0, -1.0, 1.0])
    position = rotation.T @ (d.xpos[hand] - d.xpos[base])
    rotvec = Rotation.from_matrix(rotation.T @ d.xmat[hand].reshape(3, 3)).as_rotvec()
    return np.r_[position, rotvec]


def run_expert(env: TransferEnv, out: Path, meta: dict, settle_steps: int = 40) -> dict:
    (out / "rgb").mkdir(parents=True, exist_ok=True)
    env.start_expert()
    rows, phases, info = [], [], {}
    obs, done = env.observe(), False
    while True:
        if not done:
            action, phase, done = env.expert_action()
        elif info.get("success") or settle_steps <= 0:
            break
        else:
            settle_steps -= 1
            phase = "settle"
        Image.fromarray(obs["image"]).save(out / "rgb" / f"{len(rows):06d}.png")
        command_m = 2.0 * float(action[6])
        rows.append(
            np.r_[
                len(rows) * env.control_dt,
                env.data.qpos[env.arm_q],
                env.data.qvel[env.arm_dof],
                base_tcp(env),
                1.0 - command_m / STROKE_M,
                command_m,
                env.data.qpos[env.finger_q].sum(),
            ]
        )
        phases.append(phase)
        obs, info = env.step(action)
    rows = np.asarray(rows)
    np.savez_compressed(
        out / "data.npz",
        t=rows[:, 0],
        q=rows[:, 1:7],
        qd=rows[:, 7:13],
        tcp=rows[:, 13:19],
        gripper_close_cmd=rows[:, 19],
        gripper_command_m=rows[:, 20],
        gripper_opening_m=rows[:, 21],
        phase=np.asarray(phases),
    )
    meta.update(
        steps=len(rows),
        hz=1.0 / env.control_dt,
        world_tilt_deg=-45.0,
        success=bool(info["success"]),
        image="policy_cam, square = centre crop of the real 640x480 frame",
        conventions="UR controller joints (rad) and base-frame TCP pose [x,y,z,rx,ry,rz], no joint_offset",
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def collect(args: argparse.Namespace) -> None:
    env = TransferEnv(image_size=args.image_size)
    jobs = [("real", path) for path in args.real] + [("seed", seed) for seed in range(args.episodes)]
    try:
        for index, (kind, source) in enumerate(jobs, start=args.start_index):
            out = args.out / f"episode_{index:04d}"
            if out.exists():
                raise SystemExit(f"{out} exists; choose another --out or --start-index")
            if kind == "real":
                env.reset(seed=0, randomize=False)
                green = real_grasp_point(env, source)[:2]
                place_cubes(env, green, green + np.array([args.gap, 0.0]))
                meta = {"source": "sim_expert", "real_episode": str(source)}
            else:
                env.reset(seed=int(source), randomize=True)
                meta = {"source": "sim_expert", "seed": int(source)}
            meta.update(green=env.task.green_start.tolist(), red=env.task.red_start.tolist())
            meta = run_expert(env, out, meta)
            print(f"{out}: {meta['steps']} steps, success={meta['success']}, green={np.round(meta['green'], 3)}")
    finally:
        env.close()


def load_episode(path: Path) -> dict:
    record = dict(np.load(path / "data.npz"))
    meta = json.loads((path / "meta.json").read_text())
    closed = np.flatnonzero(record["gripper_close_cmd"] > CLOSED)
    if not len(closed):
        raise ValueError(f"{path}: the gripper never closes")
    n = len(record["q"])
    record["events"] = np.array([0, closed[0], min(closed[-1] + 1, n - 1), n - 1], dtype=float)
    record["world"] = record["tcp"][:, :3] @ world_from_base(meta.get("world_tilt_deg", -45.0)).T
    record["path"], record["hz"] = path, float(meta.get("hz", 20.0))
    return record


def frame(record: dict, index: int, size: int) -> np.ndarray:
    image = Image.open(record["path"] / "rgb" / f"{index:06d}.png").convert("RGB")
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return np.asarray(image.crop((left, top, left + side, top + side)).resize((size, size), Image.BILINEAR))


def progress(record: dict) -> np.ndarray:
    """Per-frame position on a common axis: each segment between events takes one third."""
    return np.interp(np.arange(len(record["q"])), record["events"], np.linspace(0.0, 1.0, len(EVENTS)))


def labelled(image: np.ndarray, text: str) -> np.ndarray:
    tile = Image.fromarray(image)
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 8 * len(text) + 8, 16), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(tile)


def write_video(path: Path, real: dict, sim: dict, size: int) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=int(sim["hz"]))
        stream.width, stream.height, stream.pix_fmt = 2 * size, size, "yuv420p"
        stream.options = {"crf": "20"}
        for i in range(len(sim["q"])):
            r = int(round(np.interp(i, sim["events"], real["events"])))
            image = np.concatenate(
                [labelled(frame(real, r, size), f"real {r}"), labelled(frame(sim, i, size), f"sim {i}")], 1
            )
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def keyframes(real: dict, sim: dict, size: int) -> np.ndarray:
    marks = np.linspace(0.0, 1.0, 2 * len(EVENTS) - 1)
    rows = []
    for record, name in ((real, "real"), (sim, "sim")):
        p = progress(record)
        tiles = [frame(record, int(np.argmin(np.abs(p - m))), size) for m in marks]
        rows.append(np.concatenate([labelled(t, f"{name} {m:.2f}") for t, m in zip(tiles, marks)], 1))
    return np.concatenate(rows, 0)


def plot(real: dict, sim: dict, path: Path) -> None:
    """TCP x/y/z (go_home world) and fingertip distance on the event-aligned axis; blue real, orange sim."""
    panels = [("x (m)", lambda r: r["world"][:, 0]), ("y (m)", lambda r: r["world"][:, 1]),
              ("z (m)", lambda r: r["world"][:, 2]), ("opening (m)", lambda r: r["gripper_opening_m"])]
    width, height, margin = 900, 190, 60
    canvas = Image.new("RGB", (width, height * len(panels)), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (name, value) in enumerate(panels):
        top = k * height
        series = [(progress(r), value(r), color) for r, color in ((real, (31, 119, 180)), (sim, (255, 127, 14)))]
        lo = min(float(v.min()) for _, v, _ in series)
        hi = max(float(v.max()) for _, v, _ in series)
        hi = hi if hi - lo > 1e-6 else lo + 1e-3

        def xy(p: np.ndarray, v: np.ndarray) -> list[tuple[float, float]]:
            px = margin + p * (width - 2 * margin)
            py = top + height - 25 - (v - lo) / (hi - lo) * (height - 45)
            return list(zip(px.tolist(), py.tolist()))

        draw.rectangle((margin, top + 20, width - margin, top + height - 25), outline=(150, 150, 150))
        for j, event in enumerate(EVENTS):
            x = margin + j / (len(EVENTS) - 1) * (width - 2 * margin)
            draw.line((x, top + 20, x, top + height - 25), fill=(210, 210, 210))
            draw.text((x - 15, top + height - 20), event, fill=(0, 0, 0))
        for p, v, color in series:
            draw.line(xy(p, v), fill=color, width=2)
        draw.text((5, top + 3), f"{name}   [{lo:.3f}, {hi:.3f}]   blue real, orange sim", fill=(0, 0, 0))
    canvas.save(path)


def summary(real: dict, sim: dict) -> dict:
    def describe(record: dict) -> dict:
        grasp, release = (int(i) for i in record["events"][1:3])
        world = record["world"]
        return {
            "duration_s": len(record["q"]) / record["hz"],
            "grasp_tcp_world": world[grasp].round(4).tolist(),
            "release_tcp_world": world[release].round(4).tolist(),
            "lift_m": float(world[grasp:release + 1, 2].max() - world[grasp, 2]),
            "transfer_dx_m": float(world[release, 0] - world[grasp, 0]),
            "closed_opening_m": float(np.median(record["gripper_opening_m"][grasp + 10 : release])),
        }

    real_d, sim_d = describe(real), describe(sim)
    diff = {
        "grasp_tcp_world_m": (np.subtract(sim_d["grasp_tcp_world"], real_d["grasp_tcp_world"])).round(4).tolist(),
        "release_tcp_world_m": (np.subtract(sim_d["release_tcp_world"], real_d["release_tcp_world"]))
        .round(4)
        .tolist(),
        "start_joint_max_rad": float(np.abs(sim["q"][0] - real["q"][0]).max()),
        "start_tcp_m": float(np.linalg.norm(sim["tcp"][0, :3] - real["tcp"][0, :3])),
    }
    return {"real": real_d, "sim": sim_d, "sim_minus_real": diff}


def compare(args: argparse.Namespace) -> None:
    real, sim = load_episode(args.real), load_episode(args.sim)
    args.out.mkdir(parents=True, exist_ok=True)
    plot(real, sim, args.out / "trajectory.png")
    Image.fromarray(keyframes(real, sim, 240)).save(args.out / "keyframes.png")
    write_video(args.out / "side_by_side.mp4", real, sim, args.size)
    result = summary(real, sim)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Wrote {args.out}/side_by_side.mp4, keyframes.png, trajectory.png, summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    col = sub.add_parser("collect", help="Run the expert and save episodes in the ur5_record.py layout")
    col.add_argument("--real", type=Path, nargs="*", default=[], help="Real episodes whose grasp point places green")
    col.add_argument("--episodes", type=int, default=0, help="Additional randomized resets (seeds 0..N-1)")
    col.add_argument("--gap", type=float, default=0.122, help="Red minus green x (m) for --real episodes")
    col.add_argument("--image-size", type=int, default=480)
    col.add_argument("--start-index", type=int, default=0)
    col.add_argument("--out", type=Path, default=Path("data/sim_raw"))
    cmp = sub.add_parser("compare", help="Event-aligned video, key frames, TCP plot and summary")
    cmp.add_argument("--real", type=Path, required=True)
    cmp.add_argument("--sim", type=Path, required=True)
    cmp.add_argument("--size", type=int, default=320, help="Video tile size in pixels")
    cmp.add_argument("--out", type=Path, default=Path("logs/compare"))
    args = parser.parse_args()
    if args.command == "collect" and not args.real and args.episodes <= 0:
        parser.error("give --real episodes and/or --episodes N")
    collect(args) if args.command == "collect" else compare(args)


if __name__ == "__main__":
    main()
