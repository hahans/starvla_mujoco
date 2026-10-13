"""Real-robot inference for the MuJoCo-trained QwenFast SFT or SFT + GRPO LoRA policy.

Subcommands:

- ``serve``: load the policy on a GPU machine and answer ``{image, state}`` requests
  over websocket with one chunk of physical joint targets.
- ``run``: closed-loop control through a ``Robot`` adapter. The policy is either loaded
  locally (``--checkpoint``) or queried remotely (``--server host:port``); the remote
  mode imports neither torch nor the model code, so it runs on a light robot PC.

``--robot sim`` drives the identical loop in MuJoCo; use it to verify a deployment
before connecting hardware. ``--robot ur5`` drives the real left UR5 over RTDE
(``mujoco_task.ur5_robot``). Any other class implementing ``Robot`` is selected with
``--robot package.module:ClassName``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

STATE_DIM = 21
ACTION_DIM = 7
FINGER_MAX_M = 0.035


class Robot(Protocol):
    """Hardware adapter. Units and joint order must match the MuJoCo training robot.

    Joints are ``shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3`` of the
    arm carrying the black gripper, in radians, with the same zero and sign
    conventions as the simulator. Fingers are two per-finger openings in metres
    (``0`` closed, ``0.035`` fully open per finger).
    """

    def observe(self) -> dict:
        """Return ``image`` (HxWx3 uint8 RGB), ``joint_pos`` [6], ``joint_vel`` [6], ``finger_pos`` [2]."""

    def command(self, action: np.ndarray) -> None:
        """Send six absolute joint targets and one common finger target, held for one control period."""

    def close(self) -> None:
        """Stop the robot and release resources."""


def prepare_image(image: np.ndarray, size: int, center_crop: bool = True) -> np.ndarray:
    """Convert a camera frame to the square RGB resolution rendered during training."""
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"Expected an HxWx3 uint8 RGB image, got {image.shape} {image.dtype}")
    if center_crop:
        h, w = image.shape[:2]
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        image = image[top : top + side, left : left + side]
    if image.shape[:2] != (size, size):
        image = np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))
    return np.ascontiguousarray(image)


def build_state(
    joint_pos: np.ndarray, joint_vel: np.ndarray, finger_pos: np.ndarray, last_command: np.ndarray
) -> np.ndarray:
    """Assemble the 21-value proprio vector in the simulator's order."""
    parts = [(joint_pos, 6), (joint_vel, 6), (finger_pos, 2), (last_command, ACTION_DIM)]
    values = []
    for value, size in parts:
        value = np.asarray(value, dtype=np.float32).reshape(-1)
        if value.shape != (size,):
            raise ValueError(f"Expected {size} state values, got {value.shape}")
        values.append(value)
    state = np.concatenate(values)
    if not np.isfinite(state).all():
        raise ValueError("Robot state contains non-finite values")
    return state


class SafetyFilter:
    """Apply the simulator's per-step slew limit and keep joints inside the training range.

    The simulator only limits joints to +/-2pi, which is no protection on hardware, so
    targets are additionally confined to the train-split action range plus ``margin``.
    """

    def __init__(self, low: np.ndarray, high: np.ndarray, max_delta: float = 0.12, margin: float = 0.05) -> None:
        low = np.asarray(low, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        if low.shape != (ACTION_DIM,) or high.shape != (ACTION_DIM,) or np.any(high <= low):
            raise ValueError("Expected seven increasing action bounds")
        if max_delta <= 0 or margin < 0:
            raise ValueError("max_delta must be positive and margin non-negative")
        self.low = np.r_[low[:6] - margin, 0.0]
        self.high = np.r_[high[:6] + margin, FINGER_MAX_M]
        self.max_delta = float(max_delta)

    def __call__(self, action: np.ndarray, last: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the command actually sent and whether any value was altered."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("Action must be seven finite values")
        last = np.asarray(last, dtype=np.float64)
        # Range first, slew last: from a pose outside the range the arm walks in gradually.
        target = np.clip(action, self.low, self.high)
        q = np.clip(target[:6], last[:6] - self.max_delta, last[:6] + self.max_delta)
        applied = np.r_[q, target[6]].astype(np.float32)
        return applied, bool(np.max(np.abs(applied - action)) > 1e-6)


class LocalPolicy:
    """Load SFT ``.pt`` weights or a GRPO checkpoint directory (SFT + LoRA adapter)."""

    def __init__(
        self,
        checkpoint: Path,
        device: str = "cuda",
        base_vlm: str | None = None,
        fast_tokenizer: str | None = None,
        torch_dtype: str | None = "bfloat16",
        sft_checkpoint: Path | None = None,
        decoding: str = "auto",
        repetition_penalty: float | None = None,
        center_crop: bool = True,
        no_state: bool = False,
    ) -> None:
        import torch
        from omegaconf import OmegaConf

        from .evaluate import _greedy_rows, checkpoint_run_dir, decoding_options, load_policy, use_robot_state

        self._torch = torch
        self._greedy_rows = _greedy_rows
        checkpoint = Path(checkpoint).resolve()
        self.decoding, self.repetition_penalty = decoding_options(checkpoint, decoding, repetition_penalty)
        policy, cfg, self.norm = load_policy(checkpoint, device, base_vlm, fast_tokenizer, torch_dtype, sft_checkpoint)
        self.use_state = use_robot_state(cfg, no_state)
        self.predictor = policy
        if self.decoding != "legacy":
            from .grpo_policy import FastGRPOPolicy

            self.predictor = FastGRPOPolicy(
                policy,
                temperature=1.0,
                strict_fast=self.decoding == "constrained",
                repetition_penalty=self.repetition_penalty,
            )
        self.max_new_tokens = int(cfg.framework.action_model.get("max_new_tokens", 256))
        saved = checkpoint_run_dir(checkpoint) / "grpo_config.yaml"
        if saved.is_file() and "max_new_tokens" in OmegaConf.load(saved):
            self.max_new_tokens = int(OmegaConf.load(saved).max_new_tokens)
        contract = self.norm.metadata["contract"]
        self.center_crop = center_crop
        self.metadata = {
            "checkpoint": str(checkpoint),
            "train_mode": "lora" if (checkpoint / "adapter").is_dir() else "sft",
            "decoding": self.decoding,
            "repetition_penalty": self.repetition_penalty,
            "use_state_prompt": self.use_state,
            "torch_dtype": str(next(policy.parameters()).dtype),
            "instruction": contract["instruction"],
            "image_size": int(contract["image_size"]),
            "training_obs_image_size": [int(contract["image_size"])] * 2,
            "control_hz": int(contract["control_hz"]),
            "action_horizon": int(cfg.framework.action_model.action_horizon),
            "max_joint_delta_per_step": float(contract["max_joint_delta_per_step"]),
            "action_low": [float(x) for x in self.norm.low],
            "action_high": [float(x) for x in self.norm.high],
            "joint_names": list(contract["joint_names"]),
            "state": contract["proprio"],
            "action": contract["action"],
        }

    def predict_action(
        self, image: np.ndarray, state: np.ndarray | None = None, instruction: str | None = None
    ) -> dict:
        """Return one chunk of physical targets, or a ``failure`` message for invalid FAST output.

        ``state`` is required only when the checkpoint puts robot state into its prompt.
        """
        start = time.monotonic()
        image = prepare_image(image, self.metadata["image_size"], self.center_crop)
        example = {"image": [Image.fromarray(image)], "lang": instruction or self.metadata["instruction"]}
        if self.use_state:
            if state is None:
                raise ValueError("This checkpoint uses robot state; send 'state'")
            state = np.asarray(state, dtype=np.float32).reshape(-1)
            if state.shape != (STATE_DIM,):
                raise ValueError(f"Expected {STATE_DIM} state values, got {state.shape}")
            example["state"] = state
        with self._torch.inference_mode():
            row = self._greedy_rows(self.predictor, [example], self.max_new_tokens)[0]
        latency = time.monotonic() - start
        if row["failure"]:
            return {"failure": row["failure"], "actions": None, "latency_s": latency}
        return {
            "failure": None,
            "actions": self.norm.unnormalize(row["actions"]).astype(np.float32),
            "normalized_actions": np.asarray(row["actions"], dtype=np.float32),
            "fast_token_ids": [int(t) for t in row["fast_token_ids"]],
            "latency_s": latency,
        }


class RemotePolicy:
    """Websocket client for ``serve``; needs only numpy, websockets and msgpack."""

    def __init__(self, address: str) -> None:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        host, _, port = address.rpartition(":")
        if not host or not port.isdigit():
            raise ValueError("--server must be host:port")
        self.client = WebsocketClientPolicy(host=host, port=int(port))
        self.metadata = self.client.get_server_metadata()
        self.use_state = bool(self.metadata.get("use_state_prompt", True))

    def predict_action(
        self, image: np.ndarray, state: np.ndarray | None = None, instruction: str | None = None
    ) -> dict:
        """Forward one observation; raise if the server reports an error."""
        payload = {"image": np.asarray(image, dtype=np.uint8)}
        if self.use_state:
            payload["state"] = np.asarray(state, dtype=np.float32)
        if instruction:
            payload["instruction"] = instruction
        response = self.client.predict_action(payload)
        if not response.get("ok"):
            raise RuntimeError(f"Policy server error: {response.get('error')}")
        result = response["data"]
        if result["actions"] is not None:
            result["actions"] = np.asarray(result["actions"], dtype=np.float32)
        return result

    def close(self) -> None:
        """Close the websocket."""
        self.client.close()


class SimRobot:
    """MuJoCo stand-in for hardware, used to validate the deployment loop end to end."""

    def __init__(self, seed: int = 0, randomize: bool = True, image_size: int = 128) -> None:
        import os

        os.environ.setdefault("MUJOCO_GL", "osmesa")
        from .env import TransferEnv

        self.env = TransferEnv(image_size=image_size)
        self.obs = self.env.reset(seed, randomize=randomize)
        self.info = self.env.task.info()

    def observe(self) -> dict:
        """Split the simulator proprio back into the hardware-style fields."""
        proprio = self.obs["proprio"]
        return {
            "image": self.obs["image"],
            "joint_pos": proprio[:6],
            "joint_vel": proprio[6:12],
            "finger_pos": proprio[12:14],
        }

    def command(self, action: np.ndarray) -> None:
        """Advance one 20 Hz control period."""
        self.obs, self.info = self.env.step(action)

    @property
    def success(self) -> bool:
        """Task success, available only in simulation."""
        return bool(self.info["success"])

    def close(self) -> None:
        """Release the renderer."""
        self.env.close()


def load_robot(spec: str, args: argparse.Namespace, image_size: int, control_hz: float) -> Robot:
    """Build ``sim``, the UR5 adapter, or import a hardware adapter given as ``module:ClassName``."""
    if spec == "sim":
        return SimRobot(args.sim_seed, not args.sim_no_randomize, image_size)
    kwargs = json.loads(args.robot_kwargs) if args.robot_kwargs else {}
    if spec == "ur5":
        from .ur5_robot import UR5Robot

        kwargs.setdefault("control_hz", control_hz)
        kwargs.setdefault("read_only", args.no_send)
        return UR5Robot(**kwargs)
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError("--robot must be 'sim', 'ur5' or 'package.module:ClassName'")
    return getattr(importlib.import_module(module), name)(**kwargs)


def initial_command(observation: dict) -> np.ndarray:
    """Treat the current pose as the previous command, as the simulator does at reset."""
    return np.r_[
        np.asarray(observation["joint_pos"], dtype=np.float32).reshape(6),
        np.float32(np.mean(observation["finger_pos"])),
    ].astype(np.float32)


def control_loop(
    robot: Robot,
    policy,
    safety: SafetyFilter,
    execute: int,
    max_steps: int,
    control_hz: float,
    send: bool = True,
    confirm: bool = False,
    instruction: str | None = None,
    log_path: Path | None = None,
) -> dict:
    """Observe, predict a chunk, then send ``execute`` filtered targets at ``control_hz``."""
    period = 1.0 / control_hz
    observation = robot.observe()
    last = initial_command(observation)
    step, failure = 0, None
    log = log_path.open("a") if log_path else None
    try:
        while step < max_steps:
            state = build_state(observation["joint_pos"], observation["joint_vel"], observation["finger_pos"], last)
            prediction = policy.predict_action(image=observation["image"], state=state, instruction=instruction)
            if prediction["failure"]:
                failure = prediction["failure"]
                print(f"[step {step}] invalid policy output, robot holds its last target: {failure}", flush=True)
                break
            chunk = np.asarray(prediction["actions"])[: min(execute, max_steps - step)]
            if confirm:
                print(np.array2string(chunk, precision=4, suppress_small=True))
                if input("Execute this chunk? [y/N] ").strip().lower() != "y":
                    failure = "operator_abort"
                    break
            record = {
                "step": step,
                "time": time.time(),
                "latency_s": float(prediction["latency_s"]),
                "state": state.tolist(),
                "predicted": chunk.tolist(),
                "applied": [],
                "clipped": [],
            }
            deadline = time.monotonic()
            for action in chunk:
                applied, clipped = safety(action, last)
                if send:
                    robot.command(applied)
                last = applied
                step += 1
                record["applied"].append(applied.tolist())
                record["clipped"].append(clipped)
                deadline += period
                time.sleep(max(0.0, deadline - time.monotonic()))
                if getattr(robot, "success", False):
                    break
            if log:
                log.write(json.dumps(record) + "\n")
                log.flush()
            print(
                f"[step {step}] latency={prediction['latency_s']:.3f}s clipped={sum(record['clipped'])}/{len(chunk)}",
                flush=True,
            )
            if getattr(robot, "success", False):
                break
            observation = robot.observe()
    finally:
        if log:
            log.close()
    return {"steps": step, "failure": failure, "success": getattr(robot, "success", None)}


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """Arguments shared by ``serve`` and local ``run``."""
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="SFT steps_*_pytorch_model.pt, or a GRPO checkpoints/iteration_* directory (SFT + LoRA)",
    )
    parser.add_argument("--sft-checkpoint", type=Path, help="Relocated SFT .pt used under a GRPO adapter")
    parser.add_argument("--base-vlm", help="Local Qwen2.5-VL-3B-Instruct-Action directory")
    parser.add_argument("--fast-tokenizer", help="Local FAST tokenizer directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--decoding", choices=["auto", "legacy", "aligned", "constrained"], default="auto")
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--no-center-crop", action="store_true", help="Resize the full frame instead of a center square")
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Image + instruction only; checkpoints trained with use_state_prompt=false are detected automatically",
    )


def make_local_policy(args: argparse.Namespace) -> LocalPolicy:
    """Build a local policy from parsed arguments."""
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --server is given")
    return LocalPolicy(
        args.checkpoint,
        args.device,
        args.base_vlm,
        args.fast_tokenizer,
        args.torch_dtype,
        args.sft_checkpoint,
        args.decoding,
        args.repetition_penalty,
        center_crop=not args.no_center_crop,
        no_state=args.no_state,
    )


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Serve the policy over websocket")
    add_policy_args(serve)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=10093)

    run = sub.add_parser("run", help="Closed-loop control of a robot")
    add_policy_args(run)
    run.add_argument("--server", help="host:port of a running 'serve'; skips local model loading")
    run.add_argument("--robot", default="sim", help="'sim', 'ur5' or 'package.module:ClassName'")
    run.add_argument("--robot-kwargs", help="JSON keyword arguments for the robot class, e.g. UR5Robot's")
    run.add_argument("--execute", type=int, default=8, help="Targets executed per predicted chunk (<= horizon)")
    run.add_argument("--max-steps", type=int, default=500)
    run.add_argument("--control-hz", type=float, help="Defaults to the training 20 Hz")
    run.add_argument("--max-joint-delta", type=float, help="rad per control step; defaults to the training 0.12")
    run.add_argument("--range-margin", type=float, default=0.05, help="rad allowed beyond the train action range")
    run.add_argument("--instruction", help="Override the training instruction")
    run.add_argument("--no-send", action="store_true", help="Predict and log without commanding the robot")
    run.add_argument("--confirm", action="store_true", help="Ask before executing each chunk")
    run.add_argument("--log", type=Path, help="JSONL file for states, predictions and applied commands")
    run.add_argument("--sim-seed", type=int, default=0)
    run.add_argument("--sim-no-randomize", action="store_true")
    args = parser.parse_args()

    if args.command == "serve":
        import logging

        from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer

        logging.basicConfig(level=logging.INFO, force=True)
        policy = make_local_policy(args)
        print(json.dumps(policy.metadata, indent=2), flush=True)
        WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()
        return

    if args.no_send and args.robot == "sim":
        parser.error("--no-send would freeze the simulator; use it only with hardware")
    policy = RemotePolicy(args.server) if args.server else make_local_policy(args)
    meta = policy.metadata
    if args.server and args.no_state and policy.use_state:
        parser.error("The server checkpoint uses robot state; restart 'serve' with --no-state")
    if not 1 <= args.execute <= meta["action_horizon"]:
        parser.error(f"--execute must be between 1 and {meta['action_horizon']}")
    safety = SafetyFilter(
        meta["action_low"],
        meta["action_high"],
        args.max_joint_delta or meta["max_joint_delta_per_step"],
        args.range_margin,
    )
    control_hz = args.control_hz or meta["control_hz"]
    robot = load_robot(args.robot, args, meta["image_size"], control_hz)
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"policy: {meta['checkpoint']} ({meta['train_mode']}, decoding={meta['decoding']}, "
        f"robot_state_prompt={meta.get('use_state_prompt', True)})",
        flush=True,
    )
    try:
        result = control_loop(
            robot,
            policy,
            safety,
            args.execute,
            args.max_steps,
            control_hz,
            send=not args.no_send,
            confirm=args.confirm,
            instruction=args.instruction,
            log_path=args.log,
        )
        print(json.dumps(result), flush=True)
    finally:
        robot.close()
        if isinstance(policy, RemotePolicy):
            policy.close()


if __name__ == "__main__":
    main()
