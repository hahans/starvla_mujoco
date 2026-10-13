"""Grouped evaluation must keep active seeds in one left-padded generate call."""

import numpy as np
from omegaconf import OmegaConf

from mujoco_task.evaluate import run_group, use_robot_state


class _Env:
    def __init__(self, succeed_at: int) -> None:
        self.succeed_at = succeed_at
        self.steps = 0
        self.seed = None
        self.task = self

    def reset(self, seed: int) -> dict:
        self.seed = seed
        self.steps = 0
        return {"image": np.zeros((2, 2, 3), np.uint8), "proprio": np.array([seed], dtype=np.float32)}

    def info(self) -> dict:
        return {"success": self.steps >= self.succeed_at, "applied_action": [float(self.steps)]}

    def step(self, action: np.ndarray, observe: bool = True) -> tuple[dict, dict]:
        self.steps += 1
        observation = {"image": np.zeros((2, 2, 3), np.uint8), "proprio": np.array([self.seed], dtype=np.float32)}
        return (observation if observe else {}), self.info()


def test_no_state_drops_checkpoint_robot_prompt():
    cfg = OmegaConf.create({"framework": {"use_state_prompt": True}})
    assert use_robot_state(cfg, False) is True
    assert use_robot_state(cfg, True) is False
    assert cfg.framework.use_state_prompt is False


def test_finished_episode_leaves_the_next_generate_batch():
    batches = []

    def predict(examples):
        batches.append([int(example["state"][0]) for example in examples])
        return [
            {"failure": None, "actions": np.zeros((2, 7)), "fast_token_ids": [1], "generated_token_ids": [1, 7]}
            for _ in examples
        ]

    finished = run_group(
        [_Env(2), _Env(4), _Env(4), _Env(4)],
        [0, 1, 2, 3],
        predict,
        lambda actions: actions,
        "move the green block",
        execute=2,
        max_steps=4,
        video=False,
    )
    assert batches == [[0, 1, 2, 3], [1, 2, 3]]
    assert [item[0]["seed"] for item in finished] == [0, 1, 2, 3]
    assert [item[0]["steps"] for item in finished] == [2, 4, 4, 4]
    assert [item[0]["success"] for item in finished] == [True, True, True, True]


def test_invalid_episode_leaves_the_batch_without_stopping_the_others():
    batches = []

    def predict(examples):
        seeds = [int(example["state"][0]) for example in examples]
        batches.append(seeds)
        rows = []
        for seed in seeds:
            if seed == 1 and len(batches) == 1:
                rows.append(
                    {"failure": "bad tokens", "actions": None, "fast_token_ids": None, "generated_token_ids": None}
                )
            else:
                rows.append(
                    {"failure": None, "actions": np.zeros((1, 7)), "fast_token_ids": [1], "generated_token_ids": [1, 7]}
                )
        return rows

    finished = run_group(
        [_Env(1), _Env(1), _Env(1)],
        [0, 1, 2],
        predict,
        lambda actions: actions,
        "move the green block",
        execute=1,
        max_steps=1,
        video=False,
    )
    assert batches == [[0, 1, 2]]
    assert [item[0]["failure"] for item in finished] == [None, "bad tokens", None]
    assert [item[0]["success"] for item in finished] == [True, False, True]
