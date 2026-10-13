"""Bounded scene sampling, GRPO grouping, and reproducible resume regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task.grpo import (
    collect_episodes,
    collect_rollout,
    group_advantages,
    group_reset_seeds,
    training_seed_range,
    validate_config,
)


def config():
    return OmegaConf.load("configs/mujoco/qwenfast_grpo_v2.yaml")


def test_random_seeds_include_both_endpoints_and_allow_repeats():
    cfg = config()
    validate_config(cfg)
    drawn = [seed for iteration in range(200) for seed in group_reset_seeds(iteration, cfg)]
    assert set(drawn) == set(range(21))
    assert len(drawn) > len(set(drawn))
    assert 20 in training_seed_range(cfg) and 21 not in training_seed_range(cfg)
    cfg.train_seed_min = cfg.train_seed_max = 20
    assert group_reset_seeds(17, cfg) == [20] * cfg.groups_per_iteration


def test_resume_and_batching_do_not_change_scene_sequence(tmp_path, monkeypatch):
    cfg = config()
    expected = [group_reset_seeds(i, cfg) for i in range(10)]
    state = np.random.get_state()
    group_reset_seeds(8, cfg)
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    path = tmp_path / "grpo_config.yaml"
    OmegaConf.save(cfg, path)
    restored = OmegaConf.load(path)
    # Resume at iteration 5 without consuming earlier draws, in any evaluation order.
    for i in [8, 5, 9, 6, 7]:
        assert group_reset_seeds(i, restored) == expected[i]
    restored.seed += 1
    assert [group_reset_seeds(i, restored) for i in range(10)] != expected

    def fake_rollout(envs, actor, norm, seeds, cfg):
        assert len(envs) == len(seeds)
        return seeds

    monkeypatch.setattr("mujoco_task.grpo.rollout_batch", fake_rollout)
    for num_envs in [1, 3, 8]:
        actual = collect_episodes([None] * num_envs, None, None, 5, cfg)
        assert actual == [seed for seed in expected[5] for _ in range(cfg.group_size)]


def test_legacy_sequential_reset_rule_is_preserved():
    cfg = config()
    cfg.seed = 4000000
    cfg.groups_per_iteration = 2
    cfg.rollout_epoch = 1
    cfg.train_seed_min = cfg.train_seed_max = None
    validate_config(cfg)
    assert group_reset_seeds(3, cfg) == [4000006, 4000007]
    assert training_seed_range(cfg) == range(4000000, 4000200)
    del cfg.train_seed_min
    del cfg.train_seed_max
    assert group_reset_seeds(3, cfg) == [4000006, 4000007]
    cfg.rollout_epoch = 16
    assert group_reset_seeds(3, cfg, 0) == [4000006, 4000007]
    assert group_reset_seeds(3, cfg, 1) == [4000290, 4000291]


def test_rollout_epoch_draws_a_new_scene_each_pass(monkeypatch):
    cfg = config()
    cfg.rollout_epoch = 3
    cfg.minibatch_size = cfg.groups_per_iteration * cfg.group_size * cfg.rollout_epoch
    validate_config(cfg)
    per_pass = [
        [seed for seed in group_reset_seeds(5, cfg, epoch) for _ in range(cfg.group_size)]
        for epoch in range(cfg.rollout_epoch)
    ]
    assert per_pass[0] == [seed for seed in group_reset_seeds(5, cfg) for _ in range(cfg.group_size)]
    assert len({tuple(seeds) for seeds in (group_reset_seeds(5, cfg, epoch) for epoch in range(cfg.rollout_epoch))}) > 1
    seen = []

    def fake_rollout(envs, actor, norm, seeds, train_cfg):
        assert len(envs) == len(seeds)
        seen.extend(seeds)
        return [SimpleNamespace(seed=seed, steps=1, reward=0.0, info={"success": False}, failure=None) for seed in seeds]

    monkeypatch.setattr("mujoco_task.grpo.rollout_batch", fake_rollout)
    episodes = collect_rollout([None], None, None, 5, cfg)
    expected = [seed for expanded in per_pass for seed in expanded]
    assert seen == expected
    assert [episode.seed for episode in episodes] == expected

    def fake_advantage(rewards, loss_mask, group_size):
        grouped = rewards.view(-1, group_size)
        return (grouped - grouped.mean(dim=-1, keepdim=True)).reshape(1, -1), None

    rewards = []
    for epoch in range(cfg.rollout_epoch):
        for _group in range(cfg.groups_per_iteration):
            rewards.extend([1.0, 0.0] * (cfg.group_size // 2) if epoch == 0 else [0.0] * cfg.group_size)
    advantages = group_advantages(rewards, cfg.group_size, fake_advantage)
    solo = group_advantages(rewards[: cfg.group_size], cfg.group_size, fake_advantage)
    assert torch.equal(advantages[: cfg.group_size], solo)
    later = cfg.groups_per_iteration * cfg.group_size
    assert torch.count_nonzero(advantages[later : later + cfg.group_size]) == 0


def test_minibatch_must_divide_the_trajectory_buffer():
    cfg = config()
    cfg.minibatch_size = 100
    with pytest.raises(ValueError, match="minibatch_size"):
        validate_config(cfg)


@pytest.mark.parametrize("rollout_epoch", [0, -1, 1.5])
def test_invalid_rollout_epoch_fails_before_training(rollout_epoch):
    cfg = config()
    cfg.rollout_epoch = rollout_epoch
    with pytest.raises(ValueError, match="rollout_epoch"):
        validate_config(cfg)


@pytest.mark.parametrize("lower,upper", [(None, 20), (0, None), (-1, 20), (21, 20), (0.5, 20), (0, True)])
def test_invalid_random_seed_bounds_fail_before_training(lower, upper):
    cfg = config()
    cfg.train_seed_min, cfg.train_seed_max = lower, upper
    with pytest.raises(ValueError, match="train_seed"):
        validate_config(cfg)
