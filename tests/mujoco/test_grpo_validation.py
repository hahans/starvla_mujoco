"""Validation must not change the stochastic training stream or retain rollouts."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task import grpo_validation
from mujoco_task.data import write_json
from mujoco_task.grpo import Episode, load_validation_seeds, validate_config


@pytest.mark.parametrize("fail", [False, True])
def test_validation_restores_rng_and_mode_even_on_failure(monkeypatch, fail):
    model = torch.nn.Linear(1, 1).train()
    model.device = torch.device("cpu")
    actor = SimpleNamespace(model=model, grammar=object(), repetition_penalty=1.05)
    calls = []

    def fake_rollout(envs, policy, norm, seeds, cfg, deterministic, store_samples):
        assert deterministic and not store_samples
        assert cfg.kl_beta == 0.0
        assert not torch.is_grad_enabled()
        policy.model.eval()
        torch.rand(17)
        calls.append(seeds)
        if fail:
            raise RuntimeError("simulated validation error")
        return [Episode([], 0, seed, 8, {"success": seed == 11}, None) for seed in seeds]

    monkeypatch.setattr(grpo_validation, "rollout_batch", fake_rollout)
    before = torch.get_rng_state().clone()
    if fail:
        with pytest.raises(RuntimeError, match="simulated"):
            grpo_validation.evaluate_actor(actor, [object(), object()], None, [10, 11, 12], 8, 500, 256)
    else:
        result = grpo_validation.evaluate_actor(
            actor, [object(), object()], None, [10, 11, 12], 8, 500, 256, split="train"
        )
        assert calls == [[10, 11], [12]]
        assert result["successes"] == 1 and result["invalid"] == 0
        assert result["success_rate"] == 1 / 3
        assert result["decoding"] == "constrained"
        assert result["split"] == "train"
        assert [episode["seed"] for episode in result["episodes"]] == [10, 11, 12]
    assert model.training
    torch.testing.assert_close(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "key,value",
    [
        ("validation_interval", 0.5),
        ("early_stop_patience", -1),
        ("validation_episodes", 0),
        ("validation_split", "test"),
    ],
)
def test_invalid_validation_configuration(key, value):
    cfg = OmegaConf.load("configs/mujoco/qwenfast_grpo_v2.yaml")
    cfg[key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(cfg)


def test_monitor_split_uses_manifest_order_and_only_allows_train_overlap(tmp_path):
    cfg = OmegaConf.load("configs/mujoco/qwenfast_grpo_v2.yaml")
    cfg.validation_data_root = str(tmp_path)
    metadata = {"manifest_sha256": "dataset"}
    for split, seeds in (("train", [7, 2, 0, 3, 9]), ("val", [100, 101, 102, 103, 104])):
        path = tmp_path / f"block_{split}/meta/source.json"
        path.parent.mkdir(parents=True)
        write_json(path, dict(metadata, episodes=[{"seed": seed} for seed in seeds]))
    assert cfg.validation_split == "train"
    assert load_validation_seeds(cfg, metadata) == [7, 2, 0, 3]
    cfg.validation_split = "val"
    assert load_validation_seeds(cfg, metadata) == [100, 101, 102, 103]
    del cfg.validation_split  # older configs preserve their original validation split
    assert load_validation_seeds(cfg, metadata) == [100, 101, 102, 103]
    cfg.train_seed_max = 100
    with pytest.raises(ValueError, match="overlap"):
        load_validation_seeds(cfg, metadata)
    cfg.validation_split = "train"
    cfg.validation_episodes = 6
    with pytest.raises(ValueError, match="more evaluation episodes"):
        load_validation_seeds(cfg, metadata)
    cfg.validation_episodes = 4
    with pytest.raises(ValueError, match="differs"):
        load_validation_seeds(cfg, {"manifest_sha256": "other dataset"})


def test_selection_rejects_different_splits_even_with_identical_seeds():
    with pytest.raises(ValueError, match="split"):
        grpo_validation.improves({"split": "train"}, {"split": "val"})
