"""Native StarVLA wiring and accumulation boundaries, using only test doubles."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task.data import read_json
from starVLA.dataloader import build_dataloader
from starVLA.dataloader.mujoco_lerobot import EpisodeSampler

ROOT = Path(__file__).resolve().parents[2]


def config_for(converted, tmp_path):
    _, root, _ = converted
    cfg = OmegaConf.load(ROOT / "configs/mujoco/qwenfast.yaml")
    cfg.datasets.vla_data.data_root_dir = str(root)
    cfg.datasets.vla_data.num_workers = 0
    cfg.run_root_dir = str(tmp_path / "runs")
    cfg.output_dir = str(Path(cfg.run_root_dir) / cfg.run_id)
    return cfg


def native_module(monkeypatch):
    monkeypatch.setenv("STARVLA_DISABLE_DEEPSPEED", "1")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    import starVLA.training.train_starvla as native

    return native


def test_official_dataloader_route_preserves_contract(converted, tmp_path):
    cfg = config_for(converted, tmp_path)
    loader = build_dataloader(cfg, cfg.datasets.vla_data.dataset_py)
    sample = next(iter(loader))[0]
    assert sample["state"].shape == (21,)
    assert sample["action"].shape == (8, 7)
    assert read_json(Path(cfg.output_dir) / "normalization.json")["fit_split"] == "train"
    assert max(loader.dataset.normalizer.high[:6]) < 1.01
    sampler = EpisodeSampler(loader.dataset, 42)
    first = list(sampler)
    sampler.set_epoch(1)
    assert sorted(first) == sorted(sampler) == list(range(len(loader.dataset)))
    assert first != list(sampler)
    cfg.datasets.vla_data.data_mix = "mujoco_block_test"
    with pytest.raises(ValueError, match="only"):
        build_dataloader(cfg, cfg.datasets.vla_data.dataset_py)


def test_native_main_owns_training_wiring(monkeypatch, converted, tmp_path):
    native = native_module(monkeypatch)
    cfg = config_for(converted, tmp_path)
    events = []
    monkeypatch.setattr(native, "build_framework", lambda cfg: events.append("framework") or object())
    monkeypatch.setattr(native, "setup_optimizer_and_scheduler", lambda **kw: (object(), object()))

    class SpyTrainer:
        def __init__(self, **kw):
            assert len(kw["vla_train_dataloader"].dataset) == 10
            assert kw["accelerator"].gradient_accumulation_steps == 16

        def prepare_training(self):
            events.append("prepare")

        def train(self):
            events.append("native_train_dispatch")

    monkeypatch.setattr(native, "VLATrainer", SpyTrainer)
    native.main(cfg)
    assert events == ["framework", "prepare", "native_train_dispatch"]


def test_native_accumulation_does_not_clear_earlier_microbatches(monkeypatch):
    native = native_module(monkeypatch)
    trainer = object.__new__(native.VLATrainer)
    pending, applied, events = [], [], []

    class FakeAccelerator:
        sync_gradients = False
        optimizer_step_was_skipped = False
        step = 0

        @contextmanager
        def accumulate(self, model):
            self.step += 1
            self.sync_gradients = self.step % 2 == 0
            yield

        def backward(self, loss):
            pending.append(float(loss))

        def clip_grad_norm_(self, parameters, limit):
            events.append("clip")

    accelerator = FakeAccelerator()

    def step():
        if accelerator.sync_gradients:
            applied.append(sum(pending))

    def zero_grad():
        if accelerator.sync_gradients:
            pending.clear()

    trainer.accelerator = accelerator
    trainer.optimizer = SimpleNamespace(step=step, zero_grad=zero_grad)
    trainer.model = SimpleNamespace(
        forward=lambda batch: {"action_loss": torch.tensor(float(batch))}, parameters=lambda: []
    )
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(gradient_clipping=1.0))
    trainer.lr_scheduler = SimpleNamespace(step=lambda: events.append("lr"))
    trainer._train_step(1)
    trainer._train_step(2)
    assert applied == [3.0]  # No real optimizer, gradient update, or VLA training executed.
    assert events == ["clip", "lr"]


def test_official_checkpoint_schema(monkeypatch, tmp_path):
    native = native_module(monkeypatch)
    trainer = object.__new__(native.VLATrainer)
    trainer.config = SimpleNamespace(output_dir=str(tmp_path), trainer=SimpleNamespace(save_format="pt"))
    trainer.completed_steps = 1000
    trainer.checkpoint_dir = str(tmp_path / "checkpoints")
    Path(trainer.checkpoint_dir).mkdir()
    trainer.model = torch.nn.Linear(1, 1)
    trainer.accelerator = SimpleNamespace(is_main_process=True, print=lambda *a: None, wait_for_everyone=lambda: None)
    trainer._save_checkpoint()
    path = Path(trainer.checkpoint_dir) / "steps_1000_pytorch_model.pt"
    state = torch.load(path, weights_only=True)
    np.testing.assert_array_equal(state["weight"].numpy(), trainer.model.weight.detach().numpy())
    assert not (tmp_path / "accelerator_state").exists()


def test_native_periodic_hooks_run_once_per_optimizer_step(monkeypatch):
    native = native_module(monkeypatch)
    trainer = object.__new__(native.VLATrainer)
    trainer.completed_steps = 0
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(max_train_steps=2, eval_interval=0, save_interval=1))
    trainer.accelerator = SimpleNamespace(sync_gradients=False, is_local_main_process=False)
    microsteps, saved, logged, evaluated = [], [], [], []
    trainer._log_training_config = lambda: None
    trainer._create_data_iterators = lambda: None
    trainer._get_next_batch = lambda: None

    def train_step(batch):
        microsteps.append(1)
        trainer.accelerator.sync_gradients = len(microsteps) % 2 == 0
        return {}

    trainer._train_step = train_step
    trainer._save_checkpoint = lambda: saved.append(trainer.completed_steps)
    trainer._log_metrics = lambda metrics: logged.append(trainer.completed_steps)
    trainer.eval_action_model = lambda metrics: evaluated.append(1) or metrics
    trainer._finalize_training = lambda: None
    trainer.train()
    assert len(microsteps) == 4
    assert saved == logged == [1, 2]
    assert evaluated == []
