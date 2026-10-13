"""Optional real-processor + tiny random Qwen forward/export test (no training).

Runs only when the small pinned processor metadata and FAST assets are locally
available. Never downloads weights and never calls backward or an optimizer.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image


def test_real_qwen_processor_forward_and_export(monkeypatch, tmp_path):
    metadata = Path("playground/processor_check")
    fast = Path("playground/Pretrained_models/fast")
    if not (metadata / "tokenizer.json").exists() or not (fast / "tokenizer.json").exists():
        pytest.skip("Pinned small Qwen/FAST processor assets not present; no automatic downloads")
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
    )

    from mujoco_task.data import write_json
    from mujoco_task.evaluate import load_policy
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    # The local system-site test env contains an unusable optional DeepSpeed.
    # Match the documented clean task environment, which does not install it.
    monkeypatch.setattr("accelerate.utils.other.is_deepspeed_available", lambda: False)
    cfg = OmegaConf.load("configs/mujoco/qwenfast.yaml")
    tiny = Qwen2_5_VLConfig(
        vocab_size=153713,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "fullatt_block_indexes": [0],
        },
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )
    backbone = Qwen2_5_VLForConditionalGeneration(tiny).eval()
    processor = AutoProcessor.from_pretrained(metadata, min_pixels=50176, max_pixels=50176)
    codec = get_action_model(cfg)
    # AutoProcessor is the same class object in both modules: bypass FAST loading
    # inside this temporary patch, then restore all real loaders for export reload.
    with monkeypatch.context() as m:
        m.setattr(
            "starVLA.model.modules.vlm.QWen2_5.Qwen2_5_VLForConditionalGeneration.from_pretrained",
            lambda *a, **kw: backbone,
        )
        m.setattr("starVLA.model.modules.vlm.QWen2_5.AutoProcessor.from_pretrained", lambda *a, **kw: processor)
        m.setattr("starVLA.model.framework.VLM4A.QwenFast.get_action_model", lambda config: codec)
        policy = build_framework(wrap_config(cfg)).eval()
    example = {
        "image": [Image.new("RGB", (128, 128), "green")],
        "lang": "move green right",
        "state": np.zeros(21, dtype=np.float32),
        "action": np.zeros((8, 7), dtype=np.float32),
    }
    with torch.no_grad():
        result = policy(examples=[example])
    assert torch.isfinite(result["action_loss"])
    assert int(result["num_action_tokens"]) > 1
    policy.qwen_vl_interface.model.save_pretrained(tmp_path / "Qwen2.5-VL-Action")
    processor.save_pretrained(tmp_path / "Qwen2.5-VL-Action")
    codec.fast_tokenizer.save_pretrained(tmp_path / "fast")
    OmegaConf.save(cfg, tmp_path / "task_config.yaml")
    write_json(tmp_path / "normalization.json", {"fit_split": "train", "min": [-1] * 7, "max": [1] * 7})
    (tmp_path / "COMPLETE").write_text("test fixture, not a trained checkpoint\n")
    restored, _, _ = load_policy(tmp_path, "cpu")
    with torch.no_grad():
        restored_result = restored(examples=[example])
    torch.testing.assert_close(restored_result["action_loss"], result["action_loss"], atol=1e-5, rtol=1e-5)

    # Native StarVLA checkpoints are raw framework state dicts plus run metadata.
    native_run = tmp_path / "native_run"
    native_weights = native_run / "checkpoints/steps_1000_pytorch_model.pt"
    native_weights.parent.mkdir(parents=True)
    torch.save(policy.state_dict(), native_weights)
    OmegaConf.save(cfg, native_run / "config.full.yaml")
    write_json(native_run / "normalization.json", {"fit_split": "train", "min": [-1] * 7, "max": [1] * 7})
    native_restored, _, _ = load_policy(
        native_weights, "cpu", base_vlm=str(tmp_path / "Qwen2.5-VL-Action"), fast_tokenizer=str(tmp_path / "fast")
    )
    with torch.no_grad():
        native_result = native_restored(examples=[example])
    torch.testing.assert_close(native_result["action_loss"], result["action_loss"], atol=1e-5, rtol=1e-5)
