"""Test the actual QwenFast forward/generation contracts without model training."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.model.framework.VLM4A.fast_utils import (
    action_labels,
    state_instruction,
    strict_decode,
)


def test_loss_mask_includes_only_action_and_eos():
    ids = torch.tensor([[0, 11, 12, 100, 101, 9, 7], [11, 12, 102, 9, 0, 0, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0]])
    labels = action_labels(ids, mask, 100, 103, 9)
    assert labels.tolist() == [[-100, -100, -100, 100, 101, 9, -100], [-100, -100, 102, 9, -100, -100, -100]]
    with pytest.raises(ValueError, match="No action"):
        action_labels(torch.tensor([[1, 2]]), torch.ones(1, 2), 100, 103, 9)
    with pytest.raises(ValueError, match="Missing"):
        action_labels(torch.tensor([[100, 5]]), torch.ones(1, 2), 100, 103, 9)


def test_state_prompt_no_privileged_fields():
    example = {"lang": "move green", "state": np.arange(21), "green_position": [999, 999, 999]}
    prompt = state_instruction(example, True)
    assert "previous_finger_target_m=[20.00000]" in prompt
    assert "999" not in prompt
    assert state_instruction(example, False) == "move green"
    with pytest.raises(ValueError):
        state_instruction({"lang": "x", "state": np.zeros(20)}, True)


def test_invalid_codec_does_not_become_zero_action():
    bpe = SimpleNamespace(get_vocab=lambda: {"a": 0}, decode=lambda ids: "a")
    processor = SimpleNamespace(bpe_tokenizer=bpe, min_token=0, scale=10)
    with pytest.raises(ValueError, match="Malformed"):
        strict_decode(processor, [[0]], 8, 7)
    with pytest.raises(ValueError, match="Empty"):
        strict_decode(processor, [[]], 8, 7)


@pytest.fixture
def mock_framework(monkeypatch):
    from starVLA.model.framework.VLM4A import QwenFast

    class Backbone(torch.nn.Module):
        device = torch.device("cpu")

        def generate(self, input_ids, **kwargs):
            assert kwargs["max_new_tokens"] == 256
            return torch.cat((input_ids, torch.tensor([[100, 9]])), dim=1)

    class Interface(torch.nn.Module):
        _ACTION_TOKEN_MIN, _ACTION_TOKEN_MAX = 100, 103

        def __init__(self):
            super().__init__()
            self.model = Backbone()
            self.processor = SimpleNamespace(tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda x: 9))
            self.prompts = []
            self.loss = torch.tensor(1.25)

        def build_qwenvl_inputs(self, images, instructions, solutions=None):
            self.prompts.append(instructions)
            # Deliberate action-looking token in prompt: extraction must use suffix only.
            return (
                {"input_ids": torch.tensor([[100, 4]]), "labels": torch.tensor([[-100, 100]])}
                if solutions
                else {"input_ids": torch.tensor([[100, 4]])}
            )

        def forward(self, **kwargs):
            return SimpleNamespace(loss=self.loss)

    iface = Interface()
    codec = SimpleNamespace(fast_tokenizer=SimpleNamespace(), encoder_action2fastoken=lambda actions: [[0]])
    monkeypatch.setattr(QwenFast, "get_vlm_model", lambda config: iface)
    monkeypatch.setattr(QwenFast, "get_action_model", lambda config: codec)
    seen = []

    def decode(processor, tokens, horizon, dim):
        seen.append(tokens)
        return np.zeros((1, horizon, dim))

    monkeypatch.setattr(QwenFast, "strict_decode", decode)
    cfg = OmegaConf.create(
        {
            "framework": {"name": "QwenFast", "use_state_prompt": True, "action_model": {"action_horizon": 8}},
            "datasets": {"vla_data": {}},
        }
    )
    model = QwenFast.Qwenvl_Fast(cfg)
    return model, iface, seen


def test_real_framework_prompt_and_generated_suffix(mock_framework):
    model, iface, seen = mock_framework
    from PIL import Image

    sample = {
        "image": [Image.new("RGB", (32, 32))],
        "lang": "move green",
        "state": np.zeros(21),
        "action": np.zeros((8, 7)),
    }
    assert model(examples=[sample])["action_loss"].item() == 1.25
    result = model.predict_action([sample])
    assert iface.prompts[0] == iface.prompts[1]
    assert seen == [[[0]]]
    assert result["generated_token_ids"] == [[100, 9]]
    assert result["normalized_actions"].shape == (1, 8, 7)


def test_nan_loss_is_a_failure(mock_framework):
    model, iface, _ = mock_framework
    iface.loss = torch.tensor(float("nan"))
    sample = {"image": [], "lang": "x", "state": np.zeros(21), "action": np.zeros((8, 7))}
    with pytest.raises(FloatingPointError):
        model(examples=[sample])
