"""Byte-level grammar, replay masking, and validation selection regressions."""

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from mujoco_task.fast_constraints import FastGrammar, FastGrammarProcessor, advance_utf8
from mujoco_task.grpo_validation import improves


@pytest.fixture(scope="module")
def codec():
    if not Path("playground/Pretrained_models/fast/tokenizer.json").exists():
        pytest.skip("Local FAST assets not present")
    from starVLA.model.modules.action_model.fast_ActionHeader import get_action_model

    return get_action_model(OmegaConf.load("configs/mujoco/qwenfast.yaml")).fast_tokenizer


@pytest.fixture(scope="module")
def grammar(codec):
    return FastGrammar(codec.bpe_tokenizer, 56, torch.device("cpu"))


def test_real_codec_all_demonstration_tokens_are_accepted(codec, grammar):
    rng = np.random.default_rng(12)
    actions = rng.uniform(-1, 1, (20, 8, 7)).astype(np.float32)
    for tokens in codec(actions):
        sequence = torch.tensor([*tokens, 2048])
        states = grammar.replay_states(sequence)
        assert int(states[-1]) == grammar.terminal
        assert not grammar.valid[states[:-1], -1].any()
        assert grammar.valid[states[-1]].sum() == 1  # exactly EOS at 56 coefficients


@pytest.mark.parametrize(
    "data,valid",
    [
        (b"\xc0\x80", False),
        (b"\xed\xa0\x80", False),
        (b"\xf4\x90\x80\x80", False),
        (b"\x80", False),
        ("😀".encode(), True),
        ("界".encode(), True),
    ],
)
def test_utf8_overlong_surrogate_and_unicode_boundaries(data, valid):
    state, count = 0, 0
    for byte in data:
        state, completed = advance_utf8(state, byte)
        count += completed
        if state < 0:
            break
    assert (state == 0 and count == 1) == valid


def test_early_eos_and_oversized_chunks_are_rejected(codec, grammar):
    with pytest.raises(ValueError, match="violate"):
        grammar.replay_states(torch.tensor([2048]))
    tokens = codec.bpe_tokenizer("界" * 57)["input_ids"]
    with pytest.raises(ValueError, match="violate"):
        grammar.replay_states(torch.tensor(tokens))


def test_sampling_masks_equal_replay_masks_including_split_characters(codec, grammar):
    tokens = codec.bpe_tokenizer("界" * 55 + "😀")["input_ids"] + [2048]
    allowed = torch.arange(3, 2052)
    processor = FastGrammarProcessor(grammar, allowed, prefix_length=2)
    prefix = torch.tensor([[1, 2]])
    states = grammar.replay_states(torch.tensor(tokens))
    logits = torch.linspace(-1, 1, 2052)[None, :]
    for index, token in enumerate(tokens):
        masked = processor(prefix, logits)
        online = masked[0, allowed].log_softmax(-1)[token]
        replay_logits = logits[0, allowed].masked_fill(~grammar.valid[states[index]], -torch.inf)
        torch.testing.assert_close(online, replay_logits.log_softmax(-1)[token])
        assert torch.isfinite(online)
        prefix = torch.cat([prefix, torch.tensor([[token + 3]])], dim=1)
    assert online.item() == 0.0  # forced EOS carries no spurious policy gradient


def test_model_selection_keeps_sft_on_ties_and_rejects_mismatched_eval():
    baseline = dict(
        seeds=[1, 2],
        execute=8,
        max_steps=500,
        decoding="constrained",
        torch_dtype="torch.bfloat16",
        repetition_penalty=1.05,
        successes=1,
        invalid=0,
    )
    assert not improves(dict(baseline), baseline)
    assert not improves(dict(baseline, successes=0), baseline)
    assert improves(dict(baseline, successes=2), baseline)
    with pytest.raises(ValueError, match="decoding"):
        improves(dict(baseline, decoding="legacy"), baseline)
