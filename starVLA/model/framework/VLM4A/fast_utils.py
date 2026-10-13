"""Strict, independently testable FAST contracts used in training and rollout."""

from __future__ import annotations

import numpy as np
import torch
from scipy.fft import idct


def state_instruction(example: dict, enabled: bool = False) -> str:
    """Append measured robot state with identical formatting in train and eval."""
    instruction = example["lang"]
    if not enabled:
        return instruction
    state = np.asarray(example["state"], dtype=np.float64).reshape(-1)
    if state.shape != (21,) or not np.isfinite(state).all():
        raise ValueError("Expected 21 finite raw robot-state values")
    groups = [
        ("joint_rad", state[:6]),
        ("joint_velocity_rad_s", state[6:12]),
        ("finger_m", state[12:14]),
        ("previous_joint_target_rad", state[14:20]),
        ("previous_finger_target_m", state[20:]),
    ]
    return (
        instruction
        + "\nRobot state: "
        + "; ".join(f"{name}=[" + ",".join(f"{x:.5f}" for x in values) + "]" for name, values in groups)
    )


def strict_decode(processor, tokens: list[list[int]], horizon: int, dim: int) -> np.ndarray:
    """Decode FAST DCT coefficients, raising instead of silently returning zeros."""
    decoded = []
    vocab = processor.bpe_tokenizer.get_vocab()
    valid_ids = set(vocab.values())
    for sequence in tokens:
        if not sequence or any(int(t) not in valid_ids for t in sequence):
            raise ValueError("Empty FAST sequence or token outside the tokenizer vocabulary")
        text = processor.bpe_tokenizer.decode(sequence)
        if len(text) != horizon * dim:
            raise ValueError(f"Malformed FAST chunk: {len(text)} coefficients; expected {horizon * dim}")
        coeff = np.asarray([ord(c) + processor.min_token for c in text], dtype=np.float64)
        actions = idct(coeff.reshape(horizon, dim) / processor.scale, axis=0, norm="ortho")
        if not np.isfinite(actions).all():
            raise ValueError("FAST decoded a non-finite action")
        decoded.append(actions)
    return np.stack(decoded).astype(np.float32)


def action_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, action_min: int, action_max: int, end_id: int
) -> torch.Tensor:
    """Supervise the contiguous assistant action sequence and its end token only."""
    labels = torch.full_like(input_ids, -100)
    for row, ids in enumerate(input_ids):
        active = attention_mask[row].bool()
        positions = torch.nonzero((ids >= action_min) & (ids <= action_max) & active).flatten()
        if not len(positions):
            raise ValueError("No action labels: checkpoint must contain <robot_action_*> tokens")
        first, last = int(positions[0]), int(positions[-1])
        if last - first + 1 != len(positions):
            raise ValueError("Action tokens must form one contiguous assistant segment")
        if last + 1 >= len(ids) or int(ids[last + 1]) != end_id or not active[last + 1]:
            raise ValueError("Missing assistant end token after action sequence")
        labels[row, first : last + 2] = ids[first : last + 2]
    return labels
