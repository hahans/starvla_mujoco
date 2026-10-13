"""Compare adapter math with RLinf's complete embodied preprocessing path."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from transformers import RepetitionPenaltyLogitsProcessor

from mujoco_task.evaluate import decoding_options
from mujoco_task.grpo import (
    Episode,
    bounded_reference_kl,
    group_advantages,
    importance_weights,
    load_rlinf,
    update_actor,
)
from mujoco_task.grpo_policy import ActionSample, replay_repetition_penalty


@pytest.fixture(scope="module")
def algorithms():
    root = Path(__file__).resolve().parents[3] / "Rlinf4/Rlinf4"
    if not root.exists():
        pytest.skip("Configured RLinf checkout is absent")
    return load_rlinf(root)[:2]


def test_advantages_match_rlinf_embodied_with_variable_episode_lengths(algorithms):
    from rlinf.algorithms.registry import calculate_adv_and_returns

    scores = torch.tensor([0.0, 1, 1, 0, 0, 1])
    lengths = torch.tensor([2, 4, 3, 1, 3, 4])
    rewards = torch.zeros(4, 6, 1)
    rewards[lengths - 1, torch.arange(6), 0] = scores
    mask = (torch.arange(4)[:, None] < lengths)[..., None]
    dones = (torch.arange(5)[:, None] >= lengths)[..., None]
    result = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        reward_type="action_level",
        rewards=rewards,
        dones=dones,
        loss_mask=mask,
        group_size=3,
    )
    actual = group_advantages(scores.tolist(), 3, algorithms[0])
    torch.testing.assert_close(result["advantages"], actual[None, :, None] * mask)


def test_episode_mean_matches_rlinf_masked_length_correction(algorithms):
    from rlinf.algorithms.registry import policy_loss

    # One token per control step makes both reductions directly comparable.
    lengths = torch.tensor([2, 4, 3])
    mask = torch.arange(4)[None, :] < lengths[:, None]
    logits = torch.linspace(-2, -0.1, 12).reshape(3, 4, 1).requires_grad_()
    old = logits.detach() + torch.linspace(-0.5, 0.5, 12).reshape(3, 4, 1)
    advantage = torch.tensor([-1.0, 0.5, 1.0])[:, None].expand(3, 4)
    result, _ = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=1,
        logprobs=logits,
        old_logprobs=old,
        advantages=advantage,
        loss_mask=mask,
        loss_mask_sum=lengths[:, None].expand(3, 4),
        max_episode_steps=4,
        clip_ratio_low=0.1,
        clip_ratio_high=0.1,
    )
    expected = sum(
        algorithms[1](
            logprobs=logits[i, :length, 0],
            old_logprobs=old[i, :length, 0],
            advantages=advantage[i, :length],
            loss_mask=torch.ones(length, dtype=torch.bool),
            clip_ratio_low=0.1,
            clip_ratio_high=0.1,
        )[0]
        / 3
        for i, length in enumerate(lengths)
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(torch.autograd.grad(result, logits)[0], torch.autograd.grad(expected, logits)[0])


def test_bounded_kl_matches_rlinf_and_remains_finite(algorithms):
    from rlinf.algorithms.utils import kl_penalty

    logp = torch.tensor([-1000.0, -20, -2, 0.0, 0.01, 2, 20, 1000], requires_grad=True)
    reference = torch.zeros_like(logp)
    kl = bounded_reference_kl(logp, reference)
    torch.testing.assert_close(kl, kl_penalty(logp, reference, "k3"), atol=1e-6, rtol=1e-5)
    kl.sum().backward()
    assert torch.isfinite(kl).all() and torch.isfinite(logp.grad).all()


def test_zero_advantage_still_restores_sft_with_kl(algorithms):
    from test_grpo import toy_actor

    actor = toy_actor()
    sample = ActionSample(
        {"input_ids": torch.tensor([[1, 2, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
        torch.tensor([3, 5, 7]),
        torch.zeros(3),
        None,
        None,
    )
    sample.ref_logprobs = actor.logprobs(sample).detach()
    with torch.no_grad():
        actor.model.weight.add_(0.1)
    sample.old_logprobs = actor.logprobs(sample).detach()
    before = bounded_reference_kl(sample.old_logprobs, sample.ref_logprobs).mean()
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.001)
    cfg = OmegaConf.create(dict(update_epochs=1, clip_ratio=0.2, kl_beta=0.1, max_grad_norm=1.0))
    result = update_actor(actor, [Episode([sample], 0, 0, 1, {}, None)], torch.zeros(1), optimizer, algorithms[1], cfg)
    after = bounded_reference_kl(actor.logprobs(sample), sample.ref_logprobs).mean()
    assert result["updated"] and result["kl_only"] and after < before


def test_importance_weights_match_rlinf_floor():
    recompute = torch.tensor([0.0, 0.0, -1.0], requires_grad=True)
    rollout = torch.tensor([0.5, -1.0, 0.0])
    weights = importance_weights(recompute, rollout, 1.0)
    expected = (recompute.detach() - rollout).exp().clamp(min=1.0)
    torch.testing.assert_close(weights, expected)
    assert weights[0].item() == 1.0  # exp(-0.5) is raised to the floor
    assert weights[1].item() > 2.0  # no upper cap
    assert not weights.requires_grad


def test_replay_mismatch_continues_with_rollout_denominator(algorithms):
    from test_grpo import toy_actor

    actor = toy_actor()
    sample = ActionSample(
        {"input_ids": torch.tensor([[1, 2, 1]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)},
        torch.tensor([3, 5, 7]),
        torch.zeros(3),
        None,
        None,
    )
    before = actor.logprobs(sample).detach().clone()
    sample.old_logprobs = before - 0.5619  # the mismatch that previously aborted training
    behavior = sample.old_logprobs.clone()
    cfg = OmegaConf.create(
        dict(
            update_epochs=1,
            clip_ratio=0.1,
            kl_beta=0.0,
            max_grad_norm=1.0,
            importance_sampling_fix=True,
            importance_sampling_clip=1.0,
        )
    )
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.001)
    metrics = update_actor(
        actor, [Episode([sample], 1, 0, 1, {}, None)], torch.ones(1), optimizer, algorithms[1], cfg
    )
    expected_weight = float((before - behavior).exp().mean())
    assert metrics["replay_logprob_abs_max"] == pytest.approx(0.5619, abs=1e-4)
    assert metrics["importance_weight_mean"] == pytest.approx(expected_weight, rel=1e-5)
    assert metrics["clip_fraction"] == 1.0  # PPO ratio still uses the rollout log-prob
    torch.testing.assert_close(sample.old_logprobs, behavior)
    torch.testing.assert_close(sample.replay_logprobs, before)


def test_repetition_replay_matches_hf_causal_prefixes():
    prompt, tokens = torch.tensor([[1, 3, 3]]), torch.tensor([5, 3, 5, 7])
    allowed = torch.tensor([3, 4, 5, 7])
    logits = torch.linspace(-8, 8, 32).reshape(4, 8)
    processor = RepetitionPenaltyLogitsProcessor(1.05)
    expected = torch.cat(
        [
            processor(torch.cat([prompt, tokens[:i][None, :]], dim=1), logits[i : i + 1].clone())[:, allowed]
            for i in range(len(tokens))
        ]
    )
    actual = replay_repetition_penalty(logits[:, allowed], prompt, tokens, allowed, 1.05)
    torch.testing.assert_close(actual, expected)
    assert actual[0, 2] == logits[0, 5]  # current label is not part of its own prefix


def test_eval_defaults_preserve_old_policy_and_restore_v2_processors(tmp_path):
    (tmp_path / "config.full.yaml").write_text("framework: {}")
    assert decoding_options(tmp_path, "auto", None) == ("legacy", None)
    OmegaConf.save(OmegaConf.create(dict(strict_fast=True, repetition_penalty=1.05)), tmp_path / "grpo_config.yaml")
    assert decoding_options(tmp_path, "auto", None) == ("constrained", 1.05)
    assert decoding_options(tmp_path, "aligned", 1.0) == ("aligned", 1.0)
