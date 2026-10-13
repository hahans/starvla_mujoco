"""Stochastic FAST policy with matching rollout and teacher-forced probabilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from starVLA.model.framework.VLM4A.fast_utils import state_instruction, strict_decode

from .fast_constraints import FastGrammar, FastGrammarProcessor


class ActionVocabulary(LogitsProcessor):
    """Restrict sampling to FAST tokens and the assistant terminator."""

    def __init__(self, allowed: torch.Tensor) -> None:
        self.allowed = allowed

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        masked = torch.full_like(scores, -torch.inf)
        masked[:, self.allowed] = scores[:, self.allowed]
        return masked


@dataclass
class ActionSample:
    """One sampled action chunk; CPU tensors retain the exact behavior tokens."""

    inputs: dict[str, torch.Tensor]
    tokens: torch.Tensor
    old_logprobs: torch.Tensor
    actions: np.ndarray | None
    failure: str | None
    ref_logprobs: torch.Tensor | None = None
    replay_logprobs: torch.Tensor | None = None


class FastGRPOPolicy:
    """Use one LoRA backbone; disabling its adapter gives the frozen SFT reference."""

    def __init__(
        self,
        policy: torch.nn.Module,
        temperature: float = 0.7,
        reference: FastGRPOPolicy | None = None,
        strict_fast: bool = False,
        repetition_penalty: float = 1.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("GRPO sampling temperature must be positive")
        self.reference = reference
        self.policy = policy
        self.interface = policy.qwen_vl_interface
        self.model = self.interface.model
        self.temperature = temperature
        if not np.isfinite(repetition_penalty) or repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be finite and positive")
        self.repetition_penalty = repetition_penalty
        self.end_id = self.interface.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.action_min = self.interface._ACTION_TOKEN_MIN
        self.action_max = self.interface._ACTION_TOKEN_MAX
        self.allowed = torch.tensor(
            [*range(self.action_min, self.action_max + 1), self.end_id], device=self.model.device
        )

        self.grammar = None
        if strict_fast:
            self.grammar = FastGrammar(
                policy.action_model.fast_tokenizer.bpe_tokenizer,
                policy.action_horizon * int(policy.config.framework.action_model.action_dim),
                self.model.device,
            )

    def sample(self, example: dict, max_new_tokens: int) -> ActionSample:
        """Sample one environment using the same implementation as batched rollout."""
        return self.sample_batch([example], max_new_tokens)[0]

    def predict_action(self, examples: list[dict]) -> dict:
        """Greedy evaluation with the same support and logit processors as training."""
        samples = self.sample_batch(
            examples, int(self.policy.config.framework.action_model.get("max_new_tokens", 256)), do_sample=False
        )
        for sample in samples:
            if sample.failure:
                raise ValueError(sample.failure)
        return {
            "normalized_actions": np.stack([sample.actions for sample in samples]),
            "fast_token_ids": [[t - self.action_min for t in sample.tokens[:-1].tolist()] for sample in samples],
            "generated_token_ids": [sample.tokens.tolist() for sample in samples],
        }

    @torch.no_grad()
    def reference_logprobs(self, sample: ActionSample) -> torch.Tensor:
        """Score with a separate frozen SFT model or the disabled LoRA adapter."""
        if self.reference is not None:
            return self.reference.logprobs(sample).cpu()
        if not hasattr(self.model, "disable_adapter"):
            raise RuntimeError("Full-parameter KL requires a frozen reference model")
        with self.model.disable_adapter():
            return self.logprobs(sample).cpu()

    @torch.no_grad()
    def sample_batch(self, examples: list[dict], max_new_tokens: int, do_sample: bool = True) -> list[ActionSample]:
        """Generate together with left padding; retain unpadded prompts for replay."""
        if not examples:
            return []
        if self.grammar is not None and max_new_tokens < self.grammar.max_tokens:
            raise ValueError(f"Strict FAST needs max_new_tokens >= {self.grammar.max_tokens}")
        self.model.eval()
        prompts = []
        for example in examples:
            instruction = state_instruction(example, self.policy.config.framework.get("use_state_prompt", False))
            prompts.append(
                dict(self.interface.build_qwenvl_inputs(images=[example["image"]], instructions=[instruction]))
            )
        inputs = batch_prompts(prompts, self.interface.processor.tokenizer.pad_token_id)
        generation = GenerationConfig(
            do_sample=do_sample,
            temperature=self.temperature if do_sample else 1.0,
            top_k=0,
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.end_id,
            pad_token_id=self.interface.processor.tokenizer.pad_token_id,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            repetition_penalty=self.repetition_penalty,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.model.device.type == "cuda"):
            output = self.model.generate(
                **inputs,
                generation_config=generation,
                logits_processor=LogitsProcessorList(
                    [
                        FastGrammarProcessor(self.grammar, self.allowed, inputs["input_ids"].shape[1])
                        if self.grammar is not None
                        else ActionVocabulary(self.allowed)
                    ]
                ),
                # HF >=4.50 can replace GenerationConfig's default-valued fields
                # with checkpoint defaults (this SFT has top_p=0.001!).
                use_model_defaults=False,
                top_k=0,
                top_p=1.0,
                repetition_penalty=self.repetition_penalty,
            )
        results = []
        for row, prompt in enumerate(prompts):
            tokens = output.sequences[row, inputs["input_ids"].shape[1] :]
            ends = torch.nonzero(tokens == self.end_id).flatten()
            if len(ends):
                tokens = tokens[: int(ends[0]) + 1]  # include EOS, exclude post-EOS padding
            old = torch.stack(
                [
                    score[row, token].float() - torch.logsumexp(score[row].float(), dim=-1)
                    for token, score in zip(tokens, output.scores[: len(tokens)], strict=True)
                ]
            )
            actions, failure = None, None
            try:
                if not len(tokens) or tokens[-1].item() != self.end_id:
                    raise ValueError("FAST generation reached max_new_tokens without EOS")
                action_ids = tokens[:-1].tolist()
                if any(t < self.action_min or t > self.action_max for t in action_ids):
                    raise ValueError("Non-FAST token in generated action")
                actions = strict_decode(
                    self.policy.action_model.fast_tokenizer,
                    [[t - self.action_min for t in action_ids]],
                    self.policy.action_horizon,
                    int(self.policy.config.framework.action_model.action_dim),
                )[0]
            except (ValueError, OverflowError, FloatingPointError) as exc:
                failure = str(exc)
            results.append(
                ActionSample(
                    {key: value.detach().cpu() for key, value in prompt.items()},
                    tokens.detach().cpu(),
                    old.detach().cpu(),
                    actions,
                    failure,
                )
            )
        return results

    def logprobs(self, sample: ActionSample) -> torch.Tensor:
        """Score only sampled tokens (including EOS), using the sampling distribution."""
        inputs = {key: value.to(self.model.device) for key, value in sample.inputs.items()}
        tokens = sample.tokens.to(self.model.device)
        prefix_len = inputs["input_ids"].shape[1]
        inputs["input_ids"] = torch.cat([inputs["input_ids"], tokens.unsqueeze(0)], dim=1)
        inputs["attention_mask"] = torch.cat([inputs["attention_mask"], torch.ones_like(tokens).unsqueeze(0)], dim=1)
        # Qwen caches generation rope deltas on the model; a full forward must recompute them.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.model.device.type == "cuda"):
            output = self.model(**inputs, use_cache=False, return_dict=True)
        logits = output.logits[0, prefix_len - 1 : prefix_len + len(tokens) - 1]
        restricted = logits.index_select(-1, self.allowed).float()
        if self.repetition_penalty != 1.0:
            restricted = replay_repetition_penalty(
                restricted, sample.inputs["input_ids"].to(tokens.device), tokens, self.allowed, self.repetition_penalty
            )
        restricted = restricted / self.temperature
        labels = torch.where(tokens == self.end_id, len(self.allowed) - 1, tokens - self.action_min)
        if self.grammar is not None:
            states = self.grammar.replay_states(labels)
            restricted = restricted.masked_fill(~self.grammar.valid[states], -torch.inf)
        return restricted.gather(-1, labels[:, None]).squeeze(-1) - torch.logsumexp(restricted, dim=-1)


def replay_repetition_penalty(
    logits: torch.Tensor, prompt: torch.Tensor, tokens: torch.Tensor, allowed: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Match HF's processor at each causal prefix, before temperature and masking.

    Only earlier generated tokens and the prompt count as repetitions. Including
    the current label here would leak the target and corrupt every PPO ratio.
    """
    prompt_seen = (prompt.reshape(-1, 1) == allowed).any(dim=0)
    generated = tokens[:, None] == allowed
    seen = prompt_seen | ((generated.cumsum(dim=0) - generated.long()) > 0)
    adjusted = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen, adjusted, logits)


def attach_lora(policy: torch.nn.Module, rank: int, alpha: int) -> None:
    """Train language attention projections while leaving vision and SFT weights fixed."""
    from peft import LoraConfig, get_peft_model

    model = policy.qwen_vl_interface.model
    targets = [
        name
        for name, _ in model.named_modules()
        if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) and "visual" not in name
    ]
    if not targets:
        raise ValueError("No Qwen language attention projections found for LoRA")
    policy.qwen_vl_interface.model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=0.0,
            target_modules=targets,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    policy.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Matching probabilities require dropout-free rollout, reference, and update forwards.
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def batch_prompts(prompts: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    """Left-pad text and concatenate Qwen image patches/grids in row order."""
    keys = set(prompts[0])
    supported = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    if not keys <= supported or any(set(prompt) != keys for prompt in prompts):
        raise ValueError("Expected consistent single-image Qwen prompt fields")
    length = max(prompt["input_ids"].shape[1] for prompt in prompts)
    result = {}
    for key in keys:
        if key in ("input_ids", "attention_mask"):
            value = pad_id if key == "input_ids" else 0
            result[key] = torch.cat(
                [
                    torch.nn.functional.pad(prompt[key], (length - prompt[key].shape[1], 0), value=value)
                    for prompt in prompts
                ],
                dim=0,
            )
        else:
            result[key] = torch.cat([prompt[key] for prompt in prompts], dim=0)
    return result


def configure_training(policy: torch.nn.Module, cfg) -> FastGRPOPolicy:
    """Configure LoRA or FP32 full weights with bf16 autocast and a frozen reference."""
    reference = None
    if cfg.get("train_mode", "lora") == "lora":
        attach_lora(policy, cfg.lora_rank, cfg.lora_alpha)
    else:
        if cfg.kl_beta > 0:
            frozen = copy.deepcopy(policy)
            frozen.to(dtype=torch.bfloat16 if cfg.device == "cuda" else torch.float32)
            frozen.requires_grad_(False).eval()
            frozen.qwen_vl_interface.model.gradient_checkpointing_disable()
            reference = FastGRPOPolicy(
                frozen,
                cfg.temperature,
                strict_fast=cfg.get("strict_fast", False),
                repetition_penalty=cfg.get("repetition_penalty", 1.0),
            )
        # Keep master weights, gradients and Adam moments in FP32. Autocast is bf16.
        policy.float().requires_grad_(True)
        policy.qwen_vl_interface.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    if reference is not None:
        for module in reference.policy.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
    return FastGRPOPolicy(
        policy,
        cfg.temperature,
        reference=reference,
        strict_fast=cfg.get("strict_fast", False),
        repetition_penalty=cfg.get("repetition_penalty", 1.0),
    )
