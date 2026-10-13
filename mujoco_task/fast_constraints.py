"""Exact UTF-8/FAST coefficient grammar shared by sampling and policy loss."""

from __future__ import annotations

import json

import torch
from transformers import LogitsProcessor


def advance_utf8(state: int, byte: int) -> tuple[int, int]:
    """Return (next UTF-8 state, completed codepoints), rejecting invalid Unicode."""
    if state == 0:
        if byte <= 0x7F:
            return 0, 1
        if 0xC2 <= byte <= 0xDF:
            return 1, 0
        if byte == 0xE0:
            return 4, 0
        if byte == 0xED:
            return 5, 0
        if 0xE1 <= byte <= 0xEF:
            return 2, 0
        if byte == 0xF0:
            return 6, 0
        if byte == 0xF4:
            return 7, 0
        if 0xF1 <= byte <= 0xF3:
            return 3, 0
    elif state in (1, 2, 3) and 0x80 <= byte <= 0xBF:
        return state - 1, int(state == 1)
    elif state == 4 and 0xA0 <= byte <= 0xBF:
        return 1, 0
    elif state == 5 and 0x80 <= byte <= 0x9F:
        return 1, 0
    elif state == 6 and 0x90 <= byte <= 0xBF:
        return 2, 0
    elif state == 7 and 0x80 <= byte <= 0x8F:
        return 2, 0
    return -1, 0


class FastGrammar:
    """DFA accepting exactly horizon * action_dim Unicode DCT coefficients.

    FAST's ByteLevel BPE tokens can split a UTF-8 character across several tokens.
    Counting decoded tokens independently would incorrectly count replacement
    characters. This DFA follows the original bytes, including partial characters.
    """

    def __init__(self, tokenizer, coefficients: int, device: torch.device) -> None:
        backend = json.loads(tokenizer.backend_tokenizer.to_str())
        if backend["decoder"]["type"] != "ByteLevel" or backend.get("added_tokens"):
            raise ValueError("Strict FAST requires ByteLevel BPE without added special tokens")
        vocab = tokenizer.get_vocab()
        if sorted(vocab.values()) != list(range(len(vocab))):
            raise ValueError("FAST vocabulary ids must be contiguous")
        # Invert ByteLevel's reversible mapping of all 256 bytes to Unicode symbols.
        visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        order = visible + [b for b in range(256) if b not in visible]
        symbols = visible + list(range(256, 256 + 256 - len(visible)))
        inverse = {chr(symbol): byte for byte, symbol in zip(order, symbols, strict=True)}
        pieces = [b""] * len(vocab)
        for token, index in vocab.items():
            pieces[index] = bytes(inverse[char] for char in token)
        if any(not piece for piece in pieces):
            raise ValueError("Empty FAST token would allow nonterminating generation")
        transitions = torch.full(((coefficients + 1) * 8, len(vocab) + 1), -1, dtype=torch.long)
        for initial in range(8):
            for token, piece in enumerate(pieces):
                state, count = initial, 0
                for byte in piece:
                    state, completed = advance_utf8(state, byte)
                    if state < 0:
                        break
                    count += completed
                if state < 0:
                    continue
                for before in range(coefficients + 1 - count):
                    after = before + count
                    if after < coefficients or state == 0:
                        transitions[before * 8 + initial, token] = after * 8 + state
        self.terminal = coefficients * 8
        transitions[self.terminal, -1] = self.terminal
        self.cpu_transitions = transitions
        self.transitions = transitions.to(device)
        self.valid = self.transitions >= 0
        self.max_tokens = 4 * coefficients + 1

    def replay_states(self, local_tokens: torch.Tensor) -> torch.Tensor:
        """Return the grammar state before each recorded token, including EOS."""
        states, state = [], 0
        for token in local_tokens.detach().cpu().tolist():
            states.append(state)
            state = int(self.cpu_transitions[state, token])
            if state < 0:
                raise ValueError("Recorded tokens violate the active FAST grammar")
        return torch.tensor(states, device=self.transitions.device, dtype=torch.long)


class FastGrammarProcessor(LogitsProcessor):
    """Incrementally apply the grammar while retaining the model's valid logits."""

    def __init__(self, grammar: FastGrammar, allowed: torch.Tensor, prefix_length: int) -> None:
        self.grammar = grammar
        self.allowed = allowed
        self.prefix_length = prefix_length
        self.states = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] == self.prefix_length:
            self.states = torch.zeros(input_ids.shape[0], dtype=torch.long, device=scores.device)
        else:
            last = input_ids[:, -1]
            # Finished HF batch rows contain padding. Keep those rows terminal.
            local = torch.where(
                (last == self.allowed[-1]) | (self.states == self.grammar.terminal),
                len(self.allowed) - 1,
                last - self.allowed[0],
            )
            self.states = self.grammar.transitions[self.states, local]
            if (self.states < 0).any():
                raise ValueError("Generation left the FAST grammar")
        restricted = scores[:, self.allowed].masked_fill(~self.grammar.valid[self.states], -torch.inf)
        masked = torch.full_like(scores, -torch.inf)
        masked[:, self.allowed] = restricted
        return masked
