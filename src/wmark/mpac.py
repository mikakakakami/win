"""MPAC: Multi-bit watermarking via Position Allocation + Colorlisting (Yoo et al. NAACL 2024).

Reference: arxiv 2308.00221, https://github.com/bangawayoo/mb-lm-watermarking

Simplified faithful implementation (Pilot-A scope):
- Every generated token is a "carrying position" (paper's P_position=1 default).
- Per position t with previous token p_t:
    chunk_idx = position allocation k(t) in [0, K)
    seed_part = hash(secret_key, p_t, "part") -> RNG to partition vocab into r colored sublists
  Apply +delta to the sublist whose index equals m_radix[k(t)].
- Decoder: same hashing per position, find which sublist the observed token fell in, vote per chunk.

Round-trip unit test (no compression) must achieve exact_match >= 0.95
on LLaMA-3-8B-Instruct with 20-bit msg, 200 tokens, default params -- if not, fix before Pilot-A.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch

try:
    from transformers import LogitsProcessor as _Base
except ImportError:
    _Base = object  # type: ignore[misc,assignment]

from .utils import bits_to_radix, derive_seed, radix_to_bits, torch_generator

_SUBLIST_CACHE_MAX = 512
_sublist_cache: OrderedDict[tuple[int, int, int], torch.Tensor] = OrderedDict()


def clear_sublist_cache():
    _sublist_cache.clear()


@dataclass
class MPACConfig:
    radix: int = 4              # r, must be power of 2
    delta: float = 2.0          # logit bias on the chosen colored sublist
    secret_key: int = 0xC0FFEE  # PRNG key (per-deployment, not per-message)
    skip_first: int = 1         # do not watermark the very first generation step (no prev token at start of prompt; transformers handles this but we also gate)


class MPACLogitsProcessor(_Base):
    """LogitsProcessor that injects MPAC bias for a given message m (binary list).

    The processor tracks the position-relative-to-prompt by remembering the input length.
    It expects to be the LAST processor in the chain (after temperature / top_p), so we apply
    bias on the raw logits before sampling. transformers convention: it's called BEFORE
    sampling, BEFORE temperature. delta=2.0 stays meaningful at temperature=1.0.
    """

    def __init__(self, message_bits: list[int], cfg: MPACConfig, prompt_len: int):
        self.cfg = cfg
        self.K_target_bits = len(message_bits)
        self.m_radix = bits_to_radix(message_bits, cfg.radix)
        self.K = len(self.m_radix)
        self.prompt_len = prompt_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: (batch, seq_so_far). scores: (batch, vocab) raw next-token logits.
        bsz, vocab = scores.shape
        # gen step (0-indexed within new tokens)
        gen_step = input_ids.shape[1] - self.prompt_len
        if gen_step < self.cfg.skip_first:
            return scores

        # we need a previous token to seed the hash. by convention prev = last token of input_ids.
        prev_tokens = input_ids[:, -1].tolist()
        out = scores.clone()
        for b, p_tok in enumerate(prev_tokens):
            chunk_idx = _chunk_index(gen_step, self.K, self.cfg)
            chunk_val = self.m_radix[chunk_idx]
            sublist = _sublist_assignment(p_tok, vocab, self.cfg)
            mask = sublist.to(scores.device) == chunk_val
            out[b, mask] += self.cfg.delta
        return out


def _sublist_assignment(prev_token_id: int, vocab: int, cfg: MPACConfig) -> torch.Tensor:
    """Map every token id -> sublist index in [0, r). Deterministic given (key, prev_token).

    Results are LRU-cached (up to _SUBLIST_CACHE_MAX entries) to avoid
    recomputing the expensive randperm(vocab) on repeated prev_token values.
    """
    key = (cfg.secret_key, cfg.radix, prev_token_id, vocab)
    if key in _sublist_cache:
        _sublist_cache.move_to_end(key)
        return _sublist_cache[key]

    seed = derive_seed(cfg.secret_key, prev_token_id, "part")
    g = torch_generator(seed, device="cpu")
    perm = torch.randperm(vocab, generator=g)
    assignments = torch.empty(vocab, dtype=torch.long)
    sub_size = vocab // cfg.radix
    for i in range(cfg.radix):
        if i < cfg.radix - 1:
            sl = perm[i * sub_size : (i + 1) * sub_size]
        else:
            sl = perm[i * sub_size :]
        assignments[sl] = i

    if len(_sublist_cache) >= _SUBLIST_CACHE_MAX:
        _sublist_cache.popitem(last=False)
    _sublist_cache[key] = assignments
    return assignments


def _chunk_index(gen_step: int, K: int, cfg: MPACConfig) -> int:
    """Keyed round-robin position allocation for message chunks."""
    offset = derive_seed(cfg.secret_key, K, "chunk_offset") % K
    return (gen_step - cfg.skip_first + offset) % K


def decode_message(
    generated_ids: torch.LongTensor,
    prompt_len: int,
    total_bits: int,
    vocab_size: int,
    cfg: MPACConfig,
) -> list[int]:
    """Recover binary message from a single generated sequence.

    generated_ids: (seq_len,) full tokens (prompt + new). We start voting from gen_step=skip_first.
    """
    log2r = (cfg.radix - 1).bit_length()
    K = (total_bits + log2r - 1) // log2r

    votes = torch.zeros(K, cfg.radix, dtype=torch.long)
    seq = generated_ids.tolist()
    for t in range(prompt_len + cfg.skip_first, len(seq)):
        gen_step = t - prompt_len
        prev_tok = seq[t - 1]
        cur_tok = seq[t]
        chunk_idx = _chunk_index(gen_step, K, cfg)
        sublist = _sublist_assignment(prev_tok, vocab_size, cfg)
        cur_sub = int(sublist[cur_tok].item())
        votes[chunk_idx, cur_sub] += 1

    digits = votes.argmax(dim=1).tolist()
    return radix_to_bits(digits, cfg.radix, total_bits)


def round_trip_no_model(
    message_bits: list[int],
    cfg: MPACConfig,
    vocab_size: int = 32000,
    prompt_len: int = 30,
    gen_len: int = 200,
    prev_token_seed: int = 1234,
) -> list[int]:
    """Sanity test: simulate that the model perfectly samples the bias-favored token.

    For each position, we look up the chunk_val and sublist assignment, and 'sample' by
    picking ANY token from the favored sublist. Then decode. With no noise, this should
    perfectly recover the message.
    """
    log2r = (cfg.radix - 1).bit_length()
    K = (len(message_bits) + log2r - 1) // log2r
    m_radix = bits_to_radix(message_bits, cfg.radix)

    g = torch.Generator().manual_seed(prev_token_seed)
    seq = torch.randint(0, vocab_size, (prompt_len,), generator=g).tolist()
    for gen_step in range(gen_len):
        prev_tok = seq[-1]
        if gen_step < cfg.skip_first:
            seq.append(int(torch.randint(0, vocab_size, (1,), generator=g).item()))
            continue
        chunk_idx = _chunk_index(gen_step, K, cfg)
        chunk_val = m_radix[chunk_idx]
        sublist = _sublist_assignment(prev_tok, vocab_size, cfg)
        cand = (sublist == chunk_val).nonzero(as_tuple=True)[0]
        # pick first candidate deterministically (or random — doesn't matter for round-trip)
        nxt = int(cand[0].item())
        seq.append(nxt)

    full = torch.tensor(seq, dtype=torch.long)
    return decode_message(full, prompt_len, len(message_bits), vocab_size, cfg)
