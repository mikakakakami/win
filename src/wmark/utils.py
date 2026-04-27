"""Hash + RNG helpers.

We need deterministic, key-keyed pseudo-randomness keyed by (secret_key, prev_token_id, tag).
Used by MPAC to derive (sublist partition, chunk index) per generation position.
"""
from __future__ import annotations

import hashlib
import struct

import torch


def derive_seed(secret_key: int, prev_token_id: int, tag: str = "") -> int:
    """Return a deterministic 64-bit seed for (secret_key, prev_token, tag)."""
    payload = struct.pack("<qq", int(secret_key), int(prev_token_id)) + tag.encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<Q", digest[:8])[0] & 0x7FFFFFFFFFFFFFFF


def torch_generator(seed: int, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def bits_to_radix(bits: list[int] | torch.Tensor, r: int) -> list[int]:
    """Convert a binary list to r-radix digit list. r must be a power of 2."""
    assert r > 1 and (r & (r - 1)) == 0, "r must be a power of 2"
    log2r = (r - 1).bit_length()
    if isinstance(bits, torch.Tensor):
        bits = bits.tolist()
    pad = (-len(bits)) % log2r
    bits = list(bits) + [0] * pad
    digits = []
    for i in range(0, len(bits), log2r):
        chunk = bits[i : i + log2r]
        v = 0
        for b in chunk:
            v = (v << 1) | int(b)
        digits.append(v)
    return digits


def radix_to_bits(digits: list[int], r: int, total_bits: int) -> list[int]:
    """Inverse of bits_to_radix. Trims to total_bits."""
    assert r > 1 and (r & (r - 1)) == 0
    log2r = (r - 1).bit_length()
    bits: list[int] = []
    for d in digits:
        chunk = []
        for _ in range(log2r):
            chunk.append(d & 1)
            d >>= 1
        chunk.reverse()
        bits.extend(chunk)
    return bits[:total_bits]


def random_message(b: int, seed: int) -> list[int]:
    rng = torch.Generator().manual_seed(int(seed))
    return torch.randint(0, 2, (b,), generator=rng).tolist()
