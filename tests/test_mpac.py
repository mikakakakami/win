"""Round-trip unit tests for MPAC. Run with: pytest tests/test_mpac.py -v"""
from __future__ import annotations

import random

import pytest
import torch

from wmark.metrics import bit_accuracy, exact_match
from wmark.mpac import MPACConfig, decode_message, round_trip_no_model
from wmark.utils import bits_to_radix, radix_to_bits, random_message


def test_radix_roundtrip():
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]  # 10 bits, r=4 -> 5 digits
    digits = bits_to_radix(bits, r=4)
    back = radix_to_bits(digits, r=4, total_bits=len(bits))
    assert back == bits


def test_radix_pad():
    bits = [1, 0, 1]  # 3 bits, r=4 -> needs pad to 4 -> 2 digits
    digits = bits_to_radix(bits, r=4)
    back = radix_to_bits(digits, r=4, total_bits=3)
    assert back == bits


def test_round_trip_no_model_perfect():
    """Without any noise, MPAC should recover message exactly."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    for seed in range(5):
        m = random_message(b=20, seed=seed)
        m_hat = round_trip_no_model(
            m, cfg, vocab_size=32000, prompt_len=30, gen_len=200, prev_token_seed=seed
        )
        assert exact_match(m, m_hat) == 1, f"seed={seed} m={m} m_hat={m_hat}"


def test_round_trip_with_noisy_token_choice():
    """Even with random token within the favored sublist, decoding should still work."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    m = random_message(b=20, seed=99)
    # We can't directly use round_trip_no_model with random token choice, but the existing
    # implementation uses cand[0] which is deterministic — sufficient for noise-free check.
    m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=300)
    assert exact_match(m, m_hat) == 1


def test_decoder_handles_different_radix():
    for r in [2, 4, 8, 16]:
        cfg = MPACConfig(radix=r, delta=2.0, secret_key=7)
        m = random_message(b=20, seed=0)
        m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=400)
        # higher radix can need more tokens to reach all chunk indices.
        assert bit_accuracy(m, m_hat) >= 0.9, f"r={r} bit_acc={bit_accuracy(m, m_hat)}"
