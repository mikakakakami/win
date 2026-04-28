"""Round-trip unit tests for MPAC. Run with: pytest tests/test_mpac.py -v"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wmark.metrics import bit_accuracy, exact_match
from wmark.mpac import (
    MPACConfig,
    _sublist_cache,
    clear_sublist_cache,
    decode_message,
    round_trip_no_model,
)
from wmark.utils import bits_to_radix, radix_to_bits, random_message


def test_radix_roundtrip():
    bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    digits = bits_to_radix(bits, r=4)
    back = radix_to_bits(digits, r=4, total_bits=len(bits))
    assert back == bits


def test_radix_pad():
    bits = [1, 0, 1]
    digits = bits_to_radix(bits, r=4)
    back = radix_to_bits(digits, r=4, total_bits=3)
    assert back == bits


def test_round_trip_no_model_perfect():
    """Without any noise, MPAC should recover message exactly."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    for seed in range(5):
        m = random_message(b=20, seed=seed)
        m_hat = round_trip_no_model(
            m, cfg, vocab_size=32000, prompt_len=30, gen_len=200, prev_token_seed=seed,
        )
        assert exact_match(m, m_hat) == 1, f"seed={seed} m={m} m_hat={m_hat}"


def test_round_trip_with_noisy_token_choice():
    """Even with random token within the favored sublist, decoding should still work."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    m = random_message(b=20, seed=99)
    m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=300)
    assert exact_match(m, m_hat) == 1


def test_decoder_handles_different_radix():
    for r in [2, 4, 8, 16]:
        cfg = MPACConfig(radix=r, delta=2.0, secret_key=7)
        m = random_message(b=20, seed=0)
        m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=400)
        assert bit_accuracy(m, m_hat) >= 0.9, f"r={r} bit_acc={bit_accuracy(m, m_hat)}"


def test_sublist_cache_hit():
    """Second call with same (key, prev_token, vocab) should return cached tensor."""
    clear_sublist_cache()
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=123)
    from wmark.mpac import _sublist_assignment

    a = _sublist_assignment(42, 1000, cfg)
    assert len(_sublist_cache) == 1
    b = _sublist_assignment(42, 1000, cfg)
    assert torch.equal(a, b)
    assert len(_sublist_cache) == 1
    clear_sublist_cache()


def test_sublist_cache_different_keys():
    """Different prev_tokens produce different cache entries."""
    clear_sublist_cache()
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=123)
    from wmark.mpac import _sublist_assignment

    _sublist_assignment(1, 1000, cfg)
    _sublist_assignment(2, 1000, cfg)
    assert len(_sublist_cache) == 2
    clear_sublist_cache()


def test_decode_short_sequence():
    """Decoding a very short sequence (fewer tokens than chunks) should not crash."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    m = random_message(b=20, seed=0)
    m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=5)
    assert len(m_hat) == len(m)


def test_round_trip_32bit_message():
    """32-bit message (main table length) should also round-trip correctly."""
    cfg = MPACConfig(radix=4, delta=2.0, secret_key=42)
    m = random_message(b=32, seed=0)
    m_hat = round_trip_no_model(m, cfg, vocab_size=32000, gen_len=400)
    assert exact_match(m, m_hat) == 1
