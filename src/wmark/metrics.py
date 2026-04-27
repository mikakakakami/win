"""Bit-level metrics."""
from __future__ import annotations

import numpy as np


def bit_accuracy(m_true: list[int], m_hat: list[int]) -> float:
    """Per-bit Hamming agreement, mean."""
    assert len(m_true) == len(m_hat), f"len mismatch {len(m_true)} vs {len(m_hat)}"
    eq = [int(a == b) for a, b in zip(m_true, m_hat)]
    return float(np.mean(eq))


def exact_match(m_true: list[int], m_hat: list[int]) -> int:
    return int(m_true == m_hat)


def aggregate(records: list[dict]) -> dict:
    """records: [{m_true, m_hat}, ...] -> {bit_acc_mean, bit_acc_sem, exact_match_rate}."""
    bit_accs = [bit_accuracy(r["m_true"], r["m_hat"]) for r in records]
    em = [exact_match(r["m_true"], r["m_hat"]) for r in records]
    n = len(records)
    return {
        "n": n,
        "bit_acc_mean": float(np.mean(bit_accs)),
        "bit_acc_sem": float(np.std(bit_accs) / max(np.sqrt(n), 1.0)),
        "exact_match_rate": float(np.mean(em)),
        "exact_match_sem": float(np.std(em) / max(np.sqrt(n), 1.0)),
    }
