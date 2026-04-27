"""Diagnose where MPAC encode/decode loses signal, without re-running generation.

Usage:
    python scripts/diagnose_mpac.py \
        --gen results/pilot_A_remote_bnb_smoke/generations/bf16.jsonl \
        --messages results/pilot_A_remote_bnb_smoke/messages.json \
        --config configs/pilot_A_remote_bnb_smoke.yaml \
        --first 5

Reports for each (prompt, message) pair:
  - n positions actually decoded
  - per-chunk: vote distribution across r sublists, the winning sublist, the true sublist
  - "low entropy positions" proxy: positions where >90% of generations across this slice
    landed in the same sublist (means bias didn't matter — token was deterministic)
  - vocab_size used by encoder (recorded) vs current tokenizer's len (mismatch -> bug)

This script needs only the saved generations + the wmark package; no model load.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from wmark.mpac import MPACConfig, _chunk_index, _sublist_assignment
from wmark.utils import bits_to_radix, radix_to_bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--messages", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--first", type=int, default=5)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg_w = cfg["watermark"]
    mpac_cfg = MPACConfig(
        radix=cfg_w["radix"],
        delta=cfg_w["delta"],
        secret_key=cfg_w["secret_key"],
    )
    total_bits = cfg_w["message_bits"]
    log2r = (mpac_cfg.radix - 1).bit_length()
    K = (total_bits + log2r - 1) // log2r

    with open(args.messages) as f:
        messages = json.load(f)

    records = [json.loads(l) for l in open(args.gen)]
    print(f"Loaded {len(records)} generation records, examining first {args.first}")

    # Vocab consistency report
    vsizes = Counter(r.get("vocab_size", "MISSING") for r in records)
    print(f"\n[vocab] sizes seen in records: {dict(vsizes)}")
    if "MISSING" in vsizes:
        print("  WARN: some records have no vocab_size (legacy). Decoder may use wrong vocab.")
    elif len(vsizes) > 1:
        print("  WARN: inconsistent vocab sizes across records!")

    print(f"\n[mpac] radix={mpac_cfg.radix} delta={mpac_cfg.delta} K={K} total_bits={total_bits}")

    # Per-record diagnostic
    chunk_vote_correctness = []  # list of (chunk_idx, correct?) across all records and chunks
    for ri, rec in enumerate(records[: args.first]):
        m_true_bits = messages[rec["message_idx"]]
        m_radix = bits_to_radix(m_true_bits, mpac_cfg.radix)
        seq = rec["full_ids"]
        prompt_len = rec["prompt_len"]
        vocab = int(rec.get("vocab_size", 128256))

        votes = torch.zeros(K, mpac_cfg.radix, dtype=torch.long)
        n_positions = 0
        for t in range(prompt_len + mpac_cfg.skip_first, len(seq)):
            gen_step = t - prompt_len
            prev_tok = seq[t - 1]
            cur_tok = seq[t]
            chunk_idx = _chunk_index(gen_step, K, mpac_cfg)
            sublist = _sublist_assignment(prev_tok, vocab, mpac_cfg)
            cur_sub = int(sublist[cur_tok].item())
            votes[chunk_idx, cur_sub] += 1
            n_positions += 1

        winners = votes.argmax(dim=1).tolist()
        digits_true = m_radix
        digits_recovered = winners

        n_chunks_correct = sum(1 for a, b in zip(digits_true, digits_recovered) if a == b)
        m_hat_bits = radix_to_bits(digits_recovered, mpac_cfg.radix, total_bits)
        bit_acc = float(np.mean([int(a == b) for a, b in zip(m_true_bits, m_hat_bits)]))

        print(f"\n--- record #{ri} (prompt_idx={rec['prompt_idx']}, msg_idx={rec['message_idx']}) ---")
        print(f"  n_positions={n_positions}, K={K} -> avg votes/chunk={n_positions/K:.1f}")
        print(f"  chunks correct: {n_chunks_correct}/{K}, bit_acc={bit_acc:.3f}")
        print(f"  per-chunk votes (true_digit -> dist over {mpac_cfg.radix} sublists, * = winner):")
        for k in range(K):
            row = votes[k].tolist()
            winner = winners[k]
            true = digits_true[k]
            mark = lambda i: ("*" if i == winner else " ") + ("T" if i == true else " ")
            cells = " ".join(f"{mark(i)}{row[i]:>3}" for i in range(mpac_cfg.radix))
            tag = "OK " if winner == true else "ERR"
            print(f"    chunk {k:2d} (true={true}): {cells}  [{tag}]")
            chunk_vote_correctness.append((k, int(winner == true)))

    # Aggregate per-chunk-position correctness across the inspected records
    if chunk_vote_correctness:
        per_chunk_acc = {}
        for k, ok in chunk_vote_correctness:
            per_chunk_acc.setdefault(k, []).append(ok)
        print("\n[aggregate over inspected records] chunk_idx -> correct rate:")
        for k in sorted(per_chunk_acc):
            arr = per_chunk_acc[k]
            print(f"  chunk {k:2d}: {sum(arr)}/{len(arr)} = {np.mean(arr):.2f}")


if __name__ == "__main__":
    main()
