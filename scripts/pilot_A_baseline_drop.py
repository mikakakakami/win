"""Pilot-A: confirm MPAC degrades under model compression.

Usage:
    python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage all
    python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage generate
    python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage decode
    python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage margin_flip
    python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage report

Stages are checkpointed under <output.root>/. Re-running a stage overwrites its outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# allow running from repo root with src/ layout
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root / "src"))

from wmark.compress import compress_awq4, compress_gptq4, compress_wanda, load_model
from wmark.data import load_c4_prompts
from wmark.metrics import aggregate
from wmark.mpac import MPACConfig, MPACLogitsProcessor, decode_message
from wmark.utils import random_message


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_prepare(cfg: dict, out_root: Path) -> dict:
    """Materialize prompts + messages once, persist to disk."""
    out_root.mkdir(parents=True, exist_ok=True)
    prompts_path = out_root / "prompts.jsonl"
    messages_path = out_root / "messages.json"

    if not prompts_path.exists():
        prompts = load_c4_prompts(
            n=cfg["prompts"]["n_prompts"],
            prompt_tokens=cfg["generation"]["prompt_words"],
        )
        with open(prompts_path, "w", encoding="utf-8") as f:
            for i, p in enumerate(prompts):
                f.write(json.dumps({"prompt_idx": i, "text": p}) + "\n")
    if not messages_path.exists():
        nm = cfg["prompts"]["num_messages"]
        b = cfg["watermark"]["message_bits"]
        base = cfg["prompts"]["message_seed_base"]
        msgs = [random_message(b, seed=base + i) for i in range(nm)]
        with open(messages_path, "w") as f:
            json.dump(msgs, f)
    print(f"[prepare] prompts -> {prompts_path}, messages -> {messages_path}")

    with open(prompts_path) as f:
        prompts = [json.loads(line)["text"] for line in f]
    with open(messages_path) as f:
        messages = json.load(f)
    return {"prompts": prompts, "messages": messages}


def stage_compress(cfg: dict):
    compress_fn = {
        "gptq4": compress_gptq4,
        "awq4": compress_awq4,
        "wanda50": lambda mid, sd, **kw: compress_wanda(mid, sd, sparsity=0.5, **kw),
    }
    for p in cfg["pipelines"]:
        if not p["enabled"]:
            continue
        fn = compress_fn.get(p["kind"])
        if fn is None:
            continue
        fn(
            cfg["model"]["hf_id"],
            p["save_dir"],
            n_calib=p.get("calib_n", 128),
            seq_len=p.get("calib_seq_len", 2048),
        )


def _build_mpac_cfg(cfg: dict) -> MPACConfig:
    return MPACConfig(
        radix=cfg["watermark"]["radix"],
        delta=cfg["watermark"]["delta"],
        secret_key=cfg["watermark"]["secret_key"],
    )


def stage_generate(cfg: dict, out_root: Path, prompts: list[str], messages: list[list[int]]):
    """For each enabled pipeline, generate (prompt_idx, message_idx) pairs."""
    from transformers import AutoTokenizer

    mpac_cfg = _build_mpac_cfg(cfg)
    base_seed = cfg["generation"]["seed"]
    n_prompts = len(prompts)
    nm = len(messages)
    # round-robin assignment: prompt_idx -> message_idx = prompt_idx % nm
    pairs = [(pi, pi % nm) for pi in range(n_prompts)]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["hf_id"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for pipe in cfg["pipelines"]:
        if not pipe["enabled"]:
            continue
        out_path = out_root / "generations" / f"{pipe['name']}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"[generate] {out_path} exists, skipping")
            continue

        model_path = pipe.get("save_dir") or cfg["model"]["hf_id"]
        print(f"[generate] loading model for pipeline {pipe['name']} from {model_path}")
        model = load_model(model_path, method=pipe["kind"])
        model.eval()
        device = next(model.parameters()).device
        # Use the model's actual output dim, NOT len(tokenizer): for LLaMA-3 these can differ.
        # Encoder applies bias on the model's full output dim; decoder must use the same.
        vocab_used = int(model.config.vocab_size)
        len_tok = int(len(tokenizer))
        if vocab_used != len_tok:
            print(f"[generate] WARN model.config.vocab_size={vocab_used} != len(tokenizer)={len_tok}; using model.config.vocab_size for both encode and decode")

        with open(out_path, "w", encoding="utf-8") as f:
            for prompt_idx, msg_idx in tqdm(pairs, desc=f"gen[{pipe['name']}]"):
                prompt_text = prompts[prompt_idx]
                msg = messages[msg_idx]
                ids = tokenizer(prompt_text, return_tensors="pt").to(device)
                prompt_len = ids.input_ids.shape[1]
                processor = MPACLogitsProcessor(msg, mpac_cfg, prompt_len=prompt_len)

                gen = torch.Generator(device=device).manual_seed(base_seed + prompt_idx)
                with torch.no_grad():
                    out = model.generate(
                        **ids,
                        max_new_tokens=cfg["generation"]["max_new_tokens"],
                        min_new_tokens=cfg["generation"]["min_new_tokens"],
                        do_sample=cfg["generation"]["do_sample"],
                        top_p=cfg["generation"]["top_p"],
                        temperature=cfg["generation"]["temperature"],
                        logits_processor=[processor],
                        pad_token_id=tokenizer.pad_token_id,
                    )
                full_ids = out[0].tolist()
                rec = {
                    "prompt_idx": prompt_idx,
                    "message_idx": msg_idx,
                    "prompt_len": prompt_len,
                    "vocab_size": vocab_used,
                    "full_ids": full_ids,
                    "text": tokenizer.decode(full_ids, skip_special_tokens=True),
                }
                f.write(json.dumps(rec) + "\n")
        del model
        torch.cuda.empty_cache()


def stage_decode(cfg: dict, out_root: Path, messages: list[list[int]]):
    mpac_cfg = _build_mpac_cfg(cfg)
    b = cfg["watermark"]["message_bits"]

    decoded_root = out_root / "decoded"
    decoded_root.mkdir(parents=True, exist_ok=True)

    for pipe in cfg["pipelines"]:
        if not pipe["enabled"]:
            continue
        gen_path = out_root / "generations" / f"{pipe['name']}.jsonl"
        out_path = decoded_root / f"{pipe['name']}.jsonl"
        if not gen_path.exists():
            print(f"[decode] missing {gen_path}, skipping")
            continue
        records: list[dict] = []
        with open(gen_path) as f:
            for line in f:
                rec = json.loads(line)
                ids = torch.tensor(rec["full_ids"], dtype=torch.long)
                # Use the vocab_size that the encoder actually used during generation.
                # Fall back to model.config.vocab_size lookup if missing (legacy records).
                if "vocab_size" in rec:
                    vocab = int(rec["vocab_size"])
                else:
                    from transformers import AutoConfig
                    vocab = int(AutoConfig.from_pretrained(cfg["model"]["hf_id"]).vocab_size)
                m_hat = decode_message(ids, rec["prompt_len"], b, vocab, mpac_cfg)
                m_true = messages[rec["message_idx"]]
                records.append({
                    "prompt_idx": rec["prompt_idx"],
                    "message_idx": rec["message_idx"],
                    "m_true": m_true,
                    "m_hat": m_hat,
                })
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        agg = aggregate(records)
        print(f"[decode] {pipe['name']}: {agg}")


def stage_margin_flip(cfg: dict, out_root: Path):
    """Diagnostic for H2: compute Spearman ρ(margin_FP16, top1-flip) on GPTQ-4bit."""
    if not cfg["diagnostics"]["margin_flip"]["enabled"]:
        return
    bf_pipe = next((p for p in cfg["pipelines"] if p["kind"] == "bf16" and p["enabled"]), None)
    cmp_pipe = next((p for p in cfg["pipelines"] if p["kind"] != "bf16" and p["enabled"]), None)
    if bf_pipe is None or cmp_pipe is None:
        print("[margin_flip] need bf16 + at least one compressed pipeline enabled, skipping")
        return

    from scipy.stats import spearmanr
    from transformers import AutoTokenizer

    bf_gens = [json.loads(l) for l in open(out_root / "generations" / f"{bf_pipe['name']}.jsonl")]
    bf_gens = bf_gens[: cfg["diagnostics"]["margin_flip"]["n_prompts"]]

    print(f"[margin_flip] loading bf16 model for logits...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["hf_id"], use_fast=True)
    bf_model = load_model(cfg["model"]["hf_id"], method="bf16")
    bf_model.eval()
    device = next(bf_model.parameters()).device

    margins: list[float] = []
    flips_per_prompt: list[list[int]] = []
    bf_top1_per_prompt: list[list[int]] = []

    with torch.no_grad():
        for rec in tqdm(bf_gens, desc="bf16 logits"):
            ids = torch.tensor([rec["full_ids"]], device=device)
            out = bf_model(ids)
            logits = out.logits[0]  # (seq, vocab)
            top2 = torch.topk(logits, k=2, dim=-1)
            mar = (top2.values[:, 0] - top2.values[:, 1]).cpu().tolist()
            top1 = top2.indices[:, 0].cpu().tolist()
            # only watermarked positions matter (skip prompt)
            P = rec["prompt_len"]
            margins.extend(mar[P - 1 : -1])  # logit at position t predicts token t+1
            bf_top1_per_prompt.append(top1[P - 1 : -1])
    del bf_model
    torch.cuda.empty_cache()

    print(f"[margin_flip] loading {cmp_pipe['name']} model for logits...")
    cmp_model = load_model(
        cmp_pipe.get("save_dir") or cfg["model"]["hf_id"], method=cmp_pipe["kind"],
    )
    cmp_model.eval()
    flips: list[int] = []
    with torch.no_grad():
        for rec, bf_top1 in zip(bf_gens, bf_top1_per_prompt):
            ids = torch.tensor([rec["full_ids"]], device=next(cmp_model.parameters()).device)
            out = cmp_model(ids)
            logits = out.logits[0]
            cmp_top1 = logits.argmax(dim=-1).cpu().tolist()
            P = rec["prompt_len"]
            cmp_slice = cmp_top1[P - 1 : -1]
            for a, b in zip(bf_top1, cmp_slice):
                flips.append(int(a != b))
    del cmp_model
    torch.cuda.empty_cache()

    assert len(margins) == len(flips), f"len mismatch {len(margins)} vs {len(flips)}"
    rho, p = spearmanr(margins, flips)
    res = {
        "n": len(margins),
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "flip_rate": float(np.mean(flips)),
        "margin_mean": float(np.mean(margins)),
    }
    out_path = out_root / "stability" / "correlation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[margin_flip] {res}")


def stage_report(cfg: dict, out_root: Path):
    decoded_root = out_root / "decoded"
    summary = {}
    for pipe in cfg["pipelines"]:
        if not pipe["enabled"]:
            continue
        path = decoded_root / f"{pipe['name']}.jsonl"
        if not path.exists():
            continue
        records = [json.loads(l) for l in open(path)]
        summary[pipe["name"]] = aggregate(records)

    # decision
    em = {k: v["exact_match_rate"] for k, v in summary.items()}
    fp16 = em.get("bf16")
    decisions: list[str] = []
    if fp16 is None:
        decisions.append("WARN: bf16 baseline missing")
    else:
        if fp16 < cfg["decision"]["fp16_min_exact_match"]:
            decisions.append(
                f"FAIL_IMPL: bf16 exact_match={fp16:.3f} < {cfg['decision']['fp16_min_exact_match']}"
                " -- MPAC reimpl is broken, fix before continuing"
            )
        max_drop = 0.0
        for k, v in em.items():
            if k == "bf16":
                continue
            drop = fp16 - v
            max_drop = max(max_drop, drop)
            decisions.append(f"{k}: drop={drop:.3f}")
        if max_drop >= cfg["decision"]["drop_threshold"]:
            decisions.append(f"H1: PASS (max drop={max_drop:.3f} >= {cfg['decision']['drop_threshold']})")
        elif max_drop < 0.05:
            decisions.append(f"H1: FAIL_NO_DROP (max drop={max_drop:.3f}) -- trigger R1 mitigation")
        else:
            decisions.append(f"H1: GREY (max drop={max_drop:.3f}) -- extend pilot to 1000 prompts")

    corr_path = out_root / "stability" / "correlation.json"
    if corr_path.exists():
        corr = json.load(open(corr_path))
        rho = corr["spearman_rho"]
        decisions.append(f"H2: rho={rho:.3f}, threshold={cfg['decision']['spearman_threshold']}")
        if rho <= cfg["decision"]["spearman_threshold"]:
            decisions.append("H2: PASS")
        else:
            decisions.append("H2: FAIL -- trigger R3 (empirical flip-rate gate instead of margin)")

    rep = {"summary": summary, "decisions": decisions}
    out_path = out_root / "report.json"
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "prepare", "compress", "generate", "decode", "margin_flip", "report"],
    )
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    out_root = Path(cfg["output"]["root"])

    stages = ["prepare", "compress", "generate", "decode", "margin_flip", "report"] if args.stage == "all" else [args.stage]
    state = {}
    for s in stages:
        if s == "prepare":
            state.update(stage_prepare(cfg, out_root))
        elif s == "compress":
            stage_compress(cfg)
        elif s == "generate":
            if "prompts" not in state:
                state.update(stage_prepare(cfg, out_root))
            stage_generate(cfg, out_root, state["prompts"], state["messages"])
        elif s == "decode":
            if "messages" not in state:
                state.update(stage_prepare(cfg, out_root))
            stage_decode(cfg, out_root, state["messages"])
        elif s == "margin_flip":
            stage_margin_flip(cfg, out_root)
        elif s == "report":
            stage_report(cfg, out_root)


if __name__ == "__main__":
    main()
