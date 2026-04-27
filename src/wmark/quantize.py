"""Compression wrappers: GPTQ-4bit, WANDA-50%.

For Pilot-A we only need GPTQ-4bit and WANDA-50%. INT8 / AWQ / SparseGPT / distillation
are added later for the main table.

Both routines persist quantized models to disk so we can reload without re-running the
compression step. Calibration: 128 samples from C4 train, 2048-token sequences.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import json

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_calibration_samples(tokenizer, n: int = 128, seq_len: int = 2048, seed: int = 0) -> list[str]:
    local_prompt_file = os.environ.get("WMARK_CALIB_PROMPTS")
    if local_prompt_file:
        samples = _calibration_from_prompt_file(local_prompt_file, n=n, seq_len=seq_len)
        if samples:
            return samples

    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples: list[str] = []
    for row in ds:
        if len(samples) >= n:
            break
        ids = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=seq_len)
        if ids.input_ids.shape[1] >= seq_len:
            samples.append(row["text"])
    return samples


def _calibration_from_prompt_file(path: str, n: int, seq_len: int) -> list[str]:
    texts: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text") or obj.get("prompt")
            if text:
                texts.append(str(text))
    if not texts:
        return []

    base = "\n\n".join(texts)
    repeated = (base + "\n\n") * max(4, seq_len // max(len(base.split()), 1) + 2)
    return [repeated for _ in range(n)]


def quantize_gptq_4bit(model_id: str, save_dir: str, n_calib: int = 128, seq_len: int = 2048):
    """GPTQ-4bit via auto-gptq. Persists to save_dir."""
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "config.json").exists():
        print(f"[gptq] {save_dir} exists, skipping quantization")
        return str(save_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    qcfg = BaseQuantizeConfig(bits=4, group_size=128, desc_act=True, sym=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_id, quantize_config=qcfg, torch_dtype=torch.bfloat16)
    samples = get_calibration_samples(tokenizer, n=n_calib, seq_len=seq_len)
    examples = [tokenizer(s, return_tensors="pt", truncation=True, max_length=seq_len) for s in samples]
    model.quantize(examples, batch_size=1)
    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)
    return str(save_dir)


def prune_wanda_50(model_id: str, save_dir: str, n_calib: int = 128, seq_len: int = 2048):
    """WANDA 50% unstructured pruning.

    NOTE: wanda is officially at https://github.com/locuslab/wanda. We need to either
    vendor that repo or pip install. As of now there's no pip package; we shell out
    to a small subprocess if vendored. For the pilot-A code skeleton, we provide the
    interface; actual integration is a TODO once the repo is cloned.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "config.json").exists():
        print(f"[wanda] {save_dir} exists, skipping pruning")
        return str(save_dir)

    raise NotImplementedError(
        "WANDA pruning requires vendoring locuslab/wanda. "
        "Clone to external/wanda and integrate, then remove this stub."
    )


def load_compressed(model_path: str, kind: str):
    """Load a previously compressed model from disk.

    kind in {'bf16', 'gptq4', 'bnb4', 'wanda50'}
    """
    if kind == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    if kind == "gptq4":
        from auto_gptq import AutoGPTQForCausalLM
        return AutoGPTQForCausalLM.from_quantized(model_path, device="cuda:0", use_safetensors=True)
    if kind == "bnb4":
        from transformers import BitsAndBytesConfig

        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=qcfg, device_map="auto"
        )
    if kind == "wanda50":
        # WANDA persists a regular HF checkpoint with zeroed weights
        return AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    raise ValueError(f"unknown kind {kind}")
