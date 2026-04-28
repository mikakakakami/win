"""Model compression: quantization, pruning, distillation.

Supported methods (kind string):
  bf16        no compression (baseline)
  int8        bitsandbytes INT8
  bnb_nf4     bitsandbytes NF4 (4-bit NormalFloat)
  gptq4       GPTQ 4-bit (gptqmodel / auto_gptq)
  awq4        AWQ 4-bit (autoawq)
  wanda50     WANDA 50% unstructured pruning (Sun et al., ICLR 2024)
  sparsegpt50 SparseGPT 50% unstructured (stub)
  distill     knowledge distillation (stub)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

SUPPORTED_METHODS = [
    "bf16", "int8", "bnb_nf4", "gptq4", "awq4", "wanda50", "sparsegpt50", "distill",
]


# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

def get_calibration_data(
    tokenizer, n: int = 128, seq_len: int = 2048,
    dataset_name: str = "allenai/c4", split: str = "train",
) -> list[str]:
    local = os.environ.get("WMARK_CALIB_PROMPTS")
    if local:
        texts = _load_local_texts(local, n, seq_len)
        if texts:
            return texts
    from datasets import load_dataset
    ds = load_dataset(dataset_name, "en", split=split, streaming=True)
    samples: list[str] = []
    for row in ds:
        if len(samples) >= n:
            break
        ids = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=seq_len)
        if ids.input_ids.shape[1] >= seq_len:
            samples.append(row["text"])
    if len(samples) < n:
        raise RuntimeError(f"Only collected {len(samples)}/{n} calibration samples")
    return samples


def _load_local_texts(path: str, n: int, seq_len: int) -> list[str]:
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


# ---------------------------------------------------------------------------
# GPTQ-4bit
# ---------------------------------------------------------------------------

def compress_gptq4(
    model_id: str, save_dir: str,
    n_calib: int = 128, seq_len: int = 2048,
    group_size: int = 128, desc_act: bool = True,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "config.json").exists():
        print(f"[gptq4] {save_dir} already exists, skipping")
        return str(save_dir)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    samples = get_calibration_data(tokenizer, n=n_calib, seq_len=seq_len)
    examples = [
        tokenizer(s, return_tensors="pt", truncation=True, max_length=seq_len)
        for s in samples
    ]

    # Try gptqmodel (actively maintained, prebuilt CUDA wheels)
    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        qcfg = QuantizeConfig(bits=4, group_size=group_size, desc_act=desc_act)
        model = GPTQModel.load(model_id, qcfg)
        model.quantize(examples)
        model.save(save_dir)
        tokenizer.save_pretrained(save_dir)
        return str(save_dir)
    except ImportError:
        pass

    # Fallback: auto_gptq
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        qcfg = BaseQuantizeConfig(bits=4, group_size=group_size, desc_act=desc_act, sym=True)
        model = AutoGPTQForCausalLM.from_pretrained(
            model_id, quantize_config=qcfg, torch_dtype=torch.bfloat16,
        )
        model.quantize(examples, batch_size=1)
        model.save_quantized(save_dir)
        tokenizer.save_pretrained(save_dir)
        return str(save_dir)
    except ImportError:
        pass

    raise ImportError(
        "GPTQ requires 'gptqmodel' (recommended) or 'auto-gptq'. "
        "Install: pip install gptqmodel"
    )


# ---------------------------------------------------------------------------
# AWQ-4bit
# ---------------------------------------------------------------------------

def compress_awq4(
    model_id: str, save_dir: str,
    n_calib: int = 128, seq_len: int = 2048, group_size: int = 128,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "config.json").exists():
        print(f"[awq4] {save_dir} already exists, skipping")
        return str(save_dir)

    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoAWQForCausalLM.from_pretrained(model_id)
    model.quantize(tokenizer, quant_config={
        "w_bit": 4, "q_group_size": group_size,
        "zero_point": True, "version": "GEMM",
    })
    model.save_quantized(save_dir)
    tokenizer.save_pretrained(save_dir)
    return str(save_dir)


# ---------------------------------------------------------------------------
# WANDA: Pruning by Weights AND Activations (Sun et al., ICLR 2024)
# ---------------------------------------------------------------------------

def compress_wanda(
    model_id: str, save_dir: str, sparsity: float = 0.5,
    n_calib: int = 128, seq_len: int = 2048,
) -> str:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "config.json").exists():
        print(f"[wanda] {save_dir} already exists, skipping")
        return str(save_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    _wanda_prune_inplace(model, tokenizer, sparsity, n_calib, seq_len)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    return str(save_dir)


def _wanda_prune_inplace(model, tokenizer, sparsity: float, n_calib: int, seq_len: int):
    """importance = |W| * ||X||_2, per-row unstructured pruning."""
    device = next(model.parameters()).device
    calib_texts = get_calibration_data(tokenizer, n=n_calib, seq_len=seq_len)

    hooks = []
    input_norms: dict[str, torch.Tensor] = {}

    def _make_hook(name):
        def _hook(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            sq = x.float().pow(2).sum(dim=0)
            if name in input_norms:
                input_norms[name] += sq.to(input_norms[name].device)
            else:
                input_norms[name] = sq
        return _hook

    skip = {"lm_head", "embed_tokens"}
    linears: dict[str, torch.nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not any(s in name for s in skip):
            hooks.append(mod.register_forward_hook(_make_hook(name)))
            linears[name] = mod

    model.eval()
    with torch.no_grad():
        for text in tqdm(calib_texts, desc="wanda calibration"):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            ids = {k: v.to(device) for k, v in ids.items()}
            model(**ids)
    for h in hooks:
        h.remove()

    total_pruned = total_params = 0
    for name, module in linears.items():
        if name not in input_norms:
            continue
        W = module.weight.data
        X_norm = input_norms[name].sqrt().to(W.device)
        importance = W.float().abs() * X_norm.unsqueeze(0)
        n_prune = int(W.shape[1] * sparsity)
        if n_prune == 0:
            continue
        threshold = torch.kthvalue(importance, n_prune, dim=1, keepdim=True).values
        mask = (importance > threshold).to(W.dtype)
        module.weight.data.mul_(mask)
        total_pruned += (mask == 0).sum().item()
        total_params += mask.numel()

    if total_params > 0:
        print(f"[wanda] pruned {total_pruned}/{total_params} = {total_pruned / total_params:.1%}")
    del input_norms
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# SparseGPT (stub)
# ---------------------------------------------------------------------------

def compress_sparsegpt(model_id: str, save_dir: str, sparsity: float = 0.5, **kw) -> str:
    raise NotImplementedError(
        "SparseGPT requires the official repo: https://github.com/IST-DASLab/sparsegpt"
    )


# ---------------------------------------------------------------------------
# Knowledge distillation (stub)
# ---------------------------------------------------------------------------

def compress_distill(teacher_id: str, student_id: str, save_dir: str, **kw) -> str:
    raise NotImplementedError("Distillation pipeline not yet implemented.")


# ---------------------------------------------------------------------------
# Generic loader
# ---------------------------------------------------------------------------

def load_model(path_or_id: str, method: str = "bf16"):
    from transformers import AutoModelForCausalLM

    if method in ("bf16", "wanda50", "sparsegpt50"):
        return AutoModelForCausalLM.from_pretrained(
            path_or_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
    if method == "int8":
        return AutoModelForCausalLM.from_pretrained(
            path_or_id, load_in_8bit=True, device_map="auto",
        )
    if method == "bnb_nf4":
        from transformers import BitsAndBytesConfig
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            path_or_id, quantization_config=qcfg, device_map="auto",
        )
    if method in ("gptq4", "awq4"):
        return AutoModelForCausalLM.from_pretrained(path_or_id, device_map="auto")
    raise ValueError(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")


# Backward compat alias
load_compressed = load_model
