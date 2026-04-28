"""Tests for WANDA pruning logic. CPU-only, no external deps."""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _TinyModel(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.linear1 = nn.Linear(d, d, bias=False)
        self.linear2 = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.linear1.weight)
        nn.init.normal_(self.linear2.weight)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


def _prune_wanda_manual(model: nn.Module, sparsity: float, n_samples: int = 64):
    """Standalone WANDA pruning (mirrors compress._wanda_prune_inplace logic)."""
    d = next(model.parameters()).shape[0]

    # Simulate calibration: collect input norms from random data
    input_norms: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name):
        def _hook(module, inp, out):
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            sq = x.float().pow(2).sum(dim=0)
            if name in input_norms:
                input_norms[name] += sq
            else:
                input_norms[name] = sq
        return _hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(_make_hook(name)))

    model.eval()
    with torch.no_grad():
        for _ in range(4):
            model(torch.randn(n_samples, d))
    for h in hooks:
        h.remove()

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or name not in input_norms:
            continue
        W = mod.weight.data
        X_norm = input_norms[name].sqrt()
        importance = W.float().abs() * X_norm.unsqueeze(0)
        n_prune = int(W.shape[1] * sparsity)
        if n_prune == 0:
            continue
        threshold = torch.kthvalue(importance, n_prune, dim=1, keepdim=True).values
        mask = (importance > threshold).to(W.dtype)
        mod.weight.data.mul_(mask)


def test_wanda_prune_sparsity():
    """After WANDA 50%, roughly half the weights should be zero."""
    model = _TinyModel(d=32)
    _prune_wanda_manual(model, sparsity=0.5)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            zeros = (mod.weight.data == 0).sum().item()
            total = mod.weight.data.numel()
            ratio = zeros / total
            assert 0.4 <= ratio <= 0.6, f"{name}: sparsity {ratio:.2f} not near 0.5"


def test_wanda_zero_sparsity_no_change():
    """Sparsity=0 should leave weights unchanged."""
    model = _TinyModel(d=16)
    orig = {n: m.weight.data.clone() for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    _prune_wanda_manual(model, sparsity=0.0)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            assert torch.equal(mod.weight.data, orig[name])


def test_wanda_high_sparsity():
    """90% sparsity should zero out most weights."""
    model = _TinyModel(d=32)
    _prune_wanda_manual(model, sparsity=0.9)

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            zeros = (mod.weight.data == 0).sum().item()
            total = mod.weight.data.numel()
            ratio = zeros / total
            assert ratio >= 0.85, f"{name}: expected ~90% sparsity, got {ratio:.2f}"


def test_wanda_preserves_important_weights():
    """Weights with high importance (large |W| * ||X||) should survive pruning."""
    model = _TinyModel(d=16)
    # Make one weight very large — it should survive
    model.linear1.weight.data[0, 0] = 100.0
    _prune_wanda_manual(model, sparsity=0.5)
    assert model.linear1.weight.data[0, 0] != 0, "Large weight should survive pruning"
