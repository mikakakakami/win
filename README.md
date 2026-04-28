# Multi-bit LLM Watermarking under Model Compression

现有多比特 LLM 水印 (MPAC, ECC-Multibit, SAEMark 等) 仅在文本层攻击 (改写 / 替换 / 裁剪) 下评估鲁棒性。
**模型压缩 (量化 / 剪枝 / 蒸馏) 对多比特消息恢复的破坏尚无系统研究。** 本项目填补这一空白。

**Target:** EMNLP 2026 &nbsp;|&nbsp; **Status:** Pilot-A (基线退化验证)

## Why This Matters

Zero-bit 水印在压缩下几乎不退化 (PRO: GPTQ-4bit AUC >= 0.987)，因为检测信号靠 N 个 token 聚合，单点扰动被吸收。
但 multi-bit 必须 **每个 chunk 都解码正确** —— `exact_match ~ (1-p)^K`，即使 per-chunk 错误率仅 1.5%，K=10 时 exact_match 就从 ~1.0 塌到 ~0.86。

> 检测信号还活着，消息内容塌了。

## Repository

```
win/
├── src/wmark/              核心包
│   ├── mpac.py             MPAC 编解码 (Yoo et al., NAACL 2024)
│   ├── compress.py         模型压缩 (GPTQ / AWQ / WANDA / INT8 / bnb-NF4)
│   ├── metrics.py          bit_accuracy, exact_match
│   ├── data.py             Prompt 加载 (C4, ...)
│   └── utils.py            Hash, RNG, radix 转换
├── scripts/
│   ├── pilot_A_baseline_drop.py   Pilot-A 完整流水线
│   └── diagnose_mpac.py           MPAC chunk vote 诊断
├── configs/
│   ├── pilot_A.yaml               正式 Pilot-A 配置 (500 prompts)
│   └── pilot_A_remote_bnb_smoke.yaml  远端 smoke 替代配置
├── tests/                  单元测试
├── docs/                   设计文档 (阅读顺序: 00 → 02 → 04)
│   ├── 00_problem_lock.md         问题锁定 + threat model + claim
│   ├── 01_method_spec.md          方法数学规格 (Pilot-A 后填)
│   ├── 02_pilot_A_protocol.md     Pilot-A 规程
│   ├── 03_pilot_B_protocol.md     Pilot-B 占位
│   ├── 04_full_experiment_plan.md 全表实验设计 (P0/P1/P2)
│   ├── 05_baselines_reimpl.md     baseline 重实现笔记
│   └── 06_decision_log.md         关键决策 + 时间戳
├── paper/                  LaTeX (待建)
└── results/                实验产出 (gitignored)
```

## Quick Start

```bash
# 安装 (core deps only, no CUDA extensions)
pip install -e .

# 安装量化依赖 (需要 CUDA)
pip install -e ".[quant]"

# 单元测试 (必须先过)
PYTHONPATH=src pytest tests/ -v
```

## Experiment Pipeline

### Pilot-A: 验证 baseline 在压缩下退化

```bash
# 一次跑完全部 stages
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage all

# 或逐步执行
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage prepare
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage compress
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage generate
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage decode
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage margin_flip
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage report
```

决策规则见 `docs/02_pilot_A_protocol.md`：
- **H1 PASS** (exact_match drop >= 0.15) → 进入方法设计
- **H1 GREY** (0.05 <= drop < 0.15) → 扩展到 1000 prompts
- **H1 FAIL** (drop < 0.05) → 触发 R1 应对

## Compression Methods

| Kind | 实现 | 状态 |
|---|---|---|
| `bf16` | baseline, no compression | OK |
| `int8` | bitsandbytes INT8 | OK |
| `bnb_nf4` | bitsandbytes NF4 | OK |
| `gptq4` | gptqmodel / auto-gptq | OK (需 CUDA) |
| `awq4` | autoawq | OK (需 CUDA) |
| `wanda50` | 从零实现 (Sun et al., ICLR 2024) | OK |
| `sparsegpt50` | IST-DASLab/sparsegpt | stub |
| `distill` | KD → 3B student | stub |

## Baselines (主表)

| # | Method | Source | 类型 |
|---|---|---|---|
| 1 | No WM | — | control |
| 2 | MPAC | Yoo et al., NAACL 2024 | logit-bias |
| 3 | ECC-Multibit | Qu et al., USENIX Sec 2025 | logit-bias + ECC |
| 4 | Adaptive (re-purposed) | Liu & Bu, 2024 | logit-bias |
| 5 | PRO (re-purposed) | arxiv 2510.23891 | trainable |
| 6 | **Ours** | — | logit-bias |

## Current Status

- [x] 文献调研 + gap 验证 (2026-04-27)
- [x] 问题锁定 + threat model + 4 个 claim
- [x] Pilot-A 规程 + 代码框架
- [x] MPAC 编解码 + 单元测试
- [x] 压缩模块 (GPTQ / AWQ / WANDA / INT8 / bnb-NF4)
- [ ] **Pilot-A 正式运行** ← 当前阶段
- [ ] 方法定型 (`docs/01_method_spec.md`)
- [ ] Pilot-B
- [ ] 全表实验 (~138 GPU-h on A800)
- [ ] Paper draft

## Key References

- **MPAC:** Yoo et al., "Advancing Beyond Identification: Multi-bit Watermark for LLMs", NAACL 2024. [2308.00221](https://arxiv.org/abs/2308.00221)
- **ECC-Multibit:** Qu et al., "Provably Robust Multi-bit Watermarking via ECC", USENIX Sec 2025. [2401.16820](https://arxiv.org/abs/2401.16820)
- **PRO:** "Enabling Precise and Robust Text Watermark for Open-Source LLMs", 2510.23891.
- **Pan 2025:** "Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?", ACL 2025. [2502.11598](https://arxiv.org/abs/2502.11598)
- **WaterPark:** Liang et al., EMNLP-Findings 2025. [2411.13425](https://arxiv.org/abs/2411.13425)
- **WANDA:** Sun et al., "A Simple and Effective Pruning Approach for LLMs", ICLR 2024.
- **GPTQ:** Frantar et al., ICLR 2023. &nbsp; **AWQ:** Lin et al., MLSys 2024. &nbsp; **SparseGPT:** Frantar & Alistarh, ICML 2023.

## Compute

- Pilot-A: ~5 GPU-h (1x A800)
- 全表 P0+P1: ~138 GPU-h
- 详见 `docs/04_full_experiment_plan.md`
