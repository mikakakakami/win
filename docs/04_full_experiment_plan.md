# 04. 全表实验计划 (Full Experiment Plan)

**Status:** plan only;未开始执行,等 Pilot-A 通过后启动
**Owner:** TBD
**Total estimated cost:** ~138 GPU-h (P0 + P1) / ~152 GPU-h (含 P2)

> 三层优先级:**P0 必须有(审稿人会先看)→ P1 强烈建议(证明 method 部件)→ P2 锦上添花(可砍)**。

---

## P0 主表 1: Multi-bit Attribution Robustness under Compression

**核心结果表**。Cell 报 `bit_accuracy / exact_match` (32-bit message, 200 token, mean over 500 prompts × 20 messages)。

| Compression Pipeline | No WM | MPAC (Yoo 24) | ECC-Multibit (Qu 24) | Adaptive (Liu&Bu 24) | PRO (re-purpose) | **Ours** |
|---|---|---|---|---|---|---|
| No compression (BF16) | — | — | — | — | — | — |
| INT8 (bitsandbytes) | — | — | — | — | — | — |
| GPTQ-4bit | — | — | — | — | — | — |
| AWQ-4bit | — | — | — | — | — | — |
| WANDA-50% | — | — | — | — | — | — |
| SparseGPT-50% | — | — | — | — | — | — |
| Distill → 3B student | — | — | — | — | — | — |

**模型**: LLaMA-3-8B-Instruct + Qwen-2.5-7B-Instruct(两个家族证明 generalization)

**算力**: 7 × 6 × 2 = 84 cells,~30min/cell on A800 ≈ **42 GPU-h** + watermark training (PRO + Ours, 4 模型方法对) ≈ **48 GPU-h** = **~90 GPU-h 总计**

**Baselines 重实现细节**: 见 `05_baselines_reimpl.md`。**特别注意 PRO 的 re-purpose 必须在 paper appendix 写清楚**(见 §A 风险)。

---

## P0 主表 2: Generation Quality

证明 watermark 不显著降质量。

| Method | PPL (BF16) | PPL (GPTQ-4bit) | MMLU 5-shot | GSM8K 8-shot |
|---|---|---|---|---|
| No WM | — | — | — | — |
| MPAC | — | — | — | — |
| ECC-Multibit | — | — | — | — |
| Adaptive WM | — | — | — | — |
| PRO | — | — | — | — |
| **Ours** | — | — | — | — |

**算力**: 6 方法 × 2 模型,每组 ~2h ≈ **24 GPU-h**

**评测协议(每个 metric 不同,这点关键)**:
- **PPL**: 在 watermark 生成的文本上,用一个**未水印的参考 LM**(同模型 BF16)算 PPL。这是 watermark 文本流畅度的代理(Yoo 2024 / Qu 2024 协议)。
- **MMLU 5-shot**: logprob-based 评测(逐选项算 logP,argmax),**watermark logits processor 不介入**(无 generation)。这一栏只衡量"压缩对 base capability 的影响"以及"训练式方法(PRO)的训练过程是否破坏了 base capability"。
- **GSM8K 8-shot**: chain-of-thought 生成 + 抽数字答案,**watermark logits processor 在 CoT 生成时介入**。这一栏衡量 watermark + 压缩双重影响下的推理保持。

报告时把这三个分别看,不要混着读。

---

## P0 消融 1: Component Ablation

证明方法每个部件都必要。**每加一项 exact_match 应单调上升**;若不单调,说明 component 设计有问题。

| Variant | GPTQ-4bit | AWQ-4bit | WANDA-50% | Distill-3B |
|---|---|---|---|---|
| Baseline (MPAC + LDPC) | — | — | — | — |
| + entropy-aware token gate (Yoo/Qu 默认) | — | — | — | — |
| + **margin-stability gate** (Ours, component a) | — | — | — | — |
| + **compression-aware bit allocation** (Ours, component b) = **Ours full** | — | — | — | — |

**注**: 原计划三 component 简化为两个(见 `00_problem_lock.md` §4)。adaptive ECC rate 内化在 component (b) 里,不单列。

**算力**: 4 variants × 4 compression × 1 model (LLaMA-3-8B) ≈ **15 GPU-h**

---

## P1 消融 2: Token Selection Criterion(强烈建议)

**直接验证 §C2 (机理 claim)**:entropy 不是 compression robustness 的对的量,margin stability 才是。

| Token Selection Criterion | GPTQ-4bit | AWQ-4bit | WANDA-50% | Avg |
|---|---|---|---|---|
| Random | — | — | — | — |
| High entropy (Liu & Bu 24) | — | — | — | — |
| High top-1 logit margin (FP16-only) | — | — | — | — |
| **High margin stability under noise** (Ours) | — | — | — | — |

**关键**: 第三行 vs 第四行的差异(margin 大不等于 margin 在压缩下还稳)证明"stability"才是关键,而不是简单 margin。这是 paper 一个独立 finding,可以独立成段。

**若最后一行不显著最好** → §C2 不成立,需要重写 intro 第 4 段以及 method 命名。

**算力**: 4 criteria × 3 compression × 1 model ≈ **9 GPU-h**

---

## P2 可选 1: Bit Capacity vs Robustness Trade-off

| Message length | 8 bits | 16 bits | 32 bits | 64 bits |
|---|---|---|---|---|
| Ours, GPTQ-4bit (exact match) | — | — | — | — |
| Ours, AWQ-4bit | — | — | — | — |

**算力**: 4 × 2 ≈ **8 GPU-h**。可砍。

---

## P2 可选 2: Robustness to Combined Attacks

| Attack | bit accuracy |
|---|---|
| GPTQ-4bit only | — |
| GPTQ-4bit + paraphrasing (Dipper) | — |
| GPTQ-4bit + 20% token substitution | — |
| GPTQ-4bit + cropping (50% prefix) | — |

**算力**: ~6 GPU-h。可砍。

---

## 算力总账

| 优先级 | 内容 | 算力 |
|---|---|---|
| **P0** | 主表 1 (attribution robustness) | ~90 |
| **P0** | 主表 2 (quality) | ~24 |
| **P0** | 消融 1 (component) | ~15 |
| **P1** | 消融 2 (token criterion) | ~9 |
| **P2** | 可选 1 (capacity) | ~8 |
| **P2** | 可选 2 (combined attack) | ~6 |
| **P0+P1** | | **~138 GPU-h** |
| **全部** | | **~152 GPU-h** |

**1× A800**: P0+P1 约 6 天连续。**4× A800**: 约 1.5 天(若实现支持 model parallel)。

---

## §A 风险与 reviewer 防御点

### A.1 PRO re-purpose

PRO 是 zero-bit。要在主表 1 里作为 multibit baseline,我们需要 re-purpose:
> 用 message 作为 watermark policy 的 conditioning 输入,把 policy 输出的 logit bias direction 离散化为 bit channel。

**必须在 appendix 写清楚算法,并提供消融**(我们 vs 我们魔改的 PRO multibit),让 reviewer 检查是否故意做弱了 baseline。**最好把这段算法发给 PRO 原作者请确认**(若可能)。

### A.2 Distillation 一行预期不好

`00_problem_lock.md` §6 R2 已 hedge:Pan 25 已证 zero-bit 都被打穿,multi-bit 在 distillation 下我们大概率不胜出。**这一行就报实情,不硬撑**。在 limitation 写"distillation robustness in multi-bit watermarking remains open;our method does not improve over baseline here, consistent with Pan 2025 findings on zero-bit"。

### A.3 模型族选择

LLaMA-3-8B + Qwen-2.5-7B 两家族对 reviewer 通常够。**若审稿坚持要更多**(rare,但 EMNLP 偶尔会),Phi-2 / LLaMA-3.2-3B 作为 appendix 用。算力预留 ~20 GPU-h。

### A.4 Calibration 集污染

GPTQ/WANDA 校准用 C4,生成测试也用 C4,会让 baseline 看起来过于鲁棒(in-domain 偏置)。**主表生成 prompt 必须用 OpenGen + Essays**,与 Qu 2024 evaluation 保持一致;C4 只用于校准 + Pilot-A。

### A.5 Random seed 与方差

每个 cell 的报告值是 500 (prompt × message) pairs 的 mean。需要 SEM 或 95% CI(简单 bootstrap 即可)。**两 seed 的差异若 > 报告值的差异,paper 立刻被打回**。所有主表实验跑 **3 个 seed**,报 mean ± std。算力相应 × 3 → 主表 1 实际 270 GPU-h(但若时间紧,只对 ours + 最强 baseline 跑 3 seed,其它 1 seed)。

---

## 数据集 (paper level)

| 用途 | 数据集 |
|---|---|
| Watermark generation prompts | C4 (calibration), OpenGen (test), Essays (test) |
| Quality eval - PPL | C4 validation, WikiText-103 |
| Quality eval - reasoning | MMLU 5-shot, GSM8K 8-shot |
| Quantization calibration | C4 train, 128 samples × 2048 tokens |

跟 Qu 2024 + PRO 对齐,reviewer 不能说我们 cherry-pick benchmark。

---

## 时间线(建议)

假设 Pilot-A 在第 0 周通过:

| 周 | 任务 |
|---|---|
| 1 | Method 设计稿 → `01_method_spec.md`,实现 + 单元测试 |
| 2 | Pilot-B(method 在 1 个 model + 1 个 compression 上 vs MPAC baseline) |
| 3 | P0 主表 1(LLaMA-3 + 6 compression × 6 method, 1 seed) |
| 4 | P0 主表 1(Qwen-2.5 + 6 compression × 6 method, 1 seed) + P0 主表 2 |
| 5 | P0 消融 1 + P1 消融 2 + 主表 1 补 seed 2/3 给 Ours + 最强 baseline |
| 6 | Paper draft v1 |
| 7 | P2 可选 + 内审 + 修订 |
| 8 | Paper draft v2 + 投稿 |

8 周从 Pilot-A 通过到投稿,**算力预算够**(P0+P1 ~140 GPU-h on 1× A800 = 6 天纯计算 + 充裕 buffer)。
