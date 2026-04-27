# 02. Pilot-A 规程: Baseline Drop Confirmation

**Status:** ready to implement
**Owner:** TBD
**Estimated cost:** ~5 GPU-hours on 1× A800 80G
**Decision:** Pilot-A 通过 → 进入 method 设计 (`01_method_spec.md`);未通过 → 回 `00_problem_lock.md` 修 §1 / §3 / §4

> **关于 Pilot-A 的定位**:此阶段**不验证我们的方法**(method 还没定义)。只验证两件事:(a) baseline (Yoo et al. 2024 NAACL, MPAC) 在压缩下显著退化,以确认问题真实;(b) 退化的 token-level pattern 与 logit margin 的相关性符号正确,以指导 method 设计。

---

## 1. 假设

**H1 (项目存活假设)**:
> Yoo 2024 MPAC (vanilla) (20-bit, 200 token) 在以下任一压缩 pipeline 下,exact_match 相对 FP16 绝对下降 **≥ 0.15**:
> - GPTQ-4bit
> - WANDA-50% unstructured prune

**H2 (方法方向假设)**:
> 在 FP16 → GPTQ-4bit 的压缩中,token-level "top-1 是否翻转" 与 "FP16 时该位置的 logit margin `m`" 在 token 集合上的 Spearman correlation **≤ −0.4**(即 margin 越大越不容易翻)。

H1 失败 → 项目核心 claim 站不住,需要 pivot 到更激进设置或重新定义问题。
H2 失败 → 需要把 method 的 component (a) 从"理论驱动的 margin gate"改为"经验校准的 flip-rate gate"(不影响项目存活,但弱化理论卖点;触发 R3 应对方案)。

---

## 2. Decision Rules(必须在跑前写死)

| 测量 | FP16 | GPTQ-4bit | WANDA-50% | 决策 |
|---|---|---|---|---|
| `exact_match` | A | B | C | 若 A − B ≥ 0.15 或 A − C ≥ 0.15 → H1 通过 |
| `bit_accuracy` | a | b | c | 仅作辅助参考,不作决策 |
| Spearman ρ(margin, flip) | — | — | (主测) | 在 GPTQ-4bit 上,ρ ≤ −0.4 → H2 通过 |

并列定义:
- 若 `A < 0.85`,即 baseline 在 FP16 下都不达标,**先调实现** —— 不是 baseline 问题,是我们 re-impl 错。
- 若 `A − B < 0.05` 且 `A − C < 0.05`,**几乎不退化**,触发 R1 应对(切换到 32-bit / 100 token / WANDA-70%)。
- 0.05 ≤ max-drop < 0.15:**灰区**。延展 pilot 至 1000 prompt 减小方差,再判。

---

## 3. 实验设置

### 3.1 模型

| 项 | 值 | 备注 |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3-8B-Instruct` | 单模型够 pilot;Qwen 主表再加 |
| Precision | bf16 (FP16 的稳定替代) | A800 上 bf16 比 fp16 数值更稳 |
| Generation | nucleus sampling, top_p=0.95, temperature=1.0 | 与 Yoo/Qu 一致 |
| Max new tokens | 200 | 与 Qu 24 Table 2 对齐 |
| Min length | 180 (强制) | 避免短样本 |

### 3.2 Watermark 方案: Yoo et al. 2024 (MPAC) vanilla

**实现来源**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak, "Advancing Beyond Identification: Multi-bit Watermark for Large Language Models", NAACL 2024.
- arxiv: **2308.00221**
- 官方代码: **https://github.com/bangawayoo/mb-lm-watermarking**

**关键参数**(对齐 Yoo 默认):
- Message length: **20 bits**(pilot 用 20-bit;FP16 下应 ≥ 0.97 exact match,留够下降空间)
- 无 ECC(故意,看 raw 退化幅度)
- Radix r = 4(colorlist size, MPAC 默认)→ 每 message-carrying token 携带 log2(r) = 2 bits → 20-bit msg 切成 K=10 个 r-radix chunk
- Sublist ratio γ = 1/r = 0.25
- Logit bias δ = 2.0
- Hash window: 1 prev token

**MPAC 算法骨架** (§3 Yoo 2024):
1. message m ∈ {0,1}^20 切成 r-radix 数字串 (m_1, ..., m_K)
2. 对每个生成位置 t,通过 hash(prev_token_t) 决定 (a) 该位是否为 message-carrying position,(b) 若是,carry 哪个 chunk index k(t)
3. 把词表分成 r 个 colored sublist `C_0, ..., C_{r-1}`(每个 size = |V|/r);message-carrying 位置 t 上,对 sublist `C_{m_{k(t)}}` 加 δ
4. 解码: 对每个候选 message m',对应取出 colored sublist 序列,统计观测 token 落在 "正确 sublist" 上的频率;每个 chunk index 独立投票

**实现注意**:
- **优先直接 fork bangawayoo/mb-lm-watermarking**;不要从头复现 (hash + colorlist 细节多,从头容易翻车)
- 单元测试:20-bit message 在 FP16 LLaMA-3-8B 下 exact_match ≥ 0.95 才算复现合格,与 Yoo paper Table 2 对齐

### 3.3 压缩 Pipeline

| 名 | 实现 | 配置 |
|---|---|---|
| **GPTQ-4bit** | `auto-gptq` | group_size=128, desc_act=True, calibration: C4 dev, 128 samples × 2048 token |
| **WANDA-50%** | 官方 repo `mit-han-lab/wanda` 或 `EleutherAI/lm-eval-harness` 的 wanda 集成 | 50% unstructured, calibration: C4 dev 128 samples |

**关键约束**:压缩**不能**接触 watermarked 模型权重——压缩是对 base model 做(因为我们是 logit-bias 推理时水印,模型本身没变)。所以流程是:
```
Base LLaMA-3-8B
  → 分支1: bf16 → 加 watermark logit bias → generate → text_fp16
  → 分支2: GPTQ-4bit → 加 watermark logit bias → generate → text_gptq
  → 分支3: WANDA-50% → 加 watermark logit bias → generate → text_wanda
```

每个分支用**相同的 message + 相同的 prompt + 相同的 random seed**(seed 控制 nucleus 采样)。这保证退化来自压缩本身,而非随机性。

### 3.4 数据

**Prompts**: C4 validation split,前 500 条样本的前 30 token 作为 prompt(继续生成 200 token)。

**Messages**: 用 fixed seed 生成 20 个不同的 20-bit message。每个 message 应用到 25 个 prompt → 25 个生成 → 共 500 (message, prompt, generation) triplets。这种结构便于:
- 估 exact_match per message,看是否 message-dependent
- 估 bit_acc per bit position,看 paper §3.1 position allocation 是否均匀

### 3.5 评估指标

主指标:
1. `exact_match` = mean over (message, prompt) pairs of `I[m == m̂]`
2. `bit_accuracy` = mean over (message, prompt, bit_index) of `I[m_i == m̂_i]`

诊断指标(给 H2 / method 设计用):
3. `top1_flip_rate(t)` = P(token at position t differs from FP16) per token position t
4. `margin_at_pos(t)` = `ℓ_(1)(t) − ℓ_(2)(t)` measured under FP16
5. Spearman ρ over all (prompt, position) pairs of `(margin_at_pos, top1_flip_rate)`

实现 hint:
- 收集 FP16 generation 的每位 logit (top-K=10 即可)
- 在压缩后模型上,**用 FP16 生成的 prefix 重新 forward**,看 top-1 是否仍然是 FP16 时选的那个
- 这样就能 in-place 做 margin vs flip 的相关性

**注**: top-1 翻转是 partition 错误的**充分但非必要**条件——同 partition 内换一个 token,partition 不变 → MPAC 解码不出错;不同 partition 换 token → 出错。所以 `top1_flip_rate` 是 partition error 的上界。Pilot-A 同时记录"colorlist 一致性" (top-1 token 的 colorlist 索引是否变),作为更精确的 partition error 度量。

---

## 4. 算力预算

| 步骤 | 估时 | 注 |
|---|---|---|
| 环境搭建 + 模型下载 | 1h | LLaMA-3-8B ~16G,A800 一次拉完 |
| GPTQ-4bit 量化校准 | 0.5h | 128 calib samples |
| WANDA-50% pruning | 0.3h | 128 calib samples |
| FP16 generation (500 × 200 tok) | 0.4h | bs=8 batched generation |
| GPTQ generation | 0.5h | 慢一点 |
| WANDA generation | 0.5h | dense forward |
| 解码 + bit accuracy 计算 | 0.2h | CPU 即可 |
| Margin-flip 相关性测量 | 0.5h | logit 收集 + 重 forward |
| Buffer (debug, 重跑) | 1.5h | 实际经验 30% buffer |
| **总计** | **5.4h** | 单 A800 一晚跑完 |

---

## 5. 输出 (artifacts)

`results/pilot_A/` 下应有:

```
pilot_A/
├── config.yaml                       # 实际跑的配置(从 configs/pilot_A.yaml 拷贝)
├── messages.json                     # 20 个 random 20-bit messages
├── prompts.jsonl                     # 500 prompts from C4
├── generations/
│   ├── fp16.jsonl                    # 500 (prompt, message_idx, text, sampled_logits_top10)
│   ├── gptq4.jsonl
│   └── wanda50.jsonl
├── decoded/
│   ├── fp16.jsonl                    # (prompt_idx, message_idx, m_hat, exact_match, bit_acc)
│   ├── gptq4.jsonl
│   └── wanda50.jsonl
├── stability/
│   ├── token_flips.csv               # (prompt_idx, position, token_fp16, token_compressed, margin_fp16, did_flip)
│   └── correlation.json              # Spearman ρ + p-value per pipeline
├── tables/
│   └── pilot_A_summary.csv           # 一张表:rows=pipeline, cols=exact_match/bit_acc/spearman
└── report.md                         # 跑完手写的解读 + 决策
```

最后一份 `report.md` 是 critical:**作者必须明确写**通过 / 未通过 / 灰区,以及对应触发的下一步动作。

---

## 6. 实现 checklist (实现时一条条划掉)

代码骨架(`scripts/pilot_A_baseline_drop.py`):

- [ ] 加载 LLaMA-3-8B-Instruct (bf16)
- [ ] 实现 Yoo 2024 MPAC `WatermarkLogitsProcessor` (优先 fork bangawayoo/mb-lm-watermarking,否则按 paper §3 复现 + 单元测试)
- [ ] 加载 C4 validation,采样 500 prompts
- [ ] FP16 generation 路径:`text_fp16 + per_position_top10_logits`
- [ ] GPTQ-4bit 量化(`auto-gptq`)持久化到 `models/llama3-8b-gptq4/`
- [ ] WANDA-50% pruning(`wanda` repo)持久化到 `models/llama3-8b-wanda50/`
- [ ] 在 GPTQ / WANDA 模型上分别跑同 prompt + 同 message + 同 seed 的 generation
- [ ] 实现 MPAC decoder,在三组 generation 上算 exact_match + bit_accuracy
- [ ] Margin-flip 测量:用 FP16 prefix 在 compressed model 上 forward,记录 top-1 是否一致 + FP16 margin
- [ ] 输出汇总表 + report.md 模板

---

## 7. 已知坑

1. **MPAC 复现的 fidelity**: bangawayoo/mb-lm-watermarking 已开源,优先 fork。若 API 与我们整合困难,按 paper §3 完整复现并通过单元测试。**不要凭"看起来像 KGW-multibit"自己编一个**——后续主表对照都依赖这一份。
2. **GPTQ 校准集 bias**: 校准用 C4 但生成测试也用 C4,可能让 GPTQ 看起来过于鲁棒。**Pilot-A 暂时 OK**(我们要看下降 lower bound),但主表必须用 in-domain 校准 + out-of-domain 测试。
3. **bf16 vs fp16**: Yoo 2024 paper 用 fp16,我们用 bf16。bf16 数值更稳,可能让 baseline 看起来更鲁棒。**报告时明确说**;对照实验若边界值附近,补一组 fp16 sanity check。
4. **生成长度强制 200**: 实测短生成会让 exact_match 偏高(更少 chunk)。强制 min_length=180,过滤 < 180 的样本(应该 < 5%)。
5. **多 GPU 并行**: 当前只 1 A800,串行跑;后续加卡时改 batch parallel,注意复现性(seed)。
