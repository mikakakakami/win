# 00. 问题锁定 (Problem Lock-in)

**Locked on:** 2026-04-27
**Owner:** TBD
**Status:** v0.1 (会随 Pilot-A 结果更新)

> 这份文档锁定了项目的 **核心问题、threat model、核心 claim、理论根据、风险**。一旦后续工作偏离,**先回来改这份**,再改下游。所有 paper section、pilot、实验设计都引用这一份。

---

## 1. 核心问题

**Multi-bit LLM 水印在模型层压缩(quantization / pruning / distillation)下的 message recovery 鲁棒性。**

具体地:给定 watermarked LLM `M`,生成包含 b-bit 消息 `m ∈ {0,1}^b` 的文本 `T`。当模型被压缩为 `M' = Compress(M)` 之后,从 `M'` 生成的文本 `T'` 上,我们能否仍然以高概率恢复 `m` 完整正确?

测度:
- `bit_accuracy(m, m̂) = (1/b) Σ I[m_i == m̂_i]`
- `exact_match(m, m̂) = I[m == m̂]`(更严格,paper 主指标)

---

## 2. Gap 验证(为什么这个问题没人做)

文献状态(2026-04 截至,以下都验证过):

| 论文 | multi-bit | 测压缩 | 备注 |
|---|---|---|---|
| Kirchenbauer 23 (KGW) | × | × | zero-bit 元祖 |
| **Yoo 24 NAACL (MPAC)** | ✔ | × | text-edit 攻击 only;主流 multibit baseline |
| Qu 24 USENIX Sec 25 | ✔ + ECC | × | edit distance ≤ 17 |
| Robust Binary Code (2406.10281) | ✔ | × | edits/del/translation only |
| SAEMark NeurIPS 25 | ✔ (SAE feat) | × | 3 个 text 攻击 |
| WaterPark EMNLP-Findings 25 | benchmark | × | **12 攻击全是 text-level**,明确把压缩列为 open gap |
| PRO arxiv 2510 | × **(zero-bit)** | ✔ | GPTQ/AWQ/HQQ + WANDA/SparseGPT + SLERP + FT,AUC ≥ 0.987 |
| Pan 25 ACL | n/a (攻击) | distillation | 证明 zero-bit 都能被蒸馏抹掉 |

**结论**: multi-bit × model-compression 的格子是空的。PRO 把 zero-bit 解决了反而强化痛点 ——"检测信号在压缩下还活着,但消息内容塌了"是值得做的故事。

---

## 3. Threat Model

我们的设置如下,在 paper 第 3 节会形式化:

**目标**: 模型所有者 want to attribute generated text to a specific user/session/license.

**Actors**:
- *Provider*(我们):嵌入 b-bit message,后续解码
- *Adversary*: 拿到 watermarked model `M` 后,**通过模型层压缩(非恶意,合法部署优化)**得到 `M'`,然后正常使用 `M'` 生成内容

**关键设定**:
- Adversary **非主动攻击者**——他不知道 watermark 存在,也不试图擦除。压缩是 deployment-level 优化(降显存、加速)。这是和 Pan 2025 的关键区别(那篇是恶意 distill)
- Provider 拿到生成文本(无 access 给 `M'`),用 secret key 尝试恢复 `m`
- 单条文本长度 N 固定(默认 200 token)

**评估的压缩 pipeline**(P0 主表覆盖):
1. INT8 (bitsandbytes)
2. GPTQ-4bit
3. AWQ-4bit
4. WANDA-50% (unstructured prune)
5. SparseGPT-50% (unstructured prune)
6. Distillation → 3B student

为什么把 distillation 放进去:它是文献最近的痛点(Pan 25),即使我们方法在这一行表现不好,**实事求是报告为 limitation 比避而不谈更受 reviewer 待见**。

---

## 4. 核心 Claim

我们计划讲 4 个递进的 claim:

**C1 (诊断)**: Multi-bit watermark 在模型压缩下显著退化,且退化程度 >> zero-bit 检测信号。
> 这条由 Pilot-A 直接验证;不通过项目终止。

**C2 (机理)**: 退化的主要来源是 **per-token logit margin 在压缩噪声下的不稳定性**,而当前 SOTA(Yoo/Qu/Liu&Bu/SAEMark)的 token 选择准则(高熵、随机、句法位置)与 margin stability **不对齐**。
> 由消融表 2(token selection criterion ablation)验证。

**C3 (方法)**: 一个针对压缩失真设计的 multi-bit 嵌入框架,其核心是
- (a) **Margin-stability gate**: 选择在压缩噪声下 top-1 不易翻转的 token 位嵌入消息位
- (b) **Compression-aware bit allocation**: 不同压缩 pipeline 对应不同 failure mode,在 ECC redundancy 与 token 位选择上做差异化分配

> 由消融表 1(component ablation)验证每部件贡献。

**C4 (系统)**: 在 6 种压缩 pipeline × 2 模型族(LLaMA-3-8B + Qwen-2.5-7B)的主表上,我们的方法在 exact_match 上一致优于现有 5 个 baseline,同时不显著降生成质量(MMLU/GSM8K/PPL)。

> 由主表 1 + 主表 2 验证。

---

## 5. 理论基础

### 5.1 为什么 zero-bit 在压缩下"看起来 OK"

KGW z-score:
```
z = (n_green - γ·N) / sqrt(N·γ·(1-γ))
```
- 信号 ∝ N,噪声 ∝ √N → SNR ∝ √N
- **聚合统计**:每个 token 的 green/red 偏差被平均,单点扰动可被吸收
- 这与 PRO 实验数据(GPTQ-4bit AUC = 0.987)完全一致

### 5.2 为什么 multi-bit 必塌

Yoo/Qu 类 multi-bit 的解码需要**每个 chunk 位置的 partition 选择正确**(MPAC 用"colorlisting"——把词表分成 r 个 colored list,r-radix 表示消息;Qu 用 LDPC ECC 后再走 KGW-style 嵌入)。设单 chunk 错误率 `p`,K 个 chunk:
```
exact_match ≈ (1 - p)^K
```
即使 `p = 0.015`(1.5% 单 chunk 错),K=32 时 exact_match ≈ 61%。**zero-bit 的"聚合容错"机制不复用到 multi-bit。**

### 5.3 为什么 token margin 稳定性是关键

设 token 在词表上的 logit 排序 margin 为 `m = ℓ_{(1)} − ℓ_{(2)}`,压缩近似引入扰动 `Δℓ ~ N(0, σ²)`(各 token 独立)。则 top-1 翻转概率:
```
P(flip) ≈ Φ(−m / (σ·√2))
```
其中 Φ 是 N(0,1) CDF。σ 是直觉量级,**实际值会在 Pilot-A 中直接测**(用 FP16 vs 压缩模型在同一上下文下的最后一层 logit 差):
- INT8: 极小(<0.1)
- GPTQ-4bit / AWQ-4bit: 中等(预期 0.1–0.3 量级)
- WANDA-50%: 较大且 layer-specific

注意: GPTQ/WANDA 的扰动严格来说不是 i.i.d. Gaussian——量化是 deterministic + structured(取决于 weight 分布和校准集),WANDA 是 sparse perturbation。但中心极限论证下,在 logit 层(很多 weight 求和后)用 Gaussian 近似一阶矩是合理的。这一点会在 paper 里附 sanity check。

**关键观察**: `flip` 发生即意味着 watermark partition 选择可能出错。**控制 m(选高 margin 位置)可以指数压低翻转率。**

### 5.4 为什么"高熵选择"不等于"高 margin 稳定"

反例:
- `[0.3, 0.3, 0.3, 0.1]`: 熵 = 1.84 nat (高),m ≈ 0 (脆)
- `[0.6, 0.2, 0.1, 0.1]`: 熵 = 1.57 nat (低),m = 0.4 (稳)

**所以基于熵的 token 选择(Yoo 2024 MPAC / Qu 2024 / Liu&Bu 2024 Adaptive)在压缩下不是最优。**

这是项目独立的 finding,可以独立成段在 method section 里讲。

---

## 6. 风险登记

按风险等级排序:

### R1【高】Pilot-A 跑出 baseline 几乎不退化
**触发条件**: Yoo 2024 MPAC 在 GPTQ-4bit + WANDA-50% 下,20-bit message 的 exact_match 仍 ≥ 0.85(从 ~0.97 只下降 ≤ 0.12)。
**应对**: (a) 切换到更激进设置:32-bit / 100 token / WANDA-70% 看是否退化更明显;(b) 若仍不退化,改 paper positioning 到 distillation-only 场景(pivot 风险高)。
**决策时间窗**: Pilot-A 出结果 7 天内决定。

### R2【高】Distillation 一行被打穿
**触发条件**: 我们方法在 distill-3B 下 exact_match < 0.55(接近随机)。
**应对**: 写进 limitation,讨论 Pan 2025 的根本困难,不作为方法目标。**这本身不影响 paper accept**,但要在 intro/limitation 提前 hedge。

### R3【中】Margin stability 与实证不稳定相关性弱
**触发条件**: Pilot-A 测出 token-level margin 与 quantization-induced top1-flip 的 Spearman correlation < 0.4。
**应对**: 退而求其次,改 component (a) 为"经验观测的 stability",直接用小批量校准集统计每个 token 的 flip 概率,不强行套理论。这会减弱 paper 的理论卖点,但仍然 work。

### R4【中】Multi-bit + 训练式 baseline (PRO multi-bit) 性能远超我们
**触发条件**: 我们 re-purpose PRO 为 multi-bit 后(把 watermark policy 输出当 bit channel),它在压缩下 exact_match 接近或超过我们。
**应对**: 在 paper 中明确 trade-off ——我们 training-free,PRO 需要 6h × 4×A100 训练。计算预算敏感场景下我们仍胜出。

### R5【低】LDPC 实现复杂度
**触发条件**: pyldpc 在 32-bit message 上吞吐过低,或解码不收敛。
**应对**: 退到 BCH(更简单,但纠错能力略弱)或 repetition code(toy)。

### R6【低】不同模型族 finding 不一致
**触发条件**: LLaMA-3 上 component (a) 显著有效,Qwen-2.5 上不显著。
**应对**: 报告差异,讨论为什么(tokenizer? RLHF? base distribution),作为 future work。

---

## 7. 不做什么 (out of scope)

明确划界,避免审稿质疑越界:

- 不做 closed-source LLM 水印(我们不能控制 logit)
- 不做训练式水印(PRO 路线)的 from-scratch 改进——我们走 logit-bias inference-time 路线
- 不做 vision/multimodal 水印
- 不做主动恶意 adversary(这是 Pan 25 的题)。我们的 threat model 是 deployment-time 自然压缩
- 不做 watermark stealing / forgery 防御(orthogonal)

---

## 8. 决策依据 (后续修改本文档需附依据)

任何对 §1-§5 的修改必须在 `06_decision_log.md` 记录:
- 触发证据(Pilot-X 结果 / 文献 update / reviewer feedback)
- 修改前后对比
- 时间戳 + 决策人

---

## 引用(锚定本项目位置)

- KGW: Kirchenbauer et al., "A Watermark for Large Language Models", ICML 2023.
- **Yoo et al. 2024 (MPAC)**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak, "Advancing Beyond Identification: Multi-bit Watermark for Large Language Models", NAACL 2024. [arxiv 2308.00221, code: github.com/bangawayoo/mb-lm-watermarking]
- **Qu et al. 2024**: "Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code", USENIX Security 2025. [arxiv 2401.16820]
- **Liu & Bu 2024 (Adaptive)**: Yepeng Liu, Yuheng Bu, "Adaptive Text Watermark for Large Language Models". [arxiv 2401.13927]
- Boroujeny et al. (RBC + ECC): "Watermarking Language Models with Error Correcting Codes", arxiv 2406.10281.
- SAEMark: Yu et al., NeurIPS 2025. [arxiv 2508.08211]
- WaterPark / Watermark under Fire: Liang et al., EMNLP-Findings 2025. [arxiv 2411.13425]
- PRO: "PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs", arxiv 2510.23891.
- Pan/Huang 2025: "Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?", ACL 2025. [arxiv 2502.11598]
- GPTQ: Frantar et al., ICLR 2023.
- AWQ: Lin et al., MLSys 2024.
- WANDA: Sun et al., ICLR 2024.
- SparseGPT: Frantar & Alistarh, ICML 2023.
