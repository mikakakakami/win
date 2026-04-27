# 05. Baseline 重实现笔记

每个 baseline 的 re-impl 来源、关键参数、可能的坑。

---

## 1. MPAC (Yoo et al. 2024, NAACL)

**论文**: KiYoon Yoo, Wonhyuk Ahn, Nojun Kwak, "Advancing Beyond Identification: Multi-bit Watermark for Large Language Models", NAACL 2024.
- arxiv: **2308.00221**
- 官方代码: **https://github.com/bangawayoo/mb-lm-watermarking**

**核心机制**:
- "Position allocation" + "Colorlisting":每个生成 token 通过 hash 决定它属于哪个 message chunk index;词表分成 r 个 colored sublist(r-radix),logit bias 加在与本 chunk 当前 r-radix 数字对应的 sublist 上

**关键参数**:
- Radix r = 4(默认,每 token 携带 log2(r) = 2 bits)
- Sublist ratio γ = 1/r = 0.25
- Logit bias δ = 2.0
- Hash window = 1 prev token
- Position allocation: paper §3.1 详

**坑**:
- 优先 fork 官方 repo,而非从头复现(hash + colorlist 细节多)
- 单元测试: 20-bit message 在 FP16 LLaMA-3-8B-Instruct 下 exact_match ≥ 0.95 才算复现合格(对齐 Yoo 2024 Table 2)

---

## 2. ECC-Multibit (Qu et al. 2024, USENIX Sec 25)

**论文**: Qu et al., "Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code".
- arxiv: **2401.16820**
- 开源代码: 论文 supplement(若有)

**核心机制**:
- 在 KGW-style multibit 上叠 LDPC ECC
- m → ECC encode → m' → embed via greenlist

**关键参数**:
- ECC: LDPC, rate 0.5
- pyldpc 库的矩阵参数 (n, d_v, d_c) 按 paper Table 1 设
- 解码: belief propagation, max iter ≈ 50

**坑**:
- LDPC encode/decode 数值稳定性,注意校对 paper 的具体配置
- 与 MPAC 的 colorlisting 互斥(Qu 用 binary greenlist + ECC,不是 colorlist)。**两条 baseline 在主表里要分开报告,不要混淆。**

---

## 3. Adaptive Watermark (Liu & Bu 2024)

**论文**: Yepeng Liu, Yuheng Bu, "Adaptive Text Watermark for Large Language Models".
- arxiv: **2401.13927**
- 开源代码: 待确认 (作者 Yepeng Liu GitHub)

**核心机制**:
- 高熵 token 处加 watermark logit bias,低熵 token 不动
- 用 auxiliary 语义嵌入 model 决定 logit scaling factor(语义自适应,不是 fixed delta)
- 原 paper 是 **zero-bit**,在主表里我们需要 re-purpose 为 multibit(用 message hash 控制 green list 选择,跟 KGW 同思路)

**坑**:
- 这是 zero-bit paper,re-purpose 为 multibit 必须在 appendix 写清算法,与 PRO re-purpose 类似处理(诚信)
- Auxiliary 语义 model 的选择影响实验结果——按原 paper 用 SimCSE-base(开源)
- 如果 re-purpose 后 FP16 baseline 显著低于 MPAC,说明 entropy-only 准则在 multibit 任务上不占优,**这本身就是 §C2 (机理 claim) 的一个证据**

---

## 4. PRO (zero-bit → multibit re-purpose)

**论文**: "PRO: Enabling Precise and Robust Text Watermark for Open-Source LLMs", arxiv **2510.23891**.
**开源代码**: 论文应有 GitHub 链接 (待查 OpenReview pdf)

**Re-purpose 算法**(我们魔改,**必须在 appendix 写清**,可邮件原作者请 sanity check):
1. 训练 watermark policy network P_θ,输入 (hidden state, message_chunk_id ∈ {0, ..., 2^k-1}),输出 logit bias direction d ∈ R^|V|
2. 对每个生成位置,根据 chunk_id(由 message + position 决定)取相应 d,加到 base logit 上
3. 解码: 对每个候选 chunk_id,前向 P_θ 得到 d',计算与观测文本的对齐 score
4. 跨 chunk 取最大 score message

**坑**:
- 训练成本: 原 PRO 报 6h × 4×A100 (zero-bit)。multibit 版本预计 ~2× 训练量(2^k chunk space),即 ~12h × 4×A100 ≈ 48 GPU-h(per model)
- 若超预算,fall back 到 lightweight version: 把 message 作为 conditioning concat,只 fine-tune policy head
- **诚信**: re-purpose 算法不能故意做弱。在 appendix 提供 ablation:our re-purpose vs 原 PRO zero-bit detection。前者显著弱化在 multibit 任务上是 expected,但应在合理范围
- 邮件 PRO 一作请 sanity check 算法(若 EMNLP cycle 时间允许)

---

## 5. No WM (control)

直接生成,不加任何 watermark。

仅用于:
- Quality 表 (PPL/MMLU/GSM8K) 的 baseline
- Random message 解码作为 chance level 参考(20-bit message 随机猜中 exact_match = 2^-20 ≈ 0)

---

## 6. Robust Binary Code (RBC, Boroujeny et al. 2024)

**论文**: arxiv **2406.10281**, "Watermarking Language Models with Error Correcting Codes".
**开源代码**: 待查

**默认 skip**(主表已 5 个 baseline),作为 appendix optional baseline。

---

## 校验清单(每个 baseline 都要过)

- [ ] FP16 下 exact_match 复现到 paper 报告的 ±2% 范围
- [ ] 20-bit + 32-bit 都能跑(message length 可参数化)
- [ ] 单元测试 (encode → decode without compression → m == m̂ on at least 100 random messages)
- [ ] 与 Ours 在同一 LogitsProcessor / generation API 下接入
- [ ] 加 docstring 注明: paper 来源 / 复现起 commit / 已知与 paper 的差异

---

## 主表最终 baseline 列(共 5 个 + Ours = 6)

1. **No WM** (control)
2. **MPAC** (Yoo 24, NAACL)
3. **ECC-Multibit** (Qu 24, USENIX Sec 25)
4. **Adaptive (re-purposed multibit)** (Liu & Bu 24)
5. **PRO (re-purposed multibit)** (arxiv 2510)
6. **Ours**

这 5 个 baseline 覆盖了 logit-bias 路线(MPAC, ECC, Adaptive)和 trainable 路线(PRO),公允度对 reviewer 已经够。
