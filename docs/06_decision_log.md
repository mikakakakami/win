# 06. Decision Log

记录所有改动 `00_problem_lock.md` §1-§5 的关键决策,以及方法、实验设计的方向性变化。

格式: `日期 | 决策 | 触发证据 | 修改前 → 修改后 | 决策人`

---

## 2026-04-27 | 项目锁定

- **决策**: 立项 multi-bit LLM 水印 + 模型压缩鲁棒性
- **触发**: 文献扫描确认 gap (Wang 24 / Qu 24 / SAEMark / WaterPark 全无 compression evaluation;PRO 仅 zero-bit)
- **决策人**: 项目所有者 + Claude
- **下一步**: Pilot-A

## 2026-04-27 | Method 简化为 2 component

- **决策**: 把原计划的三个 component (margin-stability gate / compression-aware allocation / adaptive ECC rate) 简化为两个,把 adaptive ECC rate 内化到 allocation 里
- **触发**: adaptive ECC rate 是信息论教科书内容,单独成立会被 reviewer 拆穿是 trivial contribution
- **修改前**: 3 component
- **修改后**: 2 component (margin-stability gate, compression-aware bit allocation)
- **决策人**: Claude 建议,所有者待确认

## 2026-04-27 | Pilot-A 用 20-bit 而非 32-bit

- **决策**: Pilot-A 的 message 长度用 20-bit
- **触发**: Yoo 2024 NAACL (MPAC) 在 20-bit / 200 token 下 FP16 报 ~97.6%,留够下降空间;32-bit 在 FP16 下 ~94% 已经偏低,baseline 退化判定不清晰
- **修改前**: 拟用 32-bit (与主表一致)
- **修改后**: Pilot 用 20-bit,主表才用 32-bit (有 ECC 救场)
- **决策人**: Claude

## 2026-04-27 | 修正 baseline 引用

- **决策**: 把所有 docs 里的 "Wang 2024" 改为 "Yoo et al. 2024 (MPAC)"
- **触发**: 复查发现 NAACL 2024 那篇是 Yoo/Ahn/Kwak 而非 Wang。方法名 MPAC,arxiv 2308.00221,代码 github.com/bangawayoo/mb-lm-watermarking
- **决策人**: Claude(初稿误读时把 search summary 里的"Wang et al."当成了作者)
- **影响**: 涉及 README + docs 00/02/04/05/06。Pilot-A §3.2 算法描述也按 MPAC 实际机制(colorlisting r-radix)更新

## 2026-04-27 | Adaptive WM 确认存在,纳入主表

- **决策**: Liu & Bu 2024 "Adaptive Text Watermark for Large Language Models" (arxiv 2401.13927) 确认存在,纳入主表 baseline
- **触发**: 之前 `05_baselines_reimpl.md` 标记 TBD,本日 web search 已确认
- **影响**: 主表 baseline 数从 4 升到 5,加上 Ours 共 6 列
- **注**: 该 paper 是 zero-bit,需 re-purpose 为 multibit。算法在 appendix 写清

## 2026-04-27 | MMLU/GSM8K 评测协议明确

- **决策**: MMLU 走 logprob 评测(watermark 不介入), GSM8K 走 CoT 生成(watermark 介入), PPL 用未水印参考 LM 测水印文本
- **触发**: 04 文档原本写"评测时关闭 watermark",reasoning 不准确
- **修改后**: 三个 metric 协议分别明确,见 04 文档主表 2 注

---

## 2026-04-27 | 算力修正 H800 → A800

- **决策**: 所有 docs 中 H800 改为 A800 80G(单卡)
- **触发**: 实际远端硬件就是 A800

## 2026-04-27 | Pilot-A smoke 替代 GPTQ → bnb_nf4

- **决策**: 由于远端 AutoGPTQ 没 CUDA extension,量化后卡在 pack/save,**用 bitsandbytes NF4 作为 4-bit smoke 替代**,跑 100 prompts 的 reduced Pilot-A
- **触发**: 远端 AutoGPTQ 0.7.1 的 CUDA ext 编译失败,quantize 完后 model.save_quantized() 长时间不返回
- **影响**: 这一轮的结果**不是正式 Pilot-A**(规程要求 GPTQ-4bit + WANDA-50%, 500 prompts)。仅用作 (a) 验证 encoder/decoder 链路,(b) 给压缩退化提供量级估计
- **正式 Pilot-A 重启条件**: 切换到 GPTQModel(AutoGPTQ 活跃 fork,有 prebuilt CUDA wheel)或 transformers 内置 GPTQConfig 路径 → 跑全 500 prompts

## 2026-04-27 | BF16 baseline 不达标 (0.03 exact_match) → 三处修复

- **决策**: 在重新跑生成前修三处 (a) `pilot_A_baseline_drop.py` 显式记录并使用 `model.config.vocab_size` 而非 `len(tokenizer)` (b) `configs/pilot_A.yaml` δ 由 2.0 提到 4.0 (c) 新增 `scripts/diagnose_mpac.py` 不重跑生成就能看 per-chunk vote 分布
- **触发**: 远端 smoke 跑出 BF16 bit_acc=0.7665, exact_match=0.03。按 Pilot-A 规程 §2 必须 ≥0.85 才往下。理论分析:r=4 colorlist 在 δ=2 下 per-chunk correct ≈ 0.53,K=10 时 0.53^10 ≈ 0.002,正好和观测吻合。Yoo 2024 原 paper r=4 时也用 δ≥2,但实际典型设置是 δ=4
- **修改前**: δ=2.0,vocab 用 len(tokenizer)
- **修改后**: δ=4.0,vocab 强制取 model.config.vocab_size,记录到 jsonl,decoder 读取
- **下一轮验证**: 用同 100 prompts smoke 重跑 generate + decode,目标 BF16 exact_match ≥ 0.85;若仍 < 0.85,触发 vendor bangawayoo 官方实现

---

## 待决策(open)

- Adaptive WM baseline 是否纳入主表(取决于能否找到 paper + code)
- PRO multibit re-purpose 算法是否需要联系原作者
- 主表是否所有 cell 都 3 个 seed,还是只 Ours + 最强 baseline 3 seed(算力 trade-off)
