# 03. Pilot-B 规程(占位)

**Status:** TBD,Pilot-A 通过 + `01_method_spec.md` 锁定后填
**Estimated cost:** ~5 GPU-hours

---

## 目标

在 1 个 model (LLaMA-3-8B-Instruct) + 1 个 compression (GPTQ-4bit) 上,验证 method v0 显著优于 MPAC baseline (Yoo 2024)。

## 假设

**H3**: 在 GPTQ-4bit 下,Ours 的 exact_match 比 MPAC baseline (Yoo 2024) 绝对高 **≥ 0.10**(20-bit message, 500 pair, 95% CI 不重叠)。

不通过 → 回 `01_method_spec.md` 修。
通过 → 进入主表(`04_full_experiment_plan.md`)。

## 设置(待填)

模型 / 数据 / 压缩方式与 Pilot-A 完全一致,只换方法。

详细规程在 method spec 锁定后写。
