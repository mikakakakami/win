# Multi-bit LLM Watermarking under Model Compression

EMNLP submission. Status: **scoping / pre-Pilot-A**.

## 一句话定位

现有 multi-bit LLM 水印(Yoo 2024 MPAC, Qu 2024 ECC, SAEMark 2025)只在文本层攻击下评估,没人系统验证模型压缩(quantization / pruning / distillation)对 message recovery 的破坏。我们填这个空白,并给出针对压缩失真的针对性方法。

## 目录

```
win/
├── docs/
│   ├── 00_problem_lock.md          问题锁定 + 理论 + threat model + claim
│   ├── 01_method_spec.md           方法数学规格(Pilot-A 后填)
│   ├── 02_pilot_A_protocol.md      Pilot-A 规程(本周跑)
│   ├── 03_pilot_B_protocol.md      Pilot-B 占位(method 锁定后填)
│   ├── 04_full_experiment_plan.md  全表 P0/P1/P2 实验设计
│   ├── 05_baselines_reimpl.md      6 个 baseline 的重实现笔记
│   └── 06_decision_log.md          关键决策 + 时间戳
├── src/wmark/                       package(暂空)
├── scripts/                         可执行脚本
├── configs/                         实验配置
├── results/                         实验产出(gitignored)
└── paper/                           ACL latex 骨架(待建)
```

## 当前状态

- [x] 文献调研 + gap 验证(2026-04-27)
- [x] 问题锁定文档
- [x] Pilot-A 规程
- [x] 全表实验设计
- [ ] Pilot-A 实现 + 跑通
- [ ] 方法定型
- [ ] Pilot-B
- [ ] 全表
- [ ] paper draft

## 阅读顺序

新人或重新进入项目,按 `00 → 02 → 04` 的顺序读,基本能理解整个设计。

## 算力

- 当前: 1× A800 80G
- 后期可加
- 全表 P0+P1 总成本估计 ~138 GPU-hours,详见 `docs/04_full_experiment_plan.md`
