# scripts/

Pilot 入口。当前只有 Pilot-A。

## 安装(在 H800 服务器上)

```bash
cd win
pip install -e .                       # installs deps from pyproject.toml
# WANDA: clone external repo (Pilot-A 默认不开,在 configs/pilot_A.yaml 里 enabled: false)
# git clone https://github.com/locuslab/wanda external/wanda
```

## 单元测试(必须先过)

```bash
cd win
PYTHONPATH=src pytest tests/ -v
```

`test_mpac.py` 必须全 pass,否则后续 Pilot 全是 garbage。

## 跑 Pilot-A

```bash
# 1. 一次性预处理 + 量化(GPTQ-4bit ~30 min on H800)
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage compress

# 2. 生成(三个 pipeline,每个 ~30-60 min)
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage generate

# 3. 解码 + 算 bit accuracy
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage decode

# 4. Margin/flip 诊断
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage margin_flip

# 5. 汇报 + decision
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage report
```

或一把跑:
```bash
python scripts/pilot_A_baseline_drop.py --config configs/pilot_A.yaml --stage all
```

## 输出

`results/pilot_A/` 下:
- `prompts.jsonl` (500)
- `messages.json` (20 个 20-bit message)
- `generations/{bf16,gptq4,wanda50}.jsonl`
- `decoded/{bf16,gptq4,wanda50}.jsonl`
- `stability/correlation.json` (Spearman ρ)
- `report.json` (decision)

## 决策

`report.json` 末尾的 `decisions` 字段会显示 H1 / H2 是 PASS / GREY / FAIL。

- H1 PASS + H2 PASS: 进入 method 设计 (`docs/01_method_spec.md`)
- H1 PASS + H2 FAIL: 仍可进入 method,但走 component(a) 的"经验校准 flip-rate"路线 (`docs/00_problem_lock.md` R3)
- H1 FAIL_NO_DROP: 触发 R1 应对 —— 切到 32-bit / 100 token / WANDA-70%
- H1 GREY: 扩展到 1000 prompt 重跑

人工还是要看 `report.json` 决定下一步,不要让脚本自己 promote 决策。
