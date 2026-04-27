# 01. Method Spec(占位)

**Status:** TBD,等 Pilot-A 跑完后填
**Trigger to write:** Pilot-A H1 通过(或灰区扩展通过)
**Trigger to revise:** Pilot-B 不显著 / 主表某 cell 反向

---

## 待填内容

按 `00_problem_lock.md` §4 的 C3,本文档需要给出:

### 1. Margin-stability gate (component a) 的形式化

- 输入: token position t, model M, base prompt context
- 输出: stability score s(t),用于决定该位是否参与 watermark 嵌入
- 数学定义: s(t) 与 logit margin 的关系,以及如何近似 "在压缩噪声下的稳定性" (理论近似 vs 经验校准 vs hybrid)
- 计算成本(per-token forward overhead)
- 单元测试: 在 simulated noise 下,选高 s(t) 的位置应有显著低的 flip rate

### 2. Compression-aware bit allocation (component b) 的形式化

- 输入: message m ∈ {0,1}^b, 候选位置集合 {t_1, ..., t_T} 及其 stability score, 目标 compression pipeline (or unknown)
- 输出: position-to-bit mapping + 每位 ECC redundancy
- 决定: 是 pipeline-aware (部署时知道 compression) 还是 pipeline-agnostic (worst-case)
  - **建议先做 agnostic**(简单且通用),pipeline-aware 在 appendix 给作为 upper bound
- ECC 选择 (LDPC vs BCH vs Polar) 与 rate adaptation rule
- 单元测试: 在固定 noise level 下,allocation 优于 uniform

### 3. 整体 algorithm pseudocode

```
def WMEncode(M, prompt, message, key):
    # 一步前向获得每位 logit + margin
    # 应用 stability gate
    # 应用 bit allocation
    # 调整 logit (KGW-style logit bias)
    # 采样下一个 token
    # 返回 next_token + state (为下一位准备)

def WMDecode(text, key, b):
    # 在 candidate messages 上 score
    # 用 ECC decoder 还原
    # 返回 m_hat
```

### 4. 复杂度 + 实现注意

- 推理时延 overhead 上界
- 解码计算量(随 b 的 scaling)
- 与 batched generation 的兼容

### 5. 安全性论证(简短)

不是密码学 paper,但要说清:
- key 不公开下,detection AUC vs random baseline
- 信号不可被 trivially copy 到非 watermarked 文本上
- 这部分可以 follow Qu 2024 的安全性框架

---

## 提示

Pilot-A 跑完后,**先用 Pilot-A 的 token-level (margin, flip) 数据拟合 component (a) 的具体函数形式**,再写本文档。换句话说:**这份 spec 是 data-driven,不是先写 spec 再去验证**。
