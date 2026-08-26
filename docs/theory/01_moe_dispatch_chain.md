---
status: draft
date: 2026-08-24
关联EXP: EXP-009, EXP-014, EXP-015, EXP-016, EXP-017
---

# MoE dispatch 链路(vLLM fused MoE,Triton 路径)

## 1. 一句话结论

MoE decode 的成本重心随 batch 从非 routed 部分(bs=1:dense GEMV 40.9%)迁移到
routed experts 的 grouped GEMM(bs=32:56.4%,EXP-014《D1 MoE decode 分解》),因此 4090 上这条链路的
第一优化杠杆是 fused_moe 的 Triton tile config 而非 kernel 重写——EXP-015《D2 MoE config 调优》用
"中段 M 与默认启发式打平"实证了这一点。

## 2. 机制(自己的话)

一次 MoE 层 forward(vLLM fused_moe Triton 路径)分五步:

1. **routing**:hidden → gate 线性层 → top-k softmax(Qwen1.5-MoE 为 top-4/60),
   得到每 token 的专家集合与权重;
2. **moe_align_block_size**:把 token×专家 映射按专家分组、对齐到 BLOCK_SIZE_M,
   产出 sorted_token_ids——为 grouped GEMM 制造"每专家一段连续行";
3. **grouped GEMM(fused_moe_kernel)**:单个 Triton kernel 网格覆盖所有专家的
   up/gate 投影 + 激活 + down 投影;tile 形状(BLOCK_SIZE_M/N/K、GROUP_SIZE_M、
   num_warps、num_stages)按 M 档从 config JSON 查表(本仓交付档位 18 个,
   M=1–4096;缺档设备回退启发式默认并打运行时告警);
4. **unpermute / moe_sum**:按 routing 权重加权求和回 token 序;
5. TP>1 时 **allreduce** 收尾(EXP-014 实测恒 ~14–15%,TP2 固定税)。

config 文件键 = (E, N, device_name[, dtype]),E/N 由并行方式决定:EP 把 60 专家
切到 2 卡(每卡 E=30,N=1408),非 EP 每卡持全部专家但 N 减半(E=60,N=704)——
GEMM 形状不同,tile 最优解不同,所以是两个独立 tuple。

## 3. 本项目实证(必须指自家 EXP 数字)

- **EXP-014**(nsys node 级分解,`--cuda-graph-trace=node` 必需):fused_moe
  grouped GEMM 占 GPU kernel 时间 18.7%(bs=1)→ **56.4%**(bs=32);MoE/dense
  吞吐比 **2.03×(bs=1)→0.97×(bs=8,反转点)→0.82×(bs=128)**;bs=1
  roofline 对照:MoE 实测 221 tok/s / 理论 ~373(59%),dense 109/~142(77%)。
  机理:top-4/60 下 batch 增大 → 每 step 命中专家并集趋全量(~28.6GB > dense
  14.2GB),激活稀疏优势反转为读放大劣势。
- **EXP-015**(1920 配置×18 M 档×2 tuple,8/24 勘正档数):kernel A/B 两端
  改善——M=1:EP **-8.5%** / 非 EP -3.8%;M=128/256:-3.3~-3.9%;M=8–64
  与默认持平;e2e TPOT +0.8~1.2% ≈ kernel 增益 × 56.4% 占比折算(自洽,但低于
  跨会话漂移,不作 headline)。
- **EXP-009《C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据》**(缺档告警原文):fused_moe.py:1106 运行时点名 E=30,N=1408 缺失
  ——"社区空缺"的三重闭环之一。
- **EXP-016《D4 FP8 vs W4A16 同卡对比》**(相邻链路):同为 MoE,quant 分派不同路径——W4A16 走 Marlin,
  FP8 在 Ada 只能走 Triton block-scaled,decode 全 regime W4A16 胜 23–48%。

## 4. 面试追问 Q&A

- **Q:为什么 bs=1 时 MoE 有优势、大 batch 反而输?**
  A:decode 是权重读带宽受限。bs=1 每 step 只读激活专家 ~2.7GB/卡,dense 读
  7.1GB/卡,roofline 比值即上限;batch 大了专家并集趋全量(28.6GB>14.2GB),
  读放大反超 dense——EXP-014 曲线在 bs≈8 处交叉。
- **Q:中段 M 为什么调不动?**
  A:1920 配置搜索的最优在 M=8–64 与启发式默认打平——该区域 tile 选择已被
  默认覆盖;收益只在两端(极小 M 与 M≥128)。这也是 D3"不做 kernel 改动"的
  数据依据(EXP-015 §6)。
- **Q:moe_align/permute 值得优化吗?**
  A:占比 ≤4.1% / ≤0.5%(EXP-014 分解表),本 regime(TP2+EP,bs≤128)不值得。
- **Q:EPLB 在这条链路的哪里?**
  A:routing 之上的专家重排层。实测 FP8 臂 2 次真实重排(balancedness
  0.53–0.74),W4A16 被上游显式拒(routed_experts.py:151);重排改变 grouped
  GEMM 的专家分段与浮点归约顺序 → 数值性输出分歧(EXP-017《D5 EPLB gate》对照组归因)。

## 5. 延伸(源码/数据,file:line)

- 缺档告警与查表:vLLM fused_moe.py:1106(EXP-009 §5 原文引用)。
- 调优器:上游 benchmarks/kernels/benchmark_moe.py(--tune;EXP-015 §3 命令)。
- EPLB:重排入口 eplb_state.py:748;W4A16 拒绝点 routed_experts.py:151(EXP-017)。
- FP8 分派:oracle/fp8.py:103-122(capability 90/100 快路径跳过 SM89 → TRITON,
  EXP-016)。
- 数据:`moe_perf/derived/d1_kernel_share_bs1.csv` / `d1_kernel_share_bs32.csv` /
  `d1_scaling.csv`;记录 EXP-014 §5、EXP-015 §5。
