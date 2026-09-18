# EXP-014 · D1 MoE decode 分解:吞吐-batch 曲线 + nsys kernel 占比

> **一句话结论**：MoE 的吞吐-batch 曲线会**反转**：top-4/60 下 batch 增大使每步命中的专家并集趋于全量（60 专家 ~28.6GB > dense 14.2GB），bs=1 的激活权重优势变成读放大劣势。serving batch（≥8）下 fused_moe grouped GEMM 是唯一大头（56.4%）——D2 的调优对象由数据锁定，不是预设。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23（08:00–08:40Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；MoE=Qwen1.5-MoE-A2.7B-Chat TP2+EP，dense=Qwen2-7B-Instruct TP2 |
| 状态 | 完成 |
| 关联清单项 | D1；D2 目标锁定；D3 目标锁定；报告第一页图 |

## 1. 目的与假设
① MoE vs dense 的 decode 吞吐-batch 曲线（GDDR6X 1008GB/s roofline frame）；② nsys kernel 级 wall-time 分解（moe_align/permute/grouped GEMM/AllReduce…），优化目标由数据定。可证伪假设：MoE 的 bs=1 权重带宽优势随 batch 增大而衰减。

## 2. 环境与配置
- 扫描：`d1_sweep.sh`——两模型同轴（输入 128/输出 256，并发 1→128 八档，每点唯一 seed，`--save-result --save-detailed`，GPU 遥测 2s 采样）。
- nsys：`d1_nsys.sh`——`--profiler-config.profiler=cuda`（/start_profile → cudaProfilerStart）+ nsys `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown` + **`--cuda-graph-trace=node`（关键，见 §7）**；负载稳态 15s 后开 20s 采集窗；bs=1 与 bs=32 各一窗（输入 128/输出 512）。

## 3. 步骤
sweep（MoE 8 点 → dense 8 点）→ nsys bs=1 → nsys bs=32 → `d1_analyze.py`（曲线+CSV）→ `d1_kernels.py`（nsys stats → 9 桶分类，未命中打印 top-5 防静默）。

## 4. 原始数据
`moe_perf/raw/EXP-014/`：16 个 bench JSON+log、两模型 server.log、GPU 遥测 CSV、`d1_nsys_moe_bs{1,32}.nsys-rep`（node 级，167MB/44MB）+ 首次踩坑的 `*_graphlevel.nsys-rep`（保留作方法学证据）、manifest(provenance)。衍生：`derived/d1_scaling.csv`、`derived/d1_kernel_share_bs{1,32}.csv`、`figures/d1_fig1_decode_scaling.png`。

## 5. 结果
**A. 吞吐-batch 曲线（输出 tok/s，p50 TPOT ms）——发现反转点**：

| 并发 | MoE tok/s | dense tok/s | MoE/dense | MoE TPOT | dense TPOT |
|---|---|---|---|---|---|
| 1 | 221 | 109 | **2.03×** | 4.40 | 9.06 |
| 4 | 538 | 398 | 1.35× | 7.21 | 9.78 |
| 8 | 726 | 748 | **0.97×（反转点）** | 9.93 | 10.26 |
| 32 | 1691 | 2094 | 0.81× | 18.33 | 14.06 |
| 128 | 3975 | 4827 | **0.82×** | 30.63 | 23.39 |

bs=1 roofline 对照：MoE 实测 221 / 理论 ~373（59%）；dense 109 / ~142（77%）。

**B. nsys kernel 占比（GPU kernel wall-time，双 rank 合计，20s 稳态窗）**：

| 分类 | bs=1 | bs=32 |
|---|---|---|
| **grouped GEMM(fused_moe routed experts)** | 18.7% | **56.4%** |
| dense GEMM/GEMV（proj/共享专家/lm_head） | **40.9%** | 14.9% |
| AllReduce(NCCL) | 13.8% | 15.0% |
| attention | 7.2% | 7.1% |
| norm/rope/act/elementwise | 8.7% | 3.7% |
| routing(topk/softmax) | 4.9% | 1.1% |
| moe_align_block_size | 4.1% | 1.0% |
| permute/unpermute/moe_sum | 0.5% | 0.3% |
| other（top 已核） | 1.2% | 0.4% |

## 6. 分析与结论
- **反转机理**：top-4/60，batch 增大 → 每 step 命中专家并集趋于全量（60 专家 ~28.6GB > dense 14.2GB），bs=1 的激活权重优势（2.7GB/step）反转为读放大劣势；分解表印证——routed experts 的 grouped GEMM 从 18.7% 膨胀到 56.4%。
- **bs=1 的另一半故事**：dense GEMV 40.9%，其中 lm_head 巨大（vocab 151936， hidden 2048 → 0.62GB 权重，TP2 vocab 切分后每 rank 每 token 仍读 0.31GB）+ 4 个共享专家 MLP——小激活 MoE 模型在 bs=1 时非 routed 部分才是大头。
- **D2/D3 目标锁定（由数据，非预设）**：serving batch(≥8)下 fused_moe grouped GEMM 是唯一大头（56.4%）→ D2 的 E=30，N=1408 config 调优正中要害；D3 若做 kernel 级优化，对象同为 fused_moe 路径；moe_align（≤4.1%）与 permute（≤0.5%）不值得优化（regime 声明：本机 TP2+EP、bs≤128）。
- AllReduce 恒 ~14-15%：TP2 的固定税，与 B1 的 allreduce 墙结论一致。

## 7. 异常、偏差与开放问题
- **方法学陷阱（已修，graphlevel 文件保留）**：默认 `--cuda-graph-trace=graph` 下 CUDA graphs 内的 decode kernel 不单列——首采只见图外的采样器/ lm_head GEMV/AllGather，bs=32 表实为 prefill 混样（fused_moe 仅 384 实例 = 4 个 prefill step）。**必须 `--cuda-graph-trace=node`**。这与"代理不转发 profile 端点"同级，入面试弹药库。
- 采集窗 kernel 时间合计（43.1s/36.3s）≈ 2 GPU × 20s 窗上限附近，含 node 级 tracing 开销；占比为窗内相对值，绝对吞吐以 sweep JSON 为准。
- 两次踩 pkill 自杀坑（复合命令 pattern 匹配自身 shell，exit 144）——HANDOFF 的方括号技巧不可省。
- replica2 形态未扫（D1 目标是 MoE vs dense 机理，不是四臂重跑）。

## 8. 下游影响
- 报告第一页图：`figures/d1_fig1_decode_scaling.png`（反转结论句标题）。
- D2：A/B 的 before 侧可直接引用本实验 bs 档；fused_moe 56.4% 占比 = "为什么调这个 config" 的一句话答案。
- S3 简历句素材："kernel 级分解定位 fused_moe grouped GEMM 在 serving batch 下占 GPU 时间 56%，据此调优…"。
- D3 目标：fused_moe 路径（config 之外的 kernel 级机会待 D2 A/B 数据再定）。
