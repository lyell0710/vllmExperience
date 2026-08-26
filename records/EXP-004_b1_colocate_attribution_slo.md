# EXP-004 · B1 colocate 归因基线 + SLO 锁定

> **一句话结论**：colocate 单卡基线成立并据此锁定 SLO：bs=1 的 TPOT 恒定 ~16ms（≈63 tok/s）是 decode 的权重带宽约束，TTFT 随输入近线性是 prefill 的计算主导——两条曲线的形状本身就是归因。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（15:48–15:52Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；模型 Qwen/Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | B1（attribution， colocate 臂）；SLO 定义锁定 |

## 1. 目的与假设
建立四臂矩阵的无负载基线（并发=1 归因跑），并以其 TTFT p50 换算 SLO 绝对值（方案：TTFT≤5×基线、TPOT≤50ms 固定、附录做 SLO-scale 敏感性曲线）。

## 2. 环境与配置
- 服务：`CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2-7B-Instruct --port 8100 --max-model-len 16384`（其余默认；未用 enforce-eager，CUDA graphs 生效）
- 客户端：`scripts/run_point.sh colocate attribution <in> 128 - 8100 8100`（= vllm bench serve，random 数据集，num-prompts 32，seed 42，ignore-eos， max-concurrency 1，percentiles 50/90/99，--save-result --save-detailed）

## 3. 步骤
起服务（84s 就绪）→ 依次 512/2048/8192 三点 → 停服务。每点自动：before 快照 → bench → after 快照 → collect_point 追加 runs.jsonl。

## 4. 原始数据
`results/b1_matrix/runs.jsonl` 第 1–3 行； `raw/20260821T15{48,49,51}_colocate_*_bench.{json,log}`；`snapshots/` 同前缀；服务日志 `raw/colocate_server.log`。注：本臂三点跑在 GPU 遥测工装加入之前（遥测自 EXP-005《replica2/tp2 归因 + 功率帽节流调查》起），无 gpu.csv。

## 5. 结果
| 输入桶 | TTFT p50/p90/p99 (ms) | TPOT p50 (ms) | GPU·s/req | → TTFT SLO(5×) |
|---|---|---|---|---|
| 512  | 65.5 / 66.4 / 80.5 | 15.87 | 2.08 | 328 |
| 2048 | 178.3 / 181.6 / 186.2 | 15.93 | 2.20 | 891 |
| 8192 | 925.2 / 946.0 / 950.8 | 16.34 | 2.95 | 4626 |

## 6. 分析与结论
- bs=1 TPOT 恒定 ~16ms（≈63 tok/s）——decode 权重带宽约束；TTFT 随输入近线性——prefill 计算主导。
- SLO 表以本实验 p50 换算并 commit 锁定（results/README.md），此后不回改。

## 7. 异常、偏差与开放问题
- **8K 桶 TTFT 分布双段**（前 ~8 请求 702–739ms，其后 897–951ms）：当时未察觉， EXP-005 溯源为功率帽节流 → 本臂 8K p50=925 实为冷→稳态混合，稳态约 905ms（EXP-005 diag-3 证实）。SLO 维持锁定值（5× 余量 ≫ 30% 效应，且换基线=回改）。

## 8. 下游影响
SLO 锁定；colocate 基线成为其余三臂的对照；双段分布触发 EXP-005 调查与遥测工装。
