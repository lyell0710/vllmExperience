# EXP-008 · B3 有限版本对照（v0.17.1 vs v0.25.1 单实例）

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（20:39–20:52Z） |
| 环境 | ENV-A（PyPI 0.17.1, torch 2.10.0+cu130）vs ENV-B colocate v2 行 |
| 状态 | 完成（有限范围：单实例；PD-vs-PD 对照因 R0-4 课程脚本缺席而后置） |
| 关联清单项 | B3 |

## 1. 目的与假设
同 workload、同协议下两个相隔约 8 个月的 vLLM 版本的 system-version comparison。
**红线**：不做因果归因到单个组件（scheduler/kernel/默认配置皆变），只报系统级差异。

## 2. 环境与配置
- 服务：`/root/venvs/v0.17.1/bin/vllm serve Qwen/Qwen2-7B-Instruct --port 8100
  --max-model-len 16384`（GPU0，其余默认）；客户端用 ENV-B 的 vllm bench
  （OpenAI 兼容，客户端版本不影响服务端测量）。
- 协议 v2 同 seed（x042/x099），arm=colocate_v0171 入 runs.jsonl（provenance
  env=ENV-A, sha=n/a-pypi-0.17.1）。

## 3. 步骤
attribution×3（512/2048/8192，并发 1）+ saturation×3，与 ENV-B colocate v2 同协议。

## 4. 原始数据
runs.jsonl arm=colocate_v0171 六行(在 b1_matrix 下,B3 未另设目录);
raw/ 内同前缀 bench json/log/gpu csv 18 件 + snapshots 12 件。
**勘误(8/23 审计)**:v0171 的 server 日志未保留(其余臂均有),该臂服务端
证据等级降为"终端级";结论数字全部来自 bench raw 与 /metrics 快照,不受影响。

## 5. 结果
| 指标 | v0.17.1 | v0.25.1 | Δ |
|---|---|---|---|
| TTFT p50 512/2048/8192 (ms) | 66.4 / 225.4 / 929.6 | 65.4 / 224.9 / 925.2 | <1% |
| TPOT p50 (ms) | 16.00 | 15.87 | <1% |
| 饱和吞吐 512 (req/s) | **7.14** | **10.36** | **+45%** |
| 饱和吞吐 2048 | 3.59 | 3.63 | ~0 |
| 饱和吞吐 8192 | 0.89 | 0.90 | ~0 |
| 服务启动时间 | ~308s | ~58s | -81% |

## 6. 分析与结论（system-version comparison 口径）
- 无负载延迟与 decode 速度八个月间几乎不变（同为权重带宽/计算约束，物理上限未动）。
- **短请求饱和吞吐 +45%**：收益集中在每请求开销敏感的 regime；计算受限桶零差异。
  不归因到具体组件（版本间 scheduler/API server/默认参数均变）。
- 启动时间 308→58s 是显著的工程体验差异（附带观察）。

## 7. 异常、偏差与开放问题
- 本对照非 PD-vs-PD（0.17.1 P2pNccl 1P1D 需课程 proxy 脚本，R0-4 到位后可补）；
  P2pNccl 机理层面对照见 analysis/p2pnccl_bugs_id_chain.md（静态）。
- profiler 接口差异（env var→CLI）已在 EXP-003《profiling 工装验证》记录，为本对照的接口演化补充实例。

## 8. 下游影响
B4 报告第 2 段素材（演化叙事的性能维度）；S2 支撑。
