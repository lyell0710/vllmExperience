# EXP-002 · 硬件三数（R0-1 硬件画像）

> **一句话结论**：硬件画像三个数把后面所有归因钉死：单向 0.6–0.9 GB/s vs 双向 22.7 GB/s，**相差 25 倍就是「无 P2P」的定量指纹**；collective 只有 1.78 GB/s，直接预言了后来 TP2 的 prefill 零加速。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（15:04–15:07Z 主体；NIXL 大传输补测见 EXP-006） |
| 环境 | 与 vllm 无关的系统级测量（driver 610.57.04， CUDA 13.2， NCCL 2.19.7） |
| 状态 | 完成 |
| 关联清单项 | R0-1；解锁红线"P2P 受限" |

## 1. 目的与假设
量化 2×4090 的三条互联路径：GPU 间裸拷贝（p2p 路径）、TP collective 路径（NCCL allreduce）、KV 通路（NIXL，见 EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》）——为矩阵结果提供因果基线。

## 2. 环境与配置
- p2pBandwidthLatencyTest：cuda-samples（tool_sha b7c5481c55…，构建于 /root/tools/cuda-samples， sparse checkout `cpp/5_Domain_Specific/p2pBandwidthLatencyTest` + 根 `cmake/`）
- all_reduce_perf：nccl-tests（tool_sha 717b683182，MPI=0；NCCL 取自 ENV-B venv 的 nvidia-nccl wheel，nccl-home 软链方案见 hw 文件 provenance）
- 命令：`all_reduce_perf -b 1M -e 512M -f 2 -g 2`

## 3. 步骤
构建与执行命令完整记录在各结果文件的 provenance 行。

## 4. 原始数据
`pd_disagg/hw/topo.txt`、`hw/p2p_bandwidth_latency.txt`、`hw/all_reduce_perf.txt`。

## 5. 结果
- P2P：connectivity matrix=0；`topo -p2p r` = **GNS（GPU not supported，驱动级禁用）**
- 单向 D2D **0.60–0.91 GB/s**；双向 **22.6–22.8 GB/s**；GPU 间延迟 14.5–15.9µs；本卡内 memcpy ~924 GB/s
- NCCL allreduce（SHM 回退）：**avg bus bw 1.78 GB/s**，256M+ 消息 ~1.85 GB/s
- PCIe：双卡 Gen4 x16（空闲降 Gen1）；bus C1/E1

## 6. 分析与结论
- 单向 0.6–0.9 GB/s 是 cudaMemcpyPeer 无 P2P 时分段中转路径；双向 22.7 GB/s 接近 Gen4 x16 双向流水极限——单双向差 25 倍是"无 P2P"的定量指纹。
- 1.78 GB/s collective 直接预言了 TP2 prefill 零加速（EXP-005《replica2/tp2 归因 + 功率帽节流调查》证实）。
- 标注：allreduce 数字仅代表 TP collective 路径，不代表 KV 通路（红线要求）。

## 7. 异常、偏差与开放问题
- `nvidia-smi topo -m` 在本容器不可用（hwloc 读不到 PU），以 `topo -p2p r`+PCIe link 替代，已在 topo.txt 注明。
- 单向 D2D 三个矩阵单元数值分散（0.60/0.69/0.91/4.36），与测试内部分段策略有关，引用时用区间不用单值。
- 〔勘注 2026-08-29〕**allreduce 带宽复测差异**：EXP-018《NCCL allreduce size 扫描》用同二进制（tool_sha 717b683182）、同 NCCL 2.28.9、同参数复测 1M–512M 区间，得 avg bus bw **6.20 GB/s**（平台 6.4–6.5），比本记录 1.78 GB/s 高 3.5 倍。本记录的 provenance 未记 PCIe 运行态与系统负载，无法回溯归因（候选：8/21 并发负载 / PCIe 未升 Gen4 / 走了 Socket 网络路径）。〔2026-08-30 闭环：EXP-019《1.78 vs 6.2 GB/s 机制调查》判定为真实环境差异、非计时口径问题；复现实验设计见 EXP-019 §8。〕**教训（本记录违反铁律 4 精神）**：硬件测量的 provenance 必须记录 NCCL 环境变量、PCIe 运行态、NCCL_DEBUG 日志——本记录只写了 `env=n/a sha=n/a`，导致 8 天后无法定因。权威数字在复现 1.78 前不作对外引用。

## 8. 下游影响
红线"P2P 受限"解锁；三数进所有报告的硬件画像段；为 EXP-005/006 的机理解释提供基线。
