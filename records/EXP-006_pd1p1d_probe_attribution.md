# EXP-006 · pd1p1d 指标探针 + 归因 + NIXL 大传输实测

> **一句话结论**：NIXL KV 通路的有效吞吐**跨尺寸恒定在 0.26–0.27 GB/s**，原因是 descriptor 只有 ≈16KB（每 block 每层单发）造成的碎片化小拷贝——量级与 EXP-002 测到的无 P2P 单向路径吻合。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（16:49–16:58Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；模型 Qwen/Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | B1（attribution， pd1p1d 臂）；R0-1 NIXL 大传输收尾；B2 前置 |

## 1. 目的与假设
① 摸清 v0.25.1 NIXL prometheus 指标名与暴露侧（P/D），固化 gate 判定； ② pd1p1d 归因基线；③ 大传输下的 NIXL 实测（R0-1 第三数收尾）。

## 2. 环境与配置
- P：`CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 vllm serve Qwen/Qwen2-7B-Instruct --port 8100 --max-model-len 16384 --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'`
- D：GPU1:8200 / side 5601 / kv_consumer，其余同上
- proxy：`smoke/toy_proxy_v0251.py --port 8192 --prefiller-ports 8100 --decoder-ports 8200`
- **未用 enforce-eager，CUDA graphs 与 NIXL 共存正常**（相对 smoke 配置的升级）
- bench 打 8192，快照直抓 8100+8200

## 3. 步骤
起 P/D/proxy → 探针（快照→单请求→快照→diff）→ 按探针结果改 collect_point.py（精确指标名 + 字段扩展）→ 三点归因 → 提取 8K 遥测。

## 4. 原始数据
- 探针快照：`results/b1_matrix/snapshots/exp006_probe_{8100,8200}_{before,after}.prom`
- `results/b1_matrix/runs.jsonl` 第 10–12 行；`raw/`、`snapshots/` 同前缀 + `raw/pd_{P,D,proxy}.log`

## 5. 结果
**探针（指标体系）**：传输计数全在 **D 端**（Pull 语义）： `vllm:nixl_bytes_transferred_{sum,count}`、`nixl_xfer_time_seconds_sum`、 `nixl_post_time_seconds_sum`、`nixl_num_descriptors_sum`、 `prompt_tokens_by_source_total{source="external_kv_transfer"}`（逐 token 记账远端 KV）； P 端仅 `nixl_num_failed_{transfers,notifications}_total`、`nixl_num_kv_expired_reqs_total`； `_created` 系列是时间戳非计数器。单请求（~9 token prompt）：bytes=917504 = 16 tokens×57344B（block=16 取整），ext_kv_tokens=8（=9-1，D 自算最后一 token）。

**归因（p50，32 请求/点）**：
| 输入桶 | TTFT (ms) | TPOT (ms) | GPU·s/req |
|---|---|---|---|
| 512 | 214.4 | 15.86 | 4.46 |
| 2048 | 554.6 | 15.93 | 5.16 |
| 8192 | 2685.4 | 16.35 | 9.24 |

**NIXL 遥测（gate 字段增量，全部 gate PASS：failed/expired=0，transfers=32=completed）**：
| 桶 | bytes 总 | MB/xfer | xfer_sum | avg xfer | post 总 | desc/xfer | ext_kv_tok | 有效吞吐 |
|---|---|---|---|---|---|---|---|---|
| 512 | 0.940GB | 29.4 | 3.65s | 113.9ms | 121ms | 1792 | 16384 | 0.26GB/s |
| 2048 | 2.820GB | 88.1 | 10.57s | 330.2ms | 142ms | 5380 | 49184 | 0.27GB/s |
| 8192 | 14.069GB | 439.7 | 51.29s | 1602.7ms | 714ms | 26834 | 245344 | 0.27GB/s |

## 6. 分析与结论
- **有效吞吐 0.26–0.27GB/s 跨尺寸恒定**：descriptor ≈16KB/个（=每 block 每层单发，56 desc/block=28 层×K，V）→ 碎片化小拷贝，量级与 EXP-002《硬件三数》单向无 P2P 路径一致。措辞红线：只可称 telemetry-derived effective throughput， xfer 时间不与 post 相加（post 已含）。
- PD TTFT 分量对账：8K 的 2685 ≈ P prefill（~900，热态） + xfer(1603) + D 首步/代理—— 吻合。
- gate 判定已按精确指标名固化进 collect_point.py（跨端口求和；含 failed_notifications 与 external_kv_tokens 新字段）。

## 7. 异常、偏差与开放问题
- **开放问题 → B2**：D 端实拉 7668 token/req（<8192；bytes 439.7MB/57344B=7668 与 ext_kv_tokens 245344/32=7667 两计数器独立互证）。疑与 block 取整/前缀缓存/ 最后 block 自算的记账规则相关，需读 D 端调度与 nixl connector 代码定论。在定论前，KV 量引用一律用 bytes 实测值，不用"input_len×57344"推算值。

## 8. 下游影响
- R0-1 三数全部收尾（DECISION.md 硬件基线含大传输行）。
- gate 机械化完成 → sweep 阶段 PD 臂可全自动判 PASS/FAIL。
- 简历素材：S1 子结论②；B2 的 request 级归因从 xfer 直方图桶 + 开放问题切入。
