# EXP-001 · NIXL 1P1D smoke 与版本裁决

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（上午，~09:03–09:15Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）与 main（7aa248fcfe, 0.26.1rc1.dev682）双跑 |
| 状态 | 完成 |
| 关联清单项 | R0-3；裁决锁定 ENV-B 主战场 |

## 1. 目的与假设
假设：v0.25.1 的同机双卡 NIXL Pull 路径在 2×4090 上可用。附带：按预定规则（三项检查、平手取 release）裁决四臂矩阵用 v0.25.1 还是 main。

## 2. 环境与配置
- 模型 Qwen/Qwen2.5-0.5B-Instruct；P=GPU0:8100(side 5600)、D=GPU1:8200(side 5601)
- `--enforce-eager --max-model-len 2048 --gpu-memory-utilization 0.7`
- `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer|kv_consumer","kv_load_failure_policy":"fail"}'`
- toy proxy 端口 8192（`smoke/toy_proxy_v0251.py`）

## 3. 步骤
`smoke/run_1p1d_smoke.sh <venv> <label> <proxy>`（脚本即完整步骤），每版本一遍。

## 4. 原始数据
`pd_disagg/smoke/smoke_{v0.25.1,main}_{result.txt,P.log,D.log,proxy.log}`（result 首行 provenance）。注：result 里 provenance 的 sha 记的是当时 /root/vllm checkout HEAD（7aa248fc），对 v0.25.1 wheel 应为 752a3a50—— 已在 scripts/provenance.sh 修正语义，历史文件不改。

## 5. 结果
| 检查项 | v0.25.1 | main |
|---|---|---|
| 1P1D 请求 | PASS (2/2) | PASS (2/2) |
| KV Transfer metrics | PASS，avg xfer 14.128ms / 0.188MB / 13.27MB/s / desc=24 / post 0.911ms | PASS，avg xfer 14.7ms |
| failure_policy=fail | PASS（且为默认值） | PASS（且为默认值） |
| 日志 ERROR | 0 | 0 |

## 6. 分析与结论
三项平手 → 按预定规则取 release：**v0.25.1 锁定为主战场**（DECISION.md）。小传输（0.188MB）下 xfer 14ms 为延迟主导，不代表带宽（大传输实测见 EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》）。

## 7. 异常、偏差与开放问题
- setup_envs.log 末尾的 "v0.17.1 import 失败/版本号错" 为 cwd 遮蔽假故障（在 /root/vllm 源码目录裸跑 python），各 venv 经中立目录复验全部健康。

## 8. 下游影响
ENV-B 锁定；smoke 数字进硬件画像"NIXL 初值"；provenance sha 语义规则确立。
