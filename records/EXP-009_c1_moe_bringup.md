# EXP-009 · C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据

> **一句话结论**：Qwen1.5-MoE-A2.7B 在 TP2+EP 路径下上卡可用，并留下默认 config 的基线数字——D2 调优的 A/B 框架由此就绪。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（20:53–21:01Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；Qwen/Qwen1.5-MoE-A2.7B-Chat（BF16, ~28.6GB） |
| 状态 | 完成 |
| 关联清单项 | C1；C2 运行时证据；D1/D2 基线 |

## 1. 目的与假设
MoE 模型在 TP=2 + expert parallel 下上卡可用；顺带抓取 fused_moe 缺失 config 的运行时告警（C2 空缺的第一手证据）与未调优基线（D2 的 before 数字）。

## 2. 环境与配置
`CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --port 8100 --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel --gpu-memory-utilization 0.88`。启动 ~216s（含 torch.compile AOT 缓存构建）。

## 3. 步骤
上卡 → 抓告警 → smoke completion → attribution（512/2048，并发 1）+ saturation（512）。

## 4. 原始数据
raw/c1_moe_server.log（含告警原文）；runs.jsonl arm=moe_tp2ep 三有效行（另有两行 MODEL 未设导致 404 的失败行，保留，gate_pass=false，教训见 §7）。

## 5. 结果
- **C2 运行时铁证**（日志原文）： `WARNING [fused_moe.py:1106] Using default MoE config. Performance might be sub-optimal! Config file not found at .../fused_moe/configs/ E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json`——目标 tuple 与 C2 判定完全一致（TP2+EP ⇒ 每 rank 30 专家、N=1408）。
- smoke 通过（0-shot 补全正常）。
- **未调优基线**（默认 config）：
  | 点 | 数值 |
  |---|---|
  | attribution 512 TTFT p50 / TPOT p50 | 56.0ms / **4.62ms** |
  | attribution 2048 TTFT p50 / TPOT p50 | 149.7ms / 5.66ms |
  | saturation 512 | 11.50 req/s / 1472.6 tok/s 输出 |
- 对照 dense（同为 TP2）：Qwen2-7B TPOT 9.26ms → **MoE 激活 2.7B 使 bs=1 decode 快 2.0×**；饱和吞吐 11.50 vs 12.31 req/s（512）——批量化后 MoE 的权重读取优势被专家分发/路由开销部分抵消（D1 的分解对象）。

## 6. 分析与结论
TP2+EP 路径可用；D2 调优的 A/B 框架就绪（before=默认 config 的上表数字）。

## 7. 异常、偏差与开放问题
- 首轮 bench 全 404：run_point.sh 的 MODEL 默认值仍是 Qwen2-7B-Instruct，对 MoE 服务请求了错误模型名（vLLM 对未知模型回 404）。教训：**多模型阶段 MODEL 必须显式设置**；失败行按规则保留。
- 2048 桶 saturation 与 8192 桶未跑（C1 只要求上卡；D1 网格另行设计）。

## 8. 下游影响
C1 ✅；C2 证据链三重闭环（本地目录判定 + 远端查重 + 运行时告警）； D2 的 benchmark_moe.py 调优可以随时开工（baseline 在案）。
