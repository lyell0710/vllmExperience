# EXP-010 · C3 Qwen3-30B-A3B W4A16 上卡

> **一句话结论**：Qwen3-30B-A3B 的 W4A16 checkpoint 格式锁定、上卡可用、基线两点在案，为后续的吞吐-精度对比（D4）与 nsys kernel 分解（D1）备好对象。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（21:20–21:28Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1） |
| 状态 | 完成 |
| 关联清单项 | C3（D4 前置） |

## 1. 目的与假设
锁定具体 checkpoint + 量化格式并上卡（清单红线：AWQ/GPTQ/AutoRound 不许混称）。

## 2. 环境与配置
- **checkpoint 锁定**：`Qwen/Qwen3-30B-A3B-GPTQ-Int4`（Qwen 官方 GPTQ Int4 发行版， ~16.5GB）。格式口径：**GPTQ Int4 = W4A16**（权重 4bit、激活 16bit）， Ada 上经 **MarlinLinearKernel** 执行（日志确认）。
- `CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen3-30B-A3B-GPTQ-Int4 --port 8100 --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel --gpu-memory-utilization 0.88`；启动 ~184s。

## 3. 步骤
下载（后台，~16.5GB）→ 上卡 → smoke → attribution/saturation 各一点（512 桶）。

## 4. 原始数据
raw/c3_w4a16_server.log；runs.jsonl arm=moe30b_w4a16_tp2ep 两行。

## 5. 结果
- smoke 输出连贯（"The capital of France is → Paris. The capital of Italy is Rome"）。
- attribution 512：TTFT p50 71.5ms / TPOT p50 **4.93ms**；saturation 512： **10.02 req/s / 1283 tok/s 输出**。
- 与 A2.7B MoE（BF16, 4.62ms）与 dense 7B TP2（9.26ms）同框：30B 总参、~3B 激活
  + W4A16 的 decode 速度与 2.7B 激活 BF16 相当——权重读取量近似（3B×0.5B/参 ≈ 2.7B×2B/参 的量级差被专家路由与 Marlin 反量化开销平衡， D1/D4 分解的对象）。
- 未出现 fused_moe "config not found" 告警（与 A2.7B 不同）——该 shape（E=128/TP2+EP→每 rank 64， N=768）的 kernel 选择路径待 D 阶段核实（量化 MoE 走 marlin moe 路径，config 机制可能不同）；不作断言。

## 6. 分析与结论
C3 达成：checkpoint/格式锁定、上卡可用、基线两点在案。D4（吞吐-精度对比）与 D1（nsys 分解）的对象就绪。

## 7. 异常、偏差与开放问题
- 磁盘余 12GB——D4 若需 FP8/BF16 对照版本需先清理（Qwen1.5-MoE-Chat 29GB 可作候补清理对象，或挂载扩容）。
- 30B 的 MoE config 告警缺席原因未定论（见 §5）。

## 8. 下游影响
D4 前置完成；S4 可选句的素材通道打开。
