# EXP-005 · replica2/tp2 归因 + 功率帽节流调查

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（16:24–16:45Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；模型 Qwen/Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | B1(attribution, replica2/tp2 臂)；方法论修正 |

## 1. 目的与假设
replica2 与 tp2 两臂归因基线。调查分支的假设（数据倒逼产生）：
"replica2@8K 快于 colocate 是单卡持续负载下的降频所致"。

## 2. 环境与配置
- replica2：两个单卡实例（GPU0:8100 / GPU1:8200，配置同 EXP-004）+
  `matrix/rr_proxy.py --port 8300 --backends 127.0.0.1:8100 127.0.0.1:8200`；
  bench 打 8300，快照直抓 8100/8200
- tp2：`CUDA_VISIBLE_DEVICES=0,1 vllm serve ... --tensor-parallel-size 2 --port 8100`
- 客户端同 EXP-004（run_point.sh，32 请求/点，并发 1）

## 3. 步骤
replica2 三点 → 诊断跑 ×3（见下）→ 遥测采样验证 → 工装加遥测 → 换 tp2 三点。

## 4. 原始数据
- `results/b1_matrix/runs.jsonl` 第 4–9 行；`raw/`、`snapshots/` 同前缀；
  服务日志 `raw/replica2_server_{8100,8200}.log`、`raw/replica2_proxy.log`、`raw/tp2_server.log`
- 节流采样：`records/data/EXP-005_throttle_trace.csv`（1.5s×40 轮，双卡，
  列=index,temp,sm_clock,power,throttle_reason）
- **诊断跑 3 次未走 run_point 工装、未存 raw json**（证据等级：终端输出，
  完整命令与数字记录于下；由此新增卫生规则见 §8）：
  - diag-1 直连 8100：`vllm bench serve --port 8100 --random-input-len 8192
    --random-output-len 128 --num-prompts 16 --seed 42 --max-concurrency 1 --ignore-eos`
    → TTFT p50 **901.19** / p90 928.96 ms
  - diag-2 直连 8200（同参数）→ p50 **892.69** / p90 912.07 ms，TPOT p50 16.35
  - diag-3 持续负载 + 采样：`--random-output-len 16 --num-prompts 40 --seed 7` @8100
    → TTFT mean 904.41 / p50 **905.61** / p90 919.80 ms（= 稳态值）

## 5. 结果
| arm | TTFT@512 | @2048 | @8192 (p50 ms) | TPOT p50 | GPU·s/req@8K |
|---|---|---|---|---|---|
| replica2 | 65.2 | 173.5 | 714.6 | 15.87–16.35 | 5.58 |
| tp2 | 62.8 | 173.5 | 693.7 | **9.26–9.48** | 3.79 |

节流采样（GPU0，diag-3 期间）：空闲 210MHz/14W/0x1 → 负载 40→63°C、
427–443W（帽 450W）、SM 2820↔2460–2535MHz、**节流原因 0x4 = SW Power Cap**。

## 6. 分析与结论
- **发现① 功率帽节流**：colocate@8K 双段分布（EXP-004 §7）+ diag-1/2/3 全部 ~900ms
  （卡已热）+ 采样坐实 SW Power Cap → 持续 prefill 稳态 TTFT ≈905ms，
  冷启 boost 段 ≈700ms；replica2 轮转=50% 占空比维持 boost → 715ms。
  温度 63°C 排除热因。频率降 ~12% 与 TTFT +30% 不完全成比例，
  差额疑与瞬时 boost/显存时钟相关（未深究，非主线）。
- **发现② TP2 不对称收益**：decode 16→9.3ms（-42%，每卡半份权重 + 小消息
  allreduce ~1.3ms/token 代价）；8K prefill 零加速（694≈冷态单卡 700ms）——
  28 层 × 58.7MB 大消息 allreduce 受 1.78GB/s collective 带宽约束（EXP-002 印证）。

## 7. 异常、偏差与开放问题
- 诊断跑未存 raw（当时为快速排障）→ 已定规则杜绝（§8）。
- 方法论决定：attribution 各臂数字=各自占空比工况，**headline 以 sweep 为准**
  （满负载下各臂同为持续态）；attribution 表引用时必须带工况标注。

## 8. 下游影响
- 工装：run_point.sh 加 GPU 遥测（2s 采样 → runs.jsonl `gpu_telemetry`），
  自 tp2 三点起每点自带工况证据。
- 卫生规则（README 约定 #8）：**任何 GPU 跑（含诊断/临时）一律存 raw**，
  否则数字降级为终端级证据并必须在实验记录中注明。
- 简历素材：发现①②均为"异常→拆解→机理"完整链条（S1 归因子结论）。
