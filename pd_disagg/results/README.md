# B1 四臂矩阵数据规范（开跑前锁定；改 schema = 新增字段，不得改语义）

```
results/b1_matrix/
├── raw/         # vllm bench 原始输出 + 引擎/代理日志（同前缀）
├── snapshots/   # 每测量点 before/after 的 /metrics 全量快照（直抓 8100/8200，不抓代理）
├── derived/     # 汇总表 CSV（可由 runs.jsonl 完整重算）
└── runs.jsonl   # 权威数据：每行 = 一个测量点
```

前缀命名：`<UTCyyyymmddThhmm>_<arm>_<in>x<out>_<mode>[_rps<r>]`
例：`20260823T0510_pd1p1d_2048x128_sweep_rps4`

## runs.jsonl 每行字段

```jsonc
{
  "run_id": "20260823T0510_pd1p1d_2048x128_sweep_rps4",
  "arm": "colocate | replica2 | tp2 | pd1p1d",
  "mode": "attribution | sweep",        // attribution=并发1 归因跑；sweep=offered-load 扫描
  "model": "Qwen/Qwen2-7B-Instruct",
  "input_len": 2048, "output_len": 128,
  "offered_rps": 4.0,                    // attribution 模式为 null
  "num_prompts": 200,
  "wall_time_s": 0.0,
  "completed": 0, "failed_requests": 0,
  "metrics": {
    "ttft_ms": {"p50": 0, "p90": 0, "p99": 0},
    "tpot_ms": {"p50": 0, "p90": 0, "p99": 0},
    "throughput_tok_s": 0,
    "goodput_slo_rps": 0,               // SLO 定义见下，全矩阵统一，锁定后不改
    "gpu_seconds_per_request": 0        // = gpu_count × wall_time_s / completed
  },
  "gpu_count": 2,                        // colocate=1，其余=2（成本口径的分母依据）
  "gates": {                             // 非 PD 臂：nixl_* 与 expired 置 null
    "kv_load_failure_policy": "fail",
    "log_stats_enabled": true,           // 即：未带 --disable-log-stats
    "metrics_scraped_direct": true,      // 直抓引擎端口
    "nixl_bytes_delta": 0,
    "nixl_transfers_delta": 0,
    "transfers_expected": 0,             // 预期远端请求数；须 == nixl_transfers_delta
    "failed_transfers": 0,               // 必须 0
    "failed_notifications": 0,           // 必须 0
    "expired_reqs_P": 0,                 // P 端必须 0
    "pass": true
  },
  "snapshot_before": ["snapshots/<前缀>_8100_before.prom", "..._8200_before.prom"],
  "snapshot_after":  ["snapshots/<前缀>_8100_after.prom",  "..._8200_after.prom"],
  "raw": ["raw/<前缀>_bench.json", "raw/<前缀>_P.log", "raw/<前缀>_D.log"],
  "provenance": {"env": "ENV-B", "sha": "752a3a5044", "version": "0.25.1",
                  "cmd": "<完整命令>", "date": "<ISO8601>",
                  "gpu": "2x RTX 4090", "driver": "610.57.04"}
}
```

## 规则

1. `gates.pass=false` 的行**保留在 runs.jsonl**（诚实记录），但绝不进 derived/、
   figures/ 与报告。
2. **SLO 定义**（goodput 用）：在首个 sweep 之前于此处填死数值并 commit——
   `TTFT ≤ [待定] ms 且 TPOT ≤ [待定] ms`。之后不得回改（防事后挑阈值）。
3. gate 增量由 `scripts/metrics_snapshot.sh diff before after` 计算，
   人工誊入 runs.jsonl 或由跑批脚本自动写入。
4. 每个测量点重复次数与 warmup 规则同样在首跑前定死，写入本文件。
5. GPU-seconds/request 的 gpu_count 口径：按**部署占用的 GPU 数**计
   （colocate=1；replica2/tp2/pd1p1d=2），不按利用率折算——报告中如实说明。

## B3 版本对照数据

同 schema，`provenance.env=ENV-A`，文件放 `results/b3_version_compare/`；
结论只能称 **system-version comparison**（scheduler/kernel/默认配置/传输方向均不同）。
