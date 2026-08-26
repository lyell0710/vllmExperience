# 主线实验计划（修订版 v2 · 2026-08-31 报告）

> ⚠ 本文为计划文档（v2 锚点）；执行进展与最终结论见 REPORT.md（v2 定稿）、 HANDOFF.md 与 LEDGER.md 台账；其中"main=交付"段已被 DECISION.md 取代。

> 本版按评审意见逐条修订；所有引用均针对当前 checkout `main@7aa248fcfe` 核验。结论：主线判断成立。定位为：**0.17.1 = 两小时历史基线；main（当前 checkout）= 新版交付；失败分析（带 trace）= 兜底。**

---

## 0. 事实核验摘要（评审意见逐条证据）

| # | 评审意见 | 结论 | 证据 |
|---|---------|------|------|
| 1 | 不写「0.25.1/main」；main 与 v0.25.1 不是同一线性快照 | ✅ 成立 | current=`main@7aa248fcfe`，`git describe`=`v0.26.1rc0-682-g7aa248fcfe`；`v0.25.1@752a3a5044` 不是 `main` 的祖先 |
| 2 | 仓库无 `.venv/bin/python`；规则禁止裸 `python` | ✅ 成立 | `.venv/bin/python` 不存在 |
| 3 | 新版入口 = 官方 smoke test（同机双卡 NIXL 1P1D） | ✅ 成立 | `docs/features/nixl_connector_usage.md:66` 已写明 GPU0/GPU1、5600/5601 side-channel、toy proxy |
| 4 | 默认 `NixlConnector` = Pull 兼容别名；Push 留扩展点 | ✅ 成立 | `connector.py:391` `NixlConnector = NixlPullConnector` |
| 5 |「NIXL 不通换 LMCache」不成立；LMCache 示例即 LMCache-over-NIXL | ✅ 成立 | `examples/disaggregated/lmcache/README.md:32`（"using NIXL on a single node"） |
| 6 | Mooncake 直连不需要独立 master | ✅ 成立 | `docs/features/mooncake_connector_usage.md:19` P/D 直接 `vllm serve`；需 master 的是 `MooncakeStoreConnector` |
| 7 | P2pNccl 删除 = #44854（2026-06-08），v0.25.1 已包含；动机不是单一 bug | ✅ 成立 | 删除提交 `5add018beb` 时间为 2026-06-08，且是 `v0.25.1` 的祖先 |
| 8 | 0.17.1 P2pNccl 用 `request_id#layer` 当跨实例 key | ✅ 成立 | `v0.17.1:vllm/.../p2p/p2p_nccl_connector.py:220` `request_id + "#" + layer_name` |
| 9 | 新版 NIXL Pull 显式传远端 id/descriptor | ✅ 成立 | `pull_scheduler.py:269` 返回 `remote_engine_id/remote_request_id/remote_block_ids`；`metadata.py:270` `RemoteMeta(block_ids, engine_id, request_id, host, port)` |
| 10 | 单卡混部 vs 1P1D/TP=2 资源不等价 | ✅ 成立 | 1 GPU vs 2 GPU |
| 11 | NIXL 是异步传输，聚合 xfer time 除 TTFT 无因果意义 | ✅ 成立 | `nixl_connector_usage.md:422` 起 metrics 表（xfer 含 post＋数据移动） |
| 12 | 代理不转发 profile 端点 | ✅ 成立 | `tests/v1/kv_connector/nixl_integration/toy_proxy_server.py` 无 `/start_profile`/`/stop_profile` |
| 13 | all_reduce_perf 量 TP collective，非 UCX KV copy | ✅ 成立 | 需补 NIXL 实际 bytes/xfer time |
| 14 | 已有两个 4090 FP8 config | ✅ 成立 | `fused_moe/configs/`：`E=8,N=3584` 与 `E=64,N=640`（均 `dtype=fp8_w8a8`） |
| 15 | Qwen1.5 TP2+EP → E=30，N=1408；TP2 非 EP → E=60，N=704 | ✅ 成立 | 两 shape 在 `fused_moe/configs/` 缺失（`E=30` 无、`E=60` 仅 AMD）→ 调优生成 |
| 16 | `benchmark_moe.py` 支持写 config | ✅ 成立 | `benchmarks/kernels/benchmark_moe.py:697` `def save_configs(...)` |
| 17 | EPLB 默认约 3000 engine steps 才重排 | ✅ 成立 | `vllm/config/parallel.py:65` `step_interval=3000`（window=1000） |

---

## 1. 版本与基线

- **不做「0.25.1/main」对比**。`main@7aa248fcfe`（`v0.26.1rc0-682-g7aa248fcfe`）与 `v0.25.1@752a3a5044` 不是同一线性快照。
- 矩阵：
  - `main@7aa248fcfe`：**完整矩阵**（交付对象）。
  - `v0.25.1@752a3a5044`：**单点 sanity**（确认 NIXL 链路含 v0.25.1 内已删除 P2pNccl 的事实不影响结论）。
  - `v0.17.1`：**两小时历史基线**。
- 环境：每个版本独立目录 + 独立 `.venv`。按仓库规则全部走 `uv`，**禁止裸 `python`/`pip`**。
- 每个版本记录：`git SHA` + `__version__` + `__file__` + `command -v vllm`。

## 2. 功能入口（新版）

- 直接采用**官方 smoke test**：`docs/features/nixl_connector_usage.md:66` 同机双卡 NIXL 1P1D 示例（GPU0/GPU1，`VLLM_NIXL_SIDE_CHANNEL_PORT` 5600/5601，toy proxy 8192）。
- 基本路径用默认 `NixlConnector`（= Pull 兼容别名，`connector.py:391`）。Push 作为扩展点记录即可。
- **兜底链**：NIXL/UCX 安装或链路失败 → **带 trace 的失败分析**，不切换 LMCache/Mooncake：
  - 仓库 LMCache PD 示例本身就是 **LMCache-over-NIXL**（`examples/disaggregated/lmcache/README.md:32`），只作上层实现对照。
  - 直连 `MooncakeConnector` 的 P/D/proxy **不需要独立 master**（`mooncake_connector_usage.md:19`）；跳过理由 = 控制范围 + 额外 transfer-engine/RDMA 注册依赖。

## 3. 版本演化叙述（面试用，定稿三句）

> 0.17.1 的 P2pNccl 把 engine-local `request_id#layer` 当跨实例 rendezvous key（`v0.17.1:.../p2p_nccl_connector.py:220`），P/D 独立随机后缀会让 key 分叉并挂死。上游后来是在替代 connector（NIXL/Mooncake）和插件 API 成熟后收敛维护面（删除 [PR #44854](https://github.com/vllm-project/vllm/pull/44854)，2026-06-08；前期弃用讨论见 [RFC #33115](https://github.com/vllm-project/vllm/issues/33115)），而不是只为修这个 bug。当前 NIXL Pull 显式传递 `remote_engine_id/remote_request_id/remote_block_ids`（`pull_scheduler.py:269`、`metadata.py:270`），用远端内存 descriptor 寻址，把 local request ID 限定在本地异步记账中。

- **措辞纪律**：说「拆分并显式映射身份」，**不说「新版不靠 request_id」**。

## 4. 实验设计

### 4.1 资源等价
- 单卡混部（1 GPU）与 1P1D/TP=2（2 GPU）**资源不等价**。
- 对照组：补一组「**2×独立 TP1 replica**」；或全表统一报 `GPU-seconds/request`。
- 收紧结论：除非资源等价，否则不直接下「1P1D 更优」。

### 4.2 实验两类（不再只有固定 10 RPS）
1. **transfer 归因类**：低负载 / concurrency=1，抓 transfer 归因。
2. **容量类**：offered-load sweep（如 2/4/8/16/32 RPS），报 **p50/p90/p99 + SLO goodput**。
- 固定 10 RPS 只是其中一个点。

### 4.3 输入长度
- 明确为 **512 / 2K / 8K in，128 out**（消除脚注 512in 的自相矛盾）。

### 4.4 版本对比属性
- 0.17.1 → main 的 TTFT 差只叫 **system-version comparison**（scheduler、kernel、默认配置均变）。
- 传输方向不同：旧 P2pNccl 为 Push（`PUT_ASYNC` 向），新默认 NIXL Pull 为 Read 向，需注明。
- 近似同构比较时：补一个 `NixlPush` 单点。

### 4.5 指标与基准
- **不写「KV transfer 占 TTFT 百分比」**（NIXL 异步传输，聚合 xfer time ÷ TTFT 无因果意义）。
- 先报原生 `xfer/post time`、`bytes`、`descriptor 数`、`失败数`、`有效带宽`（metrics 字段见 `nixl_connector_usage.md:422` 起）。
- 百分比只从 **request 级 trace 的关键路径**计算。
- 基准固定 **`kv_load_failure_policy=fail`**，避免失败后重算伪装成正常请求。
- 硬件数：`all_reduce_perf` 是 TP collective，**不是** UCX KV copy；三组数中补 NIXL 实际 `bytes/xfer time`。

### 4.6 profiling（与正式性能分跑分开）
- Torch profiler：只抓**少量代表请求**。
- nsys：**分别包 P/D**。
- 代理不转发 `/start_profile`、`/stop_profile`（`toy_proxy_server.py` 无该端点），`bench --profile` 打代理不会自动同时抓两端；**脚本分别控制 8100/8200**（见 `docs/contributing/profiling.md:193`）。

---

## 5. MoE 部分

- **4090 并非「完全空白」**：已有 `E=8,N=3584` 与 `E=64,N=640` 两个 RTX_4090 FP8 config（`fused_moe/configs/`）。
- Qwen1.5 BF16 目标 shape：
  - **TP2+EP** → `E=30, N=1408`（含冗余/EP 语义）。
  - **TP2 非 EP** → `E=60, N=704`。
  - 两个 shape 当前 **缺失**，用调优脚本生成（`benchmarks/kernels/benchmark_moe.py:697` `save_configs` 支持写 config）。
- **config 文件缺失 ≠ PR 自动成立**：仓库禁止低价值单文件 PR。至少需要：correctness、kernel 前后、端到端收益、模型 eval、查重结果；否则只作为**报告数据**，不开 PR。
- **EPLB 30 分钟 gate ≠ 只是「能启动」**：默认 `step_interval=3000`（`vllm/config/parallel.py:65`）才重排。需**临时缩短 window/interval**，确认真实 rearrangement，再验证重排前后输出/精度。
- **W4A16 必须固定具体 checkpoint + 量化格式**：不得把 AWQ、GPTQ、AutoRound 混称一个路径。

---

## 6. PR 持续项（仓库强制要求）

- PR 描述中：**声明使用 AI assistance**。
- 查重：先 `gh issue view`（target issue comments）+ `gh pr list --search`（issue 号与短关键词），并在 PR 中**解释为何不重复已有 PR**。
- 模型相关变化：**必须给 eval**（搜 `tests/evals/` 或 `vllm bench`，结果并入 PR）。

---

## 7. 产物目录

```
experiments/disagg_pd_mainline/
├── EXPERIMENT_PLAN.md          # 本文档
├── environments/
│   ├── v0_17_1/                # 独立 .venv + 版本记录
│   ├── v0_25_1/
│   └── main_7aa248fcfe/
├── runs/                       # 每轮：配置、日志、toy proxy 输出
├── metrics/                    # NIXL raw metrics、evals
└── traces/                     # request 级 trace（关键路径百分比）
```

每个版本记录文件 `env_record.txt`：`git SHA`、`__version__`、`__file__`、`command -v vllm`。