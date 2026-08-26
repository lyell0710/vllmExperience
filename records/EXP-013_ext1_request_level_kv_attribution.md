# EXP-013 · EXT-1 request 级 KV-wait 关联(解锁"KV 占 TTFT%"红线)

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23(07:49–07:56Z) |
| 环境 | ENV-B(752a3a5044, vllm 0.25.1)+ 本地 patch `ext1/nixl_req_telemetry_v0251.patch`;Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | EXT-1;B2 归因层收官;B4 v2 §2.2/§5 升级 |

## 1. 目的与假设
给 NIXL telemetry 加 request 关联的本地最小 patch,使 P / D / NIXL 三段在
**同一 request identity + 同一时钟域**下逐请求关联,解锁红线声明
"D 等待远端 KV 对 TTFT 的关键路径贡献占比"。
可证伪假设:v1 报告分量对账给出的"传输占 54–64%"能在 request 级因果分解下复现。

## 2. 环境与配置
- **上游查重先行**(`ext1/DEDUP.md`):开放 draft PR **#52859**(NVIDIA,
  lifecycle tracing for NIXL push/pull)已覆盖上游化方向 → 按 AGENTS.md
  fail-closed,**EXT-1 定位为本地测量 patch,不投上游**。
- **Patch**(`ext1/nixl_req_telemetry_v0251.patch`,打在 ENV-B site-packages,
  16 行全带 `# EXT1` 标记,原件备份 `ext1/orig/`):
  1. `pull_worker.start_load_kv`:D connector 首见请求时记
     `(perf_counter, epoch)`(含 handshake 等待);
  2. `base_worker._pop_done_transfers`:per-handle 累加 NIXL telemetry
     (bytes/xferDuration/postDuration/descs),全 handle DONE 时输出一行
     `EXT1_KV req_id=... remote_request_id=... kv_wait_ms=... t0_epoch=...
     done_epoch=... bytes=...`;
  3. `_handle_failed_transfer`:清理两个 dict 防泄漏。
- **身份链**(全链核实):client 自定 `X-Request-Id`(`ext1-<桶>-<序号>-<hex6>`)
  → proxy 原样转发 P 与 D(`serving.py:117 _base_request_id` 从 header 取 id)
  → D 端 connector req_id 与 remote_request_id(P 端 id)均内嵌该串。
- **时钟域**:同机 1P1D,client/proxy/P/D 全部同一 host 墙钟;kv_wait 用
  perf_counter 差值(单调),跨进程对齐用 epoch。
- 测量栈:`ext1/run_ext1.sh` = EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》同配置 1P1D pull(fail policy、
  max-model-len 16384、无 enforce-eager)+ `ext1_proxy.py`(toy proxy +
  6 个 epoch 打点 + 透传 X-Request-Id)+ `ext1_client.py`(并发 1 流式,
  每请求唯一 seed,3 桶 × 12 请求,max_tokens=32)。

## 3. 步骤
起 P/D → instrumented proxy(8192)→ metrics before 快照(直抓 8100/8200)→
36 请求 → metrics after 快照 → 提取 EXT1_KV / EXT1_PROXY 行 →
`analyze_ext1.py` 三方 join(client × proxy × D-connector)。

## 4. 原始数据
`pd_disagg/ext1/raw/EXP-013/`:client.jsonl(首行 provenance)、
{P,D,proxy}.log、ext1_kv_lines.txt、ext1_proxy_lines.txt、
metrics_{8100,8200}_{before,after}.prom。
衍生:`ext1/derived/ext1_per_request.csv`(36 行全字段)。

## 5. 结果
**Gate(全 PASS)**:36 传输 = 36 远端请求;bytes 7.39875GB;
failed transfers / failed notifications / expired = 0;metrics 直抓引擎端口。

**request 级 TTFT 因果分解**(并发 1,p50,每桶 n=11,排除 idx=0 首请求):

| 桶 | TTFT (ms) | kv_wait (ms) | **KV 占 TTFT** | p10–p90 | P 段 (ms) | post-KV (ms) | 闭环误差 |
|---|---|---|---|---|---|---|---|
| 512 | 218.3 | 118.2 | **54.2%** | 52.4–55.6% | 63.8 | 24.4 | 0.08% |
| 2048 | 726.7 | 455.7 | **62.5%** | 61.8–63.7% | 221.7 | 24.0 | 0.04% |
| 8192 | 2738.0 | 1763.8 | **64.2%** | 63.6–64.8% | 899.5 | 35.0 | 0.02% |

**互证三条**:
1. EXT1_KV 逐请求 bytes 求和 = 7398752256 = Prometheus
   `nixl_bytes_transferred_sum` **分毫不差**;
2. kv_wait(墙钟)− xferDuration(NIXL telemetry)= 0.3–1.9ms
   → 等待窗口≈传输本身,step 轮询开销可忽略;
3. 六段分解(pre_proxy + P 段 + P→D gap + D pre-KV + kv_wait + post-KV)
   求和 vs client TTFT,闭环误差 p50 <0.1%（最差桶 0.084%,逐请求最大 0.11%）。

**无扰动证明**:本次 TTFT p50 218/727/2738 vs 未打 patch 的矩阵数据
219/719/2719(EXP-006/007)——观测开销在噪声内。

**身份匹配**:36/36 的 client rid 同时出现在 D 端 req_id 与
remote_request_id(P 端 id)中——身份拆分显式映射的实测确认。

idx=0 首请求(含 NIXL handshake,单列不进统计):kv_wait 409.8/462.8/1770.3ms
——512 桶首请求 +292ms 即 handshake 成本一次性摊销的直接观测。

## 6. 分析与结论
- **红线解锁**:"D 等待远端 KV 对 TTFT 的关键路径贡献"现可作**因果占比声明**:
  同 request 身份 + 同时钟域下,KV 等待占 TTFT 54.2%/62.5%/64.2%(512/2K/8K),
  与 v1 报告的分量对账值(54–64%)吻合——对账法追认有效。
- 占比随输入长度增长而饱和于 ~64%:P 段(prefill)与传输同为 O(输入长),
  二者比值趋于常数;短输入时固定开销(HTTP/调度 ~40ms)稀释占比。
- kv_wait ≈ xferDuration 说明 D 端调度对传输完成的响应及时(step 级轮询
  延迟 ~0.3–1.8ms,随桶长缓增;各桶 p50 = 0.33/0.71/1.78ms),瓶颈就是
  传输本身,不在轮询机制。

## 7. 异常、偏差与开放问题
- max_tokens=32(矩阵 attribution 用 128):TTFT/kv_wait 与输出长无关,
  不影响本实验结论;TPOT 类指标本实验不产出。
- analyze 脚本首版被 provenance 头行绊倒(含 EXT1_PROXY 字样),已修
  (跳 `#` 行);数据无损。
- patch 仅覆盖 pull 路径;push 侧如需同样关联可仿做(EXT-2 已关闭,不做)。

## 8. 下游影响
- README 红线表:"KV 传输占 TTFT X%" 🚫→✅(引用本记录)。
- B4 v2 §2.2:分量对账升级为因果占比声明;§5 方法论补三重互证。
- S1 简历句可写"以 request 级三段关联(同身份同时钟域)测得 KV 等待占
  TTFT 54–64%"。
- 上游不投(#52859 在途);若其合入,未来引用上游机制替代本地 patch。
