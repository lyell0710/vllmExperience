# 白板图 1 · vLLM v1 调度链(请求从 HTTP 到 token)

> P2 持续项。目标:3 分钟白板画完 + 每个环节能答"数据结构是什么/为什么在这层"。
> 基于 v0.25.1 源码(ENV-B),锚点已在本机核对。

## 图(白板版)

```
HTTP POST /v1/completions
   │  X-Request-Id header ──> request_id(serving.py:117 _base_request_id)
   ▼
AsyncLLM (APIServer 进程)
   │  tokenize → EngineCoreRequest(request_id 可能加随机后缀)
   ▼ zmq IPC
EngineCore (独立进程,主循环 step())
   │
   ├─ Scheduler.schedule()
   │    waiting 队列 ──(kv_cache_manager.allocate_slots)──> running
   │    · continuous batching:每 step 重组 batch
   │    · chunked prefill:长 prompt 切 token_budget 块
   │    · prefix caching:block hash 命中免算
   │    输出 SchedulerOutput{scheduled_new_reqs, scheduled_cached_reqs,
   │                         num_scheduled_tokens, (kv_connector_metadata)}
   ▼
ModelExecutor / Worker(TP 时每 rank 一进程)
   │    prepare_inputs → forward(FlashAttention, paged KV)→ sample
   ▼
ModelRunnerOutput{sampled_token_ids} ──> Scheduler.update_from_output()
   │    检测 stop/length → free blocks
   ▼ zmq
AsyncLLM detokenize → SSE stream 回客户端
```

## 各环节一句话(面试追问层)

| 环节 | 数据结构 | 为什么在这层 |
|---|---|---|
| waiting→running | `Request`,按 FCFS+优先级 | 准入即显存承诺:allocate 不到 block 就不进 running |
| KV block | `KVCacheBlock`(16 token/块),block_pool | paged attention:显存碎片→逻辑块表 |
| chunked prefill | num_computed_tokens 游标 | 长 prompt 不独占 step,和 decode 混批控 TTFT/TPOT 平衡 |
| connector 挂点 | SchedulerOutput.kv_connector_metadata | PD 分离仅是调度器的旁路元数据,不改主循环 |

## PD 概念三段式(0.17 → 现在 → 为什么)

- **调度侧 KV 注入**:0.17 P2pNccl 在 worker 层 save/recv hook 里做,身份靠
  request_id 字符串约定;现在 v1 connector 拆 scheduler 侧(决定谁远端拉)+
  worker 侧(执行传输),元数据走 SchedulerOutput 显式传递;**为什么**:调度决策
  (allocate、何时可跑)与数据面(怎么搬)解耦,失败可回退(failure_policy)。
- 本机实证:D 端请求在 `reqs_to_recv` 出现(worker 首见)到传输完成的窗口
  = EXP-013 kv_wait,占 TTFT 54–64%。
