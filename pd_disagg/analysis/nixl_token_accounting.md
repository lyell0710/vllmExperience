# NIXL Pull 模式 token 记账溯源（"7668 token" 开放问题关闭）

> provenance： 2026-08-21 静态源码分析（AI 辅助，file：line 已在 venv 0.25.1 = g752a3a504 核对，main@7aa248fc 对照语义一致）。关闭 records/EXP-006 §7 的开放问题；同时揭示 bench 协议的前缀缓存污染问题（触发 sweep 协议 v2）。

## 结论

1. **两个计数器分毫不差，"差 1 token/请求"是双重舍入假象。** bytes 增量必为整块（917,504B=16tok）倍数，14.069GB 唯一反解 = 15,334 块 = **245,344 token**，与 ext_kv 增量 245,344 完全相等（"7668/7667" 来自 14.069GB→439.7MB→÷57344 的两次舍入；真实均值 7,667.0 tok/req）。
2. **缺口 262,144−245,344 = 16,800 token = 1,050 块，全部是 D 端本地 prefix cache 命中**（connector 在申报时扣除、worker 物理跳过）。其中 **511 块可源码定罪**： `vllm bench serve` 正式压测前用 `input_requests[0]` 发一次 test 请求（serve.py：824-871）→ 正式跑请求 #0 时 D 命中 floor((8192−1)/16)=511 块、只补拉 1 块。其余 539 块最可能为**同 seed 历史运行的缓存残留**（同 seed ⇒ prompt 逐 token 相同）——用 `prompt_tokens_by_source{local_cache_hit}` 增量即可运行时定论。
3. **"ext_kv = N−1" 的假设不成立**：connector 对非 Mamba 模型申报**完整 N**（pull_scheduler.py：62-66 → base_scheduler.py：323-328），"最后 1 token 由 D 重算"（恰 1 token，非 1 block；scheduler.py：2436-2437）发生在传输完成之后、只改 num_computed_tokens，**不回写任何指标**。探针 ext=8 ⇒ 服务端实际 prompt 是 8 token（"~9" 估计偏 1）。

## 机理链（file:line，venv 0.25.1）

1. P 端 `request_finished` 交出 kv_transfer_params（remote_block_ids/engine_id/ request_id/host/port/remote_num_tokens）→ pull_scheduler.py:181-275。
2. D 端调度：先本地 prefix cache `get_computed_blocks`（v1/core/sched/ scheduler.py:723-726）→ 再问 connector `get_num_new_matched_tokens`（:736-742 → pull_scheduler.py:34-66：`count = N − num_computed_tokens`）→ 记账 `prefill_stats.set(..., num_external_cached_tokens)`（:777-784， **ext_kv 计数器取值点，无 −1**）。
3. 分配：`allocate_slots(..., delay_cache_blocks=True)`（：903-915）； `update_state_after_alloc`（pull_scheduler.py：112-179）只取 **未哈希新块**（kv_cache_manager.py：95-105）——本地命中块排除；请求进 WAITING_FOR_REMOTE_KVS（：947-967）。
4. worker：`_read_blocks_for_req`（pull_worker.py：101）→ `_apply_prefix_caching`（base_worker.py：2165-2189）**远端块列表尾对齐裁剪到本地未命中块数**→ 按块生成 descriptor，`make_prepped_xfer("READ")` RDMA 读（pull_worker.py：303-313）。
5. 完成：`_pop_done_transfers`（base_worker.py：2027-2072，**bytes 计数器在此按 NIXL telemetry totalBytes 录入**，失败不计字节）→ scheduler 缓存块 + `num_computed_tokens = N−1`（scheduler.py：2430-2437）→ 1-token 前向取 logits。

## 两计数器口径

| 计数器 | 侧 | 递增点 | 口径 |
|---|---|---|---|
| `prompt_tokens_by_source{external_kv_transfer}` | D scheduler | loggers.py：1158-1163 ← scheduler.py：780-784 | **token 粒度承诺值**（N−本地命中），不感知传输失败 |
| `nixl_bytes_transferred_sum` | D worker | base_worker.py：2040-2044（DONE 时） | **块粒度实测字节**（非对齐 prompt 向上取整） |

块对齐 prompt（8192）两者每请求恒等；分离仅发生于非对齐 prompt、传输失败、重传。

## 随机数据集结构（污染机理）

- `--random-prefix-len` 默认 0（datasets.py:1952-1954, DEFAULT_PREFIX_LEN=0）。
- 生成：`allowed_tokens[(offset_i + i + arange(len)) % V]`（datasets.py:753-755）， offsets 由 `default_rng(seed)` 决定（utils.py:98-101）。
- ⇒ **同批 32 prompt 互不共享前缀**（斜坡错位是公共子串非前缀）； **同 seed 跨运行 prompt 逐 token 相同**；且 **seed 相同时短桶 prompt 是长桶对应 prompt 的精确前缀**（同 offsets 数组）——这解释了同 session 顺序跑 512→2048→8192 时 2048 桶实测 25% 缓存命中（16,352=32×511 整）、8192 桶 8.6%。

## 对实验协议的影响（→ 协议 v2）

1. sweep/saturation 每点**唯一 seed**（桶×点位派生，跨臂复用同 seed 保可比性）。
2. 旧 seed-42 行保留 runs.jsonl（诚实），derived 按 seed 字段区分协议版本。
3. attribution 臂间对比不受影响（各臂协议相同、污染相同），绝对值注记（2048 桶约 −25% prefill 工作量、8192 约 −8.6%）；SLO 绝对值维持锁定（预注册原则），敏感性附录覆盖偏差影响。
4. bench 的 test-request 使每次运行的请求 #0 天然带前缀命中（~1/N 请求），footnote。

## 60 秒面试版

见文末 records/EXP-011 引用（口径：D 端"承诺值 vs 实测字节"两本账、块对齐时逐 token 相等、缺口=本地命中且 511 块可源码定罪到 bench 的 test 请求）。
