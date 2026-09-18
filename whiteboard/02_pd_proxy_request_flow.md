# 白板图 2 · PD 分离请求流转(NIXL Pull,1P1D)

> P2 持续项。本机 EXP-006/013 实测数字直接标在边上——白板上画完就能报数。

## 图(白板版,标 8K 输入实测 p50)

```
client ──POST(stream)──> proxy(toy, :8192)
                           │ ① 改写:max_tokens=1, kv_transfer_params{do_remote_decode}
                           │    同一 X-Request-Id 发 P
                           ▼
                    P(:8100, kv_producer, GPU0)
                           │ ② 完整 prefill(~900ms@8K)
                           │    KV 留在 P 显存,回 kv_transfer_params:
                           │    {remote_engine_id, remote_request_id, remote_block_ids,
                           │     remote_host/port}   ←—— 身份拆分显式映射
                           ▼
                    proxy ③ 把 P 返回的 kv_transfer_params 塞进原请求 → 发 D
                           ▼
                    D(:8200, kv_consumer, GPU1)
                           │ ④ scheduler 侧:allocate 本地块,标 reqs_to_recv
                           │ ⑤ worker 侧 start_load_kv:
                           │      (首次)NIXL handshake @side-channel 5600/5601
                           │      ~300ms 一次性(EXP-013 idx=0 实测)
                           │      _read_blocks → RDMA READ 远端 descriptor
                           │      【kv_wait = 1764ms@8K = TTFT 的 64.2%】
                           │      有效吞吐 0.26GB/s(descriptor ~16KB 碎片化)
                           │ ⑥ 传输 DONE → 请求进 batch,第 1 个 decode step
                           ▼
                    首 token 回流 proxy → client(TTFT 2738ms@8K)
```

## Gate 层(答"怎么证明传输真发生")

`kv_load_failure_policy=fail` + D 端 `/metrics` 直抓：bytes 增量=预期、count=远端请求数、failed/expired=0；EXP-013《EXT-1 request 级 KV-wait 关联》再加 request 级：逐请求 bytes 求和与 Prometheus 分毫不差。

## PD 概念三段式

- **rendezvous（P、D 怎么对上同一份 KV）** 0.17：双方各自拼 `request_id#layer_name` 当 key，PUT 推流；InputProcessor 随机后缀（input_processor.py：212）使 P/D key 必然分叉 → D 无超时 `Condition.wait`(engine：317)整实例挂死（EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》实机复现）。现在：P 把 `remote_engine_id/remote_request_id/remote_block_ids` 显式还给 proxy，D 拉取时带着 P 的身份去读 P 的内存 descriptor——引擎身份/会话身份/内存寻址三层正交，随机后缀无影响（EXP-013:36/36 双端身份匹配）。为什么：隐式字符串约定 → 显式契约；身份由拥有者签发，不靠两边猜。
- **传输方向（push vs pull）** 0.17：P 主动 PUT（PUT_ASYNC 默认）。现在：默认 Pull（D 发起 RDMA READ），另有 NixlPushConnector(P WRITE)。为什么：pull 让消费者按调度节奏取数、生产者无状态化（留 KV + 过期租约即可）；本机实测方向只差 6.7%（EXP-011《EXT-2 NixlPush 单点》）——瓶颈在互联不在方向。
- **失败语义** 0.17：无超时等待，失败=挂死。现在：`kv_load_failure_policy=fail|recompute` + expiry + heartbeat 租约。为什么：PD 是分布式系统，部分失败是常态，必须可观测可回退。
