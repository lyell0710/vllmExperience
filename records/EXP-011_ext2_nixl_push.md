# EXP-011 · EXT-2 NixlPush 单点（推 vs 拉方向对照）

> **一句话结论**：推方向（NixlPush）略优于拉方向：8K TTFT -6.7%、有效吞吐 +10–13%。**但量级没变**——0.305 vs 0.27 GB/s 同贴互联墙，传输方向救不了 PD 分离。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-22（09:55–10:06Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | EXT-2（弹性）；B4 §2.4 补充 |

## 1. 目的与假设
用 **NixlPushConnector 专用 push proxy**（清单红线：不得拿 Pull 的 toy proxy 换名跑）测推方向单点，回答"传输方向能否改变 PD 在本互联上的结论"。

## 2. 环境与配置
- proxy：`examples/disaggregated/disaggregated_serving/disagg_proxy_pushconnector_demo.py`（从 v0.25.1 tag 提取 → `matrix/`；机制：D 先经 NIXL 通知向 P 注册本地块， P 用 NIXL **WRITE** 推送——与 pull 的 D 端 READ 相反）
- P/D：`NixlPushConnector`，P 带 `engine_id=prefill-engine-001`（proxy 参数须匹配）， side channel 5600/5601，`kv_load_failure_policy=fail`，其余同 pd1p1d 臂。

## 3. 步骤
起 P/D → push proxy(8192) → smoke → attribution 512/8192（seed 1042/3042，与 pull 臂同 seed 直接可比）。

## 4. 原始数据
runs.jsonl arm=pd1p1d_push 两行 + raw/ext2_push_{P，D，proxy}.log。 **注**：这两行的 gates 结构化字段为 None——collect_point 的 is_pd 判断当时用 `== "pd1p1d"` 精确匹配未命中 push 臂名（已修为 startswith）；全部计数在 `gates.kv_deltas_raw` 完整在案，本记录 §5 由 raw 对账，数据有效性不受影响。

## 5. 结果（与 pull 同 seed 对照）
| 指标 | Pull（EXP-007 v2） | Push（本实验） |
|---|---|---|
| TTFT p50 512 (ms) | 219.3 | **210.4** |
| TTFT p50 8192 (ms) | 2718.7 | **2537.1**（-6.7%） |
| 计数器所在端 | D（READ 发起方） | **P（WRITE 发起方）** |
| bytes/req 512 | 29.4MB（含缓存扣减） | 29.36MB=512tok **全量** |
| bytes/req 8192 | 439.7→469.8MB | 469.8MB **全量**（desc 28672/req） |
| avg xfer 8192 | 1602.7→1730ms | **1538ms** |
| 有效吞吐 | 0.26–0.27 GB/s | **0.278–0.305 GB/s** |
| avg post | ~4ms | 71–149ms（WRITE posting 在 P 端显著更贵） |
| failed/expired | 0 | 0（raw 中 *_total 无增量） |
| D 端 by_source | N−本地命中 | **完整 N**（16384/262144 = 32×N 整） |

## 6. 分析与结论
- 推方向略优（8K TTFT -6.7%，有效吞吐 +10–13%）：WRITE 免去 pull 的请求-应答回合，且 P 端在 prefill 完成即推、与 D 端调度解耦。
- **但量级不变**：0.305 vs 0.27 GB/s 同贴互联墙——**传输方向救不了 PD**，B4 §2.4 的"形态与互联能力错配"结论对两个方向同时成立。
- push 模式 D 不做前缀缓存扣减（注册全部本地块，P 全量推）——与 pull 的尾对齐裁剪（base_worker._apply_prefix_caching）形成机制对照，面试素材。

## 7. 异常、偏差与开放问题
- 工装 bug（is_pd 精确匹配）——已修；教训：臂名匹配用前缀不用全等。
- 单点性质：未跑 sweep（EXT 弹性定位），推方向的负载行为未测。

## 8. 下游影响
B4 v2 §2.4 补一段推/拉对照；EXT-2 关闭。
