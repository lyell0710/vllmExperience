# EXP-007 · B1 四臂 offered-load 扫描战役（协议 v2）

> **一句话结论**：四臂全矩阵扫描给出选型答案：**replica2 是吞吐王**（2048/8192 近完美 2× 扩展、全桶 goodput 最高）；**TP=2 在这条互联上是最差性价比**——decode 的 -42% 收益在批量化后被 prefill allreduce 天花板吃掉，双卡只换来 13-19% 吞吐。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-21（17:56–20:27Z，四臂连续会话） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；Qwen2-7B-Instruct |
| 状态 | 完成 |
| 关联清单项 | B1 主体（S1 headline 数字来源） |

## 1. 目的与假设
四臂在共同负载轴上的 goodput/延迟/成本边界。假设：受限互联下 replica2 扩展最好、 tp2 受 allreduce 天花板、pd1p1d 受传输墙。

## 2. 环境与配置（协议 v2）
- 各臂 fresh server；配置同 EXP-004~006，例外：**tp2 用 --gpu-memory-utilization 0.88**（0.9 在 warmup 阶段 OOM：需 150MB 仅剩 124MB，两卡同报；偏差记录在案）。
- **协议 v2 = 每点唯一 seed**（attribution x042 / saturation x099 / sweep 档位 x001..x006，x=桶号 1/2/3），跨臂同点位同 seed（工作负载可比）。动因：`analysis/nixl_token_accounting.md` 揭示同 seed 跨运行/跨桶前缀缓存污染（同 seed 下短桶 prompt 是长桶的精确前缀；实测 2048 桶 25% 命中、8192 桶 8.6%）。
- 每桶 SLO（锁定值）：TTFT ≤328/891/4626ms + TPOT ≤50ms；goodput 由 collect_point 按每请求 detailed 数组计算。
- 网格：公共档 {0.5,0.75,0.9,1.05}×colocate 饱和 = 512：{5.2,7.8,9.3,10.9} / 2048：{1.8,2.7,3.3,3.8} / 8192：{0.45,0.68,0.81,0.95}；另加各臂贴近自身饱和的档位。
- num_prompts：512:320 / 2048:192 / 8192:56（饱和探测 200-400/128-256/48-96，SAT_CONC=64）。

## 3. 步骤
每臂：起栈 → attribution×3（v2 干净基线）→ saturation×3 → sweep（12-18 点）→ 拆栈。每点自动：before 快照 → bench（--save-result --save-detailed）→ after 快照 → GPU 遥测汇入 → runs.jsonl。

## 4. 原始数据
`results/b1_matrix/runs.jsonl`（v2 协议 84 有效行 + 早期行保留）；raw/、snapshots/ 同前缀全量；服务日志 raw/v2_*。瞬断重跑：colocate 2048@2.7、tp2 2048@{1.8,3.8} 各 1 次 ServerDisconnectedError（1/192 请求，客户端连接层瞬断，服务端 0 ERROR）——失败行保留（gate_pass=false），同 seed 重跑通过。频率 3/~90 次 bench，记为已知噪声。

## 5. 结果（goodput rps；完整表由 runs.jsonl 重算）
**饱和吞吐（req/s）**：
| 桶 | colocate（1 卡） | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 10.36 | 15.58* | 12.31 | 7.84 |
| 2048 | 3.63 | 7.00 | 4.16 | 2.12 |
| 8192 | 0.90 | 1.78 | 1.02 | **0.54** |
*512 replica2 疑受 SAT_CONC=64 客户端并发或 rr_proxy 上限影响（2048/8192 为 1.93/1.98×）。

**goodput 峰值（rps @ 档位）**：colocate 8.57@9.3 / 2.41@2.7 / 0.43@0.68；replica2 12.75@14 / 4.96@5.3 / 0.90@0.95；tp2 10.18@11.8 / 2.51@2.7 / 0.60@0.81；pd1p1d 1.59@5.2 / 0.16@1.8 / 0.11@0.45。

**v2 attribution（同热工况，p50 ms）**：
| 桶 | colocate | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 65.4 | 66.2 | 64.0 | 219.3 |
| 2048 | 224.9 | 220.7 | 219.8 | 718.6 |
| 8192 | 925.2 | 902.6 | 881.3 | 2718.7 |

## 6. 分析与结论
1. **replica2 = 吞吐王**：2048/8192 近完美 2× 扩展；全桶 goodput 最高。
2. **tp2 双卡只换来 13-19% 吞吐**：decode -42% 的收益在批量化后被 prefill allreduce 天花板吃掉（8K 归因仅 -5%，同工况）——TP=2 在此互联下是最差性价比。
3. **pd1p1d 全负载段溃败**：512 桶 66% 饱和度时 goodput 已仅 1.59（传输延迟 114-220ms 直接吃光 SLO 余量）；8K 饱和 0.54 rps 与 0.27GB/s 传输墙理论上限（~0.57） 吻合——**传输带宽即容量上限**的直接证据。
4. **成本口径（per-GPU goodput）**：colocate 单卡最优；replica2 与其打平（无跨卡开销的复制）；tp2/pd 均为负收益。选型结论：此类互联受限双卡， **短请求选 colocate×2（=replica2），长上下文超单卡容量才考虑 tp2；PD 不可取**。
5. v2 干净基线修正了污染值：2048 桶真实无负载 TTFT ≈225ms（原 178ms 含 25% 缓存命中）；8K ≈881-925ms（同热工况四臂 prefill 几乎无差异——功率帽整平了差距）。

## 7. 异常、偏差与开放问题
- tp2 0.9 显存利用率 OOM（warmup 150MB > 剩 124MB）→ 0.88；对高负载 KV 容量的影响未单独量化（各臂 KV 预算本就不同，成本口径不受影响）。
- ServerDisconnected 偶发（~3% 运行），根因未深究（客户端 aiohttp keepalive 侧）。
- replica2@512 饱和疑欠饱和（SAT_CONC/代理上限），如需引用 512 扩展效率先复测 SAT_CONC=128。
- SLO 绝对值基于 v1 污染基线锁定（预注册不回改）；v2 干净基线下 2048 桶 SLO 实为 3.96×（891/225）而非 5×——敏感性附录必须覆盖，引用时注明。
- 8/24 勘注（审计收尾）：snapshots/ 存在 3 个 0 字节空快照（`20260821T1911_tp2_{512,2048,8192}x128_attribution_8100_before.prom`）——疑为 tp2 臂该重启窗口 8100 端口未起时抓空；runs.jsonl 无任何行引用该前缀（tp2 归因跑实际引用 1643–1645 与 1913–1915 前缀的完整快照），数据侧无影响； raw 不可变，空文件原地保留仅登记。

## 8. 下游影响
S1 headline 全部就位；B4 报告主体数据齐；figures/ 待出图〔已出：pd_disagg/figures/fig1–fig7〕；PD 溃败的定量机理衔接 B2（传输延迟分解）。

- 〔勘注 2026-09-15：**512 桶的 conc64 数字已被 conc128 口径取代**——EXP-023《replica2@512 饱和复测（SAT_CONC=128）》与 EXP-024《512 桶四臂统一到 conc128 口径》补测得 colocate 12.81 / replica2 20.87 / tp2 12.30 / pd1p1d 8.15 req/s，扩展效率 1.63×。其中 tp2 与 pd1p1d 在 conc64 时就已到顶（饱和 / 传输墙），只有 colocate/replica2 真欠饱和。本记录的 conc64 值作为史料保留；现行口径见 LEDGER B1 行与 fig7。2K/8K 桶不受影响。〕
