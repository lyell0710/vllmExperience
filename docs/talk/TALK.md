# 面试讲稿(现行版 · 2026-08-24)

> 现行只留一份；被取代版本移 docs/archive/ 并标 superseded。数字全部带 EXP 锚点，与 LEDGER.md 证据台账一致；开讲前先过 LEDGER.md「措辞红线状态」表。整合来源：RESUME_EVIDENCE.md 各节面试防御 + `pd_disagg/analysis/` 两篇口径草稿（p2pnccl_bugs_id_chain.md / nixl_token_accounting.md）。

## 0. 开场 30 秒

在 2×RTX 4090（无 NVLink、P2P 驱动级禁用）上做了两条线。第一条，vLLM 四种部署形态的选型基准——单卡混部/双副本/TP2/NIXL PD 分离，512/2K/8K 输入×多档负载 60+ 测量点（EXP-007《B1 四臂 offered-load 扫描战役》），并用 request 级三段关联把 PD 的 TTFT 拆到因果占比：KV 等待占 54–64%（EXP-013《EXT-1 request 级 KV-wait 关联》）。第二条，MoE 性能优化——nsys kernel 级分解定位 fused MoE grouped GEMM 占 serving batch GPU 时间 56.4%（EXP-014《D1 MoE decode 分解》），据此为社区空缺的 4090 BF16 config 完成上游标准调优、PR 材料备齐（EXP-015《D2 MoE config 调优》），外加 FP8 vs W4A16 的量化选型边界（EXP-016《D4 FP8 vs W4A16 同卡对比》）。

## 1. 主线一:部署选型(S1,EXP-007/EXP-013)

**90 秒讲法**：消费级双卡的互联画像先钉死——P2P 驱动级禁用（GNS）、单向 D2D 0.60–0.91 GB/s、NCCL collective 带宽受限（EXP-002《硬件三数》，实测值待复核）。四臂结论： 双副本近线性 2×（2K 输入 7.00 vs 3.63 req/s）且 goodput 全场最高；TP2 decode 提速 42%（带宽分摊）但 prefill 零加速（大消息 allreduce 受 collective 带宽约束），吞吐仅 +13–19%；PD 分离全负载段被传输压垮——NIXL 有效吞吐恒 0.26–0.27GB/s（telemetry-derived，~16KB/descriptor 碎片化）。再用 EXT-1(EXP-013)把"传输是不是瓶颈"从对账推断升级成因果占比：KV 等待占 TTFT 54.2/62.5/64.2% (512/2K/8K，p50)，闭环误差 p50 <0.1%。推方向也试过：push 仅 -6.7% TTFT@8K（EXP-011《EXT-2 NixlPush 单点》），方向救不了量级——结论是选型边界，不是"PD 不行"。

**防御**：
- "凭什么说传输真的发生了" → 每测量点 gate 字段（nixl bytes 增量、成功传输数= 预期远端请求数、failed=0、expired=0、/metrics 直抓引擎端口）与指标同行存于 runs.jsonl(EXP-007)。
- "54–64% 怎么来的" → EXP-013 三重互证：逐请求 bytes 和=Prometheus 分毫不差； kv_wait−xferDuration=0.3–1.9ms；六段分解闭环误差 p50 <0.1%（最差桶 0.084%）； 36/36 请求身份双端匹配；patch 前后 TTFT 噪声内（无扰动）。
- "patch 是不是改了核心" → ~16 行本地可观测性改动，非 connector 核心改造（红线表限定措辞）。
- "功率帽是什么" → 持续 prefill 降频 2820→2475MHz（SW Power Cap 遥测）， TTFT +30%；所以四臂对比全部同热工况（EXP-005/007）。

## 2. 深挖线:P2pNccl 两 bug(2 分钟口径,EXP-012 + analysis)

> 红线：只说"复现/定位/验证"，不说"发现/修复"；"吃透"说"梳理"。

我在 vLLM 0.17.1 上做 PD 分离时定位过 P2pNcclConnector 的两个缺陷，后来对照 0.25.1 的 NIXL 架构验证了修复思路——这条路线 2026 年 6 月被上游整体移除，这两个缺陷本质上是"该架构为什么必须死"的案例。

第一个是 chunked prefill 崩溃。P2pNccl 的传输单元是"请求×层"的一次性完整张量， wire protocol 没有 chunk 序号；0.17.1 的补法是 scheduler 侧攒块、末 chunk 才发， 并用 assert(connector：433)把"P 节点任何多步执行都是 prefill 续传"焊死。P 端只要出现一步 decode——比如 proxy 没把 max_tokens 钳成 1——assert 直接打死 EngineCore。实机复现精确命中：433 原生 traceback（EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》）；实证还修正了静态分析：裸直连（无地址串 id）会先崩在：518 的 parse_request_id，：433 需要地址串 id + max_tokens>1 两个条件同时成立。

第二个更隐蔽：跨实例 KV 匹配 key 是 request_id#layer，前提是两端内部 id 逐字节相等，但 InputProcessor 给每实例内部 id 追加 8 位随机后缀（input_processor.py：212）——HTTP 层看一切正常，connector 层 key 已分叉。PUT 模式下张量以 P 端 key 躺在 D 的 recv_store，D 在无超时 Condition.wait(engine：317)上按 D 端 key 死等：整个 D 实例挂死+持续漏内存（EXP-012 实机：双请求 hang、全线程 futex_wait、P /health 恒 200）； GET 模式则静默乱码。根因是隐式契约——连接器假设外部 id 等于内部 id，引擎侧早就不保证了。

0.25.1 NIXL 的答案是身份拆分显式映射：P 结束时把 remote_engine_id/request_id/ host：port/block_ids 显式交给 proxy 转发，D 装进 RemoteMeta，数据面靠 handshake 注册的 descriptor 做 RDMA READ。0.25.1 的 id 随机化依然存在，NIXL 毫发无损——显式契约对隐式契约的胜利。（全 file：line 见 `pd_disagg/analysis/ p2pnccl_bugs_id_chain.md`；机制笔记 `docs/theory/02_pd_kv_path.md`。）

**防御**："为什么 bug2 没有 Python 栈帧" → 容器 ptrace 限制（py-spy/gdb 不可用），：317 定位由 wchan + 行为学 + 静态 file：line 三方闭环，诚实标注（EXP-012 §4）。

## 3. 主线二:MoE 优化(S3,EXP-014/EXP-015)

**90 秒讲法**：先问"为什么调这个"——nsys node 级分解（EXP-014）：fused_moe grouped GEMM 占 GPU kernel 时间从 18.7%(bs=1)膨胀到 56.4%(bs=32 serving batch)；MoE/dense 吞吐比 2.03×(bs=1)→0.97×（bs=8 反转）→0.82×(bs=128)， 机理是 top-4/60 下 batch 增大、命中专家并集趋全量，激活稀疏优势变读放大劣势。据此为社区空缺的两个 tuple（E=30，N=1408 EP / E=60，N=704 非 EP；本地+远端+运行时告警三重查重）跑上游标准调优：1920 配置×18 M 档（benchmark_moe.py --tune）。结果如实说：kernel 两端改善（M=1：EP -8.5%/非 EP -3.8%；M≥128：-3.3~-3.9%）， 中段与默认启发式打平——这本身是 D3 的结论：config 即最优杠杆，不做无数据支撑的 kernel 改动。三级验证：correctness 120 passed / kernel A/B / e2e bench。

**防御**：
- "e2e 才 +1% 值得吗" → e2e TPOT +0.8~1.2% 三档一致且与 kernel 增益×56.4% 占比折算自洽，但低于跨会话漂移（±5~8%），所以主证据=kernel A/B，e2e 只作防御层数字，不进简历句（8/24 定档修正，EXP-015 §5）。
- "怎么排除 JIT 干扰" → 新 config 首跑 TTFT 1021ms 是 Triton 现场编译伪影， warmup 复测 225ms，PR 正文注明（EXP-015 §7）。
- "为什么 EP/非 EP 两个 tuple" → EP 切专家（每卡 E=30，N=1408），非 EP 切 N (E=60，N=704)，GEMM 形状不同、tile 最优解不同（docs/theory/01）。
- PR 状态措辞：材料六件套齐备、分支就绪，"提交留用户本人"——未提交不说"提交"。

## 4. 量化选型(S4,EXP-016)

W4A16(GPTQ-Int4，Marlin)decode 全 regime 胜 FP8 23–48%(TPOT 4.91 vs 7.10ms@bs1)且权重减半；FP8 仅高并发 prefill 段 TTFT 反超（497 vs 613ms@c128， 计算受限）+ wikitext PPL 相对优 3.3%（7.663 vs 7.922，同 31212 计分 token）。加分点：从 vLLM 分派逻辑解释 Ada 为何走不进 Hopper FP8 快路径—— oracle/fp8.py：103-122 capability 90/100 检查跳过 SM89 → TRITON block-scaled。防御：同 token 集 PPL 协议；regime 反转机理（decode 带宽受限 vs prefill 计算受限 + Marlin 反量化开销）。

## 5. 追问速查表

| 追问 | 要点 | 锚点 |
|---|---|---|
| EPLB 怎么不讲？ | gate 判定不上简历：W4A16 上游显式拒（routed_experts.py：151），FP8 臂 2 次真实重排（balancedness 0.53–0.74）+无-EPLB 对照组把输出分歧因果归属 EPLB（数值性） | EXP-017 |
| 记账缺口 7668 token？ | bytes 反解 245,344 token 与计数器分毫不差；缺口=D 端 prefix cache 命中，511 块源码定罪于 bench test 请求 | EXP-006 §7 关闭，`analysis/nixl_token_accounting.md` |
| 前缀缓存污染？ | 同 seed 命中缓存虚高——sweep 协议 v2 每点唯一 seed | EXP-007 |
| nsys 怎么看不到 decode kernel? | CUDA graphs 默认 graph 级 trace 藏 kernel，必须 --cuda-graph-trace=node（other 桶 77%→1.2%） | EXP-014 §7 |
| 双向 22.7 vs 单向 0.6GB/s？ | 无 P2P 时 cudaMemcpyPeer 走分段中转，双向是流水线叠加 | EXP-002，`hw/p2p_bandwidth_latency.txt` |
| 版本对照公平吗？ | 只称 system-version comparison（传输方向不同）：无负载 Δ<1%、512 桶饱和 +45%、启动 308→58s | EXP-008 |

## 6. 讲前红线自查

LEDGER.md「措辞红线状态」表逐行过：两 bug 只"复现/定位/验证"；PR 未提交不说 "提交"；带宽只说 telemetry-derived；e2e +0.8~1.2% 不作 headline；EXT-1 patch 只说"~16 行本地可观测性改动"。
