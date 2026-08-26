# 消费级双卡上的 vLLM 部署形态实验：四臂矩阵、PD 成本结构与架构演化

> **B4 报告 · v2 定稿**（2026-08-23。v1 → v2 变更：吸收 EXT-1 request 级
> KV 归因（EXP-013《EXT-1 request 级 KV-wait 关联》，§2.2/§5 升级为因果占比声明）、EXT-2 推/拉对照
> （EXP-011《EXT-2 NixlPush 单点》，§2.4）、R0-4 动态复现（EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》，§3/§4，B3 完整版表述定稿））
> 数据与图表全部可由 `results/b1_matrix/runs.jsonl` + `raw/` 重算
> （`scripts/make_figures.py`）；每个测量点带 provenance 与 gate 字段。
> 环境：2×RTX 4090 24GB · driver 610.57.04 · vLLM v0.25.1（752a3a5044）·
> Qwen2-7B-Instruct BF16 · 协议 v2（每点唯一 seed，详见 records/EXP-007）。

---

## 0. 一页结论

**问题**：只有两张消费级 4090，应该做单卡混部、双副本、TP=2，还是 Prefill-Decode
分离？

**答案**（本机实测，SLO = TTFT≤5×无负载基线 + TPOT≤50ms）：

| 场景 | 选型 | 依据 |
|---|---|---|
| 短/中请求（≤2K）追吞吐 | **双副本（replica2）** | 饱和吞吐近完美 2×（7.00 vs 3.63 req/s@2K）；goodput 全场最高 |
| 追单位 GPU 成本 | **单卡混部（colocate）** | per-GPU 峰值 goodput 最高或与 replica2 打平（fig3） |
| 需要最低 decode 延迟 | TP=2（谨慎） | TPOT 9.3 vs 16ms（-42%）；但吞吐仅 +13-19%，性价比差 |
| 模型单卡放不下 | TP=2（被迫） | 唯一选项 |
| PD 分离 | **不可取** | KV 通路有效吞吐仅 0.27GB/s（telemetry-derived）；全负载段 goodput 溃败（fig1） |

**三个机理级发现**（均为异常→拆解→实证的完整链条）：
1. **TP2 收益不对称**：decode -42%（权重带宽分摊），prefill 零加速——28 层 ×
   58.7MB 的大消息 allreduce 正好撞上实测 1.78GB/s 的 collective 带宽墙。
2. **NIXL KV 通路有效吞吐 0.26–0.27GB/s 恒定**：descriptor ~16KB/个（每 block
   每层单发）的碎片化小拷贝，在无 P2P 的 PCIe 中转路径上跑不满带宽；小传输另有
   ~12ms 延迟地板（fig5）。
3. **450W 功率帽节流**：持续 prefill 使 SM 降频 2820→~2475MHz（SW Power Cap
   0x4，63°C 非热因），TTFT 从冷态 700ms 抬升至稳态 ~905ms（+30%）——消费卡
   基准测量必须声明功率工况（本报告所有 v2 数据为同热工况）。

---

## 1. 硬件画像（所有结论的因果地基）

| 路径 | 实测 | 含义 |
|---|---|---|
| P2P | `topo -p2p r` = GNS，connectivity=0 | GeForce 驱动级禁用 |
| 单向 D2D | 0.60–0.91 GB/s | 无 P2P 的 cudaMemcpyPeer 分段中转 |
| 双向 D2D | 22.6–22.8 GB/s | Gen4 x16 双向流水极限；与单向差 25 倍=无 P2P 指纹 |
| GPU 间延迟 | 14.5–15.9 µs | — |
| NCCL allreduce | **1.78 GB/s** bus bw | 仅代表 TP collective 路径 |
| NIXL KV 通路 | **0.26–0.27 GB/s** 有效 | telemetry-derived；仅代表 KV 传输路径 |
| 卡内带宽 | ~924 GB/s | GDDR6X，decode 的物理上限 |

原始文件：`hw/`（provenance 齐备）。注意三条互联数字对应三条不同路径，不可互换。

## 2. 四臂矩阵

### 2.1 设计
四臂 × 三输入桶（512/2048/8192，输出 128）× 两跑法（并发 1 归因 + offered-load
扫描），公共负载轴（由 colocate 饱和推档：0.5/0.75/0.9/1.05×）+ 各臂自身容量档。
每点 gate：failure_policy=fail、/metrics 直抓引擎、NIXL bytes 增量=预期、
failed/expired=0、GPU 遥测同行存储。SLO 预注册锁定（328/891/4626ms + 50ms），
敏感性附录（fig6）证明臂间排序在 0.5–4× 阈值区间稳定。

### 2.2 归因层（并发 1，同热工况，p50）
| 桶 | colocate | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| TTFT 512 (ms) | 65.4 | 66.2 | 64.0 | **219.3** |
| TTFT 2048 | 224.9 | 220.7 | 219.8 | **718.6** |
| TTFT 8192 | 925.2 | 902.6 | 881.3 | **2718.7** |
| TPOT (ms) | 15.9–16.4 | 15.9–16.4 | **9.3–9.5** | 15.9–16.4 |

- 非 PD 三臂 TTFT 无实质差异（prefill 无并行收益：TP2 的计算减半被 allreduce 吃光）。
- PD 的 TTFT 溢价全部来自传输——**request 级因果占比（EXT-1，EXP-013）**：
  本地最小 patch 使 P / D / NIXL 三段在同一 request 身份 + 同一时钟域下逐请求
  关联，测得 **D 等待远端 KV 占 TTFT 54.2% / 62.5% / 64.2%**（512/2K/8K，p50，
  p10–p90 带宽 ±2% 内），六段分解闭环误差 p50 <0.1%（最差桶 0.084%）；占比随输入长饱和于 ~64%
  （P 段与传输同为 O(输入长)，比值趋常数；短输入被 ~40ms 固定开销稀释）。
  v1 的分量对账（fig4，54–64%）被逐请求数据追认。kv_wait 与 NIXL
  xferDuration 仅差 0.3–1.9ms——等待窗口就是传输本身，不在调度轮询。

### 2.3 负载扫描（headline，fig1/fig2）
| 桶 | 饱和 req/s：colo/repl/tp2/pd | goodput 峰值 rps |
|---|---|---|
| 512 | 10.36 / 15.58* / 12.31 / 7.84 | 8.57 / 12.75 / 10.18 / **1.59** |
| 2048 | 3.63 / 7.00 / 4.16 / 2.12 | 2.41 / 4.96 / 2.51 / **0.16** |
| 8192 | 0.90 / 1.78 / 1.02 / **0.54** | 0.43 / 0.90 / 0.60 / **0.11** |

*replica2@512 网格未达真实拐点（EXP-007 §7），数字保守。

- **replica2**：2K/8K 桶 1.93/1.98× 扩展——无跨卡通信的复制在受限互联上是
  "免费"的并行。
- **tp2**：双卡只换 13–19% 吞吐。decode 提速在批量化后失去分量（大 batch 下
  decode 转向计算/调度约束），prefill 的 allreduce 墙成为主导。
- **pd1p1d**：饱和 0.54 req/s@8K 与传输墙理论上限吻合（0.27GB/s ÷ 470MB/req
  ≈ 0.57）；**传输带宽即容量**。低负载区也无立足点：传输延迟（114–1730ms）
  直接吃光 SLO 余量，512 桶 66% 饱和度时 goodput 已仅 1.59（fig1 黄线贴地）。
- 成本口径（fig3）：per-GPU 峰值 goodput colocate 8.57/2.41/0.43，
  replica2 6.37*/2.48/0.45，tp2 5.09/1.25/0.30，pd 0.80/0.08/0.05。

### 2.4 对 PD 分离的公平陈述
PD 分离的价值主张（消除 prefill 对 decode 的干扰、独立扩缩 P/D 池）在
**跨节点大集群 + 高速互联（NVLink/IB/RDMA NIC）**下成立。本实验证明的是其
**适用边界**：当 KV 通路只有 ~0.27GB/s 时，1P1D 在任何负载与任何输入长度下都
无法收回传输成本。这不是 NIXL 的缺陷——是部署形态与互联能力的错配。

**传输方向对照（EXT-2，EXP-011）**：用 NixlPushConnector 专用 push proxy
（P 端 NIXL WRITE 推送，与 pull 的 D 端 READ 相反）同 seed 复测 512/8192 单点：
推方向略优——8K TTFT 2537 vs 2718ms（-6.7%），有效吞吐 0.278–0.305 vs
0.26–0.27 GB/s（+10–13%），机理是 WRITE 免去请求-应答回合且 P 完成 prefill
即推、与 D 调度解耦；但**量级不变，两个方向同贴互联墙**——"形态与互联能力
错配"的结论对推/拉同时成立。附带机制对照：push 模式 D 端注册全部本地块、
P 全量推送（8K 全量 469.8MB/req），不做 pull 侧的前缀缓存尾对齐裁剪；
WRITE 的 posting 成本在 P 端显著更高（71–149ms vs pull 的 ~4ms）。

## 3. P2pNccl → NIXL：架构演化（三句话定稿版展开）

**三句话**：
1. 0.17.1 把 engine-local 的 `request_id#layer_name` 当跨实例 rendezvous key，
   而 InputProcessor 给每个实例的内部 request_id 追加独立随机后缀
   （input_processor.py:212），key 必然分叉——PUT 模式下 D 端在无超时的
   `Condition.wait()`（p2p_nccl_engine.py:317）上挂死整实例，GET 模式静默输出乱码。
2. 上游删除 P2pNccl（#44854，2026-06，提交 5add018beb ∈ v0.25.1 祖先）是在替代
   connector 与插件 API 成熟后收敛维护面，非为单个 bug。
3. 当前 NIXL Pull 显式传递 `remote_engine_id/remote_request_id/remote_block_ids`
   （pull_scheduler.py:265-275），数据面用 handshake 注册的内存 descriptor +
   block_ids 做 RDMA READ——**身份被拆分并显式映射**（引擎身份/会话身份/内存寻址
   三层正交），不是"不再依赖 request_id"；0.25.1 的 request_id 随机化依然存在
   而 NIXL 毫发无损，是显式契约对隐式契约的胜利。

完整机理（含 chunked_prefill assert 崩溃链 connector:433、四层 ID 传播表、全部
file:line）：`analysis/p2pnccl_bugs_id_chain.md`。

**版本对照的完整表述（B3 定稿，EXP-008《B3 有限版本对照》 + EXP-012）**，分两个正交维度：
1. **单实例 system-version comparison**（EXP-008）：无负载延迟 8 个月未变
   （物理约束未动），512 桶饱和 +45%（每请求开销路径的收益），启动 308s→58s。
2. **PD 可用性对照**（EXP-012 实机坐实）：0.17.1 P2pNccl 1P1D 在**默认配置下
   正常请求即触发 D 整实例挂死**（request_id 随机后缀分叉 → 无超时
   `recv_store_cv.wait()`，D 全线程 futex 停摆、不自愈，而 P /health 恒 200）——
   0.17.1 的 PD 不构成可用对照臂，跨版本 PD-vs-PD **没有吞吐对比可做，
   对照结论即"不可用 vs 可用"本身**。这也是为什么 B3 不给 PD 吞吐对比表。

## 4. 已知 bug 的复现与链路分析（诚实署名）

两个缺陷为课程已记录的已知问题；本工作为**复现级源码定位与机理验证**
（措辞红线：复现/定位/验证，非发现/修复）。静态分析全部 file:line 在本机
两版本源码核对；**动态复现已实机闭环（EXP-012，2026-08-23）**：
- **缺陷 1**（chunked_prefill assert）：直连 P + 地址串 request_id +
  max_tokens>1 → 精确命中 `p2p_nccl_connector.py:433` AssertionError，
  EngineCore 崩溃（一级证据：原生 traceback）。**实证修正静态分析**：裸
  request_id 直连会先崩于 `connector:518` parse_request_id（ValueError），
  根本走不到 :433——精确触发需要 id 内嵌 peer 地址串。
- **缺陷 2**（rendezvous key 分叉）：经 proxy 的正常请求 → D 整实例挂死
  （双请求 hang + 全线程 futex_wait + util 0 + P /health 恒 200，不自愈）；
  :317 定位由 wchan + 行为学 + 静态 file:line 三方闭环（py-spy 栈帧因容器
  ptrace 限制未取，诚实标注）。
- 三条触发路径（裸 id 崩 :518 / 地址串 id 崩 :433 / 经 proxy 触发 D 侧挂死）
  构成该连接器脆弱性的完整触发图景。
面试展开口径见 analysis 文末"2 分钟版"。

## 5. 归因方法论（可信度声明）

- 每测量点：provenance 行 + gate 字段同行存储 + before/after /metrics 快照
  （直抓引擎，代理不可信）+ GPU 遥测（频率/功率/节流位）。
- NIXL 数字只称 telemetry-derived effective throughput；xferDuration 含 posting，
  不与 postDuration 相加。**"KV 传输占 TTFT X%"为因果占比声明（EXT-1 解锁）**，
  依据三重互证：① 逐请求 bytes 求和与 Prometheus 计数器分毫不差；
  ② kv_wait（墙钟）≈ xferDuration（telemetry，差 0.3–1.9ms）；③ 六段全链分解
  vs client TTFT 闭环误差 p50 <0.1%（最差桶 0.084%）。观测无扰动：打 patch 后 TTFT 218/727/2738
  vs 矩阵 219/719/2719（噪声内）。patch 上游化已查重放弃（#52859 在途，
  `ext1/DEDUP.md`）。
- token 记账双计数器互证（bytes vs prompt_tokens_by_source）分毫不差；
  缺口逐块对账到前缀缓存命中（analysis/nixl_token_accounting.md）——
  测量体系自洽性的独立证明。
- 已知偏差全部登记：tp2 gpu-util 0.88（OOM 规避）、replica2@512 欠饱和疑点、
  SLO 基线含 v1 缓存污染（敏感性附录覆盖）、~3% 运行率的客户端瞬断（重跑处理）。

## 附录
- A. SLO 敏感性：fig6（0.5–4× 排序稳定）
- B. 全量数据表：results/b1_matrix/derived/sweep_summary.csv
- C. 实验记录索引：../records/（EXP-001~017）
- D. request 级 KV 归因全数据：ext1/derived/ext1_per_request.csv（EXP-013）
- E. MoE 前瞻（第 2 阶段，EXP-009/014）：Qwen1.5-MoE-A2.7B TP2+EP 未调优基线
  TPOT 4.62ms（dense 7B TP2 的 2.0×）；**decode 优势在 bs≈8 反转**
  （2.03×@bs1 → 0.82×@bs128，top-4/60 命中并集随 batch 趋全量的读放大）；
  nsys node 级分解：serving batch 下 fused_moe grouped GEMM 占 GPU 时间
  56.4%——E=30,N=1408 config 缺失（运行时告警在案）正中该热点
  （图：../moe_perf/figures/d1_fig1_decode_scaling.png），调优见
  moe_perf/（EXP-015）
