---
status: complete
关联EXP: EXP-006, EXP-007, EXP-011, EXP-013, EXP-014, EXP-015
配套: docs/theory/02_pd_kv_path.md(速查版) · pd_disagg/REPORT.md §2.2/§5(结论版) · moe_perf/PR_DRAFT.md(交付材料) · 本文=逐段走读版
---

# 深度讲义 02 · 把"传输占 TTFT 多少"从推断做成测量:PD 归因的 ~16 行 patch 与 MoE 调优的三级验证

> 读者：已读过讲义 01（四臂形态、互联三数、功率帽）的人。读法：不跳步。每个论断后跟证据锚（EXP 编号 / 文件：行号 / derived 路径）； 凡本文从仓内 raw/derived 现算的量，一律标注"本文现算"。

## 1. 这一篇回答什么问题

PD 分离在纸面上赢在哪、在这台机器上为什么全负载段输；以及一句话——"D 等待远端 KV 占 TTFT 54.2% / 62.5% / 64.2%"——怎么从**分量对账的推断**升级成**逐请求的因果占比**。读完你应当能： ①手推 PD 的三条账(单请求传输量 $B_{kv}(n)$、传输时间、容量上限 $BW_\mathrm{eff}/B_{kv}$)， 并解释实测饱和 0.54 req/s@8K 为何与算式的 0.57 对得上；②讲清 ~16 行本地可观测性改动为什么 **恰好**打在那几处（请求身份 + 同时钟域），以及六段闭环误差 p50 <0.1% 不是"噪声小"而是 **缺口具名**——本文把这 0.18–0.52 ms 的残差逐桶指认到了具体一段代码；③从遥测反解 0.27 GB/s 的碎片化根因（descriptor 恰 16,384 B/个，传输时间对 descriptor **计数**线性而非对字节线性）； ④把同一套"先分解、再归因、后验证"搬到 MoE：nsys node 级分解定位 fused_moe grouped GEMM 占 56.4%(bs=32)→ config 搜索 → correctness / kernel A/B / e2e 三级验证 → 材料齐备（**未提交**）；⑤答上"你凭什么说那 54% 是因果""闭环误差 0.02% 是不是自证循环"这类追问， 并诚实说出它不覆盖什么。

### 1.1 本篇要建立的五条能力

1. **归因能力**：能分清"分量对账"（把总量减去几个别处测来的分量，剩下的归给某一项） 与"因果占比"（分子分母同属一条请求、同一时钟域）。**同一个百分比，证据等级差一个量级**（§3.2、§4 段 4）。
2. **观测能力**：知道给一个分布式系统加测量点时，哪几处是**不可替代**的落点，以及 "观测改变被观测对象"要用什么对照来排除（§3.3）。
3. **反解能力**：能从遥测里的几个整数（bytes / descs / xferDuration）反推出访问模式， 进而定出优化方向，并给这个方向一个**可证伪的收益上界**(§3.6)。
4. **硬件语义能力**：能把 Triton 的六个 config 旋钮映射到 Ada 的硬件约束——共享内存每 block 99 KB、每 SM 四个 warp scheduler、cp.async 的 commit-wait 组语义、 mma 的 fragment 布局——并据此说清搜索空间的边界（§3.8）。
5. **诚实能力**：能说出"折算对上了不等于可以当卖点"（效应量 vs 噪声量，§3.7 ⑤）、 "闭环误差小不等于测得准"（§6 误区 2）、"负结论照常报告"(§7 Q8)。

### 1.2 符号与口径约定

| 符号 | 含义 | 本机取值/来源 |
|---|---|---|
| $B_{kv}(n)$ | 单请求 KV 字节数 | $n\times 57{，}344$ B，按 16-token 块向上取整 |
| $BW_\mathrm{eff}$ | NIXL 有效吞吐 | 0.26–0.27 GB/s(telemetry-derived,EXP-006/007) |
| kv_wait | D 端 connector 首见请求 → 全部 handle DONE | perf_counter 差，含握手（§3.3 ①） |
| xferDuration | NIXL 自报的纯传输时间 | 已含 posting，**不与 postDuration 相加** |
| descs | 一次传输的 descriptor 个数 | 每（层， K 或 V， block） 一个，恒 16 KiB |
| $\alpha$ / $\beta$ | 每次传输的固定开销 / 渐近带宽 | $t(m)=\alpha+m/\beta$（讲义 01 公理 B） |
| M | fused_moe kernel 的 token 维 | config JSON 的档位键；实际查表用最近邻（§3.8.4） |
| E / N | 每 rank 专家数 / 专家中间维 | EP：30 / 1408；非 EP：60 / 704 |

模型口径：PD 线为 Qwen2-7B-Instruct（28 层、28 Q 头、4 KV 头、head_dim 128、hidden 3584， 本机 `config.json`）;MoE 线为 Qwen1.5-MoE-A2.7B-Chat（24 层、hidden 2048、 `num_experts` 60、`num_experts_per_tok` 4、`moe_intermediate_size` 1408、 `shared_expert_intermediate_size` 5632、vocab 151936，本机 `config.json`）。

### 1.3 本篇引用的一级文献(详细出处见 §8.3)

- MoE 架构：Lepikhin et al., "GShard", arXiv:2006.16668;Dai et al., "DeepSeekMoE", arXiv:2401.06066;Gale et al., "MegaBlocks", arXiv:2211.15841。
- PD 分离与可观测性对照：Zhong et al., "DistServe", arXiv:2401.09670; Patel et al., "Splitwise", arXiv:2311.18677;W3C Trace Context / OpenTelemetry。
- 硬件与工具语义：PTX ISA §9.7.9.26.3.1–3.3（cp.async 组语义）、§9.7.15.5.8（mma.m16n8k16 fragment 布局）;NVIDIA Ada GPU Architecture Tuning Guide §1.4.1.1/§1.4.2; Ada 白皮书 SM 结构与 Appendix A Table 2;Nsight Systems User Guide (`--cuda-graph-trace`);Triton `triton.Config` 文档；Python `time` 模块文档。

## 2. 直觉与第一性原理

**先想没有 PD 分离的世界。** 一个引擎实例里 prefill 与 decode 抢同一批 SM：一条 8K 输入做 prefill 时，所有正在解码的请求都在等，ITL 被顶起来——经典的队头阻塞。PD 分离的价值主张只有两条：**①消除 prefill 对 decode 的干扰；②P 池与 D 池独立扩缩、各自选最优并行度**(REPORT §2.4)。**代价只有一条，但很硬：KV 必须搬家**，搬运量不是常数， 是 $B_{kv}(n) = n \times（\text{每 token KV 字节}）$，随输入长线性增长。

**日常类比与失效点。** 像中央厨房备菜、门店出餐：备菜与出餐不再抢同一个灶。类比在两处失效：①连锁店之间"运菜"相对烹饪是二阶小量，而本机 KV 通路只有 0.26–0.27 GB/s (telemetry-derived effective throughput，EXP-006/007)，搬运成了主项；②"独立扩缩"要求 P：D 比例可调，而 1P1D 是这个形态最退化的样子——**收不到扩缩红利，却全额支付传输成本**。

**判据（第一性原理形式）**：PD 值不值，取决于付出的 $t_\mathrm{xfer}(n)$ 与省下的 $\Delta t_\mathrm{interference}$ 孰大。这台机器上两件事让不等式必然向左倾斜：**并发 1 时右边恒为 0**——没有别的请求可被干扰，归因表（EXP-007《B1 四臂 offered-load 扫描战役》 §5，同热工况 p50）里 colocate 与 pd1p1d 的 TPOT 都是 15.9–16.4 ms，PD 没让 decode 变快，因为本来就没有干扰可消除；**满负载时左边爆炸**——8K 桶每请求要搬 469.8 MB（EXP-011《EXT-2 NixlPush 单点》 push 臂全量口径；pull 臂经前缀裁剪实拉 439.7 MB，EXP-006），按 0.27 GB/s 需 1.63 s。所以本仓的结论不是"PD 不好"，是**形态与互联能力错配**；真正要论证的是下一句：那条溢价**确实**由传输造成——问题于是从"部署选型"推进到 "归因方法"。

### 2.1 把判据写成不等式,然后逐项定价

上面那句判据可以写死：PD 分离相对混部的净收益是 $$\Delta = \underbrace{\Delta t_\mathrm{interference}(\lambda)}_{\text{省下的干扰}} \；-\；\underbrace{t_\mathrm{xfer}(n)}_{\text{付出的传输}} \；-\；\underbrace{o_\mathrm{handshake}+o_\mathrm{proxy}}_{\text{一次性与常数项}}.$$

三项各自的定价方式完全不同，这是本篇后面所有工作的分工：

- $\Delta t_\mathrm{interference}(\lambda)$ 是**负载的函数**，并发 1 时恒为 0——所以它只能在 sweep 里测，不能在 attribution 里测。本仓的 sweep 结果（讲义 01 §5.3） 已经给了答案：全负载段 pd 的 goodput 都低于 colocate，即这一项从未大到能翻盘。
- $t_\mathrm{xfer}(n)$ 是**输入长度的函数**，可以在并发 1 下干净地测——这正是 §3–§5 的主题。
- $o_\mathrm{handshake}$ 是**一次性**的：EXP-013《EXT-1 request 级 KV-wait 关联》的 idx=0 首请求 kv_wait 409.8/462.8/1770.3 ms，512 桶比后续请求高 292 ms，就是它的直接观测。

**方法论要点**：一个由多项组成的判据，不同项要用**不同的实验设计**去量。把它们混在一个数字里报，就永远说不清哪一项在起作用。

### 2.2 三条贯穿全篇的公理

- **公理 A（同身份同时钟）**：两个量能相减，当且仅当它们属于同一个对象、来自同一个时钟域。这两条在 PD 架构下都不平凡（§3.2），本篇 ~16 行改动的全部意义就是把它们建立起来。
- **公理 B（校验量必须落盘）**：一次性的自我保证不算证据。`identity_match`、 `sum_segments_ms` 都是**落盘字段**，任何人都能从 CSV 重算（§4 段 4）。
- **公理 C（效应量要和噪声量比，不是和零比）**：方向对不等于可报告。MoE 调优的 e2e +0.8~1.2% 方向一致、机理自洽，但低于跨会话漂移 ±5~8%，因此**不作 headline** (§3.7 ⑤、§5.3)。

### 2.3 观测一个分布式系统需要哪三个前置条件

后面 §3.3 的四个落点不是拍脑袋选的，它们分别满足下面三个条件中的一个：

| 条件 | 为什么不平凡 | 本篇怎么满足 |
|---|---|---|
| **身份可 join** | 一条请求在 client / P / D 三处有三个 id | client 自定 `X-Request-Id` 贯穿三方，D 端日志同时打出本地 req_id 与 remote_request_id |
| **时钟可相减** | 四个进程，时长要单调钟、对齐要墙钟 | kv_wait 用 `perf_counter`，两端另记 `time.time()` epoch（§3.4.2 讲两把尺的语义） |
| **观测零扰动** | 加了测量点就不再是原来的系统 | 打 patch 前后 TTFT 218/727/2738 vs 219/719/2719，噪声内（§3.5 对照臂） |

三条缺任何一条，后面的"54.2%"都退回成一个好看的数字。

## 3. 完整推导与机制

### 3.1 三条账:传输量、传输时间、容量上限

一步一理由：①**每 token KV 字节**——Qwen2-7B 28 层，每层 K、V 各一份，每份 4 个 KV 头 × 128 维，BF16 每元素 2 B：$28 \times 2 \times 4 \times 128 \times 2 = 57{，}344$ B/token（GQA 下 KV 头数 4 不等于注意力头数，这步最常算错）；②**块粒度**——KV 按 block(16 token)管理，$16 \times 57{，}344 = 917{，}504$ B/block，传输按块取整，非块对齐 prompt 向上取整（analysis/nixl_token_accounting.md 的"两计数器口径"表）；③**单请求传输量**——8192 token = 512 块 = 469,762,048 B ≈ 469.8 MB，EXP-013 实测 36/36 请求的 bytes 与该式**逐字节相等**(§3.5)；④**传输时间**——$t_\mathrm{xfer} = B_{kv}/BW_\mathrm{eff} = 469.8\，\mathrm{MB}/0.27\，\mathrm{GB/s} \approx 1.63$ s，EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》实测 avg xfer 1602.7 ms；⑤**容量上限**——传输在关键路径且串行，则 $\mathrm{req/s}_{\max} \approx 0.27/0.470 \approx 0.57$，EXP-007 实测饱和 0.54 req/s@8K。**传输带宽即容量**。

TTFT 侧对照（EXP-007 §5 归因表，并发 1、同热工况、p50）：PD 溢价 = 219.3−65.4 = 153.9 ms(512)、 718.6−224.9 = 493.7(2K)、2718.7−925.2 = 1793.5(8K)；溢价随输入长增长的形状与 $B_{kv}(n)$ 一致——但**形状一致不等于因果**。

#### 3.1.1 57,344 这个数字里,GQA 占了 7 倍

$B_{kv}$ 的每 token 系数完全由 config 决定：$2\，L\，KVH\，D\，s = 2\times28\times4\times128\times2 = 57{，}344$ B。**四个因子里最容易搞错的是 $KVH$**： Qwen2-7B 的 `num_attention_heads` 是 28 而 `num_key_value_heads` 是 4，用前者会把整笔账放大 7 倍。GQA 的原始动机就是压这一项（Ainslie et al.， arXiv：2305.13245，§2）。

**这条对 PD 分离的意义比对单机推理更大**：单机上 GQA 省的是显存与读带宽，PD 分离上它直接省的是**跨卡搬运量**。假想 MHA 版本的 Qwen2-7B，8K 请求的 KV 是 3.29 GB，按 0.27 GB/s 要 12.2 s——**本机 PD 的溃败程度已被 GQA 缓解了 7 倍，仍然溃败**。

#### 3.1.2 块粒度不是本仓的约定,是 vLLM 的默认值

"16 token 一块"来自 PagedAttention 论文的块大小扫描，原文："In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads. Accordingly, vLLM sets its default block size as 16."(Kwon et al., arXiv:2309.06180,§7.2)。同节给出两侧的失效机理： 块太小则"vLLM may not fully utilize the GPU's parallelism for reading and processing KV cache"，块太大则"internal fragmentation increases and the probability of sharing decreases"。

**它同时决定了两个层次的粒度**：块粒度（917,504 B）是**记账**粒度，descriptor 粒度（16,384 B）是**传输**粒度，两者差 56 倍（28 层 × K/V）。§3.6 会说明决定有效吞吐的是后者——**"块大小选得好"并不意味着"传输效率高"**。

### 3.2 从"分量对账"到"因果占比":缺的到底是什么

v1 报告（fig4）是**分量对账**：拿 colocate 的无负载 TTFT 当 PD 的 P 段，拿遥测 avg xfer 当传输段，剩下归"其余"。它给出 54–64%，方向没错，但有三个结构性弱点：①**跨臂替代**——P 段用的是另一个臂的数字，两臂工况不必相同；②**口径错配**——avg xfer 是一个测量点内所有传输的聚合均值，不是"这一条请求"的；③**边界不清**——握手、调度轮询、块分配被一股脑塞进"传输"或"其余"。

要升级成**因果占比**，只需两个条件，缺一不可：**同一请求身份**（分子与分母属于同一条请求）、 **同一时钟域**（两个量能相减）。这两条在 PD 架构下都不平凡。**身份**：一条请求有三个 id（client 自定 / P 端引擎内部 / D 端引擎内部）；NIXL 恰恰把"引擎身份 / 会话身份 / 内存寻址"三层 **正交拆开**（theory/02 §2，pull_scheduler.py：265-275 显式交出 remote_engine_id / remote_request_id / remote_block_ids）——这对健壮性是优点（0.17.1 P2pNccl 正因隐式 key 分叉而挂死，EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》），对观测则意味着**必须显式建立 join 键**。**时钟**：client / proxy / P / D 是四个进程，时长要用单调钟（perf_counter），跨进程对齐要用同一把墙钟（epoch）；本实验是同机 1P1D， epoch 天然同域——这是方法成立的前提，也是它的边界（§6）。

#### 3.2.1 "因果"这个词在这里到底能主张到什么程度

必须先把话说小：本文的"因果占比"**不是干预实验意义上的因果**（那需要改变 $BW_\mathrm{eff}$ 再看 TTFT 怎么动）。它主张的是一个更弱但可验证的命题：

> 在同一条请求的时间轴上，存在一段连续的、可命名的窗口 $[t_0， t_\mathrm{done}]$， 该请求在此窗口内**除等待远端 KV 之外无事可做**，且该窗口长度占 TTFT 的 54.2% / 62.5% / 64.2%。

这个命题比"相关"强，因为它是**同一对象、同一时间轴上的区间分解**，不是两组数字的统计关联； 它比"干预因果"弱，因为它不排除"如果传输变快，别的段会变慢"这类补偿效应。

**本仓确实有一个近似的干预臂**：EXP-011 换传输方向（push）使 $BW_\mathrm{eff}$ 从 0.26–0.27 升到 0.278–0.305 GB/s(+10–13%)，8K TTFT 相应从 2718.7 降到 2537.1 ms(−6.7%)。 **预测检验（本文现算）**：若 TTFT 中只有 kv_wait 随 $BW_\mathrm{eff}$ 变，则 $\Delta\mathrm{TTFT} \approx 1763.8\times(1 - 0.27/0.29) \approx 122$ ms， 即 −4.5%；实测 −181.6 ms(−6.7%)。**同量级、同方向，实测幅度略大**——差额的候选是 push 免去了 pull 的请求-应答回合（EXP-011 §6），即它不只改了带宽还改了一个常数项。这条对照把"占比"往干预因果的方向推了一步，但**不足以完全跨过去**，如实登记。

#### 3.2.2 身份的三层正交:优点与代价是同一件事

NIXL 把"引擎身份 / 会话身份 / 内存寻址"三层显式拆开（theory/02 §2； `pull_scheduler.py:265-275` 交出 remote_engine_id / remote_request_id / remote_block_ids）。按官方文档，agent 持全局唯一 ID、内存以 Memory Section 注册、传输由本地与远端两份 descriptor 列表加上目标 agent 名与操作（read/write）构成（ai-dynamo/nixl `docs/nixl.md`， Design / Memory Sections / Transfer 三节）。

**优点**：三层解耦意味着"谁是引擎""这是哪条会话""要读哪块内存"互不隐式绑定。 0.17.1 的 P2pNccl 正是因为把它们隐式塞进一个 rendezvous key 而在动态场景下分叉挂死（EXP-012 实机复现）。**代价**：没有任何一层天然携带"这是 client 的哪条请求"， 所以观测方必须**自己建 join 键**。

**一句话总结**：**健壮性来自解耦，可观测性来自重新耦合。** 本篇的 ~16 行改动做的就是后者——在不动前者的前提下，把三层重新串成一根线。

### 3.3 ~16 行改动的设计:为什么恰好是这几个落点

改动打在 v0.25.1 的 NIXL connector（D 端），逐行带 `# EXT1` 标记、原件备份可还原（`ext1/orig/`）；口径上是**本地可观测性改动，不是 NIXL/Connector 核心改造**（措辞约定）。三个落点各解决一个不可替代的问题：①**起点在 connector 首见请求处**(`pull_worker.start_load_kv`)而非 scheduler，因为 kv_wait 要**含握手**（首次与远端 P 建 NIXL agent 连接的一次性成本），放在 scheduler 会把握手甩到窗口外，首请求 +292 ms（512 桶，EXP-013 §5）就观测不到；②**聚合按 req_id 逐 handle 累加**(`base_worker._pop_done_transfers`)而非用现成 Prometheus counter，因为 counter 是**全局聚合量**，并发下无法归属到单条请求，而一条请求可能拆成多个 handle；③**输出是一行绑定三段身份与两段时钟的日志**(`_ext1_emit`)——跨行拼接要额外假设日志顺序，而日志顺序在多线程下不可靠。外加第 4 处：失败路径（`_handle_failed_transfer`）清理两个 dict，这不是记账需要，是**防泄漏**——一个典型的"观测代码把被观测系统弄坏"的失败模式。

**为什么不动核心记账路径**：一旦改 counter 语义，新数据与既有测量点（EXP-007）不再可比；而本次关键论据之一恰恰是"打 patch 前后 TTFT 218/727/2738 vs 219/719/2719，噪声内"——**观测零扰动的对照，必须建立在被观测路径未被改写之上**。

#### 3.3.1 四个落点的反事实检查

判断一个测量点是不是"不可替代"，办法是问：**把它挪到别处，哪个观测会消失、而自洽性仍然全绿？** 逐条过：

| 落点 | 挪到别处 | 消失的观测 | 自洽性还绿吗 |
|---|---|---|---|
| ① `start_load_kv` 记双钟 | 挪到"块分配完成后" | 首请求 +292 ms 的握手成本（EXP-013 §5） | **仍然全绿**——最危险的一类错误 |
| ② 按 req_id 逐 handle 聚合 | 用现成 Prometheus counter | 并发下无法归属到单条请求 | 单请求场景下也绿，并发一上就错 |
| ③ 一行日志绑定身份与两钟 | 拆成多行再离线拼 | 无——但要额外假设日志顺序 | 多线程下顺序不可靠，静默错配 |
| ④ 失败路径清理两个 dict | 不做 | 无 | 绿，但长跑内存单调上涨 |

**①这一行是本篇方法论的核心教训**：一个把语义悄悄改掉、却不破坏任何自洽性检查的改动， 是最难被发现的。防它的唯一办法是**把口径写进 docstring 并当成契约**（patch：36-41 的 docstring 就写死了"incl. handshake wait"）。

#### 3.3.2 "~16 行"这个规模本身是一条论据

改动小不是为了炫技，是为了让"观测零扰动"这条主张**先验上就可信**：四处改动全部是 ① 字典写入、② 整型累加、③ 一行惰性格式化日志、④ 字典删除。没有一处进入数据平面（不碰 descriptor 构造、不碰 NIXL 调用、不碰调度决策），所以**被观测路径的指令流没有变**。再加上后验的对照臂（打 patch 前后 TTFT 噪声内），先验与后验两侧都成立。

**反例长什么样**：如果为了记账在传输完成回调里加一次 `torch.cuda.synchronize()`， 或者把 `logger.info` 换成 f-string（即使日志级别关闭也要拼字符串），都会把开销引进关键路径。**"改动小"是可验证的性质，不是修辞**——`ext1/orig/` 留了原件，`diff` 一遍即可。

### 3.4 闭环误差 p50 <0.1% 是怎么来的:六段分解的望远镜性质

六段定义（口径见 `analyze_ext1.py:9-14` 的 docstring）：pre_proxy = `t_recv(proxy) − t_send(client)`（client→proxy 网络）；p_segment = `t_p_done − t_p_send`（P 端 prefill 含 HTTP）；gap_p_to_d = `t_d_send − t_p_done`（proxy 搬运 kv_transfer_params）；d_pre_kv = `t0_epoch(D conn) − t_d_send`（D 端 HTTP+排队+调度）； kv_wait = `done_epoch − t0_epoch`（D 等远端 KV）；post_kv = `t_first_token(client) − done_epoch`（D 首步 + 流式回传）。

**推导**：六段按定义展开求和，中间项两两相消（望远镜求和）：

$$\sum_{6} = (t_\mathrm{first\_token} - t_\mathrm{send}) - (t_\mathrm{p\_send} - t_\mathrm{recv}) = \mathrm{TTFT} - \delta$$

$\delta = t_\mathrm{p\_send} - t_\mathrm{recv}$ 是 proxy **内部**那一段：收到请求之后、发给 P 之前，代码在做 `await request.json()` 与选实例（`ext1_proxy.py:162-169`）。它 **没有被列进六段**，所以闭环误差不是测量噪声，而是一个**结构性的、可具名的缺口**——必然为正，且应随 prompt 体积增长。**实测验证（本文现算，可从仓内数据复算）**：

| 桶 | 闭环误差 p50(EXP-013 §5) | 残差 = TTFT − Σ六段（p50） | proxy 内部 `t_p_send − t_recv` |
|---|---|---|---|
| 512 | 0.08% | 0.176 ms | 0.169–0.239 ms |
| 2048 | 0.04% | 0.300 ms | 0.290–0.386 ms |
| 8192 | 0.02% | 0.515 ms | 0.349–0.819 ms |

三列逐桶对齐：**残差区间与 proxy 内部段区间完全重合**（全体 36 请求 0.169–0.820 ms， `raw/EXP-013/ext1_proxy_lines.txt`），且随 prompt 体积单调增长——与 JSON 解析成本的预期一致（推断，非独立实测）。闭环误差至此被 100% 解释。注意误差**百分比** 512 桶最大（0.084%）而残差**绝对值**最小（0.176 ms）：分母从 218.3 涨到 2738.0 ms，涨得比分子快（核验 $218.3\times0.00084=0.183$、$2738.0\times0.000189=0.518$ ms，与现算对上）—— **"误差小"看绝对量，"误差率小"可能只是分母大**（§6 误区 2）。

#### 3.4.1 望远镜求和为什么必然只剩一项(代数细节)

把六段按定义写开（记 client 侧发送为 $t_s$、收到首 token 为 $t_f$；proxy 侧收到为 $t_r$、发给 P 为 $t_{ps}$、P 返回为 $t_{pd}$、发给 D 为 $t_{ds}$；D 侧 connector 首见为 $t_0$、全部 handle DONE 为 $t_{dn}$）：

$$ \begin{aligned} \Sigma_6 =& (t_r - t_s) + (t_{pd} - t_{ps}) + (t_{ds} - t_{pd}) \\ &+ (t_0 - t_{ds}) + (t_{dn} - t_0) + (t_f - t_{dn}) \\ =& (t_f - t_s) - (t_{ps} - t_r). \end{aligned} $$

$t_{pd}$、$t_{ds}$、$t_0$、$t_{dn}$ 各出现两次且符号相反，逐对相消；**只有 $t_{ps}$ 与 $t_r$ 各出现一次**，因为六段里没有任何一段以 $t_r$ 为起点、以 $t_{ps}$ 为终点。所以残差 $\delta = t_{ps} - t_r$ 是**结构性的、必然为正的、可具名的**。

**这条代数有两个后果，必须一起讲**：
1. **正面**：残差不是噪声，是缺口。它应随 prompt 体积增长（JSON 解析成本）， 实测三桶 0.176 / 0.300 / 0.515 ms 单调上升，且与 proxy 内部段的实测区间完全重合。
2. **反面**：望远镜求和**不能证明任何一段的数值正确**。六段同时偏移一个常数 $c$， 相消之后 $\Sigma_6$ 一点不变。**闭环是完整性检验，不是精度检验**——这正是 §7 压力问 9 要回答的，也是为什么必须有链 1 与链 2 这两条"另一端不由我定义"的证据。

#### 3.4.2 两把尺:perf_counter 与 time.time 的语义为什么不能互换

Python 官方文档把这两者的契约写得很清楚：

- `time.perf_counter()`:"Return the value (in fractional seconds) of a performance counter, i.e. a clock with the highest available resolution to measure a short duration. It does include time elapsed during sleep. ... **The reference point of the returned value is undefined, so that only the difference between the results of two calls is valid.**"（CPython 实现上与 `time.monotonic()` 同源，单调不回退）
- `time.time()`:"Return the time in seconds since the epoch as a floating-point number. ... **While this function normally returns non-decreasing values, it can return a lower value than a previous call if the system clock has been set back between the two calls.**"

两条契约正好互补：**perf_counter 有单调性但没有共同原点，time.time 有共同原点但可能回退**。本篇的用法因此是唯一正确的组合——**时长用 perf_counter（单调、高分辨率）， 跨进程对齐用 epoch（有共同原点）**。

**换错会怎样，可以精确说**：若把 `done_epoch` 换成 perf_counter 值，则 $post\_kv = t_f（\text{client 的 epoch}） - done\_epoch（\text{D 的 perf\_counter}）$ 两个量没有共同原点，差值是一个**任意大的常数**（取决于两个进程各自的进程启动时刻）， post_kv 会算出荒谬值——而**闭环校验会立刻把它抓出来**，因为 $\Sigma_6$ 与 TTFT 会差同一个常数。这是"多条互相独立的校验"在实践中的价值：一个口径错误会同时打破多处。

**边界**：这套用法的前提是**同机**。跨节点时 epoch 不再同源，PTP/NTP 的残差（通常数十微秒到毫秒量级）会直接进入分解，0.1% 量级的闭环无从谈起（§6 边界①）。

### 3.5 三重互证:三条正交证据链,各自排除什么

**链 1 · 账目对不对（bytes 与 descriptor 双恒等）**。逐请求 bytes 求和 = **7,398,752,256** = Prometheus `vllm:nixl_bytes_transferred_sum`，分毫不差（EXP-013 §5）。本文进一步核验 descriptor：逐请求 Σdescs = **451,584** = `vllm:nixl_num_descriptors_sum` 的 after 值（before 全 0，`raw/EXP-013/metrics_8200_{before,after}.prom`），同样分毫不差。它排除：日志行丢失、重复计入、handle 漏聚合。再算两步（本文现算，纯算术）： $7{，}398{，}752{，}256 / 451{，}584 = 16{，}384$ B——每 descriptor **恰 16 KiB**； $7{，}398{，}752{，}256 / 57{，}344 = 129{，}024$ token $= 12 \times (512+2048+8192)$——**恰等于协议期望的全部 prompt token**，即零本地前缀命中、零舍入。对照 EXP-006 的固定 seed 协议：那次 262,144 − 245,344 = 16,800 token 的缺口全部是 D 端本地 prefix cache 命中（analysis/nixl_token_accounting.md 逐块定罪，其中 511 块源码定罪于 bench 的 test 请求）。同一套记账在两种协议下给出两种结果而两次都对上账——这是**协议 v2（每请求唯一 seed） 有效性的独立验证**。

**链 2 · 归因边界对不对（kv_wait ≈ xferDuration）**。kv_wait 是**墙钟**等待窗口，xferDuration 是 NIXL **自报**的纯传输时间，来源完全独立；实测差 0.3–1.9 ms（各桶 p50 = 0.33/0.71/1.78 ms， EXP-013 §6）。它排除"kv_wait 里混了大量调度轮询/握手/块分配"。**注意**：xferDuration 已含 posting，**不与 postDuration 相加**。**链 3 · 分解完不完整（六段闭环）**。§3.4 已证：误差 p50 <0.1%（最差桶 0.084%，逐请求最大 0.11%），且残差被指认到 proxy 内部解析段；它排除"某段被重复计入或漏掉"。

**外加对照臂 · 观测有无扰动**：打 patch 后 TTFT p50 218/727/2738 vs 未打 patch 的矩阵 219/719/2719(EXP-006/007)，噪声内——没有这条，前三条都可能是"被观测系统已经变了样"的自洽假象。三条链 + 一条对照，才撑起"KV 等待占 TTFT 54.2% / 62.5% / 64.2%（512/2K/8K，p50，每桶 n=11， 排除 idx=0 首请求）"这句**因果占比**声明。

#### 3.5.1 三条链的独立性是怎么保证的

"三条证据"只有在**各自的另一端不由同一个人定义**时才算独立。逐条检查：

| 链 | 我方的量 | 对方的量 | 对方由谁定义 | 排除了什么 |
|---|---|---|---|---|
| 1 · 账目 | EXT1_KV 日志逐请求 bytes / descs 求和 | Prometheus `nixl_bytes_transferred_sum` / `nixl_num_descriptors_sum` | **上游 vLLM 的既有 counter**（未被 patch 改动） | 日志行丢失、重复计入、handle 漏聚合 |
| 2 · 边界 | kv_wait（墙钟窗口） | xferDuration（**NIXL 自报**） | **NIXL 库内部** | 窗口里混了大量调度轮询/握手/块分配 |
| 3 · 完整 | 六段之和 | client 侧 TTFT | **bench 客户端**（与打点体系无关） | 某段被重复计入或漏掉 |
| 对照 | 打 patch 后 TTFT | 未打 patch 的矩阵 TTFT | **EXP-006/007 的历史数据** | 观测改变了被观测对象 |

四行的"对方"分别是：上游 counter、NIXL 库、bench 客户端、历史实验。**没有一个是本次 patch 产出的**——这就是独立性的操作定义。

**为什么必须是三条而不是一条**：链 3（闭环）在代数上只能剩一项（§3.4.1）， 所以它单独**不能**证明任何数值正确；链 1 只证明账目不漏不重，不证明窗口里装的是什么； 链 2 只证明窗口里装的是传输，不证明没有别的段被吃掉。**三条各自留有一个漏洞， 而三个漏洞互不重叠**——这才是"三重互证"这个说法成立的理由，不是"证据多就更可信"。

#### 3.5.2 链 1 的两个整数为什么值得单独算一遍

$7{，}398{，}752{，}256 / 451{，}584 = 16{，}384$ 与 $7{，}398{，}752{，}256 / 57{，}344 = 129{，}024 = 12\times(512+2048+8192)$ 这两步都是纯算术， 但它们各自封住一个可能的解释：

- 第一步**封住"descriptor 大小是变的"**：如果 descriptor 大小随传输规模变化， 总字节除以总个数不会恰好落在 $2^{14}$ 上；
- 第二步**封住"有本地前缀命中"**：129,024 恰等于协议期望的全部 prompt token 数（三桶各 12 请求），说明零命中、零舍入。

**对照面很关键**：EXP-006 用的是固定 seed 协议，同一套记账给出的是 262,144 − 245,344 = 16,800 token 的缺口，全部被 `analysis/nixl_token_accounting.md` 逐块定位为 D 端本地 prefix cache 命中（其中 511 块源码定罪于 bench 的 test 请求）。 **同一套记账在两种协议下给出两种结果、而两次都对得上账**——这既验证了记账本身， 也构成协议 v2（每请求唯一 seed）有效性的独立证据。

### 3.6 0.27 GB/s 的碎片化根因:从 descriptor 恒等式反解

链 1 里那个 16,384 B 不是巧合，可从第一性原理推出： $16\ \text{token/block} \times 4\ \text{KV 头} \times 128\ \text{维} \times 2\，\mathrm{B} = 16{，}384\，\mathrm{B}$，即**一个 descriptor =（一层， K 或 V， 一个 block）**。每 block 需 $28 \times 2 = 56$ 个 descriptor，$56 \times 16{，}384 = 917{，}504$ B/block，与 §3.1 第 2 步闭合。实测 descs 逐桶 1792 / 7168 / 28672（`derived/ext1_per_request.csv`，36/36 无一例外）= $56 \times$（32/128/512 块），逐字对上。**关键判据（本文现算）**——每请求 xferDuration ÷ descriptor 数：

| 桶 | descs | xfer p50 (ms) | **每 descriptor 耗时** | 等效吞吐 |
|---|---|---|---|---|
| 512 | 1,792 | 117.9 | 65.8 µs | 0.249 GB/s |
| 2048 | 7,168 | 454.5 | 63.4 µs | 0.258 GB/s |
| 8192 | 28,672 | 1762.7 | 61.5 µs | 0.267 GB/s |

descriptor 大小恒定 16 KiB，每 descriptor 耗时也几乎恒定（61.5–65.8 µs）——**传输时间对 descriptor 计数线性，而不是"带宽×时间"**。这就是碎片化的定量指纹：单次传输大小从没变过，吞吐被钉死在 $16{，}384\，\mathrm{B}/63\，\mu s \approx 0.26\，\mathrm{GB/s}$。**量级对照**：EXP-002《硬件三数》实测 GPU 间延迟 14.5–15.9 µs，一次 16 KiB 花 ~62 µs 约为裸延迟的 4 倍——成本主要落在每次传输的固定开销（descriptor 处理、launch、同步），不在搬字节本身； 这也解释了 fig5 上那条 ~12 ms 的"小传输延迟地板"（smoke 0.188 MB / 14.1 ms，EXP-001《NIXL 1P1D smoke 与版本裁决》）。 **工程推论（可证伪）**：要提速必须**合并 descriptor**（层维度批量成更大连续块），而不是换方向——方向已被实测排除：NixlPush 8K TTFT −6.7%、吞吐 +10–13%，**量级不变** (EXP-011)。**口径约定**：0.26–0.27 GB/s 只能称 telemetry-derived effective throughput， 不能讲成链路物理带宽。

#### 3.6.1 用两参数模型把"碎片化"写成公式(本讲义推导)

按讲义 01 公理 B，一次传输的时间是 $t(m)=\alpha+m/\beta$，有效吞吐 $$T(m)=\frac{m}{\alpha+m/\beta}.$$ 本机实测：$m=16{，}384$ B 固定，$t$ 在三个桶里是 61.5–65.8 µs，**几乎不随桶变**。把两个已知端点代进去反解：

- 若 $\alpha \gg m/\beta$，则 $T(m)\approx m/\alpha = 16{,}384/62\,\mu s \approx 0.264$ GB/s——与 telemetry 反解的 0.26–0.27 GB/s 一致；
- 用 EXP-002 的单向裸拷贝 $\beta\approx 0.6$–0.91 GB/s 反查： $m/\beta = 16{，}384/0.75\，\mathrm{GB/s}\approx 22\，\mu s$，占 62 µs 的 35%； 余下 40 µs 是 $\alpha$。**两项同量级而 $\alpha$ 略大**，与"每 descriptor 耗时几乎恒定"这一观测自洽（若 $\beta$ 项主导，耗时会随 $m$ 变——但 $m$ 本来就不变， 所以这条只能作为量级检查，不能作为 $\alpha/\beta$ 的精确分离）。
- **半带宽消息长度** $m_{1/2}=\alpha\beta \approx 40\，\mu s\times0.75\，\mathrm{GB/s} \approx 30$ KB。**16 KiB 恰好落在 $m_{1/2}$ 之下**——这就是"碎片化"的精确含义： 当前的传输粒度处在曲线的固定开销主导侧。

**口径提醒**：上面的 $\alpha$/$\beta$ 分离是**推断**，因为本仓只在单一 $m$ 下测过， 一个点解不出两个参数。要真正分离必须做 descriptor 大小扫描——本仓未做，不主张精确值。

#### 3.6.2 descriptor 大小由 connector 的 region 划分决定(源码依据)

16 KiB 不是 NIXL 的选择，是 vLLM NixlConnector 注册内存的方式决定的。v0.25.1 的 worker 侧在建立 Memory Section 时把 K 与 V 注册成**不同的 region**，源码注释原文： "K and V are now in different regions. Advantage is that we can elegantly support MLA and any cases where the K and V tensors are non-contiguous"。于是 region 数 $= L\times 2 = 56$，而 descriptor id 由 `(region_id, block_id)` 二元组线性化生成（`_compute_desc_ids`：`region_ids * num_blocks + block_arr`）。

**结论**：一个 descriptor $=$ 一层的 K 或 V 在一个 block 上的连续片段 $= 16\times KVH\times D\times s = 16{，}384$ B。这解释了 §3.6 的恒等式，也说明 **要改这个粒度必须改 region 划分，而不是改 NIXL 参数**。

#### 3.6.3 合并 descriptor 能救多少:一个可证伪的上界(本讲义推导)

工程推论是"合并 descriptor"，但收益不是无限的，可以算出上界：

- **合并到什么程度**：把一个 block 的 56 个 descriptor 合成 1 个，$m$ 从 16 KiB 涨到 917,504 B（56 倍）。
- **若 $\alpha$ 不变（40 µs）**:$T = 917{,}504/(40\,\mu s + 917{,}504/\beta)$。代入 $\beta=0.75$ GB/s：分母 $=40+1223=1263\,\mu s$,$T=0.73$ GB/s。
- **上界由谁决定**：注意此时 $\beta$ 项已经主导（1223 µs vs 40 µs）， **所以合并之后的天花板不再是 $\alpha$，而是这条链路本身的单向带宽 0.60–0.91 GB/s** (EXP-002)。

**结论（可证伪）**：合并 descriptor 的收益上界约为 **2.4–3.4×**(0.27 → 0.6–0.91 GB/s)， 不是一两个数量级。要跨过 1 GB/s，必须换互联（有 P2P/NVLink/RDMA），不是换实现。 **这条上界把"优化方向"和"优化幅度"分开说清楚了**：方向正确，但即使做到极限， 8K 桶的传输仍需 0.5–0.8 s，PD 相对 colocate 的 TTFT 溢价仍在 50% 以上—— **结论不会翻转**。本仓未做该改造，以上为推导，不作实测主张。

**与已排除方向的对照**：换传输方向（push）实测只挽回 8K TTFT 6.7%、有效吞吐 +10–13% (EXP-011)——因为它改的是常数项（免去请求-应答回合）而不是 $m$。 **两条放在一起就是一个清晰的判据：改 $m$ 有 2–3× 的空间，改协议只有 10% 的空间。**

### 3.7 同一套方法搬到 MoE:先分解、再归因、后验证

PD 那条线的骨架是：**先把总量分解到可归属的段，再用独立证据链锁死归因，最后设对照臂验证**。 MoE 线同法，只是尺子从"请求级时间"换成"kernel 级 GPU wall-time"：①**分解**——nsys 采一个 20 s 稳态窗，`cuda_gpu_kern_sum` 按 kernel 名归 9 类，bs=32 时 **fused_moe grouped GEMM 占 56.4%**，bs=1 时反而是 dense GEMM/GEMV 占 40.9%（EXP-014《D1 MoE decode 分解》 §5）；②**归因**——MoE/dense 的 decode 优势 **2.03×(bs=1) → 0.97×（bs=8，反转点）→ 0.82×(bs=128)**，机理是 top-4/60 路由下 batch 增大后每 step 命中的专家并集趋于全量（60 专家约 28.6 GB > dense 14.2 GB），bs=1 的激活权重优势（2.7 GB/step）反转为读放大劣势，分解表印证——grouped GEMM 占比 18.7% → 56.4% (EXP-014 §6)；③**目标由数据锁定**——serving batch(≥8)下唯一大头是 fused_moe，而该形状的 Triton config 在上游**社区空缺**（运行时告警在案，EXP-009《C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据》 §5），moe_align(≤4.1%)与 permute (≤0.5%)不值得动；④**三级验证**——correctness（没算错）/ kernel A/B（主证据）/ e2e（验证机理自洽）；⑤**折算式** $\Delta_\mathrm{e2e} \approx \Delta_\mathrm{kernel} \times$ 该 kernel 时间占比——kernel 端 M≥128 改善 3.3–3.9%，乘 56.4% 得 e2e 约 2% 的上限，实测 TPOT **+0.8~1.2%**，同量级、方向一致，机理自洽；但该幅度**低于跨会话漂移**(±5~8%)，按仓内措辞约定 **不作 headline**，主证据是 kernel A/B 两端数字。**折算对上了，不等于可以拿它当卖点**——判据是"效应量 vs 噪声量"，不是"方向对不对"。

#### 3.7.1 grouped GEMM 的 kernel 组织:为什么"分组"是关键词

"grouped GEMM"这个名字容易被误解成"一堆独立的小 GEMM"。vLLM 的 `fused_moe_kernel` 不是那样组织的，它把所有专家的计算**摊进同一个 kernel 的同一个网格**，靠三个索引张量把 token 与专家对上号（源码 docstring，`vllm/model_executor/layers/fused_moe/fused_moe.py:370-383`， main@7aa248fc）：

- `sorted_token_ids`:"A tensor containing the sorted indices of tokens, repeated topk times and arranged by the expert index they are assigned to";
- `expert_ids`:"A tensor containing the indices of the expert for each block. It determines which expert matrix from B should be used for each block in A";
- `num_tokens_post_padded`：补齐之后的 token 总数。

docstring 把设计意图写死了："The sorting of `sorted_token_ids` by expert index and padding ensures divisibility by `BLOCK_SIZE_M`, which is necessary to maintain consistency in block matrix multiplication across different blocks processed by the same expert."

**这句话是整个 MoE kernel 组织的核心**：按专家排序 + 每专家补齐到 `BLOCK_SIZE_M` 的整数倍， 使得**每个 M 方向的 block 恰好只属于一个专家**，于是 kernel 里只需一次 `tl.load(expert_ids_ptr + pid_m)` 就能定位该 block 该用哪份专家权重（`fused_moe.py:423`：`off_experts = tl.load(expert_ids_ptr + pid_m)`），不需要任何 block 内的分支。

**补齐的代价可以算，而且上游把公式写在代码里**： `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)` (`moe_align_block_size.py:74`)——每个专家最多浪费 $block\_size-1$ 个 token 槽位。 bs=32、top-4 时命中约 53 个专家（§3.7.3），$BLOCK\_SIZE\_M=16$ 下最坏浪费 $53\times15 = 795$ 个槽位，而实际有效槽位是 $32\times4=128$ 个——**浪费可以数倍于有效计算**。这就是"小 batch 下 MoE 的 grouped GEMM 效率低"的机理级解释，也是 §3.8.1 里 `BLOCK_SIZE_M=16` 在小 M 档被搜索选中的原因：**块越小，补齐浪费越少**。

`moe_align_block_size` 就是生成这三个张量的算子（`moe_align_block_size.py:11-73` 的 docstring 给了完整的排序-补齐示例）。它在本机 bs=1 时占 4.1%、bs=32 时占 1.0% (EXP-014)——**不是热点，但它决定了热点 kernel 能不能被写成无分支的形式**。

#### 3.7.2 专家并行:同一个 kernel 怎么表达"这个专家不在我这张卡上"

`--enable-expert-parallel` 打开后，60 个专家按 rank 切分，每 rank 持 30 个。实现方式不是换 kernel，而是**在索引层打标记**：`moe_align_block_size` 的 docstring 写明 "In the case of expert_parallel, moe_align_block_size initially considers all experts as valid and aligns all tokens appropriately. Before the function returns it marks the experts_ids that are not in the current GPU rank as -1 so the MoE matmuls could skip those blocks."(`moe_align_block_size.py`)

于是主 kernel 里只有两行 `off_experts = tl.load(expert_ids_ptr + pid_m)` / `if off_experts == -1:`（`fused_moe.py:423-424`；gptq/awq 变体在 `:166-167` 同构）——**不在本 rank 的 block 直接退出，不做任何访存**。

**这条设计决定了两个 config 文件的形状**：
- **EP(TP2 + expert parallel)**：每 rank 30 个专家，每个专家的中间维仍是完整的 1408 → 文件名 `E=30,N=1408`；
- **非 EP（纯 TP2）**：每 rank 仍见到全部 60 个专家，但每个专家的中间维被 TP 切成一半 → 文件名 `E=60,N=704`。

两者的 GEMM 形状完全不同（前者 M 小 N 大、专家少；后者专家多、N 小），所以**必须分别调优**——这正是 EXP-015《D2 MoE config 调优》要产出两个 JSON 的原因，也是 A/B 必须 EP 与非 EP 分开跑的原因（§4 段 8）。

**与文献的关系**：按 rank 切专家是 GShard 提出的专家并行的标准形态（Lepikhin et al.， arXiv：2006.16668）；Qwen1.5-MoE 的"4 个常驻共享专家 + 60 路由专家取 4" 则是 DeepSeekMoE 的细粒度切分 + 共享专家隔离两条策略的实例（Dai et al.， arXiv：2401.06066：把专家细分并"isolating $K_s$ experts as shared ones， aiming at capturing common knowledge and mitigating redundancy in routed experts"）。 **共享专家永远被激活，所以它属于 §5.2 表里的 dense GEMM 桶，不属于 grouped GEMM 桶**——分类器把它们分开，是读懂那张表的前提。

#### 3.7.3 反转点为什么在 bs≈8:一个可以手算的并集增长(本讲义推导)

每个 token 独立地从 60 个路由专家里取 4 个。若路由近似均匀，一个 batch 里 $B$ 个 token 命中的**不同专家个数**的期望是 $$E[\#\text{experts}] = 60\left(1-\left(1-\tfrac{4}{60}\right)^{B}\right).$$ 代入：

| B | 期望命中专家数 | 占 60 的比例 |
|---|---|---|
| 1 | 4.0 | 6.7% |
| 4 | 14.5 | 24% |
| **8** | **25.5** | **42%** |
| 32 | 53.4 | 89% |
| 128 | 60.0 | 100% |

**这张表就是反转的机理**。每 step 要读的路由专家权重 $\approx$ 命中比例 × 24.9 GB（全部路由专家 BF16 字节数，本文现算：24 层 × 60 专家 × 3 矩阵 × 2048 × 1408 × 2 B）， 而**分摊到每个 token 的读取量在 $B$ 小时按 $1/B$ 下降、在 $B$ 大时趋于常数**。 dense 侧没有这个结构，每 step 恒读 15.2 GB。两条曲线必然相交，交点由上式决定。

**bs=1 的核对（本文现算，与仓内 roofline 锚一致）**：激活权重 $= 4$ 专家 $\times 3 \times 2048\times1408\times2$ B $\times 24$ 层 $= 1.66$ GB（路由） $+ 1.66$ GB（共享专家，`shared_expert_intermediate_size` 5632）$+ 0.81$ GB（注意力） $+ 1.24$ GB（embedding + lm_head，`tie_word_embeddings` 为 false）$\approx 5.4$ GB， TP2 后每卡 2.7 GB——**与 `moe_perf/d1_analyze.py` 的 roofline 注释（"~2.7B 参数/~5.4GB， 双卡各 ~2.7GB → 上限 ~373 tok/s"）逐项吻合**。这条核对说明那个 roofline 锚不是估的。

**能预测到什么程度，要说实话**：上式预测 bs=32 时 MoE 每 step 读 $0.89\times24.9 + 3.7 \approx 25.9$ GB vs dense 15.2 GB，若两者都纯带宽受限， 吞吐比应为 $15.2/25.9 = 0.59$；实测是 **0.81**(1691 vs 2094 tok/s，EXP-014)。 **差距在 0.2 以上，本仓没有数据裁决**。候选：(a) bs=32 时 dense 已部分转入计算/调度约束， 其带宽利用率低于 MoE；(b) 路由不均匀（热门专家被反复命中）使实际并集小于均匀假设； (c) 权重在 L2 中的跨 step 复用。**如实登记为开放问题，不做归因**——但注意， 上式要解释的主要现象（**存在交点、交点在个位数 batch、grouped GEMM 占比从 18.7% 涨到 56.4%**） 三条都对上了。

#### 3.7.4 折算式 $\Delta_\mathrm{e2e}\approx\Delta_\mathrm{kernel}\times p$ 的合法性

这就是 Amdahl 定律的一阶形式：若某部分占总时间比例 $p$、该部分加速 $1/(1-\delta)$， 则总时间变为 $(1-p)+p(1-\delta)$，即总加速 $\approx p\delta$（当 $\delta$ 小）。 **三个前提必须同时成立**：

1. **$p$ 与被优化对象口径一致**。56.4% 是 bs=32 的 nsys 窗内 GPU kernel wall-time 占比， 而 e2e 的 TPOT 里还含 CPU 侧调度、采样、Python 开销——**分母不同**。所以折算给的是 **上界**，不是预测。
2. **优化不改变其余部分**。换 config 只改 Triton 的 tile 与流水深度，不改算子序列， 这一条成立。
3. **kernel 加速在实际 M 分布上成立**。A/B 测的是离散 M 档，而 serving 时 M 随 batch 与 token 分布变化，还要经过最近邻查表（§3.8.4）——**这是 e2e 幅度低于折算上界的第一候选**。

代入：kernel 端 serving 相关 M 档改善 3.3–3.9%，$\times 56.4\% \approx 2\%$ 上界； 实测 TPOT +0.8~1.2%，落在上界之下、同量级、同方向。**三个前提都写清楚之后， "机理自洽"这句话才有内容**。

### 3.8 Triton config 的六个旋钮:每一个都对应一条硬件约束

D2 产出的 JSON 里每个 M 档有六个键。它们不是抽象超参，每一个都直接映射到 Ada (compute capability 8.9)的一条硬件语义。**读懂这一节，才能解释"为什么搜索空间是有限的" 以及"为什么中段打平"。**

#### 3.8.1 BLOCK_SIZE_M/N/K + num_stages:共享内存的硬上限

Triton 的 GEMM 主循环把 A 的 $[BM， BK]$ 片与 B 的 $[BK， BN]$ 片搬进共享内存， 并按 `num_stages` 做多缓冲。每级缓冲的字节数（BF16，$s=2$）： $$\mathrm{smem/stage} = (BM\cdot BK + BK\cdot BN)\times 2\ \mathrm{B}.$$

硬上限来自 Ada Tuning Guide §1.4.1.1:"The shared memory capacity per SM is 100 KB." "The maximum shared memory per thread block is 99 KB." "CUDA reserves 1 KB of shared memory per thread block."（§1.4.2.2 另给可配置档位："supports shared memory capacity of 0, 8, 16, 32, 64 or 100 KB per SM"）。

**把 36 个 tuned tuple 逐个代入（本文现算）**：

| 口径 | EP 文件最大值 | 非 EP 文件最大值 | 是否越界（99 KB） |
|---|---|---|---|
| $\mathrm{smem/stage}\times num\_stages$ | 100.0 KB（M=32 档） | 100.0 KB（M=24 档） | **恰好越界 1 KB** |
| $\mathrm{smem/stage}\times(num\_stages-1)$ | 80.0 KB | 80.0 KB | 全部合法 |

**这张表推出一个可检验的结论**：既然这两个 tuple 是 `benchmark_moe.py --tune` 实际跑通并选出来的（编译不过的配置会被搜索过程剔除），那么 Triton 的流水实现必然是 **$(num\_stages-1)$ 级多缓冲**，而不是 $num\_stages$ 级。$\times num\_stages$ 的读法会给出 100.0 KB——**恰好等于每 SM 容量、恰好超过每 block 上限 1 KB**，这个"恰好"不可能是巧合。（推断，基于 Ada Tuning Guide 的两个上限值与 36 个 tuple 的实测可用性；本仓未反编译验证。）

**顺带得到搜索空间的边界**：36 个 tuple 里最大占用 80 KB，离 99 KB 还有余量， 说明**搜索没有被共享内存卡死**——限制它的是别的东西（占用率、L2 复用、补齐浪费）。

#### 3.8.2 num_stages 的硬件语义:cp.async 的 commit-wait 组

`triton.Config` 的官方描述是：num_stages 为"the number of stages that the compiler should use when software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs."。"SM80+"这个限定语泄露了实现：软件流水靠的是 Ampere 引入的异步拷贝指令 `cp.async`(PTX ISA："Requires sm_80 or higher")。

PTX ISA 把这套机制的语义写得非常精确（§9.7.9.26.3.1–3.3）：

- `cp.async` 本身是**非阻塞**的："a non-blocking instruction which initiates an asynchronous copy operation of data from ... global state space ... to ... shared state space"；其大小只能是 4/8/16 字节（"cp-size can only be 4, 8 and 16"）; `.cg` 只在 L2 缓存、`.ca` 在包括 L1 的各级缓存。
- `cp.async.commit_group` **把此前所有未提交的 cp.async 打成一组**："creates a new cp.async-group per thread and batches all prior cp.async instructions initiated by the executing thread but not committed to any cp.async-group into the new cp.async-group." 同组之内**没有顺序保证**："There is no memory ordering guarantee provided between any two cp.async operations within the same cp.async-group."
- `cp.async.wait_group N` **只等到"最多还剩 N 组在飞"**:"will cause executing thread to wait till only N or fewer of the most recent cp.async-groups are pending and all the prior cp.async-groups committed by the executing threads are complete."

**三条合起来就是 num_stages 的含义**：编译器为每一级流水发一批 `cp.async`，用 `commit_group` 封成一组，计算前用 `wait_group N` 让最老的一组落地、较新的几组继续在飞。 **"还剩 N 组在飞"这个语义就是"流水深度"的字面实现**（具体的 N 取值本仓未做 PTX 级验证）。

**由此可以解释 config 表里的一个现象**：小 M 档（1/2/4/8）普遍选 `num_stages=4~5`， 大 M 档（≥512）普遍选 2。小 M 时每级缓冲小（10–20 KB），多压几级几乎不花共享内存， 而访存延迟需要更深的流水去藏；大 M 时每级缓冲大（32–40 KB），再加深就会挤占共享内存、压低每 SM 的 block 数。**深度与占用率的取舍，被共享内存上限量化地绑在一起**(§3.8.1)。

#### 3.8.3 num_warps 的硬件语义:Ada 每个 SM 只有四个 warp scheduler

`triton.Config` 的描述：num_warps 是"the number of warps to use for the kernel when compiled for GPUs. For example, if num_warps=8, then each kernel instance will be automatically parallelized to cooperatively execute using 8 * 32 = 256 threads."

对应的硬件事实来自 Ada 白皮书的 SM 结构："the AD10x SM is divided into four processing blocks (or partitions), with each partition containing a 64 KB register file, an L0 instruction cache, **one warp scheduler, one dispatch unit**, 16 CUDA Cores ... one Ada Fourth-Generation Tensor Core, four Load/Store units, and a Special Function Unit"。

**于是 num_warps 的两个常用取值有精确含义（本讲义推导）**：
- `num_warps=4`：每个 partition 恰好分到 1 个 warp。**分区内没有第二个 warp 可切换**， 一旦该 warp 因等 `cp.async` 或等 MMA 结果而停顿，该分区的发射槽就空转——延迟只能靠软件流水（num_stages）去藏。
- `num_warps=8`：每个 partition 2 个 warp，调度器可以在两者之间切换，**多了一层硬件级的延迟隐藏**；代价是每 warp 可用的寄存器减半（64 KB/partition 固定）， 寄存器压力上升可能反过来压低占用率（Ada 每线程最多 255 个寄存器，Tuning Guide §1.4.1.1）。

上游默认启发式写的是 `num_warps = 4 if M <= 128 else 8`(`fused_moe.py:1393-1395`)， 注释给的理由是"Large batches have enough blocks to saturate the GPU， so we use more warps per block to increase arithmetic intensity"。**而搜索结果并不同意这条二分**(§3.8.5)。

#### 3.8.4 GROUP_SIZE_M 与 L2:分组排序换的是什么

kernel 的 pid 映射不是行优先，而是分组的（`fused_moe.py:385-396`），注释写明 "This is done in a grouped ordering to promote L2 data reuse"。含义是：把 `GROUP_SIZE_M` 个 M-block 编成一组，组内先走完所有 N-block 再换行——于是同一组内的 block 反复命中同一批 B（专家权重）tile，**这些 tile 在 L2 里活着**。

RTX 4090 的 L2 是 73728 KB（Ada 白皮书 Appendix A Table 2；完整 AD102 是 98304 KB， Ada Tuning Guide §1.4.2.1 称其为"16x larger than GA102"）。一个专家的一份权重 $2048\times1408\times2\，\mathrm{B} = 5.5$ MB，三份 16.5 MB——**几个专家的权重就能填满 L2 的一大块**， 所以分组的收益在专家数多、每专家 token 少的 MoE 形状下尤其明显。

上游启发式对 `GROUP_SIZE_M` 的处理是：`tokens_per_expert = M // max(E, 1)`， `group_m = 16 if tokens_per_expert > 128 else 1`，注释理由是"with many experts each one sees few tokens so grouping is useless"。**代入本机**：EP 下 $E=30$，要让 `tokens_per_expert > 128` 需要 $M > 3840$——**即在整张 config 表的 18 个档里， 只有 M=4096 这一档会开启分组**。而搜索结果在 M=1 就选了 `GROUP_SIZE_M=32`， 在 M=24/96/128/256 选了 64。**启发式的"没用"判断在这个形状上过于保守**，这是两端有收益的一个具体来源。

**查表规则的一个细节必须补上**：文件是**精确匹配**（`E=<专家数>,N=<中间维>, device_name=<GPU>.json`，不在就回退默认启发式，`fused_moe.py:1145-1166`）， 但**文件内部的 M 档是最近邻匹配**： `config = configs[min(configs.keys(), key=lambda x: abs(x - M))]`(`fused_moe.py:1441`)。所以 serving 时真正生效的档位，是离当前 token 数最近的那个键——**这就是 §3.7.4 第 3 条前提"kernel A/B 的离散 M 档不等于 serving 的实际 M 分布"的机制来源**。

#### 3.8.5 启发式 vs 搜索:36 个 tuple 无一完全一致(本文现算)

把上游默认启发式（`fused_moe.py:1366-1412` 的 bf16 分支）在 18 个 M 档上各算一遍， 与 tuned JSON 逐字段比对：

| 字段 | EP 文件一致档数 | 非 EP 文件一致档数 |
|---|---|---|
| BLOCK_SIZE_M | 7 / 18 | 6 / 18 |
| BLOCK_SIZE_N | 8 / 18 | 10 / 18 |
| BLOCK_SIZE_K | 11 / 18 | 8 / 18 |
| GROUP_SIZE_M | 6 / 18 | 8 / 18 |
| num_warps | 7 / 18 | 7 / 18 |
| num_stages | 7 / 18 | 7 / 18 |
| **六字段全一致** | **0 / 18** | **0 / 18** |

**这张表把"中段打平"的解释改写了**。原来的说法是"中段 M 恰落在启发式调得较准的区间， 搜索结果与它撞车"；但逐字段比对显示**没有一个档位撞车**，连一次都没有。所以正确的解释是：**中段的目标函数是平坦的**——存在一个宽的近优平台，启发式给的点与搜索给的点落在平台的不同位置，时间却几乎相同（A/B 实测中段 $\Delta\approx0$）。

**这个改写有实际后果**：若中段是"启发式已最优"，kernel 级优化就没有空间；若是"平台很平"， 空间存在但不在 tile 参数里，要找收益得换维度（算子融合、量化、更好的 permute 策略）。 **EXP-015 §6 的 D3 判定（"config 即最优杠杆，不做无数据支撑的 kernel 改动"）在两种解释下都成立**，但理由不同，而理由决定下一步往哪找。

### 3.9 魔法数总表:每个数字由什么决定

| 数字 | 出现在 | 由什么决定 | 换平台/换模型怎么变 |
|---|---|---|---|
| 57,344 B/token | §3.1 | **理论上界**：$2LKVH\，D\，s$，config 完全确定 | 按公式重算；MHA 会 ×7 |
| 16 token/block | §3.1.2 | **实测扫描**：PagedAttention §7.2 的块大小曲线 | vLLM 可配置 |
| 16,384 B/descriptor | §3.6.2 | **理论上界**：由 connector 的 region 划分（K/V 分开）推出 | 改 region 划分即变 |
| 56 descriptor/block | §3.6.2 | **理论上界**：$L\times2$ | 随层数变 |
| ~62 µs/descriptor | §3.6.1 | **实测扫描**：三桶 61.5–65.8 µs | 随互联与软件栈变 |
| $m_{1/2}\approx30$ KB | §3.6.1 | **推断**：$\alpha\beta$，单点数据不足以精确分离 | 需做 descriptor 尺寸扫描才能定 |
| 2.4–3.4× 合并收益上界 | §3.6.3 | **理论上界**：合并后由链路单向带宽 0.60–0.91 GB/s 封顶 | 有 P2P/RDMA 时完全不同 |
| E=30 / N=1408(EP) | §3.7.2 | **硬件约束 + 并行度**：60 专家 ÷ 2 rank；中间维不切 | 随 TP/EP 度与模型变 |
| E=60 / N=704（非 EP） | §3.7.2 | 同上：专家不切、中间维 ÷ 2 | 同上 |
| bs≈8 反转点 | §3.7.3 | **理论上界**：$60(1-(1-4/60)^B)$ 与 dense 常数读取的交点 | 随专家数与 top-k 变 |
| 99 KB 共享内存 | §3.8.1 | **硬件约束**：Ada Tuning Guide §1.4.1.1 | 随 compute capability 变 |
| $(num\_stages-1)$ 级缓冲 | §3.8.1 | **推断**：由 36 个 tuple 的可用性与 99 KB 上限反推 | 随 Triton 版本变 |
| num_warps ∈ {4, 8} | §3.8.3 | **硬件约束**：Ada 每 SM 四个 warp scheduler | 随架构变 |
| 18 个 M 档 | §5.3 | **约定**：`benchmark_moe.py` 的档位表 | 上游工具决定 |
| 56.4% / 18.7% | §5.2 | **实测扫描**：nsys node 级窗内相对值 | 随 batch 与模型变 |
| ±5~8% 会话漂移 | §3.7 ⑤ | **实测扫描**：D1/D2 同点跨会话对照 | 随散热与功率工况变 |

**用法与讲义 01 §3.8 相同**：引用任何一个数字之前先说清它属于哪一类。标"推断"的两条（§3.6.1 的 $m_{1/2}$、§3.8.1 的缓冲级数）是本篇里**证据最弱**的两处， 已各自写明还差什么实验才能坐实。

## 4. 代码逐段走读

按一次测量的执行顺序读：proxy 打点（身份贯通）→ D 端 connector 记时 → 落一行日志 → 离线三方 join；最后是 MoE 分解的采集口径。逐 handle 聚合那一处（`patch:14-26`，把 `res.totalBytes / xferDuration / postDuration / descCount` 按 req_id 累加，紧接上游原有的 `self.xfer_stats.record_transfer(res)`）机理已在 §3.3 ② 讲过，此处不再展开代码。全部引用为仓内真实代码逐字拷贝，标 文件：起-止行。

**段 1 · proxy：身份透传与中间 epoch**(`pd_disagg/ext1/ext1_proxy.py:162-179`)

```python
        t_recv = time.time()
        req_data = await request.json()
        # EXT1: honor client-supplied identity
        request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))
        timing = {"request_id": request_id, "t_recv": t_recv}

        prefill_client_info = get_next_client(request.app, "prefill")
        timing["t_p_send"] = time.time()
        response = await send_request_to_service(
            prefill_client_info, api, req_data, request_id
        )
        timing["t_p_done"] = time.time()

        response_json = response.json()
        await response.aclose()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data["kv_transfer_params"] = kv_transfer_params
```

**角色**：身份链中枢与中间 epoch 的唯一来源。**关键行为什么这么写**：①`headers.get("X-Request-Id", ...)` 而非无条件 `uuid4()`——client 自定 id 被接管，再随 header 转发 P 与 D(`:133-136`、`:148-151`)， vLLM 侧从 header 取 id 内嵌进引擎 req_id，join 键就此贯通；②`t_recv` 取在 `await request.json()` **之前**——正因如此，"解析请求体"这段没落进任何一段，成了 §3.4 那个可具名的残差；③打点全用 `time.time()`(epoch)而非 perf_counter，跨进程要与 D 端日志对齐、单调钟没有共同原点；余下两个 epoch 记在 `generate_stream()` **内部**(`:183-194`)，因为 StreamingResponse 惰性，函数返回时还没真正发出。**改错会怎样**：`t_d_send` 若写在闭包外会早于真实发送，d_pre_kv 被系统性拉长、kv_wait 占比被稀释；丢掉 header 透传则三方 join 失配，`analyze_ext1.py:92` 会把请求丢进 WARN。

**段 2 · D 端起点：connector 首见请求即记双钟** (`pd_disagg/ext1/nixl_req_telemetry_v0251.patch:82-89`)

```diff
@@ -43,6 +43,7 @@
         We check for these trnxs to complete in each step().
         """
         for req_id, meta in metadata.reqs_to_recv.items():
+            self._ext1_t0[req_id] = (time.perf_counter(), time.time())  # EXT1
             meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                 meta.local_block_ids
             )
```

**角色**：kv_wait 窗口的左端点。**为什么在这一行**：`start_load_kv` 是 D 端 connector **第一次** 看见该请求的位置，握手在它之后——起点放这里 kv_wait 才**含握手**，首请求的 409.8/462.8/1770.3 ms 与后续请求之差（512 桶 +292 ms）才成为握手一次性成本的直接观测；两个钟分工明确：perf_counter 算时长（单调、不受墙钟调整影响），time.time() 供跨进程对齐。**改错会怎样**：起点若挪到"块分配完成后"，握手成本被排除出 kv_wait、落进 d_pre_kv，首请求观测消失，而三重互证**仍然全绿**——这是最危险的一类错误：不破坏自洽性，只悄悄改变语义。

**段 3 · 一行日志绑定三段身份与两段时钟**(`nixl_req_telemetry_v0251.patch:36-66`)

```diff
+    def _ext1_emit(self, req_id: str) -> None:  # EXT1
+        """EXT1 local patch: one line per completed recv request associating
+        request identity with KV-wait span and aggregated NIXL telemetry.
+        kv_wait_ms spans from D-connector first seeing the request
+        (start_load_kv, incl. handshake wait) to all read handles DONE.
+        Epochs are host wall-clock for cross-process alignment (same host)."""
+        t0 = self._ext1_t0.pop(req_id, None)
+        agg = self._ext1_agg.pop(req_id, None)
+        if t0 is None or agg is None:
+            return
+        meta = self._recving_metadata.get(req_id)
+        remote_req = (
+            meta.remote.request_id
+            if meta is not None and meta.remote is not None
+            else ""
+        )
+        logger.info(
+            "EXT1_KV req_id=%s remote_request_id=%s kv_wait_ms=%.3f "
+            "t0_epoch=%.6f done_epoch=%.6f bytes=%d xfer_us=%d post_us=%d "
+            "descs=%d handles=%d",
+            req_id,
+            remote_req,
+            (time.perf_counter() - t0[0]) * 1e3,
+            t0[1],
+            time.time(),
+            int(agg[0]),
+            int(agg[1]),
+            int(agg[2]),
+            int(agg[3]),
+            int(agg[4]),
+        )
```

**角色**：整套方法的产物格式；`bytes/xfer_us/post_us/descs` 来自同一个喂给 Prometheus 的 `res`——这正是 §3.5 链 1 那个"分毫不差"能当**完整性校验**（检验没有 handle 被漏）而非自我复述的原因。调用点是"该请求全部 handle DONE"那一刻（`patch:31`，插在 `done_req_ids.add(req_id)` 之后）。**关键行为什么这么写**：①两个 `pop` 而非 `get`——取走即清，聚合完成的请求不再占内存，也杜绝二次发射；②`remote_req` 取 P 端 id，这一项让**身份拆分成为可测量对象**（36/36 请求的 client rid 同时出现在 D 端 req_id 与 remote_request_id 中，EXP-013 §5）；③kv_wait 用 perf_counter 差、两个 epoch 用墙钟—— **时长与对齐分用两把尺**，docstring 把口径写死；④`logger.info` 用**惰性格式化**（`%s` + 参数）而非 f-string，日志级别关掉时不做字符串拼接——"观测零扰动"结论的实现侧保证之一。**改错会怎样**：用 `get` 不 `pop`，`_ext1_t0` 只增不减、长跑内存单调上涨； 把 done_epoch 换成 perf_counter 值则跨进程对齐立刻失效，post_kv 段算出荒谬值——而闭环校验会把它抓出来。

**段 4 · 离线三方 join 与六段分解**(`pd_disagg/ext1/analyze_ext1.py:97-121`)

```python
        row = dict(
            request_id=rid,
            bucket=c["bucket"],
            idx=c["idx"],
            ttft_ms=c["ttft_ms"],
            pre_proxy_ms=(p["t_recv"] - c["t_send"]) * 1e3,
            p_segment_ms=(p["t_p_done"] - p["t_p_send"]) * 1e3,
            gap_p_to_d_ms=(p["t_d_send"] - p["t_p_done"]) * 1e3,
            d_pre_kv_ms=(k["t0_epoch"] - p["t_d_send"]) * 1e3,
            kv_wait_ms=k["kv_wait_ms"],
            post_kv_ms=(c["t_first_token"] - k["done_epoch"]) * 1e3,
            kv_share_of_ttft=k["kv_wait_ms"] / c["ttft_ms"],
            bytes=k["bytes"],
            xfer_ms=k["xfer_us"] / 1e3,
            post_ms=k["post_us"] / 1e3,
            descs=k["descs"],
            handles=k["handles"],
            remote_request_id=k["remote_request_id"],
            identity_match=rid in k["req_id"] and rid in k["remote_request_id"],
        )
        row["sum_segments_ms"] = (
            row["pre_proxy_ms"] + row["p_segment_ms"] + row["gap_p_to_d_ms"]
            + row["d_pre_kv_ms"] + row["kv_wait_ms"] + row["post_kv_ms"]
        )
        joined.append(row)
```

**角色**：三个数据源（client JSONL / proxy 行 / D 端 EXT1_KV 行）合成一行逐请求记录，落 `derived/ext1_per_request.csv`。**关键行为什么这么写**：①`kv_share_of_ttft` 的分子分母 **同属一条请求**——这一行就是"因果占比"与"分量对账"的全部区别；②`identity_match` 把身份校验做成**落盘字段**而非临时断言，事后可复核（36/36 为 True，本文复核）； ③`sum_segments_ms` 与 `ttft_ms` 并列存盘，闭环误差可由任何人从 CSV 重算——**校验量必须落盘，否则它只是一次性的自我保证**；④join 对 `len(matches) != 1` 直接 WARN 跳过（`:89-96`），不做模糊匹配。**改错会怎样**：把 `kv_share` 改成"桶级 kv_wait 中位数 ÷ 桶级 TTFT 中位数"，数字大体不变但语义退回分量对账——**同一个百分比，证据等级完全不同**。

**段 5 · MoE 分解的采集口径：node 级 trace 是硬条件**(`moe_perf/d1_nsys.sh:16-25`)

```bash
CUDA_VISIBLE_DEVICES=0,1 nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --kill=sigkill -o "$OUT" --force-overwrite=true \
  "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
  --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel \
  --gpu-memory-utilization 0.88 \
  --profiler-config.profiler=cuda \
  > "$RAW/d1_nsys_moe_bs${BS}_server.log" 2>&1 &
```

**角色**：决定分解表是否成立的那一行是 `--cuda-graph-trace=node`。**为什么这么写**：①vLLM 的 decode 步跑在 CUDA graph 里，nsys 默认 `graph` 模式把整张图记成**一个** kernel，图内的 fused_moe / attention / allreduce 全不单列——首采的"分解表"里 other 桶占 77%、fused_moe 只有 384 个实例（实为 4 个 prefill step 的混样），bs=32 表整张作废（EXP-014 §7，graphlevel 文件保留作对照证据），node 级重采后 other 降到 1.2%；②`--capture-range=cudaProfilerApi` 配 `--profiler-config.profiler=cuda`： 采集窗由 `/start_profile` 触发的 `cudaProfilerStart` 控制（全 rank），脚本先跑 15 s 让负载进 decode 主导稳态、再开 20 s 窗（`:45-50`）——**profile run 的时延数字永不进 benchmark 表**；③配套分类器（`moe_perf/d1_kernels.py:53-61`）把未命中任何桶的 kernel 收进 `unknown` 并打印 top-5(`:78-82`)——**分类器必须暴露它不认识的东西**，正是这条设计让 graph-level 采集的失败当场可见。**改错会怎样**：漏 node 级参数则分解结论指向完全错误的热点；不设稳态等待则窗内混进 prefill 潮使 grouped GEMM 占比被高估；把 other 静默归零，一次口径错误就伪装成漂亮的结论。

**段 6 · join 的失配处理：宁可丢也不猜**(`pd_disagg/ext1/analyze_ext1.py:89-96`)

```python
    for rid, c in client.items():
        p = proxy.get(rid)
        matches = [k for k in kv if rid in k["req_id"]]
        if p is None or len(matches) != 1:
            print(f"WARN unmatched {rid}: proxy={p is not None} kv={len(matches)}",
                  file=sys.stderr)
            continue
        k = matches[0]
```

**角色**：三方 join 的把关口。**关键行为什么这么写**：①`len(matches) != 1` 而不是 `len(matches) >= 1`——**多匹配和零匹配一样是失败**，因为多匹配意味着 id 前缀撞车， 取第一个会静默配错请求；②失配走 `continue` 而不是"取最像的一个"，**不做模糊匹配**； ③WARN 打到 `stderr`，与 markdown 结果分流，不会污染可解析的输出； ④`rid in k["req_id"]` 用**子串**匹配，是因为 vLLM 会在 header id 外面包一层前缀（`serving.py:117 _base_request_id` 从 header 取 id 再内嵌进引擎 req_id），而 client 自定的 `ext1-<桶>-<序号>-<hex6>` 里带 6 位随机十六进制，撞车概率可忽略——**这是"子串匹配可接受" 的前提条件，不是普适做法**。**改错会怎样**：放宽成"取第一个匹配"，一次 id 撞车就会把两条请求的段拼在一起，而闭环误差**未必**会异常（两条请求的 TTFT 量级相同）。

**段 7 · 分类器必须暴露它不认识的东西**(`moe_perf/d1_kernels.py:56-80`)

```python
        for r in rows:
            t = float(r["Total Time (ns)"])
            b = classify(r["Name"])
            agg[b] = agg.get(b, 0.0) + t
            if b == "other":
                unknown.append((t, r["Name"]))
        print(f"\n## bs={bs}  (GPU kernel wall-time 总计 {total/1e6:.1f} ms 采集窗)")
        print("| 分类 | 时间 ms | 占比 |")
        print("|---|---|---|")
        out_rows = []
        for b, t in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"| {b} | {t/1e6:.1f} | {t/total*100:.1f}% |")
            out_rows.append({"bucket": b, "time_ms": round(t / 1e6, 2),
                             "share_pct": round(t / total * 100, 2)})
        with open(DER / f"d1_kernel_share_bs{bs}.csv", "w", newline="") as f:
            f.write(f"# source: moe_perf/raw/EXP-014/d1_nsys_moe_bs{bs}.nsys-rep"
                    " (nsys stats cuda_gpu_kern_sum, node-level trace)"
                    " \u00b7 generated by moe_perf/d1_kernels.py \u00b7 "
                    + datetime.now(timezone.utc).strftime("%Y-%m-%d") + "\n")
            w = csv.DictWriter(f, fieldnames=["bucket", "time_ms", "share_pct"])
            w.writeheader()
            w.writerows(out_rows)
        unknown.sort(reverse=True)
        if unknown:
            print("top other(前5,防静默):")
```

**角色**：把 nsys 的 `cuda_gpu_kern_sum` 表归成 9 个语义桶并落 CSV。**关键行为什么这么写**： ①**分母是窗内全部 kernel 时间之和**(`total`)，不是某个模型化的总量——所以表里的百分比是"窗内相对值"，绝对吞吐必须另取（§5.2 的口径声明）；②`if b == "other": unknown.append(...)` 把未命中任何正则的 kernel **原名连同耗时收起来**，最后打印前 5——**这一条是整张表可信的关键**：分类器不暴露自己的盲区，一次口径错误就伪装成漂亮的结论。首采 graph 级 trace 时 other 占 77%，正是这个打印让失败当场可见（EXP-014 §7）；③CSV 首行写 provenance（源文件 + 采集口径 + 生成脚本），**表能反查回 nsys-rep**；④按耗时降序打印，读者第一眼看到的就是热点。**改错会怎样**：把 other 静默归零或并入邻近桶，graph 级采集的整表作废就查不出来——那次 bs=32 的 fused_moe 只有 384 个实例（实为 4 个 prefill step 的混样）， 表面上是一张正常的表。

**段 8 · A/B 的次序即协议**(`moe_perf/d2_ab.sh:50-74`)

```bash
echo "=== phase 1: kernel A (default, JSON 未装)"
kernel_bench ep_default --enable-expert-parallel
kernel_bench noep_default

echo "=== phase 2: e2e A (default)"
e2e_bench default

echo "=== phase 3: install tuned JSONs into repo configs/"
cp -v "$EP_JSON" "$CFGDIR/"
[ -n "$NOEP_JSON" ] && cp -v "$NOEP_JSON" "$CFGDIR/"

echo "=== phase 4: kernel B (tuned)"
kernel_bench ep_tuned --enable-expert-parallel
kernel_bench noep_tuned

echo "=== phase 5: e2e B (tuned)"
e2e_bench tuned

echo "=== phase 6: correctness (pytest moe 子集)"
cd /root/projects/vllm && CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m pytest \
  tests/kernels/moe/test_moe.py -q -k "not deepseek and not fp8 and not int8 and not wna16" \
  -x --no-header > "$RAW/correctness_pytest.log" 2>&1
echo "pytest rc=$?"
tail -3 "$RAW/correctness_pytest.log"
echo "D2_AB_DONE"
```

**角色**：六个 phase 把 A/B 的顺序固化成脚本，顺序本身就是协议。**关键行为什么这么写**： ①**default 必须在 phase 1–2 先测完**——一旦 `cp` 把 JSON 拷进 `fused_moe/configs/`， exact-match 文件名查找立刻生效（`fused_moe.py:1145-1166`），**没有回头路**； ②kernel 与 e2e **各自成对、不交叉**(1↔4、2↔5)，让每一对之间的间隔尽可能短——热工况漂移是本机的一阶噪声（讲义 01 §5.4），交叉编排会把漂移引进对比； ③EP 与非 EP 分别 `kernel_bench` 一次，**不共享一次 serving**——两者的 config 文件不同， 同一进程里会互相污染；④correctness 放在最后且 `-k` 排除异 dtype 分支，它要证伪的是 "config 改动改变了数值结果"，**不是覆盖率声明**；⑤每个 phase 打印 rc，失败可定位到相位。 **改错会怎样**：先装 JSON 再测 default，得到的"default"其实是 tuned，A/B 归零。

**段 9 · 客户端：每请求唯一 seed 与可 join 的 id**(`pd_disagg/ext1/ext1_client.py:53-64`)

```python
    for bucket in args.buckets:
        for i in range(args.num_per_bucket):
            seed = args.seed_base + bucket + i
            prompt = build_prompt(tokenizer, bucket, seed)
            rid = f"ext1-{bucket}-{i:03d}-{uuid.uuid4().hex[:6]}"
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
            }
```

**角色**：把讲义 01 的协议 v2（每点唯一 seed）与本篇的身份链要求合并到一处。 **关键行为什么这么写**：①`seed = seed_base + bucket + i` 让每条请求的 prompt 互不相同， **防前缀缓存跨请求命中**——链 1 里"129,024 token 零缺口"这条结论直接依赖它（§3.5.2）； ②`rid` 同时编码桶号与序号（便于分组统计）并缀 6 位随机十六进制（**防跨运行撞车**， 也是段 6 允许用子串匹配的前提）；③`stream: True` 是硬要求——TTFT 必须从**首个 chunk** 测量，非流式只能拿到 e2e；④`temperature: 0` 消除采样随机性，让重复运行的输出长度可比。 **改错会怎样**：seed 写成常数，D 端 prefix cache 会吃掉一部分 prompt， bytes 与 ext_kv_tokens 两个计数器会同时变小而**仍然互相吻合**——账还是平的， 结论却错了（EXP-006 固定 seed 协议下的 16,800 token 缺口就是这个情形）。

## 5. 实验数据怎么读

### 5.1 EXP-013 主表:每一列在防什么

| 桶 | TTFT (ms) | kv_wait (ms) | **KV 占 TTFT** | p10–p90 | P 段（ms） | post-KV (ms) | 闭环误差 |
|---|---|---|---|---|---|---|---|
| 512 | 218.3 | 118.2 | **54.2%** | 52.4–55.6% | 63.8 | 24.4 | 0.08% |
| 2048 | 726.7 | 455.7 | **62.5%** | 61.8–63.7% | 221.7 | 24.0 | 0.04% |
| 8192 | 2738.0 | 1763.8 | **64.2%** | 63.6–64.8% | 899.5 | 35.0 | 0.02% |

**口径**：并发 1、p50、每桶 n=11（排除 idx=0 首请求）、request 级因果占比。四个定语一个都不能丢——去掉"并发 1"就变成对满负载的承诺，去掉"排除首请求"就混进握手成本，去掉"request 级"就退回分量对账。

- **先看 p10–p90 而不是 p50**：三桶分布宽度都在 ±2% 内，占比是一个**稳定的结构量**，不是被少数离群请求拉出来的均值（若 p10–p90 张开到 30–80%，同一个 p50 的解释力完全不同）。
- **占比为何随输入长饱和在 ~64%**：P 段与传输段**同为 $O(n)$**——63.8→221.7→899.5 与 118.2→455.7→1763.8，比值 1.85/2.06/1.96 基本恒定，故占比趋常数；短输入被 ~40 ms 固定开销稀释， 512 桶只有 54.2%(EXP-013 §6)。
- **d_pre_kv 是隐藏的第三条 $O(n)$ 线**：9.3 / 22.3 / 41.0 ms（本文现算）——D 端也要 tokenize 完整 prompt、做块分配（推断，机制见 analysis/nixl_token_accounting.md 调度链）。
- **post_kv 几乎恒定**：24.4 / 24.0 / 35.0 ms——KV 到齐后 D 只做 1 个 token 的前向再流式吐出；这条近似常数本身就是"kv_wait 确实吃掉了全部输入相关成本"的旁证。
- **误差条怎么看**：本表不画误差条，给的是 p10–p90 区间与闭环误差列；闭环误差列**不是精度**，是完整性——它回答"六段加起来还差多少"，而 §3.4 已把这点差指认到具体代码段。

**两张配图**：`fig4_pd_ttft_decompose.png`（堆叠柱，`make_figures.py:154-184`）y 轴 PD 无负载 TTFT p50，三段自下而上是"P 端 prefill（≈colocate 无负载 TTFT）"、"NIXL KV 传输（telemetry avg xfer）"、"其余"——**注意最底段的括号**，它就是 §3.2 说的跨臂替代，所以这张图是**v1 的分量对账**而非因果占比；两者结论一致（54–64%）本身是一条独立信息：**对账法被逐请求数据追认有效**。`fig5_nixl_transfer_scaling.png`（双对数，`:187-215`）x 轴每传输 MB、y 轴每传输 ms， 两条参考线是 0.27 GB/s 斜渐近线与 ~12 ms 小传输延迟地板：大传输点贴渐近线说明有效吞吐跨尺寸恒定，smoke 点（0.188 MB / 14.1 ms）贴地板说明小传输是**延迟主导**而非带宽主导。

### 5.2 EXP-014 两张 kernel 占比表:对读才有信息

| 分类 | bs=1 | bs=32 |
|---|---|---|
| **grouped GEMM(fused_moe routed experts)** | 18.7% | **56.4%** |
| dense GEMM/GEMV（proj/共享专家/lm_head） | **40.9%** | 14.9% |
| AllReduce(NCCL) | 13.8% | 15.0% |
| attention | 7.2% | 7.1% |
| norm/rope/act/elementwise | 8.7% | 3.7% |
| routing(topk/softmax) | 4.9% | 1.1% |
| moe_align_block_size | 4.1% | 1.0% |
| permute/unpermute/moe_sum | 0.5% | 0.3% |
| other（top 已核） | 1.2% | 0.4% |

**单看一列会得出错误结论**：只看 bs=1 你会去优化 dense GEMV；只看 bs=32 你会以为 MoE 永远被 grouped GEMM 支配。**对读**才见机理：18.7%→56.4% 与 40.9%→14.9% 是同一件事的两面——batch 增大，routed 专家的命中并集扩张，而 lm_head 这类**与 batch 无关的固定读**被摊薄（vocab 151936 × hidden 2048 → 0.62 GB 权重，TP2 切分后每 rank 每 token 仍读 0.31 GB，EXP-014 §6）； AllReduce 恒 14–15% 是 TP2 的固定税，与讲义 01 的 allreduce 墙同源。**这张表防了哪些坑**： ①node 级 trace（§4 段 5）否则整表作废；②稳态窗（15 s 后开 20 s）否则混进 prefill；③9 类 + 防静默 other；④采集窗 kernel 时间合计 43.1 s / 36.3 s ≈ 2 GPU × 20 s 窗上限附近（含 node 级 tracing 开销），故**占比是窗内相对值，绝对吞吐一律以 sweep JSON 为准**(EXP-014 §7)。

配套曲线 `moe_perf/figures/d1_fig1_decode_scaling.png`（源数据 `derived/d1_scaling.csv`，x 轴并发 1→128、y 轴输出 tok/s）：读图先找**交点**(bs≈8,0.97×)，再看两侧斜率——左侧 MoE 陡（激活参数量优势），右侧 dense 反超（读放大）。bs=1 的 roofline 对照：MoE 实测 221 / 理论 ~373(59%)，dense 109 / ~142(77%)——MoE 达成率更低，量化解释了"MoE 的 bs=1 优势没有理论上那么大"(routing 4.9%
+ moe_align 4.1% 的额外开销)。

### 5.3 EXP-015 三张表:主证据、辅助证据与陷阱

**kernel A/B（主证据，µs,default→tuned）**:

| M | EP | Δ | 非 EP | Δ |
|---|---|---|---|---|
| 1 | 38.2→34.9 | **-8.5%** | 24.4→23.4 | **-3.8%** |
| 8 | 389.4→389.0 | ~0 | 250.9→252.7 | ~0 |
| 32 | 563.4→564.1 | ~0 | 506.1→507.2 | ~0 |
| 64 | 578.2→573.1 | -0.9% | 568.2→566.9 | ~0 |
| 128 | 609.8→585.9 | **-3.9%** | 601.8→579.4 | **-3.7%** |
| 256 | 621.4→597.9 | **-3.8%** | 609.2→589.0 | **-3.3%** |

**读法**：收益的**形状**比幅度重要——两端显著、中段打平。为什么是这个形状？看上游默认启发式（`vllm/model_executor/layers/fused_moe/fused_moe.py:1371-1395`，main@7aa248fc）： `block_m` 按 M 分四档（16/32/64/128）、`block_n` 按 M≤64 二分、`group_m` 只在 `tokens_per_expert > 128` 时才开 16、`num_warps` 按 M≤128 二分——一组**粗粒度阶梯**。中段 M 恰落在启发式调得较准的区间，搜索结果与它撞车（打平）；M=1 的极端 decode 形状与 M≥128 的大 tile 区，阶梯分辨率不够，搜索才有空间。这条读法直接决定了 D3 的判定：**tile 空间在中段已被启发式覆盖 → 不做无数据支撑的 kernel 改动**(EXP-015 §6)。

**A/B 的次序即协议**(`moe_perf/d2_ab.sh:50-74`)：default 必须**先**测——config 一旦拷进 `fused_moe/configs/`，exact-match 文件名查找立刻生效，没有回头路；kernel 与 e2e 各自成对、不交叉， 让每对间隔尽可能短（热工况漂移是本机一阶噪声）；EP 与非 EP 分别测，共享一次 serving 会互相污染。

**e2e（辅助证据）**：TPOT p50 c1 4.40→4.34 ms、c32 17.91→17.70、c128 28.78→28.47，**一致 +0.8~1.2%**；吞吐与 TTFT 在会话噪声内持平。**为什么只能当辅助**：跨会话漂移 ±5~8%（EXP-015 §5 以 D1 同点为对照：本轮 default c32 1596 / c128 4233 tok/s vs D1 的 1691 / 3975），**效应量小于噪声量**，仓内措辞约定因此明确 **D2 的 e2e 不作 headline**；另外 e2e 未采 GPU 遥测（热工况不可证），吞吐只写"噪声内持平"、不作方向声明，TPOT p50 对热态不敏感予以保留（EXP-015 §7）。

**陷阱 · Triton 首跑 JIT 伪影**：装入新 config 后**首个** c32 bench 的 TTFT p50 是 1021 ms (default 176 ms)，而后跑的 c128 正常（363 vs 359 ms）——新 tile 形状第一次被流量命中触发现场编译，32 路并发同时阻塞。处理不是丢数据，而是 **warmup 后复测并两版并存**：c32 吞吐 1616 vs 1596、TPOT 17.76 vs 17.91；c128 吞吐 4178 vs 4233（噪声内）、TPOT 28.52 vs 28.78 (EXP-015 §5)。**读任何"换了实现之后第一次测"的数字，先问有没有 JIT/autotune 的一次性成本混在里面。**

**交付物核对**：两个 JSON 各 **18 个 M 档** + 一个 `triton_version` 元键（仓内已更正一处计数错误： 曾把元键计入，误报 19 档）。元键不是装饰——上游加载时会 `tuned_config.pop("triton_version", None)` 再转 int 键（fused_moe.py：1155-1157），所以它**必须存在且必须被排除在 M 档之外**。correctness： `pytest tests/kernels/moe/test_moe.py::test_fused_moe` → **120 passed， 120 skipped， 0 failed**（skipped 为异平台/异 dtype 参数化）。PR 材料齐备，**提交动作留给本人，未提交**。

### 5.4 两个 config JSON 怎么读:18 个档位不是 18 个独立结论

`raw/EXP-015/configs_{ep,noep}/` 两个 JSON 是这条线的交付物。读它们要先知道三件事：

**（一）键是 M 档，取值是最近邻**。18 个键 {1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096} 覆盖两个数量级；serving 时按 `min(configs.keys(), key=lambda x: abs(x - M))` 取最近的一个（`fused_moe.py:1441`）。 **所以档位之间的空隙也是被这张表管着的**——例如 M=200 会落到 256 档而不是 128 档。

**（二）`triton_version` 是元键，必须存在且必须不被当成 M 档**。上游加载时 `tuned_config.pop("triton_version", None)` 再把其余键转 int(`fused_moe.py:1155-1157`)； 两个文件里都是 `"3.7.1"`。**交付物的规格数字要按消费方的解析逻辑数**——按顶层 key 数会数出 19，按 M 档数是 18（仓内已更正）。

**（三）每个 tuple 都要过共享内存这一关**。按 §3.8.1 的公式逐个代入，两个文件的最大占用都是 80.0 KB(在 $(num\_stages-1)$ 级缓冲口径下)，离 Ada 的 99 KB 上限还有 19 KB 余量（本文现算）。**这说明搜索没有撞到共享内存的墙**——限制它的是别的维度。

**逐档速读（EP 文件，E=30 / N=1408）**：

| M 档区间 | 典型形状 | 读法 |
|---|---|---|
| 1–8 | $BM$ 16–32、$BK$ 64–128、$stages$ 4 | decode 形状：块小（补齐浪费小）、流水深（藏访存延迟） |
| 16–128 | $BM$ 16–32、$BN$ 32–64、$stages$ 2–5 | 中段：参数跳动大而 A/B 时间几乎不变——**平台区**(§3.8.5) |
| 256–512 | $BM$ 32、$BN$ 32–64 | 过渡段 |
| 1024–4096 | $BM$ 64–128、$BN$ 128–256、$stages$ 2 | prefill 形状：块大（算术强度高）、流水浅（共享内存被大块吃掉） |

**一句话**：这张表的形状本身讲了一个故事——**从"补齐浪费主导"过渡到"算术强度主导"**， 而两端正是 A/B 里有收益的两端（M=1 与 M≥128）。中段是平台，所以打平。

### 5.5 图怎么读:四张图的分工

| 图 | 问什么 | 先看哪里 |
|---|---|---|
| `fig4_pd_ttft_decompose.png` | PD 的 TTFT 花在哪 | **最底段的括号**——它标着"≈colocate 无负载 TTFT"，即跨臂替代，这是 v1 分量对账而非因果占比（§5.1） |
| `fig5_nixl_transfer_scaling.png` | 有效吞吐是否跨尺寸恒定 | 大传输点是否贴 0.27 GB/s 斜渐近线、smoke 点是否贴 ~12 ms 延迟地板 |
| `d1_fig1_decode_scaling.png` | MoE vs dense 的 batch 标度 | **先找交点**(bs≈8, 0.97×)，再看两侧斜率：左陡右反超 |
| kernel 占比两表（§5.2） | 优化目标该选谁 | **两列对读**，单看一列必错 |

**通用读法与讲义 01 相同**：先形状、再次序、最后数值。fig5 的两条参考线尤其值得注意—— **带宽渐近线与延迟地板正是公理 B 里 $\beta$ 项与 $\alpha$ 项的可视化**；小传输落在地板上， 说明那里换更快的链路没用（§3.6.1）。

## 6. 误区与边界

1. **"nsys 采下来就是 kernel 分解。"** 本仓第一次采的 bs=32 分解表**整张作废**：默认 `--cuda-graph-trace=graph` 下 CUDA graph 内的 decode kernel 不单列，other 占 77%，fused_moe 只有 384 个实例（实为 prefill 混样）；node 级重采后 other 降到 1.2%（EXP-014 §7，graphlevel 文件保留作对照证据）。**一般形式：profiler 的默认聚合口径可能把你要找的东西整个藏起来，而结果表看上去完全正常。**
2. **"闭环误差 0.02% 说明测量很准。"** 错两次：①闭环误差衡量的是**分解的完整性**而非任何一段的精度——六段全偏移同一个常数也能闭环；②0.02% 小于 0.08% 主要因为分母大（2738 vs 218 ms），残差绝对值反而更大（0.515 vs 0.176 ms，本文现算）。正确用法是**把残差指认到具体代码段**，指认不了就不该声称闭环。
3. **"曾误计 19 个 M 档。"** 仓内已更正：`triton_version` 是元键不是 M 档，实为 18 档。**交付物的规格数字要按消费方的解析逻辑数**（上游 loader 明确 pop 掉该键，fused_moe.py：1155-1157），不是按文件里有几个顶层 key 数。
4. **"0.27 GB/s 就是这条链路的带宽。"** 它是 **telemetry-derived effective throughput**：分子是 NIXL 自报 totalBytes，分母是 NIXL 自报 xferDuration，里面含着 descriptor 碎片化的固定开销（每 16 KiB 花 ~62 µs，约为裸延迟 14.5–15.9 µs 的 4 倍）。它衡量的是**这套软件栈在这种访问模式下的有效速率，不是链路能力**；相关措辞约定：xferDuration 已含 posting，**不与 postDuration 相加**。
5. **"tuned config 全面胜出。"** 真实形状是两端显著、中段打平。把"M=1 −8.5%"讲成"MoE kernel 提速 8.5%"，既丢了 M 档定语也丢了 EP/非 EP 双口径（−8.5% / −3.8%）；e2e 的 +0.8~1.2% 更不能当 headline，它低于跨会话漂移。
6. **"PD 分离在这台机器上失败，说明 PD 分离是错的。"** 不是。收益项在并发 1 下恒为 0（无干扰可消除）、满负载下被 0.27 GB/s 的成本项压垮，这是**形态与互联能力错配**(REPORT §2.4)；换方向（push）只挽回 8K TTFT −6.7%、吞吐 +10–13%，量级不变（EXP-011），问题不在实现方向。

7. **"分类器把 kernel 归到 9 个桶，剩下的都是 other，没什么可说的。"** 恰恰相反： `other` 是**分类器的自我举报机制**。它占 77% 时整表作废，占 1.2% 时表才可信（§4 段 7）。**任何聚合统计都必须留一个"我不认识"的桶并把它打印出来**， 否则一次口径错误会伪装成一张漂亮的表。
8. **"共享内存够用，所以 BLOCK_SIZE 可以随便调大。"** Ada 的上限是每 block 99 KB、每 SM 100 KB(Ada Tuning Guide §1.4.1.1)，而块开大同时会压低每 SM 能驻留的 block 数（上限 24）。两者是一对相反的力，**tile 参数的搜索空间正是被这对力围起来的**(§3.8.1)。本仓 36 个 tuple 的最大占用 80 KB，说明搜索没撞到这堵墙——**没撞到墙不等于墙不存在**。
9. **"num_warps 越大越好，线程多就快。"** Ada 每个 SM 只有四个 warp scheduler（每 partition 一个，白皮书 SM 结构）。`num_warps=4` 恰好每个 partition 一个 warp， `num_warps=8` 是两个；再往上每 warp 的寄存器份额继续下降（每 partition 64 KB 固定）， 可能反过来压低占用率。**它是一个有最优点的参数，不是单调的**(§3.8.3)。
10. **"config 文件装上就一定生效。"** 文件名是**精确匹配** (`E=<专家数>,N=<中间维>,device_name=<GPU>.json`，`fused_moe.py:1145-1166`)， E 或 N 算错一位就静默回退默认启发式并打印一行 warning。EXP-009 的运行时告警原文（`WARNING [fused_moe.py:1106] Using default MoE config...`）正是这条机制的正面用例——**它同时是"社区空缺"这一判定的第一手证据**。
11. **"kernel A/B 有收益，e2e 就该有同比例收益。"** 折算式的三个前提（§3.7.4）缺一不可， 其中最容易漏的是第三条：A/B 测的是离散 M 档，而 serving 的 M 分布经过最近邻查表（`fused_moe.py:1441`）才落到某个档上。**离散基准与连续负载之间隔着一层查表**。
12. **"同机做到 0.1% 闭环，跨节点也差不多。"** 跨节点后 `t0_epoch` 与 `done_epoch` 不再来自同一把墙钟，PTP/NTP 的残差直接进入分解。Python 文档已经写明 `time.time()` "can return a lower value than a previous call if the system clock has been set back"——**墙钟不是单调的**。跨节点要先解决时钟同步，再谈闭环量级（§3.4.2）。

**适用边界**：①**同机是本方法的硬前提**——四个进程共享一把墙钟，epoch 才能直接相减；跨节点时 `t0_epoch`/`done_epoch` 的对齐需另设时钟同步方案（PTP/NTP 残差会直接进入分解），六段闭环的 0.1% 量级不可外推。②**并发 1**——全部占比数字来自 attribution 模式（并发 1、每桶 12 请求、 max_tokens=32、排除首请求），并发上去后 kv_wait 会混入排队。③**pull 路径**——patch 只覆盖 pull，push 侧若要同样关联需仿做（EXP-013 §7）。④**平台**——单机 2×RTX 4090、P2P 驱动禁用、 vLLM 0.25.1(ENV-B)、Qwen2-7B-Instruct、1P1D（非 xPyD）；MoE 线为 ENV-C(main@7aa248fc)、 Qwen1.5-MoE-A2.7B-Chat、TP2+EP。可外推的是**方法**，不是数字。

## 7. 连环追问

1. **Q：kv_wait 到底从哪一刻算到哪一刻？** 从 D 端 connector 在 `start_load_kv` 首次看见该请求（含尚未建立时的握手等待）起，到该请求**全部** NIXL read handle 变为 DONE 止（patch：36-41 的 docstring 写死了这条口径）。 perf_counter 计时长，两端另记 epoch 供跨进程对齐。
2. **Q：你凭什么说那 54% 是因果而不是相关？** ①分子分母同属一条请求、同一时钟域（analyze_ext1.py：102-108），不是跨臂替代；②kv_wait 与 NIXL 自报的 xferDuration 只差 0.3–1.9 ms，窗口里装的确实是传输；③六段闭环误差 p50 <0.1%，残差被指认到 proxy 内部解析段。再加打 patch 前后 TTFT 噪声内一致的对照，排除"观测改变了被观测对象"。
3. **Q：16 行改动会不会把被测系统改慢了？** 设了对照：打 patch 后 TTFT p50 218/727/2738 vs 未打 patch 的矩阵 219/719/2719(EXP-006/007)， 噪声内。实现侧也做了压制：惰性 `%s` 格式化、每请求只发一行、聚合只做整型累加。
4. **Q：0.27 GB/s 为什么远低于 EXP-002 的单向 D2D 0.60–0.91 GB/s？** 访问模式不同。KV 通路是每 block 每层单发的 16 KiB 小拷贝，实测每 descriptor 61.5–65.8 µs（本文现算），约为 GPU 间裸延迟 14.5–15.9 µs 的 4 倍，固定开销主导；裸拷贝测试搬的是大块连续内存。
5. **Q：那要怎么把 KV 通路提上去？** 按 §3.6 的判据，方向是**合并 descriptor**（层维度批量化、减少发起次数），不是换传输方向——方向已被 EXP-011 排除（8K TTFT −6.7%，量级不变）。本仓未做该改造，不外推收益。
6. **Q：MoE 的 fused_moe 占 56.4%，为什么调完 config 只快 1%？** 折算式 $\Delta_\mathrm{e2e} \approx \Delta_\mathrm{kernel} \times$ 占比：kernel 端 serving 相关 M 档改善 3.3–3.9%，乘 56.4% 得约 2% 的上限，实测 TPOT +0.8~1.2%，同量级。**但该幅度低于跨会话漂移（±5~8%），按措辞约定不作 headline，主证据是 kernel A/B。**
7. **Q：为什么 M=1 反而是收益最大的一档（−8.5%）？** M=1 是最极端的形状：每个专家只分到极少 token，默认启发式的粗阶梯（fused_moe.py：1371-1395：`block_m` 四档、`group_m` 只在 `tokens_per_expert > 128` 时才开） 在这里分辨率最不够；搜索出的 tuple(BLOCK_SIZE_M=16 / N=64 / K=64 / GROUP_SIZE_M=32 / warps=4 / stages=4)与启发式给的不同，因而有空间。
8. **Q：120 passed 里有 120 skipped，是不是一半没测？为什么最后判"不做 kernel 改动"？** skipped 是异平台/异 dtype 的参数化（本机 Ada、BF16），不是被跳过的失败；correctness 的作用是**证伪"config 改动改变了数值结果"**，不是覆盖率声明。D3 的判定同样由数据给出： 中段 M 的 tuned 与 default 打平，说明 Triton tile 空间在该形状已被启发式覆盖，config 就是最优杠杆（EXP-015 §6）。**负结论照常报告**——这比硬做一个没有数据支撑的"优化"诚实。
9. **压力问 Q：闭环误差 p50 <0.1% 会不会是自证循环——六段都由你自己定义，当然加得起来？** 诚实答：部分是。六段中有五段的端点来自自己插的打点，望远镜求和在**代数上**必然只剩 $t_\mathrm{p\_send}-t_\mathrm{recv}$ 一项，所以闭环本身**不能**证明任何一段的数值正确； 它只能证明两件事：①没有整段被漏掉或重复计入；②TTFT（client 侧独立测量，与打点体系无关） 与这套打点体系一致。真正给 kv_wait 定性的是**链 2**（与 NIXL 自报 xferDuration 独立吻合） 与**链 1**（与 Prometheus counter 独立吻合）——这两条的另一端都不由我定义。这也是为什么必须是三条正交链，而不是"闭环误差很小"一条。
10. **压力问 Q：54–64% 换一台有 NVLink 的机器还剩多少？你的方法还能用吗？** 数字不能外推，方法能——但要付一次代价。数字上：占比 ≈ $t_\mathrm{xfer}$ / TTFT，而 $t_\mathrm{xfer} \propto 1/BW_\mathrm{eff}$；换到高速互联占比会塌到个位数甚至更低，PD 的收益项（消除干扰、独立扩缩）才有机会占上风——**本仓测的是该形态的下界条件，不是对形态的否定** (REPORT §2.4)。方法上：身份链（X-Request-Id 贯穿）与三重互证原样可用，**时钟域不行**——跨节点后 epoch 不再同源，必须先解决时钟同步，否则 0.1% 量级的闭环无从谈起；另外本仓只测了 1P1D、并发 1、pull 路径，xPyD 下还要多一层"P/D 配对与排队"的段，六段分解要重新设计。

11. **Q：为什么 descriptor 恰好是 16 KiB，而不是 NIXL 的某个默认值？** 它由 connector 注册内存的方式决定：v0.25.1 把 K 与 V 注册成**不同的 region**（源码注释："K and V are now in different regions"），region 数 $=L\times2=56$， descriptor id 由 `(region_id, block_id)` 线性化。于是一个 descriptor $=16\times4\times128\times2=16{，}384$ B(§3.6.2)。**要改粒度必须改 region 划分， 不是调 NIXL 参数。**
12. **Q：合并 descriptor 能把 0.27 GB/s 提到多少？** 上界约 **2.4–3.4×**（到 0.60–0.91 GB/s，即这条无 P2P 中转路径的单向带宽）， 因为合并后 $\beta$ 项接管、$\alpha$ 不再主导（§3.6.3 的算式）。**不是一两个数量级**； 要跨过 1 GB/s 必须换互联。本仓未做该改造，以上为推导，不作实测主张。
13. **Q：MoE 的反转点为什么在 bs≈8 而不是别的数？** 因为每 step 命中的不同专家数期望是 $60(1-(1-4/60)^B)$，B=8 时约 25.5 个（42%）， 此时 MoE 每 token 分摊的权重读取已经追上 dense 的常数读取（§3.7.3）。 **换成 top-2/128 专家的模型，交点位置会明显右移**——这个式子给的是可外推的机制， 不是可外推的数字。
14. **Q：为什么共享专家不算进 grouped GEMM 桶？** 因为它对每个 token 都激活，走的是普通的 dense GEMM 路径，不经过 `sorted_token_ids`/`expert_ids` 的分组索引。分类器把它归进 "dense GEMM/GEMV"（`d1_kernels.py` 的 BUCKETS 正则）。**分类口径决定了表怎么读**：bs=1 时那 40.9% 里就含着共享专家与 lm_head 两块。
15. **Q：$(num\_stages-1)$ 级缓冲这个结论有多硬？** 是**推断**，不是实测。依据：36 个 tuple 里有两个在 $\times num\_stages$ 口径下恰好算出 100.0 KB，**恰好超过 Ada 的每 block 99 KB 上限 1 KB**，而它们确实被 `benchmark_moe.py --tune` 跑通并选中（编译不过的会被剔除）。要坐实需要反编译或读 Triton 的 pipeliner 源码——**本仓没做，如实标注**(§3.8.1)。
16. **Q：为什么 nsys 的默认 `--cuda-graph-trace=graph` 会把结论带偏？** Nsight Systems 官方文档写明：`graph` 是"trace whole CUDA graphs without node activities. This reduces overhead ... and is the default when available"， `node` 才"collect node activities instead of whole-graph traces. This may cause significant runtime overhead."。vLLM 的 decode 步跑在 CUDA graph 里， 默认口径下**整张图记成一个活动**，图内的 fused_moe/attention/allreduce 全不单列。 **默认值优化的是采集开销，不是可解释性**——两者在这里正好冲突。
17. **Q：node 级 trace 有显著开销，那这张表的占比还准吗？** 准的是**相对占比**，不准的是**绝对时长**。仓内已声明：采集窗 kernel 时间合计 43.1 s / 36.3 s ≈ 2 GPU × 20 s 窗上限附近（含 node 级 tracing 开销）， 故占比为窗内相对值，**绝对吞吐一律以 sweep JSON 为准**(EXP-014 §7)。这也是仓内"profiler 环境的时延数字永不进 benchmark 表"这条规矩的由来。
18. **Q（压力）：你说中段是"平台"，可你并没有扫遍平台——万一是搜索没搜到更好的点？** 诚实答：不能排除。能说的是三件事：①搜索空间是 1920 个配置 × 18 个 M 档（EXP-015 §2），不是抽样几个；②启发式与搜索结果**六字段无一档完全一致** (§3.8.5)，而 A/B 时间差 ≈0——两个明显不同的点给出相同时间，这是"平台"的直接证据， 不是"两个点都最优"的巧合；③共享内存余量 19 KB(§5.4)说明搜索没被硬约束截断。 **但"平台的边界在哪、平台上还有没有更好的角落"，本仓没有数据**，不主张。
19. **Q（压力）：你反复说"不作 headline"，那这条 MoE 线到底交付了什么？** 交付三件可核验的东西：①两个**社区空缺**格子的 config JSON（判定依据是运行时告警原文 + 本地目录检查 + 远端查重，EXP-009 §5）；②kernel A/B 的两端数字（M=1 −8.5%/−3.8%，M≥128 −3.3~−3.9%，EP/非 EP 双口径）；③一条**负结论**——中段被平台覆盖，因此不做无数据支撑的 kernel 改动（EXP-015 §6）。 e2e 的 +0.8~1.2% 只作 supporting。**把效应量小于噪声量的结果降级，是交付的一部分， 不是交付的缺失。**

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | **PagedAttention §7.2**："vLLM sets its default block size as 16" | EXP-006 单请求探针 bytes = 917,504 = 16×57,344，逐字节吻合 | **一致**。本篇全部块粒度账建立在这条默认值上；换 block_size 则 §3.1 全部重算 |
| 2 | **NIXL 官方文档**（`docs/nixl.md`，Transfer 节）：传输由本地与远端两份 descriptor 列表 + 目标 agent 名 + read/write 构成 | 每（层， K/V， block） 一个 descriptor，恒 16,384 B；逐请求 Σdescs = Prometheus `nixl_num_descriptors_sum` 分毫不差 | **一致**。文档只定义了"descriptor 是传输单位"，**粒度由 connector 的 region 划分决定**——这一层文档没说，本仓从源码与遥测两侧确认（§3.6.2） |
| 3 | **DistServe §3.3**：OPT-66B 512-token 请求 KV ≈ 1.13 GB；10 rps 需 11.3 GB/s(≈90 Gbps)才能让传输开销"invisible" | 本机 $BW_\mathrm{eff}$ = 0.26–0.27 GB/s | **前提不成立**，不是结论冲突。本机比该门槛低约 42×；论文把带宽写成**前提条件**，本仓测的是前提被破坏后的形态 |
| 4 | **DistServe §6.3**：KV 传输占总时延 "<0.1%"，">95% 请求延迟 <30 ms" | KV 等待占 TTFT **54.2% / 62.5% / 64.2%**(EXP-013) | **同一个量，相差三个数量级**。差异全部来自 $BW_\mathrm{eff}$（NVLink 600 GB/s vs 本机 0.27 GB/s）。**这一行是本篇最有价值的对照**：它说明"传输占比"不是形态的属性，是**互联的属性** |
| 5 | **Splitwise §IV-C/§VI-A**：逐层重叠 KV 传输，非重叠残留 A100 ~8 ms、H100 ~5 ms；第二 token 时延从 +64% 降到 +16.5% | 本仓 pull/push 均为整请求粒度，8K avg xfer 1602.7 ms | **实现层次不同**。按 §3.5.4 口径估算，即使完美重叠，8K TTFT 也只能降到 ~1.8 s，仍是 colocate 的 2×——**重叠改常数，不改结论** |
| 6 | **Megatron §3 / Mooncake §5.1**：跨节点扩 TP 需每层两次（RDMA） allreduce，显著降低 prefill MFU | 讲义 01 §3.3 的 tp2 prefill 零加速 | **同一机理，不同尺度**。结论方向一致，阈值差两个数量级 |
| 7 | **GShard(arXiv：2006.16668)**：专家按设备切分的专家并行 | `--enable-expert-parallel` 下每 rank 30 专家，靠 `expert_ids = -1` 跳过非本 rank 的 block | **一致**，但实现方式值得注意：**不是换 kernel，是在索引层打标记**(§3.7.2) |
| 8 | **DeepSeekMoE(arXiv：2401.06066)**：细粒度专家切分 + 共享专家隔离 | Qwen1.5-MoE：60 路由专家取 4 + 4 个常驻共享专家（本机 `config.json`） | **一致**。共享专家因此落在 dense GEMM 桶而非 grouped GEMM 桶——**分类口径的直接后果**(§5.2) |
| 9 | **MegaBlocks §5.1.2**：在其基准里 128×128 tile "consistently perform on-par or better"，且"this same configuration is commonly selected by NVIDIA cuBLAS" | 本机 36 个 tuned tuple 里，只有 M=1536(EP)与 M=3072（非 EP）选到 $BM{=}BN{=}128$ | **形状不同，不构成冲突**。MegaBlocks 面向训练（M 极大），本机 serving 的 M 多在 1–256，补齐浪费与占用率把最优 tile 压小（§3.7.1） |
| 10 | **Triton 文档**：num_stages 是"the number of stages that the compiler should use when software-pipelining loops. Mostly useful for matrix multiplication workloads on SM80+ GPUs" | 小 M 档选 4–5、大 M 档选 2 | **一致**，且本篇给出了机理：深度受共享内存上限约束（§3.8.1–3.8.2） |
| 11 | **PTX ISA §9.7.9.26.3.3**：`cp.async.wait_group N` 等到"only N or fewer of the most recent cp.async-groups are pending" | 未直接观测（本仓不做 PTX 级反汇编） | **未核实层**。本篇只用它解释 num_stages 的语义，不主张观测到了具体的 wait_group 参数 |
| 12 | **Ada Tuning Guide §1.4.1.1**：每 block 最多 99 KB 共享内存、每 SM 100 KB、CUDA 保留 1 KB | 36 个 tuple 在 $(s-1)$ 口径下最大 80 KB；两个 tuple 在 $\times s$ 口径下恰为 100.0 KB | **交叉验证**。"恰好 100.0 KB"这个巧合反过来支持 $(s-1)$ 级缓冲的读法（§3.8.1，推断） |
| 13 | **Ada 白皮书 SM 结构**：每 SM 四个 partition，各有一个 warp scheduler 与一个 dispatch unit | tuned config 的 num_warps 在 4 与 8 之间跳，与启发式的 `M<=128 ? 4 : 8` 二分**在 11/18 档不一致** | **硬件语义解释了取值集合，但没有解释取值选择**——后者由平台的平坦性决定（§3.8.5） |
| 14 | **Nsight Systems 文档**：`--cuda-graph-trace` 默认 `graph`，"reduces overhead"；`node` "may cause significant runtime overhead" | 默认口径下 bs=32 分解表整张作废（other 77%），node 级重采后 other 降到 1.2% | **文档预言 → 实测证实**。文档把它写成开销权衡，本仓测到的是**可解释性权衡**(§7 Q16) |
| 15 | **Python `time` 文档**：`perf_counter` "reference point ... is undefined， so that only the difference ... is valid"；`time()` "can return a lower value ... if the system clock has been set back" | 本篇用 perf_counter 计时长、epoch 做跨进程对齐 | **一致**，且两条契约正好互补——这是"两把尺"设计的官方依据（§3.4.2） |
| 16 | **W3C Trace Context / OpenTelemetry**：`traceparent` 携带 version-trace_id-parent_span_id-trace_flags，逐跳传播 | 本仓用单个 `X-Request-Id` 贯穿三方 | **本仓是简化版**。少了 span 层级与父子关系，所以六段分解只能靠离线 join 拼出来；生产做法见 §8.2 |

**读法**：16 条里 6 条是正向闭环（1、2、7、8、14、15），3 条是"前提不成立"(3、4、5)， 2 条是"形状不同"(9、13)，1 条自认未核实（11），其余是交叉验证或简化说明。 **第 4 行值得单独记住**：同一个指标（KV 传输占时延比）在两台机器上差三个数量级， 而两边的测量方法都没错——**这就是"数字属于平台，方法属于领域"的最好例证**。

### 8.2 与生产实现的差距各在哪一层

- **可观测性这一层**：上游有同方向的 draft PR #52859(NVIDIA，NIXL push/pull lifecycle tracing)，本仓查重后按 fail-closed 原则把 EXT-1 定位为**本地测量 patch，不投上游** (`pd_disagg/ext1/DEDUP.md`)。生产系统会把这类关联做成一等公民（OpenTelemetry span： proxy → P → D → NIXL transfer 一条 trace），而不是一行 `logger.info` + 离线正则 join； 代价是采样与传播开销，收益是跨节点天然带时钟同步语义。
- **身份这一层**：本仓靠 client 自定 `X-Request-Id` 贯穿三方（`ext1_proxy.py:165`、 `serving.py:117 _base_request_id` 从 header 取 id）；生产网关会强制注入 trace id 并逐跳透传。 vLLM 的 NIXL 已把身份三层正交拆开（引擎身份 / 会话身份 / 内存寻址，theory/02 §2），这正是它对 0.17.1 P2pNccl 那种隐式 rendezvous key 分叉免疫的原因（EXP-012 实机复现）。
- **KV 通路这一层**：生产 PD 分离跑在 NVLink / IB / RDMA NIC 上，KV 走 GPUDirect RDMA，带宽两个数量级于本机；descriptor 碎片化在那里被大得多的链路带宽掩盖，在本机则成了主导项——**同一份 connector 代码，在不同互联上暴露的瓶颈完全不同。**
- **MoE config 这一层**：上游 config 是 exact-match 文件名查找（`E=<专家数>,N=<中间维>,device_name=<GPU>.json`），没有插值、没有邻近回退——所以"社区空缺"是一个**离散**的事实：文件在或不在。本仓补的两个 tuple(E=30，N=1408 / E=60，N=704) 就是把两个具体格子填上，材料齐备但**未提交**。**要补一句精确的**："无邻近回退"说的是 **文件层**（`fused_moe.py:1145-1166`：找不到就回退默认启发式）；**文件内部的 M 档反而是最近邻匹配**(`fused_moe.py:1441`)。两层粒度不同，混说会得出"M=200 没有 config 可用" 这样的错误结论（§3.8.4）。

### 8.3 延伸阅读(每条一句话说明它能解决什么疑问)

**论文**

1. Kwon， Li， Zhuang， Sheng， Zheng， Yu， Gonzalez， Zhang， Stoica， "Efficient Memory Management for Large Language Model Serving with PagedAttention"， arXiv：2309.06180， §7.2（block size 16 的扫描依据）、§4.2(block table)。——本篇所有块粒度账的出处； 想知道 16 这个数是怎么定的，只读 §7.2 就够。
2. Zhong et al.， "DistServe"， arXiv：2401.09670，§3.3（通信开销的带宽门槛）、 §6.3（KV 传输占时延 <0.1% 的实测）。——把 §6.3 与本篇 §5.1 并排看， 就理解了"同一个指标差三个数量级"是怎么回事。
3. Patel et al.， "Splitwise"， arXiv：2311.18677，§IV-C（逐层重叠传输）、§VI-A（重叠前后）。——想知道"传输能不能藏起来、能藏多少"，读 §IV-C 的机制描述。
4. Lepikhin et al.， "GShard： Scaling Giant Models with Conditional Computation and Automatic Sharding"， arXiv：2006.16668。——专家并行的原始形态；本篇 §3.7.2 里 "按 rank 切专家"这件事的出处。
5. Dai et al.， "DeepSeekMoE： Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"， arXiv：2401.06066。——细粒度专家切分 + 共享专家隔离两条策略； 解释 Qwen1.5-MoE 为什么是"60 路由取 4 + 4 常驻共享"。
6. Gale， Narayanan， Zaharia， Ganguli， "MegaBlocks： Efficient Sparse Training with Mixture-of-Experts"， arXiv：2211.15841，§5.1.2（tile 尺寸基准）、§5.1.3(blocked-CSR-COO)。——想知道"MoE 的 grouped GEMM 还能怎么组织"（块稀疏而非排序补齐），读这两节。

**官方文档与规范**

7. NVIDIA PTX ISA，§9.7.9.26.3.1–3.3(`cp.async` / `cp.async.commit_group` / `cp.async.wait_group`)与 §9.7.15.5.8（mma.m16n8k16 的 fragment 布局：每线程 4 个 `.f16x2` 寄存器，`groupID = %laneid >> 2`）。——想把 Triton 的 num_stages 落到指令语义上，读前三节；想理解为什么 tile 的 M 维天然是 16 的倍数，读后一节。
8. NVIDIA Ada GPU Architecture Tuning Guide,§1.4.1.1 Occupancy（共享内存 100 KB/SM、 99 KB/block、24 blocks/SM、255 寄存器/线程）、§1.4.2.1 Increased L2 capacity、 §1.4.2.2 Unified Shared Memory/L1/Texture Cache。——§3.8 全部硬件常数的出处。
9. NVIDIA Ada GPU Architecture 白皮书，SM 结构段（四个 partition、各一个 warp scheduler 与 dispatch unit、64 KB 寄存器堆）与 Appendix A Table 2（RTX 4090 的 L2 73728 KB 等）。——num_warps 取值集合的硬件依据（§3.8.3）。
10. NVIDIA Nsight Systems User Guide，`--cuda-graph-trace`（graph / node 两档的语义与开销说明）与 `--capture-range=cudaProfilerApi`。——§4 段 5 与 §7 Q16 的官方依据； 想避免"默认口径藏掉热点"这类坑，先读这一条。
11. Triton 文档 `triton.Config`（num_warps / num_stages / num_ctas 的定义）。——config JSON 六个键里四个的官方语义。
12. Python 官方文档 `time` 模块（`perf_counter` / `monotonic` / `time` 的三条契约）。——"两把尺"设计的依据；想知道为什么不能只用一把，读 `time()` 那句"can return a lower value ... if the system clock has been set back"。
13. W3C Trace Context 规范与 OpenTelemetry 的 Context Propagation 文档（`traceparent` 的 version-trace_id-parent_span_id-trace_flags 四段结构）。——本仓 `X-Request-Id` 单键方案的工业对照面；想把六段分解升级成 span 树，从这里开始。

**仓内证据**

14. `pd_disagg/ext1/nixl_req_telemetry_v0251.patch` 全文 + `ext1/orig/` 原件。——自己 `diff` 一遍确认"~16 行"这个定性，并对照 §3.3.1 的反事实表逐条想一遍 "挪到别处会怎样"。
15. `pd_disagg/analysis/nixl_token_accounting.md`、`moe_perf/raw/EXP-015/configs_{ep,noep}/` 两个 JSON、`records/EXP-013` §5–§7 与 `records/EXP-014` §7。——依次回答：16,800 token 的缺口是怎么被逐块定位的、18 个 M 档的形状讲了什么故事（§5.4）、三重互证的原始数字、以及 graph 级采集失败的完整记录。
