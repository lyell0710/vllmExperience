---
status: complete
关联EXP: EXP-002, EXP-004, EXP-005, EXP-006, EXP-007, EXP-008
配套: docs/theory/02_pd_kv_path.md(速查版) · pd_disagg/REPORT.md(结论版) · 本文=逐段走读版
---

# 深度讲义 01 · 两张消费级 4090 该怎么用:四臂形态的资源账、互联墙与功率帽探案

> 阅读前提：知道 prefill / decode 是两个阶段，会读 shell 与 Python；不要求分布式推理背景。引用规范：凡属论文或官方文档的论断一律给出处（标题 + arXiv/DOI 编号 + 章节号， 文档给 URL 路径 + 小节名）；凡本文从仓内 raw/derived 现算的量标注"本文现算"； 凡本文自己补出的推导标注"本讲义推导"；无法用检索确认的说法标注"未核实"。每个数字都带 EXP 锚，原始文件在 `pd_disagg/` 与 `records/data/` 下。

## 目录

- [1. 这一篇回答什么问题](#1-这一篇回答什么问题)
  - [1.1 本篇要建立的六条能力](#11-本篇要建立的六条能力)
  - [1.2 符号与口径约定](#12-符号与口径约定)
  - [1.3 本篇引用的一级文献(详细出处见 §8.3)](#13-本篇引用的一级文献详细出处见-83)
- [2. 直觉与第一性原理](#2-直觉与第一性原理)
  - [2.1 把四臂写成同一个三元组](#21-把四臂写成同一个三元组)
  - [2.2 三条贯穿全篇的公理](#22-三条贯穿全篇的公理)
  - [2.3 类比失效点清单](#23-类比失效点清单)
- [3. 完整推导/机制](#3-完整推导机制)
  - [3.1 互联三数:三条路径,三个数字,不可互换](#31-互联三数三条路径三个数字不可互换)
  - [3.2 decode 为什么由带宽定价:TPOT 下限的推导](#32-decode-为什么由带宽定价tpot-下限的推导)
  - [3.3 TP2 的收益与代价:一半权重 + 一份通信税](#33-tp2-的收益与代价一半权重--一份通信税)
  - [3.4 replica2:零通信的复制,近线性的扩展](#34-replica2零通信的复制近线性的扩展)
  - [3.5 pd1p1d:每请求一份 KV 的搬运账](#35-pd1p1d每请求一份-kv-的搬运账)
  - [3.6 把四臂放进同一个账本(本讲义推导)](#36-把四臂放进同一个账本本讲义推导)
  - [3.7 goodput:文献里的定义与本仓的实现](#37-goodput文献里的定义与本仓的实现)
  - [3.8 魔法数总表:每个数字由什么决定](#38-魔法数总表每个数字由什么决定)
- [4. 代码逐段走读](#4-代码逐段走读)
- [5. 实验数据怎么读](#5-实验数据怎么读)
  - [5.1 硬件三数原始文件怎么读](#51-硬件三数原始文件怎么读)
  - [5.2 84 个测量点是怎么设计的](#52-84-个测量点是怎么设计的)
  - [5.3 三张主表逐行读](#53-三张主表逐行读)
  - [5.4 功率帽探案:隐藏变量是怎么被抓住的](#54-功率帽探案隐藏变量是怎么被抓住的)
  - [5.5 自己动手复算:三个可以从 raw 重来的量](#55-自己动手复算三个可以从-raw-重来的量)
  - [5.6 图怎么读:六张图的分工](#56-图怎么读六张图的分工)
- [6. 误区与边界](#6-误区与边界)
- [7. 连环追问](#7-连环追问)
- [8. 工业对照与延伸](#8-工业对照与延伸)
  - [8.1 论文/文档怎么说 vs 本项目实测:逐条对照](#81-论文文档怎么说-vs-本项目实测逐条对照)
  - [8.2 与生产实现的差距各在哪一层](#82-与生产实现的差距各在哪一层)
  - [8.3 延伸阅读(每条一句话说明它能解决什么疑问)](#83-延伸阅读每条一句话说明它能解决什么疑问)

## 1. 这一篇回答什么问题

只有两张 RTX 4090（无 NVLink、P2P 被驱动禁用），单卡混部、双副本、张量并行 TP2、 Prefill-Decode 分离四种形态该选哪个，以及为什么在这台机器上答案是唯一的。读完你应当能： 手推 decode 的带宽 roofline 和 TP2 的收益/代价账（算式到 ms 级）；解释 1.78 GB/s、 0.26–0.27 GB/s、22.7 vs 0.6 GB/s 这三组互联数字各自代表哪条路径、怎么测出来； 面对"你的 SLO 阈值是不是挑出来的""replica2 凭什么不到 2× 也叫近线性"这类追问给出带证据锚点的回答。

### 1.1 本篇要建立的六条能力

1. **路径能力**：听到一个互联数字，先问"哪条路径、哪种消息大小、哪个软件栈"。本机三条路径三个数字差两个数量级（§3.1），混用任何一个都会把结论带偏。
2. **推导能力**：能从"decode 每步必须把权重读一遍"这一句出发，把 TPOT 下界、 TP2 的分摊收益、PD 的容量上限一路推到 ms 与 req/s，并说清每一步凭什么合法、在什么条件下失效（§3.2–§3.5）。
3. **口径能力**：知道 14.2 是 GiB 还是 GB、924 是读带宽还是读写合计、busbw 与 algbw 在 2 卡时为什么相等（§3.1.3、§3.2.3）。**任何一个魔法数都要能回答 "它是理论上界、硬件约束，还是实测扫描定的"**(§3.8)。
4. **对照能力**：能设计"反例臂"（colocate 单卡）与"隐藏变量臂"（工况遥测）， 知道对比实验里"我变快"与"对照变慢"不可区分（§5.4）。
5. **文献能力**：知道 DistServe / Splitwise / Mooncake 给 PD 分离预设了什么互联前提，以及本机把那个前提降低了两个数量级之后会发生什么（§8.1）。
6. **诚实能力**：知道哪些结论是本机实测、哪些是推断、哪些是本仓没测因而不主张（§6 边界、§7 压力问）。

### 1.2 符号与口径约定

| 符号 | 含义 | 本机取值/来源 |
|---|---|---|
| $W$ | 模型权重字节数 | Qwen2-7B-Instruct BF16，见 §3.2.4 |
| $BW_{\mathrm{mem}}$ | 卡内显存带宽 | 实测 ~924 GB/s(EXP-002)；标称 1008 GB/s（Ada 白皮书 Table 2） |
| $BW_{\mathrm{col}}$ | NCCL collective 路径带宽 | 1.78 GB/s avg busbw(EXP-002) |
| $BW_{\mathrm{eff}}$ | NIXL KV 通路有效吞吐 | 0.26–0.27 GB/s,telemetry-derived(EXP-006/007) |
| $n$ | 单请求输入 token 数 | 三桶 512 / 2048 / 8192 |
| $B_{kv}(n)$ | 单请求 KV 字节数 | $n\times 57{,}344$ B，按 16-token 块向上取整（§3.5） |
| TTFT / TPOT | 首 token 时延 / 每输出 token 时延 | bench 口径，`--percentile-metrics ttft,tpot,itl,e2el` |
| goodput | 同时满足两条 SLO 的请求数 ÷ 墙钟 | §3.7、§4 段 5 |
| $L$ / $H$ / $KVH$ / $D$ / $d$ | 层数 / Q 头 / KV 头 / head_dim / hidden | 28 / 28 / 4 / 128 / 3584 |

模型配置取自本机 `Qwen/Qwen2-7B-Instruct` 的 `config.json`:`hidden_size` 3584、 `num_hidden_layers` 28、`num_attention_heads` 28、`num_key_value_heads` 4、 `intermediate_size` 18944、`vocab_size` 152064、`tie_word_embeddings` false、 `torch_dtype` bfloat16。`head_dim` 未显式给出，由 3584/28 = 128 得到。GQA 的 4:28 分组是后文所有 KV 账的起点（GQA 定义见 Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245,§2.2）。

**三个必须一起出现的定语**：任何时延数字要带（a）臂名、(b)输入桶、(c)工况（并发 1 归因 / 饱和 / sweep 档位；冷态 / 同热工况）。丢掉任何一个，数字都不可比——§5.4 的功率帽探案就是丢掉（c） 会发生什么的实录。

### 1.3 本篇引用的一级文献(详细出处见 §8.3)

- PD 分离：Zhong et al., "DistServe", arXiv:2401.09670;Patel et al., "Splitwise", arXiv:2311.18677;Qin et al., "Mooncake", arXiv:2407.00079。
- 调度模型：Yu et al., "Orca", OSDI 2022;Kwon et al., "PagedAttention", arXiv:2309.06180;Agrawal et al., "Sarathi-Serve", arXiv:2403.02310。
- 张量并行：Shoeybi et al., "Megatron-LM", arXiv:1909.08053,§3。
- 性能模型：Williams, Waterman, Patterson, "Roofline", CACM 52(4), 2009, DOI 10.1145/1498765.1498785;Hockney, Parallel Computing 20(3), 1994。
- 硬件与库语义：NVIDIA Ada GPU Architecture 白皮书 Appendix A Table 2; CUDA C++ Programming Guide §3.4.2;NCCL User Guide 环境变量页； nccl-tests `doc/PERFORMANCE.md`;NVML `nvmlClocksThrottleReasons`; NVIDIA GPU Performance Background User's Guide §4。

## 2. 直觉与第一性原理

**先想没有这个问题的世界。** 如果模型能塞进一张卡、且一张卡吞吐够用，部署没有选型问题： 一个进程，prefill 和 decode 混跑，这就是本仓的 colocate 基线（EXP-004《B1 colocate 归因基线 + SLO 锁定》）。选型问题诞生于 "多出一张卡"——多出来的算力、显存、带宽要通过某种**组织方式**变成吞吐或延迟，而每种组织方式都要付一种代价。

**日常类比：两个厨师开餐馆。**
- replica2 = 开两家一模一样的店：菜单、灶台全复制，互不说话，客流对半分；
- tp2 = 两人同炒每一道菜：你切一半我切一半，但每道菜出锅前必须把两人的半成品合到一起；
- pd1p1d = 一人只备菜、一人只掌勺：备好的菜要整盘从一号灶端到二号灶。

类比的失效点必须点破：厨师之间"传菜"几乎免费，而 GPU 之间传数据在本机要走一条被禁用了直连（P2P）的 PCIe 路径——**合菜（allreduce）与端菜（KV 传输）的成本在这台机器上不是二阶小量，而是主项**。这是全篇的第一性原理：**部署形态的本质是"用通信换组织"，通信有多贵， 形态就有多少自由度。**

三本资源账（每臂都要各记一遍，详细算式在 §3）：

| 臂 | 显存（权重） | 算力组织 | 跨卡通信 |
|---|---|---|---|
| colocate | 1 卡 × 全量 14.2 GB | prefill/decode 混跑互相干扰 | 0 |
| replica2 | 2 卡 × 全量（复制） | 两条独立流水线 | **0** |
| tp2 | 每卡 1/2(7.1 GB) | 每个算子切一半、逐层合并 | 每层 allreduce |
| pd1p1d | 2 卡 × 全量（P、D 各一份） | 阶段专业化（P 计算受限/D 带宽受限） | 每请求整份 KV |

一眼可见：四臂中只有 tp2 和 pd1p1d 把跨卡通信放进了关键路径。所以在互联受限的平台上， **先测互联，再谈形态**——这就是本仓把 EXP-002《硬件三数》放在一切实验之前的原因。

### 2.1 把四臂写成同一个三元组

上面那张表可以形式化成一个三元组 $(\,M,\;C,\;X\,)$：

- $M$：每卡驻留的权重字节数——决定 decode 的带宽下界与可用的 KV 预算；
- $C$：一次前向被切成几份、在几张卡上并行——决定 prefill 的计算下界；
- $X$：每单位工作要跨卡搬多少字节、搬多少次——决定通信税。

| 臂 | $M$ | $C$ | $X$（每请求） | $X$（每 token） |
|---|---|---|---|---|
| colocate | $W$ | 1 卡 | 0 | 0 |
| replica2 | $W$（×2 份） | 1 卡/请求 | 0 | 0 |
| tp2 | $W/2$ | 2 卡协同 | $L\cdot k\cdot n\cdot d\cdot 2$ B(prefill) | $L\cdot k\cdot d\cdot 2$ B(decode) |
| pd1p1d | $W$（×2 份） | 1 卡/阶段 | $B_{kv}(n)$ | 0 |

其中 $k$ 是每层前向的 allreduce 次数（Megatron 切法下 $k=2$，见 §3.3.1；本仓按 $k=1$ 做下界计数）。**这张表是全篇的骨架**：§3.2 算 $M$ 的后果，§3.3 算 tp2 的 $X$，§3.5 算 pd 的 $X$，§3.4 说明 $X=0$ 意味着什么。

一个立刻可读出的结论：tp2 的通信量**与 batch 无关地按 token 计**，而 pd1p1d 的通信量 **按请求的输入长度计**。所以 tp2 的税随负载线性增长但每 token 恒定，pd1p1d 的税随输入长度线性增长——两者在长上下文场景下的恶化速度完全不同（§5.3 的三桶对比）。

### 2.2 三条贯穿全篇的公理

- **公理 A（先测路径，再谈形态）**：形态的可行域由互联能力划定，不由形态本身的优雅程度划定。本仓把 EXP-002 放在最前面，不是流程洁癖，是因果顺序。
- **公理 B（通信成本 = 固定开销 + 字节数/带宽）**：任何一次数据搬运的时间都可写成 $t(m)=\alpha+m/\beta$，$\alpha$ 是每次发起的固定开销，$\beta$ 是渐近带宽。这套两参数刻画法出自 Hockney 对 MPP 通信性能的 COMMS1 口径(R. W. Hockney， "The communication challenge for MPP： Intel Paragon and Meiko CS-2"， Parallel Computing 20(3)：389–398, 1994)。**由此立刻得到一个判据**： 当 $m \ll \alpha\beta$ 时，有效带宽 $m/t(m)\approx m/\alpha$ 与 $\beta$ 无关——**换更快的链路救不了小消息**。使有效带宽达到 $\beta/2$ 的消息大小 $m_{1/2}=\alpha\beta$ 是这条曲线的拐点（本讲义按该两参数模型自行推出）。 §3.5、§5.1 与讲义 02 的 descriptor 分析全部落在这条公理上。
- **公理 C（工况是测量的一部分）**：消费卡在功率帽下的时钟不是常数，同一段代码在冷态与稳态可以差 30% 的 TTFT(§5.4)。工况没入账的对比实验，结论不成立。

### 2.3 类比失效点清单

**"两家店 = 两倍吞吐"** 忽略了入口层（replica2 的 512 桶疑点在代理/客户端并发上限， §3.4.3）；**"合菜是二阶小量"** 忽略了 8192 token 时单层 allreduce 就是 58.7 MB， 在 1.78 GB/s 上不是小量（§3.3.2）；**"端菜就是走两步"** 忽略了每请求 469.8 MB 在 0.27 GB/s 上要 1.63 s(§3.5)——这是把整锅菜用吸管吸过去；**"专业化一定更快"** 忽略了阶段专业化的收益是**消除干扰**，而并发 1 时没有干扰可消除，收益恒为 0、成本照付。

## 3. 完整推导/机制

本节每条推导按同一模板写：**先给等式，再逐步给合法性条件，最后给失效边界**。凡本仓没有独立测量就无法确证的中间步骤一律显式标注，不用"显然"含混过去。

### 3.1 互联三数:三条路径,三个数字,不可互换

- **P2P 能力**：`nvidia-smi topo -p2p r` 返回 GNS(GPU not supported)、 p2pBandwidthLatencyTest 的 connectivity matrix 全 0(`pd_disagg/hw/topo.txt`、 `hw/p2p_bandwidth_latency.txt`，EXP-002)——GeForce 驱动层禁用，不是拓扑问题。
- **裸拷贝路径**：单向 D2D 0.60–0.91 GB/s，双向 22.6–22.8 GB/s，GPU 间延迟 14.5–15.9 µs，卡内 memcpy ~924 GB/s（同上）。单双向差约 25 倍是"无 P2P"的定量指纹： 无 P2P 时 cudaMemcpyPeer 退化为经主机内存的分段中转，单向吃满中转开销；双向两个方向的分段流水互相填空，逼近 Gen4 x16 的双向极限。
- **collective 路径**：nccl-tests `all_reduce_perf -b 1M -e 512M -f 2 -g 2`， avg bus bw **1.78 GB/s**，256 MB 以上大消息 ~1.85 GB/s(`hw/all_reduce_perf.txt`， EXP-002)。NCCL 探测不到 P2P 后回退 SHM（共享内存中转）传输。
- **KV 通路**：NIXL 的有效吞吐 0.26–0.27 GB/s(telemetry-derived，EXP-006/007)。注意这是第三条路径，既不等于裸拷贝也不等于 collective——为什么它最慢，是讲义 02 的主题。

#### 3.1.1 路径一的语义:CUDA 文档怎么定义"没有 P2P"

CUDA C++ Programming Guide 把这件事写得很明确。判定接口："Peer-to-peer memory access is supported between two devices if `cudaDeviceCanAccessPeer()` returns true for the specified devices"(§3.4.2.2 Peer-to-Peer Memory Access)。启用后的后果："If peer-to-peer access is enabled between two devices ... peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster"(§3.4.2.1 Peer-to-Peer Memory Transfers)。

把这句话反着读就是本机的处境：**P2P 不可用时，跨卡拷贝必须经主机内存中转**。这不是"慢一点"，这是路径变了——一次逻辑上的 GPU→GPU 拷贝变成 GPU0→host（PCIe 上行）+ host→GPU1（PCIe 下行）两段，中间还要过一次系统内存。

**这条语义决定了三件事**，而且每一件都能在本仓数据里找到对应：
1. 单向带宽被中转的串行开销钉死（0.60–0.91 GB/s）；
2. 双向能好得多，因为两个方向的两段可以互相填流水（22.6–22.8 GB/s）；
3. 延迟高（14.5–15.9 µs），因为多了一次主机侧的调度与同步——这个 $\alpha$ 值在讲义 02 里成为解释 0.27 GB/s 的主角。

#### 3.1.2 单双向差 25 倍:一个可以算的账(本讲义推导)

PCIe 4.0 的链路规格：每 lane 16 GT/s，128b/130b 编码，故每 lane 有效 $16\times128/130 = 15.75$ Gb/s $\approx 1.969$ GB/s；x16 即 $\approx 31.5$ GB/s **每方向**（PCI-SIG PCIe 4.0 规格；编码开销 1.54%）。本机两卡都是 Gen4 x16（EXP-002 的 PCIe link 行）。

于是：单向 D2D 实测 0.60–0.91 GB/s = 规格的 **1.9%–2.9%**，差两个数量级，说明瓶颈 **根本不在链路**而在中转路径的分段与同步（公理 B 的 $\alpha$ 项）；双向 22.6–22.8 GB/s 按"每字节两次过境 PCIe"折算约 45 GB/s 的 PCIe 流量，分摊到两条 x16 的上下行四个方向， **处在 Gen4 x16 规格的量级内**——仓内原表述"逼近 Gen4 x16 的双向流水极限"应按量级判断读，不宜读成"打满规格"（本讲义口径澄清，规格值来自 PCI-SIG，不改仓内实测数字）。两者之比 25 倍，正是"固定开销主导 vs 流水填满"的分野：单向拷贝时一段完成是下一段的前提，主机中转缓冲区的填-排空严格串行，$\alpha$ 每段付一次；双向时两个方向的段交错， 一个方向的等待期被另一个方向填上。

#### 3.1.3 路径二的口径:busbw 不是 algbw

`hw/all_reduce_perf.txt` 里有两列带宽，读错列会得到完全不同的结论。 nccl-tests 官方文档 `doc/PERFORMANCE.md` 的定义：

- **Algorithm bandwidth**:"using the most commonly used formula for bandwidth: size (S) / time (t)"，即 `algbw = S/t`;
- **Bus bandwidth**:"To provide a number which reflects how optimally the hardware is used, NCCL tests introduce the notion of 'Bus Bandwidth'"; AllReduce 的换算是 $B = \mathrm{algbw}\times\frac{2(n-1)}{n}$，理由是 "we have S elements, 2*(n-1) operations per element, and n links of bandwidth B to perform them"。

代入 $n=2$：$\frac{2(2-1)}{2}=1$——**两卡时 busbw 恰等于 algbw**。这解释了为什么仓内说"2 卡时两者相等"(§5.1)，也提醒一件事：**这个巧合只在 2 卡成立**，把本机的 1.78 GB/s 拿去和 8 卡集群的 busbw 比较，分母语义就不同了。

NCCL 会选哪条传输？官方环境变量文档写明 "SHM is used between devices when peer-to-peer cannot happen， therefore， host memory is used"（NCCL User Guide， Environment Variables 页，`NCCL_SHM_DISABLE` 条）。**这正是本机的分支**：P2P 探测失败 → 回退 SHM → 数据经主机内存。同一页给出 `NCCL_BUFFSIZE` 默认 4194304(4 MiB)， 即每对 GPU 的传输缓冲区大小——这个量级解释了为什么 all_reduce_perf 的曲线从 1 MB 起就接近平坦：消息早已超过缓冲区粒度，再大没有新红利。

#### 3.1.4 路径三:NIXL KV 通路为什么要单列

NIXL(NVIDIA Inference Xfer Library)的抽象层次与前两条完全不同。按其官方文档（ai-dynamo/nixl `docs/nixl.md`，Design/Memory Sections/Transfer 三节）：agent 在每个推理进程内实例化并持全局唯一 ID；内存以 "Memory Sections" 注册；发起方 "provide a list of local buffer descriptions and a list of remote buffer descriptors"，再"Using these descriptor lists， along with the target agent's name and the transfer operation (read or write)， a transfer handle can be created"； 后端（如 UCX）由 NIXL 自动选择。

关键在**粒度**：传输的最小单位是 descriptor，而 descriptor 的划分由 KV cache 的内存布局决定，不由链路决定。本机的 vLLM NIXL connector 把 K 与 V 注册成不同 region（v0.25.1 `base_worker.py` 的 region 设置注释原文："K and V are now in different regions"），于是每 block 每层每个 K/V 各一个 descriptor，28 层 × 2 = 56 个/块。 Qwen2-7B 的一个 16-token 块里每层每个 K 或 V 恰好是 $16\times4\times128\times2 = 16{,}384$ B，所以 **descriptor 恒为 16 KiB**——这个数字与实测 desc 计数逐字吻合（EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》的 1792 desc/xfer @512 桶）， 完整的定量指纹在讲义 02 §3.6。

**结论**：三条路径测的是三件事——链路能不能直连（路径一）、集合通信库怎么用这条链路（路径二）、上层传输库以什么粒度使用这条链路（路径三）。它们的数字**不可互换**， 这是本仓在所有对外表述里坚持给三个数字各自加限定语的原因。

#### 3.1.5 三条路径的口径对照表

| 路径 | 测什么 | 软件栈 | 消息粒度 | 本机数字 | 可比对象 |
|---|---|---|---|---|---|
| 裸拷贝 | 链路+驱动 | CUDA runtime `cudaMemcpyPeer` | 整块连续内存（测试用大块） | 单向 0.60–0.91 / 双向 22.6–22.8 GB/s | 同类无 P2P 的 PCIe 机器 |
| collective | NCCL 算法+传输选择 | NCCL（SHM 回退） | 1 MB–512 MB 扫描 | avg busbw 1.78 GB/s | 同卡数的 busbw |
| KV 通路 | NIXL+UCX+connector 布局 | vLLM NixlConnector → NIXL → UCX | 16 KiB descriptor | 0.26–0.27 GB/s(telemetry-derived) | 同 connector 同布局的部署 |

**读表法**：从左到右，软件栈越厚、粒度越碎，数字越小。三条路径的差距（单向裸拷贝 0.6 vs KV 通路 0.27，约 2.2×）不是"NIXL 比 memcpy 慢一半"这么简单的话能概括的——它是**粒度差**造成的，见公理 B。

### 3.2 decode 为什么由带宽定价:TPOT 下限的推导

一步一理由：

1. decode 每生成 1 个 token 要做一次完整前向。——自回归定义：第 t+1 个 token 依赖前 t 个 token 的 KV 与全部层权重，绕不开。
2. bs=1 时一次前向必须把全部权重从显存读一遍，且读进来的每个权重只做约 2 次浮点运算（乘、加）。——算术强度 ≈ 1 FLOP/byte，远低于 GPU 计算/带宽平衡点，处在 roofline 的内存受限侧；缓存放不下 14.2 GB，复用可忽略。
3. 因此 $\mathrm{TPOT}_{\min} = \dfrac{W_{\text{bytes}}}{BW}$。——时间由搬运字节数除以带宽给出下界，这是内存受限段 roofline 的直接读法。
4. 代入本机实测：$14.2\,\mathrm{GB} / 924\,\mathrm{GB/s} \approx 15.4\,\mathrm{ms}$。——14.2 GB 为 Qwen2-7B BF16 权重（仓内口径，`moe_perf/d1_analyze.py` roofline 注释）； 924 GB/s 用本机实测卡内 memcpy(EXP-002)而非标称 1008 GB/s，因为实测更贴近可达值（d1_analyze.py 用标称值，两口径都在仓内，引用时须注明用的哪个）。
5. 实测 TPOT p50 = 15.87–16.35 ms(EXP-004)，达成率约 94–97%。——decode 确为权重带宽受限，模型的其余一切（算子效率、调度）只在这 ~5% 余量里活动。

#### 3.2.1 每一步的合法性条件与失效边界

上面五步每一步都有前提，一条条钉死：

- **步骤 1 的前提**：稠密 decoder、无投机解码、无 early exit。MoE 打破"读全部权重"（每 step 只激活部分专家，讲义 02 §3.7），投机解码让一次前向产出多个 token。本仓 dense 臂两条都不触发。
- **步骤 2 的前提**：$W$ 远大于片上缓存。RTX 4090 的 L2 是 73728 KB = 72 MiB（Ada 白皮书 Appendix A Table 2），而 $W\approx14.2$ GiB——比值约 197:1， 即使权重访问完全按顺序，L2 能省下的也只有 $O(1/197)$。**这是"复用可忽略" 这句话的定量依据**（本讲义推导）。"2 次浮点运算"同样是精确值：每个权重在 GEMV 里参与一乘一加。
- **步骤 3 的前提**：roofline 的内存受限段。Williams et al. 把可达性能写成 $\min(\pi,\ \beta\cdot I)$（$\pi$ 峰值算力、$\beta$ 带宽、$I$ 算术强度）， 拐点（ridge point）在 $I=\pi/\beta$(CACM 52(4)：65–76, 2009， DOI 10.1145/1498765.1498785)。**它是下界不是预测**：假设带宽打满且计算访存完全重叠，所以实测必然 ≥ 该值。**如果实测小于该值，一定是分子或分母的口径错了**——§3.2.4 演示了这种情况怎么发生。
- **步骤 4/5 的前提**：bs=1。batch 增大后权重读取被摊薄，decode 逐渐离开带宽受限侧——这正是"tp2 的 -42% decode 收益在批量化后被稀释"的机理（§5.3）。

#### 3.2.2 算术强度 1.0 FLOP/B 不是约数,是恒等式(本讲义推导)

设权重元素数为 $P$，数据类型 $s$ 字节。bs=1 的一次前向：
- 浮点运算 $\approx 2P$（每个权重一乘一加）；
- 读取字节 $= sP$。

故算术强度 $I = 2P/(sP) = 2/s$。BF16 下 $s=2$，**$I = 1.0$ FLOP/B，与模型大小无关**。这就是"decode 算术强度约 1"这句话的来历——它不是某个模型的巧合，是 2 字节权重的恒等式。FP8 权重（$s=1$）会把它抬到 2.0，W4A16($s=0.5$)抬到 4.0，但都仍然远低于拐点。

拐点在哪？用 Ada 白皮书 Appendix A Table 2 的两个端点：BF16 Tensor（FP32 累加） 非稀疏峰值 165.2 TFLOPS，显存带宽 1008 GB/s，得 $$I_{\text{ridge}} = \frac{165.2\times10^{12}}{1008\times10^{9}} \approx 164\ \mathrm{FLOP/B}.$$ NVIDIA 把这个量叫 ops:byte——"the ratio of a processor's math and memory bandwidths" (GPU Performance Background User's Guide,§4 Understanding Performance)；同一节还写明"Performance of a function on a given processor is limited by one of the following three factors; memory bandwidth, math bandwidth and latency"。

**bs=1 decode 的 $I=1.0$ 比拐点低 164 倍**——这就是"decode 带宽受限"这句定性判断的定量强度：即使算力砍到 1/100 也不会变慢；任何"换更强算力的卡能提升 bs=1 decode"的说法都要先解释这 164 倍。第三个限制因子 latency 也不能忘：bs=1 每层 7 次 GEMV + 若干逐元素算子，28 层就是几百次 launch，**小模型上 launch 延迟会与带宽项同量级**——这正是 CUDA graphs 存在的理由；本仓四臂全部未加 `--enforce-eager`(EXP-004 §2)， 即 CUDA graphs 生效，launch 项被压到最低。

#### 3.2.3 分母之争:924 还是 1008(口径澄清)

两个数字都在仓内，选哪个会改变结论的方向，必须说清：

- **1008 GB/s** 是规格值（Ada 白皮书 Appendix A Table 2，`Memory Bandwidth`）。它是理论上界，任何实测都应低于它。
- **924 GB/s** 是 p2pBandwidthLatencyTest 带宽矩阵的对角线，即卡内 device-to-device memcpy 的实测值（`hw/p2p_bandwidth_latency.txt`：923.46 / 925.10 / 921.83 / 926.75）。

**它们的语义不同**：memcpy 是"读一份、写一份"的混合流，而 decode 读权重是纯读流。纯读通常比混合读写更容易接近峰值（无写分配、无读写切换开销）。所以 "用 924 更保守"这个直觉**方向不一定对**——用 memcpy 的数字去限制一个纯读工作负载， 可能反而给出一个偏紧、甚至过紧的下界。§3.2.4 会看到这正是本仓算式里发生的事。

**可操作的规矩**：同一个折算里，分子分母的口径必须配套。要么"规格字节 ÷ 规格带宽"（得到最乐观的下界），要么"实测字节 ÷ 同类工作负载的实测带宽"（得到贴近可达的下界）。本仓两个口径都留在案（`d1_analyze.py` 用 1008，本讲义 §3.2 步骤 4 沿用仓内 924）， 引用任何一个都必须注明——这条纪律比选哪个数更重要。

#### 3.2.4 14.2 这个魔法数是怎么来的(本文现算)

Qwen2-7B-Instruct 的权重体积可以精确算出来，不必估。两条独立途径：

**途径 A（读 checkpoint 元数据）**：本机模型目录的 `model.safetensors.index.json` 的 `metadata.total_size` = **15,231,233,024** 字节。折算： $15{,}231{,}233{,}024 / 10^9 = 15.231$ GB（十进制）， $15{,}231{,}233{,}024 / 2^{30} = 14.185$ GiB（二进制）。

**途径 B（从 config 逐项加，本讲义推导）**：每层 $= q(3584^2)+k(3584{\times}512) +v(3584{\times}512)+o(3584^2)+\text{gate}/\text{up}/\text{down}(3584{\times}18944\text{ 各一})=233{,}046{,}016$ 参数（不含 bias/norm）， 28 层 $=6{,}525{,}288{,}448$；embedding $152064{\times}3584=544{,}997{,}376$， `tie_word_embeddings` 为 false 故 lm_head 另算一份；合计 $\approx 7.615\times10^{9}$ 参数 $\times 2$ B $\approx 15.23$ GB。

两条途径一致。**于是 "14.2" 这个魔法数的身份确定了：它是 GiB 口径的参数字节数** (14.185 GiB)，不是十进制 GB。

这件事有后果，而且必须如实写出来：仓内算式 $14.2/924 = 15.4$ ms 里，分子是 GiB、分母是十进制 GB/s，**两个口径不配套**。把口径对齐重算：

| 口径组合 | 分子 | 分母 | 下界 | 与实测 15.87 ms 的关系 |
|---|---|---|---|---|
| 仓内原式 | 14.2（GiB 数值当 GB 用） | 924 GB/s（memcpy 实测） | 15.4 ms | 达成率 96.9%，自洽 |
| 十进制字节 ÷ memcpy 实测 | 15.231 GB | 924 GB/s | **16.48 ms** | **高于实测，下界被击穿** |
| 十进制字节 ÷ 规格带宽 | 15.231 GB | 1008 GB/s | 15.11 ms | 达成率 95.2%，自洽 |

**读法**：第二行是不可能的——实测不可能快过下界。这说明"十进制字节 + memcpy 带宽" 这个组合本身有问题，而问题出在分母：memcpy 的读写混合速率不适合当纯读流的上界（§3.2.3）。第三行是口径自洽的版本，它给出的结论与仓内原式**同量级、同方向**： decode 跑在权重带宽 roofline 的 95% 上。

**为什么仓内原式仍然给出了合理答案**：分子少算 6.6%（GiB 当 GB），分母也少算 8.3% (924 vs 1008)，两处偏差方向相反、量级相近，商基本不变。**这是巧合，不是可靠性**。这个案例是本篇"每个魔法数都要能说出它由什么决定"这条要求的最好注脚： **巧合的正确经不起换模型、换硬件**——换一张带宽 1.5 TB/s 的卡，或者换一个参数量不是 7B 的模型，两处偏差就不再抵消。

**对外表述建议**：引用 decode 下界时三个量一起给全——"权重 15.23 GB（=14.19 GiB， 来自 checkpoint 元数据）÷ 1008 GB/s（规格）= 15.11 ms，实测 p50 15.87 ms， 达成率 95%"。仓内既有的 15.4 ms / 94–97% 表述不改，引用时补一句它用的是 GiB 分子与 memcpy 分母。

#### 3.2.5 剩下的 3–6% 余量里装着什么(本讲义推导)

若下界是 15.11 ms、实测 15.87 ms，差 0.76 ms/token。这 0.76 ms 的候选去处：

1. **KV cache 读取**。8192 上下文 KV 为 $8192\times57{,}344 = 469.8$ MB，按 1008 GB/s 需 0.47 ms；512 上下文只有 29.4 MB，需 0.03 ms。**这一项随上下文线性增长**—— EXP-004 的 TPOT 正好从 512 桶的 15.87 ms 涨到 8192 桶的 16.34 ms(+0.47 ms)， **预测 0.47 ms 与实测差值逐 ms 对上**（本讲义推导）。这是本篇最干净的一条闭合。
2. **launch 与调度**（sampler、logits 处理、Python 侧调度）；3. **带宽达成率不足**（1008 是规格值，真实可达通常 85–95%）；4. **激活读写**（bs=1 只有几十 KB，可忽略）。

只有第 1 条能被数据独立验证，后三条本仓没有测量，**只列候选、不分配份额**。

### 3.3 TP2 的收益与代价:一半权重 + 一份通信税

**decode 侧（收益成立）**：每卡只持一半权重，步骤 3 的 $W$ 减半： $7.1/0.924 \approx 7.7\,\mathrm{ms}$；再加每 token 的小消息 allreduce 实测代价 ~1.3 ms（EXP-005《replica2/tp2 归因 + 功率帽节流调查》 §6），合计 ≈ 9.0 ms；实测 9.26–9.48 ms(EXP-005/007)。账能闭合： **-42% 的 decode 提速 = 权重带宽分摊 − 通信税**。

**prefill 侧（收益归零）**：prefill 是计算受限（8192 token 一批，GEMM 算术强度高）， 计算减半本应省约一半时间；但每层输出要做一次大消息 allreduce： $8192 \times 3584 \times 2\,\mathrm{B} = 58.7\,\mathrm{MB}$(hidden=3584，BF16)， 28 层合计 ~1.64 GB 通信量。若按大消息实测 1.85 GB/s 完全串行传输要 ~0.89 s——已超过实测 TTFT 693.7 ms(EXP-005)，说明真实执行存在计算-通信重叠/分块调度，该算式只能做 **量级判断**（推断，不是逐毫秒预测）：通信代价与计算减半的收益同量级，相互抵消。实测锚点： tp2 8K TTFT 693.7 ms ≈ 单卡冷态 ~700 ms(EXP-005)，**零加速**。另注：Megatron 式 TP 每层前向通常有注意力出投影、MLP 下投影两次 allreduce，仓内按每层一次做下界计数（EXP-005 §6 "28 层 × 58.7MB"），取哪个计数不改变量级结论。

**汇总到吞吐**：饱和吞吐 tp2 相对 colocate 只有 +13~19%（512/2K/8K 桶：12.31/10.36、 4.16/3.63、1.02/0.90，EXP-007《B1 四臂 offered-load 扫描战役》）——decode 的 -42% 在批量化后被稀释（大 batch 下 decode 逐渐转向计算/调度约束），prefill 的 allreduce 墙成为主导。

#### 3.3.1 Megatron 切法:为什么每层前向恰好两次 allreduce

上一段的"两次"不是经验值，是切法的必然结果。Megatron-LM 的原始论证（Shoeybi et al.， "Megatron-LM： Training Multi-Billion Parameter Language Models Using Model Parallelism"， arXiv：1909.08053，§3 Model Parallel Transformers）：

MLP 块 $Y=\mathrm{GeLU}(XA)$ 有两种切 $A$ 的方式。按行切 $A=[A_1;A_2]$、$X=[X_1,X_2]$ 得 $Y=\mathrm{GeLU}(X_1A_1+X_2A_2)$，而 "Since GeLU is a nonlinear function， $\mathrm{GeLU}(X_1A_1+X_2A_2)\neq \mathrm{GeLU}(X_1A_1)+\mathrm{GeLU}(X_2A_2)$ and this approach will require a synchronization point before the GeLU function"——**非线性不可分配律**是整个切法的第一性约束。按列切 $A=[A_1,A_2]$ 则 GeLU 可以各算各的（"This is advantageous as it removes a synchronization point"）。于是第一个 GEMM 列切、第二个行切，中间零通信，只在块末做一次 allreduce。注意力块同理：多头按头切 "doesnt require any immediate communication to complete the self-attention"， 出投影按行切。论文总结句："This enables us to perform all GEMMs in a simple transformer layer using only two all-reduces in the forward path and two in the backward path"(§3)。

**这条语义对本仓的意义有两层**：
1. **通信量的下界与上界都定了**。每层前向 1 次（仓内保守计数）到 2 次（Megatron 标准切法）allreduce，通信量因此在 1.64–3.29 GB 之间（8192 token、 28 层）。两个端点都远大于任何"二阶小量"的说法。
2. **切法决定了通信不可被消除，只能被重叠**。allreduce 的位置是非线性算子的位置决定的，不是实现偷懒。要减少它只能改并行维度（序列并行、pipeline 并行）， 不能改 kernel。Mooncake 也是这么说的：跨节点扩 TP 时 "requires two expensive RDMA-based all-reduce operations per layer， significantly reducing the MFU of prefill nodes"(arXiv：2407.00079，§5.1)。

#### 3.3.2 allreduce 字节数与 busbw 的换算(本讲义推导)

每层前向一次 allreduce 的**逻辑消息大小**是当前批的隐状态张量： $$m =(\text{本批 token 数}) \times d \times s.$$ prefill 8192 token：$8192\times3584\times2 = 58{,}720{,}256$ B $= 58.72$ MB（与 §3.3 一致）。 decode bs=1：$1\times3584\times2 = 7168$ B $= 7$ KiB。

**这 7 KiB 是理解 decode 侧 1.3 ms/token 的关键**：按 1.78 GB/s 的渐近带宽算， 7 KiB 只要 4 µs，而实测每 token 的 allreduce 代价 ~1.3 ms(EXP-005 §6)是纯带宽项的 **325 倍**——decode 侧的 allreduce **完全由固定开销 $\alpha$ 主导**，与带宽几乎无关。按 $k=2$ 分摊单次 $\alpha\approx0.65$ ms，按 $k=1$ 约 1.3 ms；对照 GPU 间裸延迟 14.5–15.9 µs，是它的 **40–90 倍**，里面装着 SHM 中转的两段拷贝、host 侧同步、 NCCL kernel 启动与 ring 的两个阶段。**本仓没有做 allreduce 分段计时**，以上分配为推断。

**一个可证伪的推论**：若 decode 侧由 $\alpha$ 主导，则 batch 增到 128 时消息涨 128 倍（7 KiB → 896 KiB）而每步 allreduce 时间几乎不变，**每 token 的通信税按 batch 反比下降**。这正好解释 §5.3 的"decode 的 -42% 在批量化后被稀释"：收益侧（权重分摊）不随 batch 变，成本侧随 batch 摊薄，叠加后 tp2 的吞吐优势稳定在 +13~19%。

#### 3.3.3 prefill 侧:两个同量级的量相减

设单卡 prefill 计算时间 $T_c$，TP2 后计算时间 $\approx T_c/2$（理想切分）， 每层通信 $t_{ar}(m)$，共 $L\cdot k$ 次。若通信完全不与计算重叠： $$T_{\mathrm{tp2}} \approx \frac{T_c}{2} + L\,k\,t_{ar}(m).$$ 零加速的条件是 $T_{\mathrm{tp2}}\ge T_c$，即 $$L\,k\,t_{ar}(m) \ \ge\ \frac{T_c}{2}.$$ 代入本机：$T_c\approx 700$ ms（冷态），右边 350 ms；左边按 $k=1$、 $m=58.72$ MB、$\beta=1.85$ GB/s 得 $28\times31.7 = 888$ ms **≫** 350 ms。 **不等式以两倍以上的余量成立**——所以"零加速"不是巧合，是结构性的。

但要诚实：左边 888 ms 加上右边 350 ms 是 1238 ms，而实测 tp2 8K TTFT 只有 693.7 ms。**这说明通信与计算确实有重叠**，上面的串行式只能当量级判断。要把它变成逐 ms 的预测，需要知道重叠比例，而本仓没有对 tp2 做 kernel 级 trace——如实登记为开放问题，不外推。

**能确定的是**：即使按最有利于 tp2 的假设（通信全部被隐藏），它也只能追平单卡， 因为实测就是 693.7 vs 700 ms。**在这台机器上，tp2 的 prefill 加速上限是 1.0×**。

#### 3.3.4 prefill 计算侧的 roofline(本讲义推导,含未解释余量)

上一小节用了 $T_c\approx700$ ms，这个数字值得单独算一遍，因为它牵出一个开放问题。

8192 token 一次 prefill 的浮点运算：
- **GEMM 部分**：$2 \times P_{\text{body}} \times n$，其中 $P_{\text{body}} = 6.525\times10^{9}$（§3.2.4 途径 B，不含 embedding/lm_head， 因为 embedding 是查表、lm_head 在 prefill 只对最后 1 个 token 算）， 得 $2\times6.525\times10^{9}\times8192 = 1.069\times10^{14}$ FLOP。
- **注意力部分**：每层 QK$^\top$ 满算 $2Hn^2D = 2\times28\times8192^2\times128 = 4.81\times10^{11}$，causal 掩掉一半 → $2.41\times10^{11}$；$\times V$ 同量， 每层合计 $4.81\times10^{11}$；28 层 $= 1.35\times10^{13}$ FLOP。
- 合计 $\approx 1.204\times10^{14}$ FLOP。

除以峰值：

| 时钟口径 | 峰值 BF16（FP32 累加） | 计算下界 | 对照实测 |
|---|---|---|---|
| 白皮书 boost 2520 MHz | 165.2 TFLOPS | **729 ms** | 冷态实测 ~700 ms—— **下界被击穿** |
| 遥测实测 2820 MHz | 184.9 TFLOPS（按频率线性折算） | 651 ms | 冷态 ~700 ms，达成率 93% |
| 节流带 2475–2535 MHz | 162.3–166.2 TFLOPS | 724–742 ms | 稳态 905.6 ms，达成率 80–82% |

**第一行是本篇第二个"下界被击穿"的例子**，而且这次的原因不是口径混用，是 **规格值不等于实际运行值**：仓内 EXP-005 的遥测明确记到未节流时 SM 时钟 2820 MHz， 比白皮书 boost 值高 11.9%。消费卡的实际 boost 常年高于标称，**把白皮书 boost 当硬上界会推出自相矛盾的结论**。这条教训与 §3.2.4 是一对：一个是分子口径错，一个是分母不是真上界。

**但第三行的 80–82% 达成率仍然偏高**，如实标注为开放问题：典型 serving 系统的 prefill MFU 更常见在 50–65%。三类候选解释本仓都没有独立数据裁决：(a) 实际参与计算的 prompt token 少于 8192（前缀命中——`analysis/nixl_token_accounting.md` 确记到 bench 的 test 请求让请求 #0 命中 511 块）；(b) FLOP 计数漏项（未计 RoPE/norm/激活， 但这些是 $O(nd)$ 而非 $O(nd^2)$，量级不足以解释）；(c) 频率线性折算不成立。 **如实登记，不做归因。**

### 3.4 replica2:零通信的复制,近线性的扩展

不切模型、不传 KV，唯一代价是权重显存翻倍（两卡各持 14.2 GB）与外置轮询代理（`pd_disagg/matrix/rr_proxy.py`）。因果链：零跨卡流量 → 互联质量与它无关 → 扩展效率只受负载均衡与客户端限制。实测 2K/8K 桶扩展 1.93×/1.98×(7.00/3.63、1.78/0.90， EXP-007)；512 桶 1.50×(15.58/10.36)带欠饱和疑点（EXP-007 §7：SAT_CONC=64 或代理上限，引用 512 扩展效率前须复测）。

#### 3.4.1 论文其实早就写了这一条

replica2 在本机是最优解，这个结论听起来"太朴素"，但它不是本仓的独创发现，而是 **PD 分离原始论文自己列出的备选项**。DistServe 在推导 decode 实例的并行方案时明确写道：

> "It is worth noting that when the model can fit into the memory of a single GPU, replication is a competitive option in addition to model parallelism for both prefill and decoding instances, to linearly scale the system's rate capacity. It may also reduce the queuing delay ... by substituting R with R/N assuming requests are equally dispatched to N replicas, at the cost of maintaining additional replicas of the model weights in GPU memory."（arXiv:2401.09670,§3.2 Analysis for Decoding Instance 末段）

三层信息全在里面：(a) 前提"模型放得进单卡"——Qwen2-7B BF16 15.2 GB 在 24 GB 卡上成立； (b) 收益"线性扩容 + 排队延迟按 $R\to R/N$ 下降"；(c) 代价"多一份权重显存"。 **本仓实测 1.93×/1.98× 正是这条论述的实例**，不是与文献冲突的孤例。

$R\to R/N$ 只在**请求被均分**时成立。排队等待随到达率单调上升且在接近服务率时发散， 所以把 $R$ 减半带来的延迟收益在高负载区远大于线性——这解释了为什么 replica2 的 goodput 优势（12.75/4.96/0.90）比饱和吞吐优势（1.50/1.93/1.98×）更明显： goodput 是带 SLO 的口径，吃排队延迟的红利。

#### 3.4.2 为什么是 1.93/1.98 而不是 2.00(三个漏损项)

1. **负载不均**：rr_proxy 是无状态轮询（`itertools.cycle`），按请求数而非工作量均分。本仓三个桶内请求长度相同（random 数据集固定 input_len），所以这一项被压到最小——**这是实验设计消除的漏损，不是它不存在**；真实流量（长度重尾）下会立刻显现。
2. **代理层开销与并发上限**：每请求多一次 HTTP 转发与一次流式转发（§4 段 7）。 512 桶的 1.50× 就落在这一项上（EXP-007 §7 登记为疑点）。
3. **客户端与测量侧**：SAT_CONC=64 对高吞吐桶是可能的天花板。

三项都在**入口层**，没有一项在 GPU 上——**这是 replica2 的性格**：它把瓶颈推到系统边界。生产用 K8s Service 或专用 router 解决，本仓的 rr_proxy 是最小实现。

#### 3.4.3 轮询之外:生产环境会怎么做

轮询是最弱的一档，往上依次是：最少连接数、最少 token 数（考虑工作量）、前缀缓存感知路由（把共享前缀的请求送到同一副本）。最后一档能拿到 rr_proxy 拿不到的收益，机制上等价于把 PagedAttention 的块级共享（arXiv：2309.06180，§4.2/§4.4 的 copy-on-write 与前缀共享）从单实例推广到副本组。**本仓不做会话亲和，replica2 的实测值是这一族方案的下限。**

### 3.5 pd1p1d:每请求一份 KV 的搬运账

Qwen2-7B 的 KV 每 token 字节数： $28\,\text{层} \times 2\,(\mathrm{K,V}) \times 4\,\text{KV 头} \times 128\,\text{维} \times 2\,\mathrm{B} = 57344\,\mathrm{B}$——与 EXP-006 单请求探针 bytes=917504 = 16 token × 57344 B 完全吻合（block=16 取整）。8K 请求全量 KV ≈ 469.8 MB（EXP-011《EXT-2 NixlPush 单点》 push 臂实测全量）；pull 臂经前缀缓存裁剪实拉 439.7 MB(EXP-006)。在 0.27 GB/s 的有效吞吐下：$439.7\,\mathrm{MB} / 0.27\,\mathrm{GB/s} \approx 1.63\,\mathrm{s}$， 与实测 avg xfer 1602.7 ms(EXP-006)对上。容量上限： $0.27\,\mathrm{GB/s} \div 470\,\mathrm{MB/req} \approx 0.57\,\mathrm{req/s}$， 实测饱和 0.54 req/s@8K(EXP-007)——**传输带宽即容量**，账在两端都闭合。

#### 3.5.1 GQA 已经把这笔账砍到了七分之一

$57{,}344$ B/token 这个数字里，$KVH=4$ 是决定性的。若 Qwen2-7B 是 MHA（28 个 KV 头）， 每 token 就是 $28\times2\times28\times128\times2 = 401{,}408$ B—— **7 倍**。 8192 token 的单请求 KV 会从 469.8 MB 涨到 3.29 GB，按 0.27 GB/s 要 12.2 s。

GQA 的原始动机正是这个：Ainslie et al. 指出多查询/分组查询注意力的收益在于减少 KV 的加载量（arXiv：2305.13245，§2）。**在 PD 分离场景里，GQA 的收益从 "少读"变成了"少传"**——这是同一个架构选择在不同系统形态下的两种兑现方式。换句话说：**本机 PD 溃败的严重程度已经被 GQA 缓解了 7 倍，仍然溃败。**

#### 3.5.2 block=16 的出处与取整规则

"KV 按 16 token 一块管理"不是本仓的约定，是 vLLM 的默认值，而这个默认值有论文依据。 PagedAttention 论文 §7.2 Impact of Block Size 做了 1–256 的扫描，结论原文： "In practice, we find that the block size 16 is large enough to efficiently utilize the GPU and small enough to avoid significant internal fragmentation in most workloads. Accordingly, vLLM sets its default block size as 16." (arXiv:2309.06180,§7.2)。同节还给出了两侧的失效机理：块太小则 "vLLM may not fully utilize the GPU's parallelism for reading and processing KV cache"；块太大则"internal fragmentation increases and the probability of sharing decreases"。

**这条默认值直接决定本仓的三个数字**：$16\times57{,}344 = 917{,}504$ B/块（EXP-006 单请求探针的 bytes 实测值）；8192 token = 512 块恰好整除，故 8K 桶的 bytes 与公式逐字节相等；非块对齐 prompt 向上取整——这是探针里 9 token 的 prompt 报出一整块 917,504 B 的原因。**魔法数归类**：16 由**实测扫描**定（论文 §7.2 的曲线），既非理论上界也非硬件约束，别的 workload 上可以是别的值——所以它是可配置项。

#### 3.5.3 容量上限:一个排队论表述(本讲义推导)

$0.27/0.470\approx0.57$ req/s 这条式子成立需要三个条件，缺一不可：

1. **传输在关键路径上**：D 端必须等 KV 到齐才能开始 decode。pull 语义下这是定义（`WAITING_FOR_REMOTE_KVS` 状态）；
2. **传输资源被串行占用**：两个请求的 KV 传输不能真正并行地各拿满带宽。若能完美并行，上限就不再是 $BW/B_{kv}$；
3. **不存在其它更早的瓶颈**：P 端 prefill 能力（约 1/0.9 ≈ 1.1 req/s@8K）与 D 端 decode 能力都必须高于 0.57。

三条在本机都成立，所以式子有效，实测 0.54 与预测 0.57 差 5%（剩余差距来自 P/D 端的其它开销与调度间隙）。

用排队论的话说这是一个**服务率上界**：串联队列（client → P → 传输 → D）的吞吐由最慢的一站决定，传输站的服务率 $\mu_{\mathrm{xfer}} = BW_{\mathrm{eff}}/B_{kv}(n)$。三个桶各自代入：

| 桶 | $B_{kv}$ | $\mu_{\mathrm{xfer}}$ 预测 | 实测饱和 | 比值 |
|---|---|---|---|---|
| 512 | 29.4 MB | 9.2 req/s | 7.84 | 0.85 |
| 2048 | ~117 MB | 2.3 req/s | 2.12 | 0.92 |
| 8192 | 469.8 MB | 0.57 req/s | 0.54 | 0.95 |

（本文现算，$B_{kv}$ 取 EXP-006 的 MB/xfer 实测值，$BW_{\mathrm{eff}}=0.27$ GB/s）

**读法**：输入越长，预测越准（0.85 → 0.92 → 0.95）。这正是"传输站成为唯一瓶颈" 的程度在加深——短桶里 P/D 两端的固定开销还占得住位置，长桶里传输把一切吃掉。 **这张表本身就是"传输带宽即容量"这句话的一条独立证据链**，它不是从 8K 一个点外推出来的。

#### 3.5.4 传输能不能藏起来:Splitwise 的答案与本机的答案

Splitwise 面对同一个问题给了工程解：把 KV 传输**与 prefill 计算重叠**。原文机制（Patel et al.， arXiv：2311.18677，§IV-C KV-cache transfer）：

> "In Splitwise, we optimize the KV-cache transfer by overlapping it with the computation in the prompt phase. As each layer in the LLM gets calculated in the prompt machine, the KV cache corresponding to that layer is also generated. At the end of each layer, we trigger an asynchronous transfer of the KV-cache for that layer while the prompt computation continues to the next layer."

效果（§VI-A）：A100 组上非重叠残留约 8 ms、H100 组约 5 ms；端到端上， 串行传输给第二个 token 增加 64% 时延，而 Splitwise 的逐层重叠只增加 16.5%。

**这套办法在本机能救多少？一个上界估计（本讲义推导）**：逐层重叠最多把传输藏进 prefill 时长里，所以可隐藏的上限就是 prefill 本身。8K 桶 prefill ≈ 0.9 s、传输 ≈ 1.63 s， **藏完之后还剩 ≈ 0.73 s 露在外面**；即使完美重叠，TTFT 也只能从 2718.7 ms 降到约 1819 ms，相对 colocate 的 925.2 ms 仍接近 2×。**互联能力不够时，重叠优化只能改善常数，改不了结论。** 这与另一条独立证据一致：换方向（push）只挽回 8K TTFT 6.7% (EXP-011)，量级不变——**问题在 $BW_{\mathrm{eff}}$，不在调度方式。**

### 3.6 把四臂放进同一个账本(本讲义推导)

现在可以给一个统一式。对输入长 $n$、输出长 $g$ 的单请求，并发 1、无排队：

$$\mathrm{TTFT} \approx \underbrace{T_{\text{prefill}}(n)/C}_{\text{计算}}
+ \underbrace{L\,k\,t_{ar}(n d s)}_{\text{TP 通信}}
+ \underbrace{B_{kv}(n)/BW_{\mathrm{eff}}}_{\text{PD 传输}}
+ \underbrace{o}_{\text{固定开销}}$$

$$\mathrm{TPOT} \approx \frac{W/C}{BW_{\mathrm{mem}}} + \frac{B_{kv}(n)/C}{BW_{\mathrm{mem}}}
+ L\,k\,t_{ar}(d s) + o'$$

四臂就是把参数填进去：

| 臂 | $C$ | $k$ | $BW_{\mathrm{eff}}$ 项 | 预测 TTFT@8K | 实测 |
|---|---|---|---|---|---|
| colocate | 1 | 0 | 无 | $T_{\text{prefill}}$ | 925.2 |
| replica2 | 1 | 0 | 无 | 同上 | 902.6 |
| tp2 | 2 | 1–2 | 无 | $T_{\text{prefill}}/2 + $ 通信 ≈ 持平 | 881.3 |
| pd1p1d | 1 | 0 | 有 | $T_{\text{prefill}} + 1.63\,\mathrm{s}$ ≈ 2.5 s | 2718.7 |

（TTFT 实测取 EXP-007 v2 attribution 同热工况 p50）

**这张表的价值在于预测力**：pd1p1d 那一行先算后测都能对上（0.9 + 1.63 = 2.53 s 预测 vs 2.72 s 实测，差 7% 落在 D 端首步与代理开销上）。唯一不能被准确预测的是 tp2 的 prefill——重叠比例未知（§3.3.3），这也是本篇唯一显式承认"只能给量级"的地方。

### 3.7 goodput:文献里的定义与本仓的实现

"吞吐高"不等于"服务好"。文献里的标准处理是引入 **goodput**——只数达标请求。 DistServe 给的定义（arXiv：2401.09670，§1/§2）：

> "per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned"

拆开看有四个可调旋钮，任何一篇报告 goodput 的文字都必须交代清楚： ① **SLO 是哪两条**——DistServe 用 TTFT + TPOT(§2.1)，本仓同样两条（TTFT ≤ 桶基线 5 倍 + TPOT ≤ 50 ms）；② **达标率门槛**——DistServe 用 90%（附录另给 99%）， **本仓不用门槛**，直接数达标请求除以墙钟；③ **分母**——DistServe 是 per-GPU， 本仓两个都报（fig1 per-system，fig3 per-GPU）；④ **SLO 绝对值怎么定**——DistServe 用应用语义（chatbot TTFT 0.25 s 等，§6.1）并另设 **SLO Scale** 做敏感性："We fix the rate and then linearly scale the two latency requirements in Table 1 simultaneously using a parameter called SLO Scale. As SLO Scale decreases， the latency requirement is more stringent."(§6.2)。**本仓 fig6 做的是同一件事**（0.5–4× 扫描，§4 段 9）， 方法论层面的独立收敛。

**两边口径的差别要讲清楚**：DistServe 的 goodput 是"给定达标率门槛下能承载的最大速率"（标量，需沿速率轴搜索），本仓的是"某个 offered load 下每秒达标请求数"（曲线）。 **后者包含前者**——在本仓曲线上找"达标率 ≥ 90% 的最右点"即还原成 DistServe 的定义。本仓选曲线口径，是因为过峰后的下降段形状本身携带信息（§5.6 读图法）。

### 3.8 魔法数总表:每个数字由什么决定

本节把全篇出现的常数逐个归类。三类来源：**理论上界**（从定义或规格推出）、 **硬件约束**（由这台机器的物理/驱动状态给定）、**实测扫描**（由测量或调参选出）。

| 数字 | 出现在 | 由什么决定 | 换机器/换模型后怎么变 |
|---|---|---|---|
| 57,344 B/token | §3.5 | **理论上界**：$2LKVH\cdot D\cdot s$，由 config 完全确定 | 按公式重算；MHA 会 ×7 |
| 16 token/block | §3.5.2 | **实测扫描**：PagedAttention §7.2 的块大小曲线 | vLLM 可配置；别的负载可能选别的值 |
| 917,504 B/block | §3.5 | **理论上界**：前两者相乘 | 同上 |
| 16,384 B/descriptor | §3.1.4 | **理论上界**：$16\times KVH\times D\times s$，由 connector 的 region 划分决定 | 改 region 划分即改变 |
| 1008 GB/s | §3.2.3 | **硬件约束**：Ada 白皮书 Table 2 规格 | 随卡型变 |
| 924 GB/s | §3.2 步骤 4 | **实测扫描**：p2pBandwidthLatencyTest 对角线 | 随卡与测法变；注意它是读写混合口径 |
| 14.2(GiB) | §3.2.4 | **理论上界**：checkpoint 元数据/config 推出的参数字节数 | 按模型重算；注意 GiB vs GB |
| 165.2 TFLOPS | §3.2.2 | **硬件约束**：Ada 白皮书 Table 2，boost 2520 MHz 口径 | 实际时钟可高于此（§3.3.4） |
| 164 FLOP/B(ridge) | §3.2.2 | **理论上界**：上两者相除 | 随卡型变 |
| 1.0 FLOP/B(decode) | §3.2.2 | **理论上界**：$2/s$ 恒等式 | 只随权重位宽变 |
| 31.5 GB/s(PCIe 4.0 x16) | §3.1.2 | **硬件约束**：PCI-SIG 规格 | 随代际变（Gen5 翻倍） |
| 1.78 GB/s | §3.1、§3.3 | **实测扫描**：nccl-tests 1M–512M 扫描的 avg busbw | 有 P2P/NVLink 时相差两个数量级 |
| 0.26–0.27 GB/s | §3.1.4、§3.5 | **实测扫描**：NIXL telemetry 反解 | 由 $\alpha$ 与 descriptor 大小共同决定 |
| 450 W / 0x4 | §5.4 | **硬件约束**：TGP 规格 + NVML 位定义 | 可用 `nvidia-smi --power-limit` 改 |
| 2820 / 2475 MHz | §5.4 | **实测扫描**：遥测采样 | 随卡个体、散热、功率帽变 |
| SLO 5× / 50 ms | §3.7、§4 段 5 | **实测扫描 + 约定**：5× 由本仓预注册，50 ms 为固定值 | 应按应用语义重设；敏感性见 fig6 |
| SAT_CONC=64 | §4 段 2 | **约定**：探顶用的客户端并发 | 512 桶疑不足（EXP-007 §7） |
| 32 / 320 / 192 / 56 | §5.2 | **约定**：每点请求数，按桶递减以控制单点时长 | 与统计精度直接相关 |

**这张表的用法**：向别人引用任何一个数字之前，先在这里找到它的行，把"由什么决定" 那一列一起说出去。标"理论上界"的可以换算，标"硬件约束"的必须带卡型， 标"实测扫描/约定"的必须带测量条件。

## 4. 代码逐段走读

单测量点的完整执行路径 = `run_point.sh`（起测）→ `vllm bench serve`（打点）→ `collect_point.py`（落账）。按执行顺序读六段，再补四段外围（代理、快照、出图、收尾）。

**段 1：快照与遥测先行（`pd_disagg/scripts/run_point.sh:20-33`）**

```bash
PREFIX=$(date -u +%Y%m%dT%H%M)_${ARM}_${IN}x${OUT}_${MODE}
[ "$RPS" != "-" ] && PREFIX=${PREFIX}_rps${RPS}

for p in "${ENGINE_PORTS[@]}"; do
  scripts/metrics_snapshot.sh snap "$p" "$R/snapshots/${PREFIX}_${p}_before.prom"
done

# GPU 遥测采样(2s): 功率帽节流会使持续 prefill 降频~12%、TTFT 抬升(8/21 实测),
# 每个测量点必须留下工况证据
GPUCSV=$R/raw/${PREFIX}_gpu.csv
( while true; do
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
      --format=csv,noheader >> "$GPUCSV"; sleep 2
  done ) & SAMPLER=$!
```

角色：一次测量的三件套（before 快照、GPU 工况流水、统一 UTC 前缀）在 bench 之前就位。为什么这么写：快照抓的是**引擎端口**（`ENGINE_PORTS` 与 bench 打的端口分离，pd 臂 bench 打代理、快照仍直抓 8100/8200）——传输是否真实发生要看引擎侧计数器增量，代理不可信； 遥测采样是功率帽事件（§5.4）之后加装的制度化产物，注释里写着动机。改错会怎样：若快照抓代理端口，PD 臂 gate 全部拿不到 nixl 计数器，整点作废；若去掉遥测，臂间差异可能被功率状态淹没且无法事后自证工况。

**采样周期 2 s 的依据**：功率帽的调频在毫秒级，2 s 采样**捕捉不到瞬态**，只给稳态分布——这是有意取舍，因为目的是回答"这一点是不是在功率帽下跑的"这个**二值**问题， 只需要节流原因位的出现与否。`clocks_event_reasons.active` 的语义由 NVML 定义（每 bit 一个原因，`0x1`=GpuIdle、`0x4`=SwPowerCap、`0x40`=HwThermalSlowdown； NVML API Reference `nvmlClocksThrottleReasons` 组），**§5.4 的定案完全建立在这三位上**。

**段 2：三种打法与唯一 seed(`run_point.sh:35-52`)**

```bash
if [ "$MODE" = attribution ]; then
  RATE_ARGS=(--max-concurrency 1 --request-rate inf)
elif [ "$MODE" = saturation ]; then
  # 饱和探测: 无限速率+高并发, 得到该臂×桶的最大吞吐 → sweep 档位按其比例取
  RATE_ARGS=(--max-concurrency "${SAT_CONC:-64}" --request-rate inf)
else
  RATE_ARGS=(--request-rate "$RPS")
fi

"$VENV/bin/vllm" bench serve \
  --host localhost --port "$BENCH_PORT" --model "$MODEL" \
  --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
  --num-prompts "$NUM" --ignore-eos --seed "${SEED:-42}" \
  "${RATE_ARGS[@]}" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir "$R/raw" \
  --result-filename "${PREFIX}_bench.json" \
  2>&1 | tee "$R/raw/${PREFIX}_bench.log"
```

角色：同一个入口跑三种模式——attribution（并发 1，量"干净"的延迟基线）、saturation（无限速率高并发，探最大吞吐，给 sweep 档位定标尺）、sweep（固定 offered rate，量负载-goodput 曲线）。为什么这么写：`--save-detailed` 落每请求的 TTFT/ITL 数组，goodput 必须逐请求判定（段 5）；`SEED` 从外部注入，协议 v2 下每个测量点唯一（§5.2 讲为什么）； `--ignore-eos` 保证输出长度恒定，否则各点输出长不可比。改错会怎样：漏 `--save-detailed` 则 goodput 无法计算；seed 写死 42 会踩前缀缓存污染（仓内实测 2048 桶 25% 命中虚高， `pd_disagg/analysis/nixl_token_accounting.md`）。

**三种模式对应三个排队学量**：attribution（并发 1）测**服务时间** $1/\mu$， 它把调度器完全旁路（队列里永远只有一个请求，连续批处理没有对象可批）； saturation 测**最大服务率** $\mu_{\max}$；sweep（固定 offered rate $\lambda$）测 $\lambda<\mu$ 区间的**响应时间曲线**，是唯一能画出 goodput 拐点的模式。 **三种缺一不可**：只有 attribution 会暴露"零负载下 PD 已经慢 3 倍"，只有 saturation 能给 sweep 定标尺，只有 sweep 能证明"pd 在 66% 饱和度时 goodput 就塌了"。

`--ignore-eos` 把输出长度钉死，代价是失去真实的输出长度分布，收益是每点工作量完全确定——**可比性优先于真实性的一次显式取舍**，因为本仓测的是形态间的相对关系。

**段 3：精确指标名取增量（`pd_disagg/scripts/collect_point.py:29-41`）**

```python
def exact_delta(deltas, metric_name, label_sub=None):
    """按精确指标名(跨引擎端口求和)取增量。名字来源: PD 探针实测,
    v0.25.1 传输计数在 D(consumer)端, P 端仅 failed/expired; _created 是时间戳需排除。"""
    total, found = 0.0, False
    for k, v in deltas.items():
        prom = k.split(":", 1)[1]          # 去掉 "port:" 前缀
        if prom.split("{")[0] != metric_name:
            continue
        if label_sub and label_sub not in prom:
            continue
        total += v
        found = True
    return total if found else None
```

角色：把 before/after 两份 Prometheus 快照的差值按**精确指标名**归并（P、D 两端口求和）。为什么这么写：指标名全部来自 EXP-006 的探针实测（快照→单请求→快照→diff），而不是猜子串——`_created` 系列是时间戳伪装成计数器，子串匹配会把它当增量收进来；传输计数只在 D 端（pull 语义，READ 发起方），跨端口求和才对。改错会怎样：早期版本用子串猜名，遇上 `pd1p1d_push` 臂（计数器移到 P 端）与 `_created` 就会算出错账。

**`prom.split("{")[0] != metric_name` 做的是"去标签后全等"**：Prometheus 一行是 `name{label="v",...} value`，同名指标可有多组标签。先切掉 `{` 之后再全等比较 = "匹配该指标的所有标签组合"，再由 `label_sub` 做标签内过滤—— `prompt_tokens_by_source_total{source="external_kv_transfer"}` 正是靠后者选出来的。这个两级过滤是"名字可重复、标签才是维度"这条数据模型的直接体现。

**段 4：gate——数字与它的合格证同行（`collect_point.py:73-104`）**

```python
    is_pd = args.arm.startswith("pd1p1d")   # 含 pd1p1d_push（EXT-2 修正）
    completed = g("completed", 0)
    duration = g("duration", 0.0)
    failed_requests = len([e for e in (g("errors") or []) if e])

    nixl_bytes = nixl_xfers = xfer_time_s = post_time_s = None
    descriptors = failed_xfers = failed_notifs = expired = ext_kv_tokens = None
    if is_pd:
        nixl_bytes = exact_delta(deltas, "vllm:nixl_bytes_transferred_sum")
        nixl_xfers = exact_delta(deltas, "vllm:nixl_bytes_transferred_count")
        xfer_time_s = exact_delta(deltas, "vllm:nixl_xfer_time_seconds_sum")
        post_time_s = exact_delta(deltas, "vllm:nixl_post_time_seconds_sum")
        descriptors = exact_delta(deltas, "vllm:nixl_num_descriptors_sum")
        failed_xfers = exact_delta(deltas, "vllm:nixl_num_failed_transfers_total")
        failed_notifs = exact_delta(deltas, "vllm:nixl_num_failed_notifications_total")
        expired = exact_delta(deltas, "vllm:nixl_num_kv_expired_reqs_total")
        ext_kv_tokens = exact_delta(
            deltas, "vllm:prompt_tokens_by_source_total",
            label_sub='source="external_kv_transfer"',
        )
        gate_pass = None
        if None not in (nixl_bytes, nixl_xfers, failed_xfers, failed_notifs, expired):
            gate_pass = (
                failed_requests == 0
                and nixl_bytes > 0
                and nixl_xfers == completed
                and failed_xfers == 0
                and failed_notifs == 0
                and expired == 0
            )
    else:
        gate_pass = failed_requests == 0
```

角色：每个测量点的"合格证"。PD 臂要过五关：零失败请求、传输字节确有增量、**成功传输数恰等于完成请求数**、零失败传输/通知、零过期。为什么这么写：PD 臂最阴险的失效模式是 "服务器照常返回、KV 其实没传"（fail policy 不设 fail 时静默回退本地重算）——只有引擎侧计数器能拆穿；`startswith` 而非 `==` 是 EXP-011 的教训：push 臂名 `pd1p1d_push` 被精确匹配漏掉，导致该臂两行的结构化 gate 字段为 None（原始计数幸存于 kv_deltas_raw， EXP-011 §4 如实登记）。改错会怎样：没有 gate，PD 臂的"好看数字"可能根本没走传输路径。

**五个条件里最强的是 `nixl_xfers == completed`**：前四条都是"不出错"型断言， 只能排除显式失败；这一条是**计数配平**，把"每个完成的请求恰好对应一次成功传输"写成等式，少一次（静默回退本地重算）或多一次（重传）都会打破它。**配平等式比阈值断言强一个等级**——阈值只能说"不太离谱"，等式能说"不多不少"（讲义 02 的三重互证是它的完整展开）。**`gate_pass` 取 `None` 而非 `False`** 也是同一种严谨：计数器取不到时 "不知道"与"没通过"是两种状态，写 `False` 会把工装故障伪装成实验失败；EXP-011 的 push 臂两行正是 `None`，原始计数仍在 `kv_deltas_raw` 里，数据没丢。

**段 5：goodput 的定义（`collect_point.py:106-116`）**

```python
    goodput = None
    if args.slo_ttft_ms is not None and args.slo_tpot_ms is not None:
        ttfts = g("ttfts") or []   # 单位: 秒(detailed 数组)
        itls = g("itls") or []
        ok = 0
        for i, t in enumerate(ttfts):
            per_itl = itls[i] if i < len(itls) else []
            tpot_ms = (sum(per_itl) / len(per_itl) * 1000) if per_itl else float("inf")
            if t * 1000 <= args.slo_ttft_ms and tpot_ms <= args.slo_tpot_ms:
                ok += 1
        goodput = round(ok / duration, 4) if duration else None
```

角色：goodput = **同时满足 TTFT 与 TPOT 两条 SLO 的请求数 ÷ 墙钟时长**(req/s)。为什么这么写：必须逐请求判定（detailed 数组），聚合分位数（如 p99≤SLO）只能给"整体过/不过"的布尔量，画不出连续的 goodput 曲线；TPOT 用该请求 ITL 均值重算，不依赖 bench 的聚合口径。SLO 数值：TTFT ≤ 5× 无负载基线（512/2K/8K = 328/891/4626 ms）+ TPOT ≤ 50 ms，基线来自 EXP-004 并预注册锁定。改错会怎样：用均值 TTFT 判定会把"半数请求超时"的点算成满分——goodput 曲线在过载段的陡降（fig1）正是逐请求判定才画得出来。

**三处细节各挡一类错误**：① `t * 1000`——detailed 数组单位是**秒**、SLO 参数是**毫秒**， 换算写在比较的同一行，让口径错误无处藏身；② `float("inf")` 兜底——只产出 1 个 token 的请求 TPOT 无定义，判成 `inf` 即判不达标，这是 fail-closed 取向（**观测缺失算失败**， 否则异常请求虚增 goodput）；③ `ok / duration` 而非 `ok / len(ttfts)`——分母是墙钟， 得到的是**速率**而非**达标率**，只有速率能与 offered load 放在同一坐标轴上比（fig1 的 x 轴）。与 §3.7 的文献口径对照：把本仓曲线反查"达标率首次跌破 90% 的位置"， 就得到 DistServe 口径的那个标量，两者不冲突，粒度不同。

**段 6：遥测汇总——工况证据入行（`collect_point.py:138-149`）**

```python
        gpu_telemetry = {}
        for idx, d in per.items():
            loaded = [s for s, p_ in zip(d["sms"], d["pws"]) if p_ > 100]
            gpu_telemetry[idx] = {
                "samples": len(d["sms"]),
                "temp_max_c": max(d["temps"]),
                "power_max_w": max(d["pws"]),
                "sm_clock_min_loaded_mhz": min(loaded) if loaded else None,
                "sm_clock_mean_loaded_mhz": round(sum(loaded) / len(loaded), 0)
                if loaded else None,
                "throttle_reasons_seen": sorted(d["reasons"] - {"0x0000000000000000"}),
            }
```

角色：把 2 s 采样流水压缩成每卡摘要（负载态最低/平均 SM 频率、峰值功率、见过的节流原因位）随行写进 runs.jsonl。为什么这么写：`p_ > 100`(W)过滤空闲样本——空闲频率 210 MHz 会把均值拉没意义；节流原因保留原始位掩码集合，`0x4`(SW Power Cap)出现即该点带功率帽工况。改错会怎样：不过滤空闲样本，"降频"永远存在；丢掉 reasons，就无法区分热节流与功率帽（§5.4 的定案证据正是 reason 位）。

**100 W 阈值的依据**：本机空闲功耗实测 13.97–15.57 W，负载态 231.97–444.65 W（EXP-005 `throttle_trace.csv`，本文现算）——**两个分布之间有 200 W 以上的空隙**， 阈值落在空隙里，不是拍脑袋。设阈值的正确姿势就是先看两类样本的分布再取间隔中央。 `sorted(...)` 只记"见过哪些原因位"不记次数，同样是因为 gate 要答的是二值问题； 需要分布时回原始 CSV 重算（§5.4.2 就是这么做的）。

**段 7：轮询代理——replica2 的全部实现（`pd_disagg/matrix/rr_proxy.py:31-46`）**

```python
async def _forward(request: Request, path: str):
    client = app.state.clients[next(app.state.rr)]
    body = await request.body()
    req = client.build_request(
        "POST", path, content=body, headers={"Content-Type": "application/json"}
    )
    resp = await client.send(req, stream=True)

    async def gen():
        async for chunk in resp.aiter_raw():
            yield chunk
        await resp.aclose()

    return StreamingResponse(
        gen(), status_code=resp.status_code, media_type=resp.headers.get("content-type")
    )
```

角色：replica2 这条臂的**全部**跨卡逻辑就是这十几行——它是"$X=0$"的字面实现。 **关键行为什么这么写**：① `next(app.state.rr)` 用 `itertools.cycle` 做无状态轮询， 选择与请求内容无关——这既是优点（零状态零同步）也是上限（不感知在途负载与前缀， §3.4.3）；② `stream=True` + `aiter_raw()` 逐块转发，**不缓冲整个响应**——若在这里聚合完再回，TTFT 会被拉到 e2e 时长，整条臂的延迟测量全部作废；③ 用 `aiter_raw` 而非 `aiter_bytes`，不做内容解码、原样透传 SSE 分块，避免代理改写 chunk 边界； ④ 每个后端一个长期 `AsyncClient`（`lifespan` 里建、`timeout=None`），连接复用。 **这段同时是一条边界声明**：没有健康检查、重试、会话亲和、背压——512 桶的欠饱和疑点（§3.4.2 漏损项 2）首先该怀疑这里，而不是 GPU。

**段 8：快照与增量——gate 的数据源（`pd_disagg/scripts/metrics_snapshot.sh:12-26`）**

```bash
case "${1:-}" in
  snap)
    curl -sf "http://localhost:$2/metrics" > "$3"
    echo "saved $3 ($(grep -cE "$PAT" "$3" || true) 行命中 KV-transfer 模式)"
    ;;
  diff)
    norm() { grep -E "$PAT" "$1" | grep -v '^#' \
             | sed -E 's/[[:space:]]+([0-9.eE+-]+)$/\t\1/' | sort; }
    join -t $'\t' -j 1 <(norm "$2") <(norm "$3") \
    | awk -F'\t' '{ d = $3 - $2
        # failed/expired 是"必须为 0"的 gate 证据，Δ=0 也要打印
        if (d != 0 || $1 ~ /failed|expired/)
            printf "%-90s Δ=%g (%g -> %g)\n", $1, d, $2, $3 }'
    echo "--- (其余 Δ=0 计数器已省略；gate 关注 bytes/transfers 的 sum/count 增量与 failed/expired=0) ---"
    ;;
```

角色：`snap` 落全量快照，`diff` 出人可读的增量。**关键行为什么这么写**： ① `curl -sf` 的 `-f` 让 HTTP 错误变成非零退出码——端口没起时立刻失败，而不是把一个 HTML 错误页写成 `.prom`（EXP-007 §7 记录过 3 个 0 字节空快照，正是这类窗口）； ② `grep -v '^#'` 去掉 HELP/TYPE 元数据行；③ `sed` 把"指标名（含标签） + 值"切成两列， 以**整行指标名（含标签）**作 join 键——标签是维度的一部分，不能丢；④ `join` 之前必须 `sort`；⑤ awk 里 `d != 0 || $1 ~ /failed|expired/` 是最有意思的一行：普通计数器只在有增量时打印，而 **failed/expired 即使 Δ=0 也强制打印**——"这一项是 0"本身就是 gate 要的证据，省掉它等于把合格证撕掉一半。**证据的"存在性"与"取值"是两件事。** 本段与段 3 分工：shell 版供人眼排查（EXP-006 探针阶段靠它认出指标名），Python 版供机器判 gate，两者读同一批 `.prom` 文件，互为交叉验证。

**段 9：SLO 敏感性——防"挑阈值"的那张图（`pd_disagg/scripts/make_figures.py:218-239`）**

```python
def fig6_slo_sensitivity(rows):
    """2048 桶、四臂：goodput@各 SLO 倍率（相对锁定 SLO 0.5/1/2/4×）——防"挑阈值"。"""
    scales = [0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for arm in ARMS:
        cand = [r for r in rows
                if r["mode"] == "sweep" and r["arm"] == arm and r["input_len"] == 2048]
        ys = []
        for s in scales:
            best = 0.0
            for r in cand:
                d = json.loads((RAW / f"{r['run_id']}_bench.json").read_text())
                tt, il = d.get("ttfts") or [], d.get("itls") or []
                ok = 0
                for i, t in enumerate(tt):
                    per = il[i] if i < len(il) else []
                    tpot = (sum(per) / len(per) * 1000) if per else 1e9
                    if t * 1000 <= 891 * s and tpot <= 50 * s:
                        ok += 1
                gp = ok / d["duration"] if d.get("duration") else 0
                best = max(best, gp)
            ys.append(best)
```

角色：把"结论会不会随阈值翻掉"这个质疑变成一张可看的图。**关键行为什么这么写**： ① 它**不读 derived 表，直接回读每点的 `_bench.json` 原始逐请求数组**——按新阈值重判需要逐请求数据，聚合值不够；这是"表图必须能从 raw 重算"的实例；② 两条 SLO **同倍率同时缩放**（`891 * s` 与 `50 * s`），只动一条会把"TTFT 敏感"与"TPOT 敏感"混淆； ③ 每个倍率取该臂**所有 sweep 档位里的最大 goodput**，即各臂在每个倍率下重新选自己的最优 offered load——**不允许用同一档位比不同阈值**，否则曲线形状被档位选择污染； ④ `1e9` 兜底与段 5 的 `inf` 同一取向。**这张图和 DistServe 的 SLO Scale 是同一件事**（§3.7 第 4 点）：四臂排序在 0.5–4× 全区间稳定，所以**排序**结论不依赖阈值， 但 goodput **绝对值**必须连阈值一起引用。

**段 10：收尾——after 快照与落账（`run_point.sh:54-67`）**

```bash
kill "$SAMPLER" 2>/dev/null || true

for p in "${ENGINE_PORTS[@]}"; do
  scripts/metrics_snapshot.sh snap "$p" "$R/snapshots/${PREFIX}_${p}_after.prom"
done

GPU_COUNT=${GPU_COUNT:-$([ "$ARM" = colocate ] && echo 1 || echo 2)}
"$VENV/bin/python" scripts/collect_point.py \
  --prefix "$PREFIX" --arm "$ARM" --mode "$MODE" \
  --input-len "$IN" --output-len "$OUT" --rps "$RPS" \
  --gpu-count "$GPU_COUNT" --engine-ports "${ENGINE_PORTS[@]}" \
  --gpu-csv "$GPUCSV" --seed "${SEED:-42}" \
  ${PROV_ENV_LABEL:+--env-label "$PROV_ENV_LABEL"} ${PROV_SHA_OVR:+--sha "$PROV_SHA_OVR"} \
  ${SLO_TTFT_MS:+--slo-ttft-ms "$SLO_TTFT_MS"} ${SLO_TPOT_MS:+--slo-tpot-ms "$SLO_TPOT_MS"}
```

角色：关采样器、抓 after 快照、把这一点的所有证据合成一行 JSONL。**关键行为什么这么写**： ① `GPU_COUNT` 默认值由臂名推出（colocate 1，其余 2）——**per-GPU 成本口径（fig3）的分母就来自这里**，写错直接改变成本结论；可被环境变量覆盖，给"同臂名不同卡数"留口子； ② `${VAR:+--flag "$VAR"}` 的含义是"变量非空才加整个选项"——**没设 SLO 时根本不传 `--slo-*`，段 5 的 goodput 保持 `None`**；写成 `"${SLO_TTFT_MS:-0}"` 会把"没设阈值" 变成"阈值为 0"，所有请求判不达标而看不出是配置缺失；③ `kill "$SAMPLER"`(pd_disagg/scripts/run_point.sh:54)写在 after 快照**之前**，否则统计会读到正在写入的不完整行；④ `$PREFIX` 从段 1 一路传到这里， **bench json、log、gpu csv、before/after 快照五类文件共用同一前缀**，这是事后把一行 JSONL 反查回全部原始文件的唯一线索。

## 5. 实验数据怎么读

### 5.1 硬件三数原始文件怎么读

`hw/p2p_bandwidth_latency.txt` 节选（EXP-002）:

```
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 923.46   0.60
     1   0.69 925.10
Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 921.83  22.62
     1  22.77 926.75
```

读法：对角线是卡内 memcpy（~924 GB/s，decode roofline 的分母）；非对角线才是跨卡。 "P2P=Enabled" 矩阵在本机与 Disabled 几乎相同——使能请求被驱动拒绝，回退同一条中转路径， 这本身就是 P2P 禁用的证据之一（连同 CANNOT Access Peer 行与 topo 的 GNS）。注意文件头的 NOTE：CUDA sample 不是精密基准，单向矩阵各单元分散（0.60/0.69/0.91/4.36），所以仓内引用一律用区间 0.60–0.91，不挑单值（EXP-002 §7）。

`hw/all_reduce_perf.txt` 读法：看 busbw 列而非 algbw——busbw 是按 allreduce 通信量归一的口径（2 卡时两者相等），1 MB 到 512 MB 消息稳定在 1.67–1.87 GB/s，avg 1.78。 **平坦的带宽曲线**说明 SHM 回退路径没有大消息红利，这预言了 TP2 prefill 的大消息 allreduce 无处可逃（§3.3）。

#### 5.1.1 "平坦"这件事本身携带信息(本讲义推导)

按公理 B，有效带宽 $m/(\alpha+m/\beta)$ **必然**随 $m$ 单调上升趋近 $\beta$。 **从 1 MB 就已平坦，说明 1 MB 远大于 $m_{1/2}=\alpha\beta$**。反解：若 $\beta\approx1.85$ GB/s 而 1 MB 处已达 1.67 GB/s（$\beta$ 的 90%），则 $\alpha\le\frac{1}{9}\cdot\frac{1\,\mathrm{MB}}{1.85\,\mathrm{GB/s}}\approx60\,\mu s$——与 §3.3.2 从 decode allreduce 反推的 0.65–1.3 ms **相差一个数量级**。

**这个矛盾必须解释。** 两个候选：(a) nccl-tests 测的是**稳态循环**（同一 buffer 反复 allreduce，连接与缓冲区都已热），而 vLLM 每步 allreduce 夹在一长串 kernel 之间， 要付同步与调度的钱；(b) 1.3 ms 是从 TPOT 差值反推的，可能混入 TP 切分本身的其它开销（更小的 GEMM 形状、额外的切分/拼接算子）。**本仓没有 tp2 的 kernel 级 trace， 无法裁决**——如实登记。方法论提醒：**微基准的 $\alpha$ 是下限，真实系统里总是更大。**

#### 5.1.2 三个文件各自排除了什么

`topo.txt`(GNS)排除"是拓扑问题，换插槽/换根桥能修"；`p2p_bandwidth_latency.txt` 的 Enabled/Disabled 两张矩阵几乎相同，排除"P2P 其实能用，只是没启用"； `all_reduce_perf.txt` 的平坦曲线排除"大消息会好起来"。**三个合起来才够**： 单看 topo 会被质疑"工具在容器里不准"，单看带宽矩阵会被质疑"CUDA sample 不是精密基准"（文件头 NOTE 自己写了），单看 allreduce 会被质疑"NCCL 没调参"。三条互相独立、指向同一结论，才构成"P2P 受限"的证据基础。

### 5.2 84 个测量点是怎么设计的

网格（EXP-007 §2，协议 v2 共 84 个通过 gate 的测量点，`results/b1_matrix/runs.jsonl`， 汇总表 `derived/sweep_summary.csv`）：

- **三个输入桶** 512/2048/8192（输出统一 128）：短请求（调度开销敏感）、中等（混合）、长上下文（prefill/传输压力），一桶一个故事，不混算。
- **每臂每桶**：attribution（并发 1）+ saturation（SAT_CONC=64 探顶）+ sweep 4 个公共档 {0.5, 0.75, 0.9, 1.05}×colocate 饱和（512：{5.2,7.8,9.3,10.9} 等）+ 若干贴近自身饱和的档位。公共档保证四臂在**同一 offered load** 下可比；自身档保证每臂的拐点都被覆盖（否则 replica2 的高容量段全是外推）。
- **num_prompts 按桶递减**(320/192/56)：保证每点时长量级相当，长桶不至于跑一小时。
- **每点唯一 seed，跨臂同点位同 seed**：唯一 seed 防前缀缓存污染——bench 的随机数据集同 seed 跨运行 prompt 逐 token 相同、且短桶 prompt 是长桶的精确前缀，实测造成 2048 桶 25% 缓存命中、8192 桶 8.6%(`analysis/nixl_token_accounting.md`)；跨臂同 seed 则保证臂间对比的工作负载逐字节相同。这一对设计合起来就是"协议 v2"。
- **防坑设计还有**：colocate 单卡基线作为所有双卡臂的"反例臂"（任何双卡方案先回答 "比一张卡好多少"）；失败点保留（gate_pass=false 行不删、不进图表）；~3% 的客户端瞬断同 seed 重跑（EXP-007 §4）。

#### 5.2.1 为什么"每点唯一 seed"比"全局固定 seed"更可复现

直觉上固定 seed 才叫可复现，这里恰好反过来，原因是**被测系统有状态**：vLLM 的 prefix cache 跨请求、跨运行持久（只要引擎不重启），于是"同一 seed 生成同一 prompt" 在有缓存的系统上变成"第二次跑的工作量比第一次少"。仓内实测把它量化了：2048 桶 25% 命中、8192 桶 8.6%(`analysis/nixl_token_accounting.md`)。**25% 命中意味着 prefill 工作量凭空少四分之一、TTFT 虚低，而 bench 报告里没有任何字段会告诉你这件事。**

正确的可复现性定义是：**记录每次运行的完整输入（含 seed），而不是让每次输入相同**。协议 v2 把 seed 按 `x042 / x099 / x001..x006`（x=桶号）编码进文件名，任何一点都能精确重放而点间不共享 prompt。**跨臂同 seed 是另一件事**，保证的是臂间可比——两条设计目标不同（防污染 / 保可比），合起来才是完整协议。

#### 5.2.2 档位怎么定:公共档 + 自身档

只用公共档会出问题：公共档按 **colocate 的饱和值**定，而 replica2 容量接近两倍， 于是它在公共档上全在低负载区，拐点根本没采到，峰值 goodput 只能外推。只用自身档也不行：四臂 offered load 不同，横向对比无从谈起。**两套档位并存是唯一解**，代价是点数翻倍——84 个点就是这么来的。**num_prompts 按桶递减（320/192/56）** 是为了让每点墙钟时长量级相当；代价是长桶样本少——8192 桶的 p99 只有 56 个样本支撑，引用时必须记住这条限制（§6 边界）。

### 5.3 三张主表逐行读

**饱和吞吐（req/s，EXP-007 §5）**：

| 桶 | colocate（1 卡） | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 10.36 | 15.58* | 12.31 | 7.84 |
| 2048 | 3.63 | 7.00 | 4.16 | 2.12 |
| 8192 | 0.90 | 1.78 | 1.02 | 0.54 |

读法：先横比（同桶四臂），再算比值：replica2/colocate = 1.50*/1.93/1.98；tp2/colocate = 1.19/1.15/1.13（即 +13~19%）；pd1p1d/colocate = 0.76/0.58/0.60——**PD 用两张卡跑不过一张卡**。带 * 的 512 桶 replica2 有欠饱和疑点（EXP-007 §7），所以"近线性 2×"只引 2K/8K 的 1.93/1.98×，这是限定语，不许丢。

**goodput 峰值（rps@档位，EXP-007 §5）**：colocate 8.57@9.3 / 2.41@2.7 / 0.43@0.68； replica2 12.75@14 / 4.96@5.3 / 0.90@0.95；tp2 10.18@11.8 / 2.51@2.7 / 0.60@0.81； pd1p1d 1.59@5.2 / 0.16@1.8 / 0.11@0.45。读法：goodput 峰值总小于饱和吞吐（饱和点上延迟已炸，SLO 达标率崩塌）；pd 的 1.59@5.2 意味着在 66% 饱和度时 goodput 已经只剩零头——传输延迟（512 桶 ~114 ms 起步）直接吃掉 328 ms SLO 的三分之一。

**v2 归因 TTFT（并发 1、同热工况，p50 ms，EXP-007 §5）**：

| 桶 | colocate | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 65.4 | 66.2 | 64.0 | 219.3 |
| 2048 | 224.9 | 220.7 | 219.8 | 718.6 |
| 8192 | 925.2 | 902.6 | 881.3 | 2718.7 |

读法：非 PD 三臂几乎无差异（prefill 无并行收益，§3.3）；pd 的溢价 154/494/1793 ms 全部来自 KV 通路——它的因果拆解（54.2/62.5/64.2%，512/2K/8K，p50，request 级因果占比）在讲义 02。图表版：`pd_disagg/figures/fig7_saturation_overview.png`（总览）、 `fig1_goodput_curves.png`（曲线族，x=offered load，y=goodput，虚实线区分臂）——看曲线族先看**过峰后的下降段**，那是各臂的失效方式：replica2 缓降，pd 贴地。

#### 5.3.1 第四张表:per-GPU 成本口径

前三张表都是"系统能干多少"，这一张是"每张卡值多少"。峰值 goodput ÷ 卡数（REPORT §2.3、fig3）：

| 桶 | colocate（1 卡） | replica2（2 卡） | tp2（2 卡） | pd1p1d（2 卡） |
|---|---|---|---|---|
| 512 | **8.57** | 6.37* | 5.09 | 0.80 |
| 2048 | 2.41 | **2.48** | 1.25 | 0.08 |
| 8192 | 0.43 | **0.45** | 0.30 | 0.05 |

**读法完全反转**：按系统吞吐 replica2 全胜，按单卡成本 colocate 在 512 桶领先、在 2K/8K 与 replica2 打平。**这不矛盾，是两个问题**："我有两张卡怎么用最好"（答：replica2）与"我该不该买第二张卡"（答：2K/8K 上是等价交换，512 桶上是负收益）。 **任何多卡方案的报告都必须同时给这两个口径**，只给前者是在用卡数换掌声。 DistServe 把成本口径直接写进指标定义（"per-GPU goodput ... for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query"，§1）， 本仓 fig3 与之同构。

#### 5.3.2 三张表的横向一致性检查(本文现算)

好数据要能互相咬合。三处交叉验证：

1. **goodput 峰值 < 饱和吞吐**，四臂三桶 12 组全部成立（如 colocate 512:8.57 < 10.36）。若某组反过来，说明 SLO 阈值宽到形同虚设。
2. **pd1p1d 的 goodput 峰值档位远低于其饱和值**：512 桶峰值在 5.2 rps 而饱和 7.84， 即 66% 饱和度；其余三臂的峰值档位都在饱和值的 90%–110%。**这个"峰值位置提前" 本身就是 SLO 被延迟吃掉的指纹**，不看曲线也能读出来。
3. **Little 定律 $L=\lambda W$ 自洽**。8192 桶 colocate： $0.90\times(0.925+128\times0.0163)\approx2.7$，即饱和时系统里约 2.7 个请求在跑； pd1p1d：$0.54\times(2.719+128\times0.0164)\approx2.6$。**两者接近**说明 PD 的容量损失不是并发度低，而是每请求占用时间长——与 §3.5.3 的"传输站是瓶颈"一致。

### 5.4 功率帽探案:隐藏变量是怎么被抓住的

这是本仓方法论含金量最高的一段（EXP-004 §7、EXP-005，日记 §9），完整链条：

1. **异常出现**：replica2@8K 归因 TTFT 714.6 ms，比 colocate 的 925.2 快 30%——并发 1 下双副本不该有任何优势，这违反 §2 的资源账。
2. **拆分布而不是信均值**：colocate 8K 的 32 个请求 TTFT 呈**双段**：前 ~8 个 702–739 ms， 之后 897–951 ms；replica2 全部 697–733 ms。→ 假设改写为"单卡持续负载下发生了劣化"。
3. **对照排除**：diag-1/2 直连两张卡分别测（901.2 / 892.7 ms）——排除"卡间个体差异"； 两卡都到 ~900，说明劣化与卡无关、与**持续负载**有关。
4. **遥测定案**：diag-3 持续 prefill 负载 + 1.5 s × 40 轮采样：温度 40→63°C、功率 427–443 W（帽 450 W）、SM 频率 2820 ↔ 2460–2535 MHz、节流原因 **0x4 = SW Power Cap** (`records/data/EXP-005_throttle_trace.csv`)。63°C 排除热节流，定案功率帽。
5. **机理闭合**：replica2 轮询把请求对半分，每卡 50% 占空比，间歇期回 boost——所以它 "快"；colocate 的 925 是冷态（~700）与稳态（~905）的混合。降频约 12%(2820→2475 MHz) 对应 TTFT +30%(700→905 ms)，不完全成比例，残差疑与瞬时 boost/显存时钟相关—— EXP-005 §6 如实写"未深究，非主线"，这也是可学的：异常解释到能定方法论决策就停， 不为完整感编故事。
6. **方法论落地**：headline 一律以 sweep 为准（满负载下各臂同为持续态，公平）； run_point.sh 加装遥测（§4 段 1）；归因表引用必须带工况标注。

防坑清单：消费卡基准必须声明功率工况；对比实验必须同热工况；冷启数字与稳态数字分开报。

#### 5.4.1 节流位的语义:为什么 0x4 能定案而 63°C 能排除

NVML 把节流原因定义成一个位掩码，每一位一个独立原因（NVML API Reference，`nvmlClocksThrottleReasons` 组）：

| 位 | 常量 | 官方描述 |
|---|---|---|
| 0x1 | GpuIdle | "Nothing is running on the GPU and the clocks are dropping to Idle state" |
| 0x2 | ApplicationsClocksSetting | "GPU clocks are limited by current setting of applications clocks" |
| 0x4 | SwPowerCap | "SW Power Scaling algorithm is reducing the clocks below requested clocks" |
| 0x8 | HwSlowdown | "HW Slowdown (reducing the core clocks by a factor of 2 or more) is engaged" |
| 0x20 | SwThermalSlowdown | 温度超出工作范围导致的降频 |
| 0x40 | HwThermalSlowdown | "HW Thermal Slowdown ... is engaged" |
| 0x80 | HwPowerBrakeSlowdown | "HW Power Brake Slowdown ... is engaged" |

**定案逻辑因此是排他性的而非相关性的**：观测到 0x4 且**没有**观测到 0x20/0x40， 就直接读出"是功率帽，不是温度"。63 °C 只是旁证，**真正的证据是位掩码里缺席的那两位**——这比"温度不高所以不是热问题"强得多，后者需要知道阈值，前者不需要。这也解释了段 6 为什么保留原始位掩码集合：压成"是否节流"就丢掉了 0x4 与 0x40 的区别，归因链断掉。

#### 5.4.2 从 40 个采样点里还能读出什么(本文现算)

`records/data/EXP-005_throttle_trace.csv` 共 80 行（双卡交错），GPU0（负载卡）40 行。按节流原因位分组重算：

| 分组 | 样本数 | SM 时钟范围 | SM 时钟均值 | 温度范围 |
|---|---|---|---|---|
| 0x4(SW Power Cap) | 23 | 2445–2805 MHz | 2533.7 MHz | 47–64 °C |
| 0x0（无节流，负载中） | 5 | 2460–2820 MHz | 2745.0 MHz | 40–57 °C |
| 0x1（空闲） | 12 | 210 MHz | 210 MHz | 28–30 °C |

三条读法：

1. **40 个样本里 23 个带 0x4**——这段持续 prefill 负载下**超过一半的采样时刻处于功率帽降频状态**，这个比例本身就是"稳态 = 降频态"的直接支撑。
2. **两个负载分组的时钟均值差 211 MHz(2745.0 → 2533.7，−7.7%)**。这与仓内引用的 "降频约 12%"是**两个不同口径**：仓内比的是峰值 boost 2820 与节流带 2475（包络之比）， 本文比的是两组样本的均值。包络之比总是大于均值之比，两个数字都对，引用必须说清是哪一种——**这是本篇第三次遇到"同一现象两个口径"**（前两次是 busbw/algbw 与 GiB/GB）。
3. **0x0 分组里出现 2460 MHz 的低值、0x4 分组里出现 2805 MHz 的高值**，说明 2 s 采样跨越了调频动作，单样本的"原因位"与"当时时钟"未必严格对应。**这是采样率不足的直接证据**，也是本仓只用这份数据做定性定案、不做定量建模的原因。

#### 5.4.3 白皮书的 450 W 与实测的 444.65 W

Ada 白皮书 Appendix A Table 2 给 RTX 4090 的 TGP(Total Graphics Power)为 **450 W**。实测负载态峰值 444.65 W（本文现算），即**帽的 98.8%**。这条对照有两重意义：

1. 它确认了 450 W 就是本机生效的功率上限（未被 vBIOS 或 `--power-limit` 改动）；
2. 它说明**这张卡在持续 prefill 下是被功率而非温度限制的**——温度上限 4090 出厂设定远高于 64 °C，而功率已经贴到帽上。

**由此得到一条可操作的建议（推断，本仓未验证）**：给 GPU 设一个略低的功率上限（例如 400 W）换取更稳定的时钟，可能让 TTFT 的**方差**下降，代价是峰值性能。本仓没做这个实验，不主张收益数字，只指出它是可测的方向。

### 5.5 自己动手复算:三个可以从 raw 重来的量

**（一）per-GPU goodput**：从 `results/b1_matrix/derived/sweep_summary.csv` 取 `mode == sweep` 行，按（arm， input_len） 分组取 `goodput_rps` 最大者除以 `gpu_count`， 即得 §5.3.1 那张表（本文现算，与 REPORT §2.3 逐位一致）。**这是检验"图有没有被手改" 最快的方法**——图里每个柱子都该能从 CSV 两步算出来。

**（二）PD 各桶的传输容量上界**：取 EXP-006 的 MB/xfer 实测值（29.4 / 88.1 / 439.7） 除以 0.27 GB/s 再取倒数，即 §3.5.3 那张表。**注意 2048 桶的 88.1 MB 与 "2048 × 57344 = 117 MB" 不等**——差的是前缀缓存裁剪（EXP-006 §7 的开放问题，已在 `analysis/nixl_token_accounting.md` 逐块定位）。§3.5.3 用的是 117 MB（协议 v2 无污染口径），那一行偏保守。**两个口径都在案，用哪个必须说。**

**（三）Little 定律自洽性**：饱和点的（吞吐 × e2e 延迟） 应约等于实际并发度（§5.3.2 第 3 条）。若乘积远大于 SAT_CONC，说明测出的"饱和"其实是客户端并发上限——replica2@512 的疑点正是这么被识别的。

### 5.6 图怎么读:六张图的分工

| 图 | 问什么 | 先看哪里 |
|---|---|---|
| fig1 goodput 曲线族 | 各臂在不同 offered load 下的服务能力 | **过峰后的下降段**，那是失效方式：缓降 vs 贴地 |
| fig2 TTFT p99 | 尾延迟随负载的恶化 | 曲线开始上翘的位置（拐点），而非绝对值 |
| fig3 per-GPU goodput | 成本口径 | colocate 柱与双卡柱的高低反转 |
| fig4 PD TTFT 分解 | PD 的溢价花在哪 | 最底段的括号（它是跨臂替代，见讲义 02 §5.1） |
| fig5 NIXL 传输标度 | 有效吞吐是否跨尺寸恒定 | 大传输点是否贴斜渐近线、小传输点是否贴延迟地板 |
| fig6 SLO 敏感性 | 结论依不依赖阈值 | 四条线的**相对次序**是否在全区间保持 |
| fig7 饱和总览 | 一页看完四臂三桶 | 同桶四柱的比值，而非绝对高度 |

**通用读法**：先读**形状**，再读**次序**，最后才读**数值**。形状回答机理（是不是带宽墙、是不是排队爆炸），次序回答选型，数值只在带全定语时才可引用。

## 6. 误区与边界

1. **"多卡跑 TP 理所当然更快。"** 本机 tp2 的 prefill 零加速（§3.3）、吞吐 +13~19%、 per-GPU goodput 为四臂最差之一（REPORT §2.3 成本口径）。TP2 只在两个场景成立：模型单卡放不下（被迫），或要压 TPOT 到单卡达不到的水平（9.3 vs 16 ms）且愿付吞吐代价。
2. **"replica2@8K 快 30%，双副本有隐藏加速。"** 仓内被证伪的假设原案（§5.4）：真相是功率帽让单卡基线变慢了。教训：**对比实验里"变快"与"对照变慢"不可区分，除非工况入账**。
3. **"benchmark 固定 seed 才科学。"** 仓内被证伪的第二案：固定 seed 让 vLLM 前缀缓存跨运行命中（2048 桶 25%），prefill 工作量凭空少四分之一，TTFT 虚低。可复现性靠 "每点唯一且记录在案的 seed"达成，不靠全局同一个 seed。
4. **"PD 分离是大厂标配，至少不会更差。"** 它的价值主张（消除 prefill 对 decode 干扰、 P/D 独立扩缩）在高速互联集群成立；本机 0.27 GB/s KV 通路下全负载段溃败，且换传输方向（push）只挽回 6.7% TTFT@8K(EXP-011)——形态与互联能力错配，不是形态本身错（REPORT §2.4 的公平陈述）。
5. **"饱和吞吐高就是好。"** tp2 512 桶饱和 12.31 高于 colocate 的 10.36，但两者 goodput 峰值 10.18 vs 8.57 的差距被两张卡的成本除回去就是负收益；服务质量要用带 SLO 的 goodput 与 per-GPU 口径双重核算。
6. **"三个互联数字里挑一个代表这台机器。"** 三条路径三个数字，差两个数量级（§3.1.5）。用 22.7 GB/s 谈 KV 传输、用 0.27 GB/s 谈 allreduce，都是把结论推到错误的方向。**引用互联数字必须同时给出路径、消息大小与软件栈。**
7. **"下界算式算出来多少就是多少。"** 本篇出现了两次"实测快过下界"的情况（§3.2.4 的 GiB/GB 混用、§3.3.4 的规格 boost 当上限）。**下界被击穿时， 错的一定是算式的口径，不是测量**——这是一条可以当断言用的规律。
8. **"L2 有 72 MB，权重能缓存住一部分。"** 72 MiB： 14.2 GiB ≈ 1:197。即使完美利用，也只能省下 0.5% 的权重读取。**片上缓存对 decode 权重流没有意义** (§3.2.1)，它的用武之地在 GEMM 的 tile 复用与 MoE 的专家权重复用（讲义 02 §5.3）。
9. **"goodput 就是带 SLO 的吞吐，定义没什么可讲的。"** 至少四个旋钮（哪两条 SLO、达标率门槛、分母是系统还是卡、阈值绝对值怎么定，§3.7）。本仓与 DistServe 的定义在第 2、3 条上不同，数字因此不可直接比。
10. **"PD 的传输可以和 prefill 重叠，所以本机的结论会被优化掉。"** 重叠的上限就是 prefill 时长。8K 桶传输 1.63 s、prefill 0.9 s， **完美重叠后仍剩 0.73 s 露在外面**(§3.5.4)。互联差两个数量级时， 调度优化只能改常数。
11. **"attribution 表就是各臂的真实延迟。"** attribution 各臂的占空比不同（replica2 每卡 50%，colocate 100%），因此工况不同。**headline 一律以 sweep 为准**（§5.4 第 6 点），引用 attribution 必须带工况标注。
12. **"84 个点足够多，结论已经很稳。"** 点多不等于每点稳：sweep 每点单次， 跨会话重复只在 MoE 线做过（那里暴露出 ±5~8% 漂移）。**点数保证的是曲线形状， 不是单点精度**（§7 压力问 10）。

**适用边界**：单机 2×RTX 4090、P2P 驱动禁用、Qwen2-7B BF16、随机负载三桶、输出 128、 1P1D（非 xPyD）。NVLink/数据中心平台三条互联数字全变，结论不可直接外推——可外推的是 **方法**：先测互联三数，再按 §2 资源账预测形态排序，最后矩阵实测验证。

**具体不覆盖什么**（逐条列，避免读者过度外推）：
- **xPyD**：P：D 比例可调正是 PD 分离的核心卖点之一（DistServe §3.2），1P1D 是它最退化的形态，本仓的结论不能当作对 xPyD 的评价；
- **分块 prefill(chunked prefill)**：Sarathi-Serve 的 stall-free 调度（arXiv：2403.02310）在**不做 PD 分离**的前提下缓解同一个干扰问题，零跨卡成本。本仓四臂里没有这一臂——**这是本仓设计的一个真实缺口**，见 §8.2；
- **量化**：W8A8/W4A16 会改变 $W$ 与 $s$，从而改变 §3.2 的全部数字；
- **长输出**：输出统一 128，decode 段占比固定；输出更长时 TPOT 的权重会上升， 四臂排序可能变化（tp2 的 -42% TPOT 收益会更值钱）；
- **真实流量分布**：随机数据集的定长 prompt 消除了轮询不均（§3.4.2），重尾分布下 replica2 的扩展效率会下降；
- **多轮会话与前缀共享**：本仓刻意消除前缀命中，而生产环境正相反——前缀缓存感知路由能让 replica2 更强（§3.4.3）。

## 7. 连环追问

1. **Q：TP2 的 TPOT 为什么是 9.3 ms 而不是 16.35/2 ≈ 8.2 ms？** A：decode 每 token 还要付一次小消息 allreduce，实测代价 ~1.3 ms/token(EXP-005 §6)； 7.7（半权重下限）+1.3 ≈ 9.0，实测 9.26–9.48 ms，账闭合。
2. **Q：1.78 GB/s 是怎么测的？** A：nccl-tests `all_reduce_perf -b 1M -e 512M -f 2 -g 2`，NCCL 探测不到 P2P 回退 SHM 传输，取 avg busbw(`hw/all_reduce_perf.txt`，EXP-002)。它只代表 collective 路径，不能代表 KV 通路（那是 NIXL 的 0.26–0.27 GB/s，telemetry-derived）。
3. **Q：单向 0.6 GB/s 与双向 22.7 GB/s 差 25 倍怎么解释？** A：无 P2P 时 cudaMemcpyPeer 走经主机的分段中转，单向暴露全部中转开销；双向两个方向的分段互相流水，逼近 Gen4 x16 双向极限。这个 25 倍差本身就是"无 P2P"的指纹（EXP-002 §6）。
4. **Q：goodput 为什么逐请求判而不用 p99 卡线？** A：要画连续曲线必须数出每个点的达标请求数（collect_point.py：106-116）；p99 卡线只给布尔结果，且会把"51% 请求超时"与"1% 超时"判成同一种失败。
5. **Q：为什么 colocate 是"反例臂"？** A：所有双卡形态必须先回答"比一张卡好多少"——pd1p1d 三桶全输给单卡（0.58–0.76×）， 没有单卡臂这一事实根本暴露不出来。
6. **Q：8K 桶四臂 prefill 几乎无差异（925/903/881），为什么？** A：同热工况下功率帽把所有持续 prefill 的臂整平到同一频率（EXP-007 §6）；TP2 的计算减半又被 allreduce 吃掉。物理约束一致，软件形态就分不出高下。
7. **Q：PD 的饱和 0.54 req/s 怎么从第一性原理预测？** A：每请求要传 ~470 MB KV，通路 0.27 GB/s，上限 0.27/0.47 ≈ 0.57 req/s，实测 0.54 (EXP-007 §6)。当传输是关键路径，容量=带宽/单请求传输量。
8. **Q：replica2 需要什么额外组件，它会成为瓶颈吗？** A：一个轮询代理（rr_proxy.py）。512 桶饱和 15.58 存在代理/客户端并发上限疑点（EXP-007 §7 如实登记），这正是"零通信"形态把瓶颈推到入口层的表现；2K/8K 未见此效应。
9. **Q：57,344 B/token 里哪一项最该被质疑？** A：`num_key_value_heads = 4`——它不是 `num_attention_heads` 的别名，GQA 下两者相差 7 倍（§3.5.1），按 28 算会把 KV 账放大 7 倍。第二该质疑的是 dtype（bf16 2 B， 开 FP8 KV cache 则减半）。**两处都在 config 里可查，不许估。**
10. **Q：为什么 16 KiB 的 descriptor 会把有效吞吐钉在 0.27 GB/s？** A：按公理 B，$\alpha$ 主导时有效带宽 $\approx m/\alpha$；实测每 descriptor 约 62 µs（讲义 02 §3.6），$16{,}384/62\,\mu s\approx0.26$ GB/s，与 telemetry 反解一致。 **要提速必须增大 $m$（合并 descriptor），不是提高 $\beta$。**
11. **Q：为什么 attribution 下 PD 的 TPOT 与 colocate 一样（15.9–16.4 ms）？** A：KV 传输只发生在第一个 token 之前，传完后 D 端就是普通的单卡 decode，权重带宽下界一模一样。**PD 分离改变 TTFT 的构成，不改变 TPOT 的物理下界**——所以它的收益只能来自"消除干扰"，而并发 1 时干扰为零（讲义 02 §2）。
12. **Q：tp2 用 `--gpu-memory-utilization 0.88` 而其余臂用默认，算不算不公平？** A：算，而且已登记（EXP-007 §7:0.9 在 warmup 阶段 OOM，需 150 MB 仅剩 124 MB）。影响面是 KV 预算；但四臂的 KV 预算本来就不同（tp2 每卡省一半权重，KV 空间反而更大）， 成本口径不受影响。**如实登记、说明影响面、不粉饰。**
13. **Q：为什么不把 chunked prefill 也做成一臂？** A：诚实答：这是本仓设计的缺口。Sarathi-Serve(arXiv：2403.02310)的 stall-free 调度在单实例内缓解 prefill 对 decode 的干扰、**零跨卡成本**，在互联受限平台上很可能是 PD 分离的正确替代品。本仓 colocate 臂跑在 vLLM 默认调度上，**没做开/关对照**， 不主张任何相关数字（§8.2）。
14. **Q：三条互联路径为什么不能互相校准，比如用 22.7 GB/s 校准 NIXL？** A：它们的 $\alpha$ 与 $m$ 都不同（§3.1.5）。裸拷贝用大块连续内存（$m$ 大、 $\alpha$ 摊薄），NIXL 用 16 KiB descriptor（$m$ 小、$\alpha$ 主导）。 **把大消息的 $\beta$ 拿去预测小消息的吞吐，是公理 B 明确禁止的操作。**
15. **Q：功率帽在数据中心卡上还存在吗？** A：机制存在（A100/H100 同样有 SW Power Cap 位），但散热与供电余量更充裕、部署时通常锁频。**本仓没有数据中心卡可测，不外推**；可外推的是"任何 GPU 基准都应该采工况"这条方法。
16. **Q（压力）：SLO 是拿被前缀缓存污染的基线锁定的，你的 goodput 结论会不会整个翻掉？** A：承认瑕疵：2048 桶 SLO 名义 5×，按 v2 干净基线实为 3.96×(891/225，EXP-007 §7)。防御是敏感性检验：阈值扫 0.5–4×，四臂 goodput **排序**全程稳定（fig6，REPORT 附录 A）。所以"replica2 最优、pd 溃败"的排序结论稳健，任何 goodput **绝对值**都必须连同阈值一起引用——这是口径边界，不辩解。
17. **Q（压力）：84 个点看起来多，但每点只有一次 bench，多轮呢？** A：诚实回答：sweep 每点单次（n=32~320 请求内含分布，报 p50/p99），跨点趋势由 12–18 点的曲线形状互相约束；瞬断点做过同 seed 重跑（3/~90 次）。但"同点跨会话重复"只在 MoE 线做过（那里暴露出 ±5~8% 会话漂移，EXP-015《D2 MoE config 调优》）。若要引用单点绝对值到 ±5% 精度， 应按仓内规范补 3 轮取 mean/std；引用排序与量级结论则现有数据足够。
18. **Q（压力）：decode 下界算式里 14.2 GB 除以 924 GB/s，这两个数的口径配套吗？** A：不配套，而且我把它算清楚了（§3.2.4）：14.2 是 GiB（真实值 14.185 GiB = 15.231 GB），924 是读写混合的 memcpy 速率而非纯读速率。两处偏差方向相反、量级相近（−6.6% 与 −8.3%），所以商基本没错——**但这是巧合**。口径对齐后的版本是 15.231 GB ÷ 1008 GB/s = 15.11 ms，实测 15.87 ms，达成率 95.2%，结论同向。仓内既有表述不改，引用时补口径说明。
19. **Q（压力）：你说 tp2 的 prefill 零加速是"结构性的"，但串行通信估计（888 ms） 和实测（693.7 ms）对不上，凭什么说结构性？** A：分两层答。**能确证的**：实测 tp2 8K TTFT 693.7 ms vs 单卡冷态 ~700 ms， 这是零加速的**直接观测**，不依赖任何估计。**只能给量级的**：串行通信估计 888 ms 说明"通信代价与计算减半的收益同量级"，但真实执行有重叠，所以该式只做量级判断（§3.3.3 已显式声明）。**我没有 tp2 的 kernel 级 trace**， 无法给出重叠比例——如实登记为开放问题，不用估算冒充测量。
20. **Q（压力）：DistServe 报了 7.4× 更高请求率，你这里 PD 连单卡都打不过， 是不是实现有问题？** A：先看前提。DistServe §3.3 明说：OPT-66B 单个 512-token 请求的 KV 约 1.13 GB， 10 rps 就需要 11.3 GB/s（约 90 Gbps）"to render the overhead invisible"， 并指出集群配 InfiniBand(800 Gbps)或退而用节点内 NVLink（A100 间 600 GB/s）。 **本机 KV 通路 0.27 GB/s，比它的"可忽略"门槛低约 42 倍，比 NVLink 低约 2200 倍。** 在这个前提下 PD 溃败是**预期内**的，不是实现缺陷。要证伪"实现有问题"， 需要指出一条本机可达而本仓没走的更快路径——EXP-011 已排除"换方向"这一条。
21. **Q（压力）：你怎么知道 0.27 GB/s 不是没调 NIXL/UCX 参数造成的？** A：不能完全排除，这是本仓的一个真实边界。**能说的**：descriptor 大小 16 KiB 由 connector 的 region 划分决定（§3.1.4），每 descriptor 耗时 61.5–65.8 µs 跨三个桶几乎恒定（讲义 02 §3.6），该恒定性指向固定开销主导而非参数未调。 **不能说的**：本仓没有做 UCX 传输参数扫描，也没有做 descriptor 合并的改造实验， 因此不主张"0.27 GB/s 是这条链路的上限"，只主张"这套软件栈在这种访问模式下的有效速率是 0.27 GB/s"（措辞：telemetry-derived effective throughput）。

## 8. 工业对照与延伸

### 8.1 论文/文档怎么说 vs 本项目实测:逐条对照

本节把"论文或官方文档说了什么"与"本仓在 2×RTX 4090 上测到什么"并排放，并诚实分析差异来源。差异不粉饰：多数来自互联代际与规模，少数来自本仓的口径或设计简化。

| # | 来源与声称 | 本仓实测（EXP 锚） | 差异分析 |
|---|---|---|---|
| 1 | **DistServe §3.3**：OPT-66B 512-token 请求 KV ≈ 1.13 GB；10 rps 需 11.3 GB/s(≈90 Gbps)才能让传输开销"invisible"；集群配 InfiniBand 800 Gbps，退而用节点内 NVLink（A100 间 600 GB/s） | KV 通路 **0.26–0.27 GB/s**(EXP-006/007)；8K 请求 469.8 MB，传输 1.63 s | **不是冲突，是前提不成立**。本机比论文的"可忽略门槛"低约 42×，比 NVLink 低约 2200×。论文自己把带宽列为**前提条件**，本仓测的正是前提被破坏后的样子 |
| 2 | **DistServe §3.2**：模型放得进单卡时，"replication is a competitive option ... to linearly scale the system's rate capacity" | replica2 2K/8K 桶扩展 **1.93×/1.98×**，goodput 全场最高（EXP-007） | **完全一致**。这是本篇唯一一条"论文预言 → 本机证实"的正向闭环。论文把它写成备选项，本机把它变成首选项——因为通信项被放大后，唯一 $X=0$ 的形态胜出 |
| 3 | **DistServe 摘要/§6.2**:7.4× 更高请求率、12.6× 更严 SLO | pd1p1d 三桶饱和吞吐为 colocate 的 **0.76/0.58/0.60×** | **口径与平台都不同**，不可比。论文的对照是 DeepSpeed-MII/vLLM 在 8×A100 集群上的 OPT-13B/66B/175B；本仓是 1P1D、2×4090、7B。**方向相反的原因在第 1 行** |
| 4 | **DistServe §6.2**：用 SLO Scale 参数线性缩放两条 SLO 做敏感性 | fig6 扫 0.5–4×，四臂排序全程稳定（§4 段 9） | **方法一致**。这是独立收敛，不是照抄——两边都发现"单一阈值的结论不可信，必须给敏感性" |
| 5 | **Splitwise §VI-A**：逐层重叠后非重叠传输时间 A100 ~8 ms、H100 ~5 ms；串行传输给第二 token 加 64% 时延，重叠后 16.5% | 本仓 pull/push 均为**整请求粒度**传输，8K 桶 avg xfer 1602.7 ms(EXP-006) | **实现层次不同**。vLLM NIXL connector 在 D 端 READ 全部块，不做逐层流水。按 §3.5.4 的上界估计，即使实现完美重叠，8K TTFT 也只能降到 ~1.8 s，仍是 colocate 的 2×。**重叠改常数，不改结论** |
| 6 | **Splitwise §II-F**：云上 InfiniBand 每 GPU 对 25–50 GB/s | 本机三条路径最快的双向 D2D 22.6–22.8 GB/s，KV 通路 0.27 GB/s | 双向裸拷贝勉强够到云 IB 的下沿，**但那条路径不是 KV 走的路径**(§3.1.5)。这正是"三个数字不可互换"要防的误读 |
| 7 | **PagedAttention §7.2**："vLLM sets its default block size as 16" | EXP-006 单请求探针 bytes = 917,504 = 16×57,344，逐字节吻合 | **一致**。本仓的块粒度账完全建立在这条默认值上；换 block_size 则 §3.5 全部数字重算 |
| 8 | **PagedAttention §1/§3.1**：既有系统 KV 显存利用率仅 20.4%–38.2% | 本仓不测 KV 显存利用率 | **不适用**。本仓四臂全部跑在 vLLM（即 PagedAttention 之后），该浪费已被消除；论文的对照是 2023 年的 FasterTransformer/Orca |
| 9 | **PagedAttention §7.1**：PagedAttention 的 attention kernel 比 FasterTransformer 慢 20–26% | 未测 | **不在范围**。本仓不做 kernel 级 attention 对标（那是 triton-kernels 线的题目）。列在这里是为了说明：**块化管理不是免费的**，它用 20–26% 的 kernel 开销换掉了 60–80% 的显存浪费 |
| 10 | **Orca §3/§4.2**：iteration-level scheduling + selective batching；调度器每次迭代重选 batch，按 `max_bs` 与 `n_slots` 双约束 | 本仓不改调度器，只在**外部**用 offered load 扫描间接观察其行为 | **层次不同**。Orca 的 `n_rsrv`（按 `req.max_tokens` 预留 KV 槽位）在 vLLM 里被 PagedAttention 的按需分块取代，所以本仓的 `--ignore-eos` 固定输出长度不会触发 Orca 式的预留浪费 |
| 11 | **Orca 摘要**：相对 FasterTransformer 同延迟下吞吐 36.9× | 本仓 EXP-008 测到 v0.17.1 → v0.25.1 的 512 桶饱和吞吐 +45%，计算受限桶零差异 | **不可比但可对读**。36.9× 是"有无连续批处理"的差距，+45% 是"连续批处理之后八个月的工程改进"的差距。**前者是范式差，后者是版本差**，两个数量级的差别本身就说明了范式改变的价值 |
| 12 | **Megatron §3**：每层前向"only two all-reduces in the forward path" | 仓内按每层 1 次做下界计数（EXP-005 §6） | **本仓保守**。按 2 次算通信量翻倍（1.64 → 3.29 GB），零加速结论只会更强。**取下界是为了让结论对计数方式不敏感** |
| 13 | **Mooncake §5.1**：跨节点扩 TP 需"two expensive RDMA-based all-reduce operations per layer， significantly reducing the MFU" | 本机 tp2 prefill 零加速（§3.3） | **同一机理，不同尺度**。Mooncake 说的是跨节点 RDMA（数十 GB/s）已经贵到不值得；本机是 1.78 GB/s 的 SHM 回退。**结论方向一致，阈值差两个数量级** |
| 14 | **Ada 白皮书 Table 2**：RTX 4090 boost 2520 MHz、1008 GB/s、L2 73728 KB、TGP 450 W | 遥测实测未节流 2820 MHz、峰值功率 444.65 W；memcpy 924 GB/s | **时钟高于规格 11.9%，功率贴帽 98.8%，带宽达规格 91.7%**。教训：白皮书 boost 是"典型值"不是硬上限（§3.3.4），而 TGP 是硬上限 |
| 15 | **CUDA Guide §3.4.2.1**：P2P 未启用时拷贝须经主机中转 | 单向 0.60–0.91 GB/s = PCIe 4.0 x16 规格的 1.9%–2.9% | **文档预言 → 实测证实**，而且量化到了具体倍数。文档只说"更慢"，本仓给出了"慢 35–50 倍"这个数 |
| 16 | **nccl-tests PERFORMANCE.md**：AllReduce busbw $= \mathrm{algbw}\times 2(n-1)/n$ | 本机 2 卡，两列相等（§5.1） | **一致**，且解释了仓内"2 卡时两者相等"这句话的来历。**这个巧合只在 2 卡成立** |
| 17 | **NCCL 环境变量文档**："SHM is used between devices when peer-to-peer cannot happen， therefore， host memory is used" | 1.78 GB/s avg busbw，1M–512M 曲线平坦 | **一致**。曲线平坦进一步说明该路径没有大消息红利（§5.1.1） |
| 18 | **NVML**：`SwPowerCap` = 0x4 "SW Power Scaling algorithm is reducing the clocks below requested clocks" | 40 个 GPU0 采样里 23 个带 0x4，无 0x20/0x40（本文现算） | **一致**，且位掩码的**缺席项**（热节流位）构成排除性证据（§5.4.1） |
| 19 | **GPU Performance Background Guide §4**：ops：byte = 处理器数学带宽与内存带宽之比；性能受三因子之一限制 | bs=1 decode 算术强度 1.0 FLOP/B，ridge 164 FLOP/B（本讲义推导） | **一致**，差 164 倍。这是"decode 带宽受限"这句定性判断的定量强度 |
| 20 | **Sarathi-Serve 摘要**：chunked prefill + stall-free 调度在 SLO 内提升吞吐，Mistral-7B 单 A100 最多 2.6× | 本仓**未做该臂的开关对照** | **本仓的设计缺口，如实登记**。它是解决同一个问题（prefill 干扰 decode）的零通信方案，在本机这种互联受限平台上很可能优于 PD 分离。不主张任何数字 |

**这张表的读法**：20 条里只有第 2、7、15、16、17、18、19 条是"预言—证实"或"定义—实现" 的正向闭环；第 1、5、6、13 条是"前提不成立"；第 3、8、9、10、11 条是"不可比"； 第 20 条是本仓的缺口。**论文的数字几乎从不能直接搬到你的机器上，能搬的是机制、前提条件与判据。**

### 8.2 与生产实现的差距各在哪一层

- **PD 分离的本源语境**：DistServe/Splitwise/Mooncake 一类系统与 vLLM 上游 disaggregated serving 示例假设的是跨节点池化 + NVLink/InfiniBand/RDMA NIC； 那里 KV 传输走 GPUDirect RDMA，带宽两个数量级于本机，"消除干扰 + 独立扩缩"的收益才能覆盖传输成本。本仓测出的是该形态的**下界条件**，不是对形态的否定（REPORT §2.4）。
- **replica2 的生产形态**：多副本 + 专业负载均衡（K8s service、cache-aware router）。本仓 rr_proxy 是最小实现（§4 段 7），不做会话亲和，**所以 replica2 的数字是这一族方案的下限**。
- **TP 的生产语境**：NVLink 平台上 allreduce 带宽高两个数量级，TP2 prefill 不再零加速； vLLM 的 custom allreduce 小消息路径也依赖 P2P，本机禁用后只剩 NCCL SHM—— **同一份代码在不同互联上走的是不同分支**。
- **调度层的缺口（本仓没做的一臂）**：Sarathi-Serve 的分块 prefill + stall-free 调度（arXiv：2403.02310）与 PD 分离解决同一个问题——prefill 长任务阻塞 decode——但它 **不需要跨卡传输任何东西**。在本机这种互联受限平台上，它在原理上就该优于 PD 分离。本仓的 colocate 臂跑在 vLLM 默认调度上，**没有做该特性的开关对照**，因此不能主张任何相关数字。这是四臂矩阵设计的一个真实缺口，补法很明确：加一臂 `colocate_no_chunked`，同协议重跑三桶。
- **调度层的历史坐标**：Orca 的 iteration-level scheduling(OSDI 2022，§3)把调度粒度从"请求"降到"迭代"，解决"早完成的请求不能提前返回、新到的必须等整批跑完"； selective batching 把 Attention 之外的算子按 token 拉平成 $[\sum L, H]$、 Attention 逐请求单算。本仓所有臂都跑在这套范式之后的 vLLM 上，**享受它但不研究它**——EXP-008《B3 有限版本对照》的版本对照（+45%@512 桶）测的是范式之上的工程演进。
- **KV 管理层**：PagedAttention 的块表（arXiv：2309.06180，§4.2）让 KV 不必连续， 代价是 attention kernel 慢 20–26%(§7.1)，收益是消除 60–80% 的显存浪费（§1）。本仓的 PD 传输账（16 token/块）完全建立在这套机制之上。
- **量化**：本仓 dense 臂只跑 BF16；生产普遍上 W8A8/W4A16。量化会同时改变 §3.2 的分子（$W$ 减半）与 §3.5 的 KV 账（FP8 KV cache），四臂排序可能变化——**本仓不外推**。

### 8.3 延伸阅读(每条一句话说明它能解决什么疑问)

1. Zhong， Liu， Chen， Hu， Zhu， Liu， Jin， Zhang， "DistServe： Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"， arXiv：2401.09670，重点 §2.1（TTFT/TPOT 的应用语义）、§3.2 末段（replication 是备选项）、§3.3（通信开销的带宽门槛计算）、§6.2（SLO Scale 敏感性法）。——想知道"PD 分离到底假设了多快的互联"以及"为什么本机 replica2 反而赢"， §3.2 末段与 §3.3 两处必须读。
2. Patel， Choukse， Zhang， Goiri， Shah， Maleki， Bianchini， "Splitwise： Efficient generative LLM inference using phase splitting"， arXiv：2311.18677，重点 §II-F（云上 IB 带宽区间）、§IV-C（逐层重叠传输的机制）、§VI-A（重叠前后的第二 token 时延 64% → 16.5%）。——想知道"传输能不能藏起来、能藏多少"，读 §IV-C。
3. Qin et al.， "Mooncake： A KVCache-centric Disaggregated Architecture for LLM Serving"， arXiv：2407.00079，重点 §5.1（跨节点 TP 的两次 RDMA allreduce 与 MFU）、 §6.1(testbed：8×A800 NVLink + 800 Gbps RDMA)。——想知道生产级 PD 分离的硬件底座长什么样，看 §6.1 的一行配置就够。
4. Yu， Jeong， Kim， Kim， Chun， "Orca： A Distributed Serving System for Transformer-Based Generative Models"， OSDI 2022，重点 §3（C1/C2 两个问题与 S1/S2 两个解）、§4.2 Algorithm 1（`max_bs` / `n_slots` / `n_rsrv` 三个旋钮）。——想弄清"为什么 Attention 不能跨请求批，而 Linear 可以"，读 §3 的三种不可批情形。
5. Agrawal， Kedia， Panwar， Mohan， Kwatra， Gulavani， Tumanov， Ramjee， "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"， arXiv：2403.02310。——想知道"不做 PD 分离还能怎么消除 prefill 对 decode 的干扰"， 这是本仓缺的那一臂的理论依据。
6. Kwon, Li, Zhuang, Sheng, Zheng, Yu, Gonzalez, Zhang, Stoica, "Efficient Memory Management for Large Language Model Serving with PagedAttention", arXiv:2309.06180(SOSP 2023)，重点 §1/§3.1（20.4%–38.2% 显存浪费的来源）、 §4.2(block table)、§4.5（FCFS + all-or-nothing 驱逐 + swap/recompute）、 §7.1（kernel 慢 20–26%）、§7.2（block size 16 的扫描依据）。——本仓一切 KV 字节账的粒度依据都在 §7.2；想知道"块化不是免费的"，读 §7.1。
7. Shoeybi， Patwary， Puri， LeGresley， Casper， Catanzaro， "Megatron-LM： Training Multi-Billion Parameter Language Models Using Model Parallelism"， arXiv：1909.08053，重点 §3（列切-行切的配对与"每层两次 allreduce"）。——想知道 TP 的通信次数为什么不能再少，读 GeLU 那段非线性论证。
8. Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", arXiv:2305.13245,§2。——想知道 `num_key_value_heads=4` 这一个字段如何把本机的 PD 溃败缓解 7 倍，读这里。
9. Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures", CACM 52(4):65–76, 2009, DOI 10.1145/1498765.1498785。——$\min(\pi,\beta I)$ 与 ridge point 的原始定义， §3.2 全部推导的模型来源。
10. Hockney， "The communication challenge for MPP： Intel Paragon and Meiko CS-2"， Parallel Computing 20(3)：389–398, 1994。——通信性能用"启动开销 + 带宽"两参数刻画的原始文献（COMMS1 口径），公理 B 的来源；想理解"为什么小消息换快链路没用"， 从这两个参数入手。
11. NVIDIA, "NVIDIA Ada GPU Architecture" 白皮书，Appendix A Table 2。——RTX 4090 的 128 SM / boost 2520 MHz / 1008 GB/s / L2 73728 KB / BF16 Tensor FP32 累加 165.2 TFLOPS / TGP 450 W 的唯一权威出处。
12. NVIDIA， CUDA C++ Programming Guide §3.4.2(Peer-to-Peer Memory Access / Transfers)与 "GPU Performance Background User's Guide" §4。——前者给 `cudaDeviceCanAccessPeer` 语义与"未启用 P2P 时须经主机中转"的官方出处（§3.1.1）， 后者给 arithmetic intensity 与 ops：byte 的官方定义（§3.2.2）。
13. NVIDIA， NCCL User Guide 环境变量页（`NCCL_P2P_DISABLE` / `NCCL_SHM_DISABLE` / `NCCL_BUFFSIZE` 默认 4 MiB）、nccl-tests `doc/PERFORMANCE.md`（algbw/busbw 与 $2(n-1)/n$ 换算）、NVML API Reference `nvmlClocksThrottleReasons` 组。——三份工具语义文档，分别支撑 §3.1.3 的回退路径、§5.1 的读列法、§5.4.1 的排除性定案。
14. `pd_disagg/hw/` 三个原始文件 + `pd_disagg/REPORT.md` §1–§2 + `figures/` 七张图。——建议对照 §5.1 的读法与 §5.6 的读图法各过一遍；三个 hw 文件各排除一类替代解释。
15. `pd_disagg/analysis/nixl_token_accounting.md`、`records/EXP-005` §4–§6（含 `records/data/EXP-005_throttle_trace.csv` 80 行采样流水）、`records/EXP-007` §2/§7。——依次回答：前缀缓存污染是怎么被逐块定位的、功率帽是怎么被位掩码定案的、 84 点网格的四条偏差是怎么如实登记的。
