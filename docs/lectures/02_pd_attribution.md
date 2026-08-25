---
status: complete
关联EXP: EXP-006, EXP-007, EXP-011, EXP-013, EXP-014, EXP-015
配套: docs/theory/02_pd_kv_path.md(速查版) · pd_disagg/REPORT.md §2.2/§5(结论版) · moe_perf/PR_DRAFT.md(交付材料) · 本文=逐段走读版
---

# 深度讲义 02 · 把"传输占 TTFT 多少"从推断做成测量:PD 归因的 ~16 行 patch 与 MoE 调优的三级验证

> 读者:已读过讲义 01(四臂形态、互联三数、功率帽)的人。
> 读法:不跳步。每个论断后跟证据锚(EXP 编号 / 文件:行号 / derived 路径);
> 凡本文从仓内 raw/derived 现算的量,一律标注"本文现算"。

## 1. 这一篇回答什么问题

PD 分离在纸面上赢在哪、在这台机器上为什么全负载段输;以及一句话——"D 等待远端 KV 占 TTFT
54.2% / 62.5% / 64.2%"——怎么从**分量对账的推断**升级成**逐请求的因果占比**。读完你应当能:
①手推 PD 的三条账(单请求传输量 $B_{kv}(n)$、传输时间、容量上限 $BW_\mathrm{eff}/B_{kv}$),
并解释实测饱和 0.54 req/s@8K 为何与算式的 0.57 对得上;②讲清 ~16 行本地可观测性改动为什么
**恰好**打在那几处(请求身份 + 同时钟域),以及六段闭环误差 p50 <0.1% 不是"噪声小"而是
**缺口具名**——本文把这 0.18–0.52 ms 的残差逐桶指认到了具体一段代码;③从遥测反解 0.27 GB/s
的碎片化根因(descriptor 恰 16,384 B/个,传输时间对 descriptor **计数**线性而非对字节线性);
④把同一套"先分解、再归因、后验证"搬到 MoE:nsys node 级分解定位 fused_moe grouped GEMM 占
56.4%(bs=32)→ config 搜索 → correctness / kernel A/B / e2e 三级验证 → 材料齐备
(**未提交**);⑤答上"你凭什么说那 54% 是因果""闭环误差 0.02% 是不是自证循环"这类追问,
并诚实说出它不覆盖什么。

## 2. 直觉与第一性原理

**先想没有 PD 分离的世界。** 一个引擎实例里 prefill 与 decode 抢同一批 SM:一条 8K
输入做 prefill 时,所有正在解码的请求都在等,ITL 被顶起来——经典的队头阻塞。PD 分离
的价值主张只有两条:**①消除 prefill 对 decode 的干扰;②P 池与 D 池独立扩缩、各自选
最优并行度**(REPORT §2.4)。**代价只有一条,但很硬:KV 必须搬家**,搬运量不是常数,
是 $B_{kv}(n) = n \times (\text{每 token KV 字节})$,随输入长线性增长。

**日常类比与失效点。** 像中央厨房备菜、门店出餐:备菜与出餐不再抢同一个灶。类比在两处
失效:①连锁店之间"运菜"相对烹饪是二阶小量,而本机 KV 通路只有 0.26–0.27 GB/s
(telemetry-derived effective throughput,EXP-006/007),搬运成了主项;②"独立扩缩"要求
P:D 比例可调,而 1P1D 是这个形态最退化的样子——**收不到扩缩红利,却全额支付传输成本**。

**判据(第一性原理形式)**:PD 值不值,取决于付出的 $t_\mathrm{xfer}(n)$ 与省下的
$\Delta t_\mathrm{interference}$ 孰大。这台机器上两件事让不等式必然向左倾斜:**并发 1 时
右边恒为 0**——没有别的请求可被干扰,归因表(EXP-007 §5,同热工况 p50)里 colocate 与
pd1p1d 的 TPOT 都是 15.9–16.4 ms,PD 没让 decode 变快,因为本来就没有干扰可消除;**满负载
时左边爆炸**——8K 桶每请求要搬 469.8 MB(EXP-011 push 臂全量口径;pull 臂经前缀裁剪实拉
439.7 MB,EXP-006),按 0.27 GB/s 需 1.63 s。所以本仓的结论不是"PD 不好",是**形态与互联
能力错配**;真正要论证的是下一句:那条溢价**确实**由传输造成——问题于是从"部署选型"推进到
"归因方法"。

## 3. 完整推导与机制

### 3.1 三条账:传输量、传输时间、容量上限

一步一理由:①**每 token KV 字节**——Qwen2-7B 28 层,每层 K、V 各一份,每份 4 个 KV 头
× 128 维,BF16 每元素 2 B:$28 \times 2 \times 4 \times 128 \times 2 = 57{,}344$ B/token
(GQA 下 KV 头数 4 不等于注意力头数,这步最常算错);②**块粒度**——KV 按 block(16
token)管理,$16 \times 57{,}344 = 917{,}504$ B/block,传输按块取整,非块对齐 prompt
向上取整(analysis/nixl_token_accounting.md 的"两计数器口径"表);③**单请求传输量**
——8192 token = 512 块 = 469,762,048 B ≈ 469.8 MB,EXP-013 实测 36/36 请求的 bytes 与
该式**逐字节相等**(§3.5);④**传输时间**——$t_\mathrm{xfer} = B_{kv}/BW_\mathrm{eff}
= 469.8\,\mathrm{MB}/0.27\,\mathrm{GB/s} \approx 1.63$ s,EXP-006 实测 avg xfer
1602.7 ms;⑤**容量上限**——传输在关键路径且串行,则 $\mathrm{req/s}_{\max} \approx
0.27/0.470 \approx 0.57$,EXP-007 实测饱和 0.54 req/s@8K。**传输带宽即容量**。

TTFT 侧对照(EXP-007 §5 归因表,并发 1、同热工况、p50):PD 溢价 = 219.3−65.4 = 153.9 ms(512)、
718.6−224.9 = 493.7(2K)、2718.7−925.2 = 1793.5(8K);溢价随输入长增长的形状与 $B_{kv}(n)$ 一致
——但**形状一致不等于因果**。

### 3.2 从"分量对账"到"因果占比":缺的到底是什么

v1 报告(fig4)是**分量对账**:拿 colocate 的无负载 TTFT 当 PD 的 P 段,拿遥测 avg xfer
当传输段,剩下归"其余"。它给出 54–64%,方向没错,但有三个结构性弱点:①**跨臂替代**
——P 段用的是另一个臂的数字,两臂工况不必相同;②**口径错配**——avg xfer 是一个测量点
内所有传输的聚合均值,不是"这一条请求"的;③**边界不清**——握手、调度轮询、块分配被
一股脑塞进"传输"或"其余"。

要升级成**因果占比**,只需两个条件,缺一不可:**同一请求身份**(分子与分母属于同一条请求)、
**同一时钟域**(两个量能相减)。这两条在 PD 架构下都不平凡。**身份**:一条请求有三个 id
(client 自定 / P 端引擎内部 / D 端引擎内部);NIXL 恰恰把"引擎身份 / 会话身份 / 内存寻址"三层
**正交拆开**(theory/02 §2,pull_scheduler.py:265-275 显式交出 remote_engine_id /
remote_request_id / remote_block_ids)——这对健壮性是优点(0.17.1 P2pNccl 正因隐式 key 分叉而
挂死,EXP-012),对观测则意味着**必须显式建立 join 键**。**时钟**:client / proxy / P / D 是四个
进程,时长要用单调钟(perf_counter),跨进程对齐要用同一把墙钟(epoch);本实验是同机 1P1D,
epoch 天然同域——这是方法成立的前提,也是它的边界(§6)。

### 3.3 ~16 行改动的设计:为什么恰好是这几个落点

改动打在 v0.25.1 的 NIXL connector(D 端),逐行带 `# EXT1` 标记、原件备份可还原
(`ext1/orig/`);口径上是**本地可观测性改动,不是 NIXL/Connector 核心改造**(措辞约定)。三个
落点各解决一个不可替代的问题:①**起点在 connector 首见请求处**(`pull_worker.start_load_kv`)而非
scheduler,因为 kv_wait 要**含握手**(首次与远端 P 建 NIXL agent 连接的一次性成本),放在 scheduler
会把握手甩到窗口外,首请求 +292 ms(512 桶,EXP-013 §5)就观测不到;②**聚合按 req_id 逐 handle
累加**(`base_worker._pop_done_transfers`)而非用现成 Prometheus counter,因为 counter 是**全局
聚合量**,并发下无法归属到单条请求,而一条请求可能拆成多个 handle;③**输出是一行绑定三段身份与
两段时钟的日志**(`_ext1_emit`)——跨行拼接要额外假设日志顺序,而日志顺序在多线程下不可靠。外加
第 4 处:失败路径(`_handle_failed_transfer`)清理两个 dict,这不是记账需要,是**防泄漏**——一个
典型的"观测代码把被观测系统弄坏"的失败模式。

**为什么不动核心记账路径**:一旦改 counter 语义,新数据与既有测量点(EXP-007)不再
可比;而本次关键论据之一恰恰是"打 patch 前后 TTFT 218/727/2738 vs 219/719/2719,噪声
内"——**观测零扰动的对照,必须建立在被观测路径未被改写之上**。

### 3.4 闭环误差 p50 <0.1% 是怎么来的:六段分解的望远镜性质

六段定义(口径见 `analyze_ext1.py:9-14` 的 docstring):pre_proxy =
`t_recv(proxy) − t_send(client)`(client→proxy 网络);p_segment = `t_p_done − t_p_send`
(P 端 prefill 含 HTTP);gap_p_to_d = `t_d_send − t_p_done`(proxy 搬运
kv_transfer_params);d_pre_kv = `t0_epoch(D conn) − t_d_send`(D 端 HTTP+排队+调度);
kv_wait = `done_epoch − t0_epoch`(D 等远端 KV);post_kv =
`t_first_token(client) − done_epoch`(D 首步 + 流式回传)。

**推导**:六段按定义展开求和,中间项两两相消(望远镜求和):

$$\sum_{6} = (t_\mathrm{first\_token} - t_\mathrm{send}) - (t_\mathrm{p\_send} - t_\mathrm{recv}) = \mathrm{TTFT} - \delta$$

$\delta = t_\mathrm{p\_send} - t_\mathrm{recv}$ 是 proxy **内部**那一段:收到请求之后、
发给 P 之前,代码在做 `await request.json()` 与选实例(`ext1_proxy.py:162-169`)。它
**没有被列进六段**,所以闭环误差不是测量噪声,而是一个**结构性的、可具名的缺口**
——必然为正,且应随 prompt 体积增长。**实测验证(本文现算,可从仓内数据复算)**:

| 桶 | 闭环误差 p50(EXP-013 §5) | 残差 = TTFT − Σ六段(p50) | proxy 内部 `t_p_send − t_recv` |
|---|---|---|---|
| 512 | 0.08% | 0.176 ms | 0.169–0.239 ms |
| 2048 | 0.04% | 0.300 ms | 0.290–0.386 ms |
| 8192 | 0.02% | 0.515 ms | 0.349–0.819 ms |

三列逐桶对齐:**残差区间与 proxy 内部段区间完全重合**(全体 36 请求 0.169–0.820 ms,
`raw/EXP-013/ext1_proxy_lines.txt`),且随 prompt 体积单调增长——与 JSON 解析成本的
预期一致(推断,非独立实测)。闭环误差至此被 100% 解释。注意误差**百分比**
512 桶最大(0.084%)而残差**绝对值**最小(0.176 ms):分母从 218.3 涨到 2738.0 ms,涨得比
分子快(核验 $218.3\times0.00084=0.183$、$2738.0\times0.000189=0.518$ ms,与现算对上)——
**"误差小"看绝对量,"误差率小"可能只是分母大**(§6 误区 2)。

### 3.5 三重互证:三条正交证据链,各自排除什么

**链 1 · 账目对不对(bytes 与 descriptor 双恒等)**。逐请求 bytes 求和 =
**7,398,752,256** = Prometheus `vllm:nixl_bytes_transferred_sum`,分毫不差(EXP-013 §5)。
本文进一步核验 descriptor:逐请求 Σdescs = **451,584** = `vllm:nixl_num_descriptors_sum`
的 after 值(before 全 0,`raw/EXP-013/metrics_8200_{before,after}.prom`),同样分毫不差。
它排除:日志行丢失、重复计入、handle 漏聚合。再算两步(本文现算,纯算术):
$7{,}398{,}752{,}256 / 451{,}584 = 16{,}384$ B——每 descriptor **恰 16 KiB**;
$7{,}398{,}752{,}256 / 57{,}344 = 129{,}024$ token $= 12 \times (512+2048+8192)$——**恰
等于协议期望的全部 prompt token**,即零本地前缀命中、零舍入。对照 EXP-006 的固定 seed
协议:那次 262,144 − 245,344 = 16,800 token 的缺口全部是 D 端本地 prefix cache 命中
(analysis/nixl_token_accounting.md 逐块定罪,其中 511 块源码定罪于 bench 的 test 请求)。
同一套记账在两种协议下给出两种结果而两次都对上账——这是**协议 v2(每请求唯一 seed)
有效性的独立验证**。

**链 2 · 归因边界对不对(kv_wait ≈ xferDuration)**。kv_wait 是**墙钟**等待窗口,xferDuration 是
NIXL **自报**的纯传输时间,来源完全独立;实测差 0.3–1.9 ms(各桶 p50 = 0.33/0.71/1.78 ms,
EXP-013 §6)。它排除"kv_wait 里混了大量调度轮询/握手/块分配"。**注意**:xferDuration 已含
posting,**不与 postDuration 相加**。**链 3 · 分解完不完整(六段闭环)**。§3.4 已证:误差
p50 <0.1%(最差桶 0.084%,逐请求最大 0.11%),且残差被指认到 proxy 内部解析段;它排除"某段被
重复计入或漏掉"。

**外加对照臂 · 观测有无扰动**:打 patch 后 TTFT p50 218/727/2738 vs 未打 patch 的矩阵
219/719/2719(EXP-006/007),噪声内——没有这条,前三条都可能是"被观测系统已经变了样"的自洽假象。
三条链 + 一条对照,才撑起"KV 等待占 TTFT 54.2% / 62.5% / 64.2%(512/2K/8K,p50,每桶 n=11,
排除 idx=0 首请求)"这句**因果占比**声明。

### 3.6 0.27 GB/s 的碎片化根因:从 descriptor 恒等式反解

链 1 里那个 16,384 B 不是巧合,可从第一性原理推出:
$16\ \text{token/block} \times 4\ \text{KV 头} \times 128\ \text{维} \times 2\,\mathrm{B}
= 16{,}384\,\mathrm{B}$,即**一个 descriptor = (一层, K 或 V, 一个 block)**。每 block 需
$28 \times 2 = 56$ 个 descriptor,$56 \times 16{,}384 = 917{,}504$ B/block,与 §3.1 第 2
步闭合。实测 descs 逐桶 1792 / 7168 / 28672(`derived/ext1_per_request.csv`,36/36 无一
例外)= $56 \times$ (32/128/512 块),逐字对上。**关键判据(本文现算)**——每请求
xferDuration ÷ descriptor 数:

| 桶 | descs | xfer p50 (ms) | **每 descriptor 耗时** | 等效吞吐 |
|---|---|---|---|---|
| 512 | 1,792 | 117.9 | 65.8 µs | 0.249 GB/s |
| 2048 | 7,168 | 454.5 | 63.4 µs | 0.258 GB/s |
| 8192 | 28,672 | 1762.7 | 61.5 µs | 0.267 GB/s |

descriptor 大小恒定 16 KiB,每 descriptor 耗时也几乎恒定(61.5–65.8 µs)——**传输时间对
descriptor 计数线性,而不是"带宽×时间"**。这就是碎片化的定量指纹:单次传输大小从没变
过,吞吐被钉死在 $16{,}384\,\mathrm{B}/63\,\mu s \approx 0.26\,\mathrm{GB/s}$。**量级
对照**:EXP-002 实测 GPU 间延迟 14.5–15.9 µs,一次 16 KiB 花 ~62 µs 约为裸延迟的 4 倍
——成本主要落在每次传输的固定开销(descriptor 处理、launch、同步),不在搬字节本身;
这也解释了 fig5 上那条 ~12 ms 的"小传输延迟地板"(smoke 0.188 MB / 14.1 ms,EXP-001)。
**工程推论(可证伪)**:要提速必须**合并 descriptor**(层维度批量成更大连续块),而不是
换方向——方向已被实测排除:NixlPush 8K TTFT −6.7%、吞吐 +10–13%,**量级不变**
(EXP-011)。**口径约定**:0.26–0.27 GB/s 只能称 telemetry-derived effective throughput,
不能讲成链路物理带宽。

### 3.7 同一套方法搬到 MoE:先分解、再归因、后验证

PD 那条线的骨架是:**先把总量分解到可归属的段,再用独立证据链锁死归因,最后设对照臂验证**。
MoE 线同法,只是尺子从"请求级时间"换成"kernel 级 GPU wall-time":①**分解**——nsys 采一个
20 s 稳态窗,`cuda_gpu_kern_sum` 按 kernel 名归 9 类,bs=32 时 **fused_moe grouped GEMM 占
56.4%**,bs=1 时反而是 dense GEMM/GEMV 占 40.9%(EXP-014 §5);②**归因**——MoE/dense 的
decode 优势 **2.03×(bs=1) → 0.97×(bs=8,反转点)→ 0.82×(bs=128)**,机理是 top-4/60 路由
下 batch 增大后每 step 命中的专家并集趋于全量(60 专家约 28.6 GB > dense 14.2 GB),bs=1 的
激活权重优势(2.7 GB/step)反转为读放大劣势,分解表印证——grouped GEMM 占比 18.7% → 56.4%
(EXP-014 §6);③**目标由数据锁定**——serving batch(≥8)下唯一大头是 fused_moe,而该形状的
Triton config 在上游**社区空缺**(运行时告警在案,EXP-009 §5),moe_align(≤4.1%)与 permute
(≤0.5%)不值得动;④**三级验证**——correctness(没算错)/ kernel A/B(主证据)/ e2e(验证
机理自洽);⑤**折算式** $\Delta_\mathrm{e2e} \approx \Delta_\mathrm{kernel} \times$ 该 kernel
时间占比——kernel 端 M≥128 改善 3.3–3.9%,乘 56.4% 得 e2e 约 2% 的上限,实测 TPOT
**+0.8~1.2%**,同量级、方向一致,机理自洽;但该幅度**低于跨会话漂移**(±5~8%),按仓内措辞约定
**不作 headline**,主证据是 kernel A/B 两端数字。**折算对上了,不等于可以拿它当卖点**——判据
是"效应量 vs 噪声量",不是"方向对不对"。

## 4. 代码逐段走读

按一次测量的执行顺序读:proxy 打点(身份贯通)→ D 端 connector 记时 → 落一行日志 → 离线三方
join;最后是 MoE 分解的采集口径。逐 handle 聚合那一处(`patch:14-26`,把 `res.totalBytes /
xferDuration / postDuration / descCount` 按 req_id 累加,紧接上游原有的
`self.xfer_stats.record_transfer(res)`)机理已在 §3.3 ② 讲过,此处不再展开代码。全部引用为仓内
真实代码逐字拷贝,标 文件:起-止行。

**段 1 · proxy:身份透传与中间 epoch**(`pd_disagg/ext1/ext1_proxy.py:162-179`)

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

**角色**:身份链中枢与中间 epoch 的唯一来源。**关键行为什么这么写**:①`headers.get("X-Request-Id",
...)` 而非无条件 `uuid4()`——client 自定 id 被接管,再随 header 转发 P 与 D(`:133-136`、`:148-151`),
vLLM 侧从 header 取 id 内嵌进引擎 req_id,join 键就此贯通;②`t_recv` 取在 `await request.json()`
**之前**——正因如此,"解析请求体"这段没落进任何一段,成了 §3.4 那个可具名的残差;③打点全用
`time.time()`(epoch)而非 perf_counter,跨进程要与 D 端日志对齐、单调钟没有共同原点;余下两个
epoch 记在 `generate_stream()` **内部**(`:183-194`),因为 StreamingResponse 惰性,函数返回时还没
真正发出。**改错会怎样**:`t_d_send` 若写在闭包外会早于真实发送,d_pre_kv 被系统性拉长、kv_wait
占比被稀释;丢掉 header 透传则三方 join 失配,`analyze_ext1.py:92` 会把请求丢进 WARN。

**段 2 · D 端起点:connector 首见请求即记双钟**
(`pd_disagg/ext1/nixl_req_telemetry_v0251.patch:82-89`)

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

**角色**:kv_wait 窗口的左端点。**为什么在这一行**:`start_load_kv` 是 D 端 connector **第一次**
看见该请求的位置,握手在它之后——起点放这里 kv_wait 才**含握手**,首请求的 409.8/462.8/1770.3 ms
与后续请求之差(512 桶 +292 ms)才成为握手一次性成本的直接观测;两个钟分工明确:perf_counter 算
时长(单调、不受墙钟调整影响),time.time() 供跨进程对齐。**改错会怎样**:起点若挪到"块分配完成
后",握手成本被排除出 kv_wait、落进 d_pre_kv,首请求观测消失,而三重互证**仍然全绿**——这是最
危险的一类错误:不破坏自洽性,只悄悄改变语义。

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

**角色**:整套方法的产物格式;`bytes/xfer_us/post_us/descs` 来自同一个喂给 Prometheus 的 `res`
——这正是 §3.5 链 1 那个"分毫不差"能当**完整性校验**(检验没有 handle 被漏)而非自我复述的原因。
调用点是"该请求全部 handle DONE"那一刻(`patch:31`,插在
`done_req_ids.add(req_id)` 之后)。**关键行为什么这么写**:①两个 `pop` 而非 `get`
——取走即清,聚合完成的请求不再占内存,也杜绝二次发射;②`remote_req` 取 P 端 id,这一项
让**身份拆分成为可测量对象**(36/36 请求的 client rid 同时出现在 D 端 req_id 与
remote_request_id 中,EXP-013 §5);③kv_wait 用 perf_counter 差、两个 epoch 用墙钟——
**时长与对齐分用两把尺**,docstring 把口径写死;④`logger.info` 用**惰性格式化**
(`%s` + 参数)而非 f-string,日志级别关掉时不做字符串拼接——"观测零扰动"结论的实现侧
保证之一。**改错会怎样**:用 `get` 不 `pop`,`_ext1_t0` 只增不减、长跑内存单调上涨;
把 done_epoch 换成 perf_counter 值则跨进程对齐立刻失效,post_kv 段算出荒谬值——而闭环
校验会把它抓出来。

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

**角色**:三个数据源(client JSONL / proxy 行 / D 端 EXT1_KV 行)合成一行逐请求记录,落
`derived/ext1_per_request.csv`。**关键行为什么这么写**:①`kv_share_of_ttft` 的分子分母
**同属一条请求**——这一行就是"因果占比"与"分量对账"的全部区别;②`identity_match` 把
身份校验做成**落盘字段**而非临时断言,事后可审计(36/36 为 True,本文复核);
③`sum_segments_ms` 与 `ttft_ms` 并列存盘,闭环误差可由任何人从 CSV 重算——**校验量必须
落盘,否则它只是一次性的自我保证**;④join 对 `len(matches) != 1` 直接 WARN 跳过
(`:89-96`),不做模糊匹配。**改错会怎样**:把 `kv_share` 改成"桶级 kv_wait 中位数 ÷
桶级 TTFT 中位数",数字大体不变但语义退回分量对账——**同一个百分比,证据等级完全不同**。

**段 5 · MoE 分解的采集口径:node 级 trace 是硬条件**(`moe_perf/d1_nsys.sh:16-25`)

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

**角色**:决定分解表是否成立的那一行是 `--cuda-graph-trace=node`。**为什么这么写**:①vLLM 的
decode 步跑在 CUDA graph 里,nsys 默认 `graph` 模式把整张图记成**一个** kernel,图内的 fused_moe /
attention / allreduce 全不单列——首采的"分解表"里 other 桶占 77%、fused_moe 只有 384 个实例(实为
4 个 prefill step 的混样),bs=32 表整张作废(EXP-014 §7,graphlevel 文件保留作对照证据),node 级
重采后 other 降到 1.2%;②`--capture-range=cudaProfilerApi` 配 `--profiler-config.profiler=cuda`:
采集窗由 `/start_profile` 触发的 `cudaProfilerStart` 控制(全 rank),脚本先跑 15 s 让负载进 decode
主导稳态、再开 20 s 窗(`:45-50`)——**profile run 的时延数字永不进 benchmark 表**;③配套分类器
(`moe_perf/d1_kernels.py:53-61`)把未命中任何桶的 kernel 收进 `unknown` 并打印 top-5(`:78-82`)
——**分类器必须暴露它不认识的东西**,正是这条设计让 graph-level 采集的失败当场可见。**改错会
怎样**:漏 node 级参数则分解结论指向完全错误的热点;不设稳态等待则窗内混进 prefill 潮使 grouped
GEMM 占比被高估;把 other 静默归零,一次口径错误就伪装成漂亮的结论。

## 5. 实验数据怎么读

### 5.1 EXP-013 主表:每一列在防什么

| 桶 | TTFT (ms) | kv_wait (ms) | **KV 占 TTFT** | p10–p90 | P 段 (ms) | post-KV (ms) | 闭环误差 |
|---|---|---|---|---|---|---|---|
| 512 | 218.3 | 118.2 | **54.2%** | 52.4–55.6% | 63.8 | 24.4 | 0.08% |
| 2048 | 726.7 | 455.7 | **62.5%** | 61.8–63.7% | 221.7 | 24.0 | 0.04% |
| 8192 | 2738.0 | 1763.8 | **64.2%** | 63.6–64.8% | 899.5 | 35.0 | 0.02% |

**口径**:并发 1、p50、每桶 n=11(排除 idx=0 首请求)、request 级因果占比。四个定语一个都不能丢
——去掉"并发 1"就变成对满负载的承诺,去掉"排除首请求"就混进握手成本,去掉"request 级"就退回分量
对账。

- **先看 p10–p90 而不是 p50**:三桶分布宽度都在 ±2% 内,占比是一个**稳定的结构量**,不是被少数
  离群请求拉出来的均值(若 p10–p90 张开到 30–80%,同一个 p50 的解释力完全不同)。
- **占比为何随输入长饱和在 ~64%**:P 段与传输段**同为 $O(n)$**——63.8→221.7→899.5 与
  118.2→455.7→1763.8,比值 1.85/2.06/1.96 基本恒定,故占比趋常数;短输入被 ~40 ms 固定开销稀释,
  512 桶只有 54.2%(EXP-013 §6)。
- **d_pre_kv 是隐藏的第三条 $O(n)$ 线**:9.3 / 22.3 / 41.0 ms(本文现算)——D 端也要
  tokenize 完整 prompt、做块分配(推断,机制见 analysis/nixl_token_accounting.md 调度链)。
- **post_kv 几乎恒定**:24.4 / 24.0 / 35.0 ms——KV 到齐后 D 只做 1 个 token 的前向再流式
  吐出;这条近似常数本身就是"kv_wait 确实吃掉了全部输入相关成本"的旁证。
- **误差条怎么看**:本表不画误差条,给的是 p10–p90 区间与闭环误差列;闭环误差列**不是
  精度**,是完整性——它回答"六段加起来还差多少",而 §3.4 已把这点差指认到具体代码段。

**两张配图**:`fig4_pd_ttft_decompose.png`(堆叠柱,`make_figures.py:154-184`)y 轴 PD 无负载
TTFT p50,三段自下而上是"P 端 prefill(≈colocate 无负载 TTFT)"、"NIXL KV 传输(telemetry
avg xfer)"、"其余"——**注意最底段的括号**,它就是 §3.2 说的跨臂替代,所以这张图是**v1 的
分量对账**而非因果占比;两者结论一致(54–64%)本身是一条独立信息:**对账法被逐请求数据追认
有效**。`fig5_nixl_transfer_scaling.png`(双对数,`:187-215`)x 轴每传输 MB、y 轴每传输 ms,
两条参考线是 0.27 GB/s 斜渐近线与 ~12 ms 小传输延迟地板:大传输点贴渐近线说明有效吞吐跨尺寸
恒定,smoke 点(0.188 MB / 14.1 ms)贴地板说明小传输是**延迟主导**而非带宽主导。

### 5.2 EXP-014 两张 kernel 占比表:对读才有信息

| 分类 | bs=1 | bs=32 |
|---|---|---|
| **grouped GEMM(fused_moe routed experts)** | 18.7% | **56.4%** |
| dense GEMM/GEMV(proj/共享专家/lm_head) | **40.9%** | 14.9% |
| AllReduce(NCCL) | 13.8% | 15.0% |
| attention | 7.2% | 7.1% |
| norm/rope/act/elementwise | 8.7% | 3.7% |
| routing(topk/softmax) | 4.9% | 1.1% |
| moe_align_block_size | 4.1% | 1.0% |
| permute/unpermute/moe_sum | 0.5% | 0.3% |
| other(top 已核) | 1.2% | 0.4% |

**单看一列会得出错误结论**:只看 bs=1 你会去优化 dense GEMV;只看 bs=32 你会以为 MoE 永远被
grouped GEMM 支配。**对读**才见机理:18.7%→56.4% 与 40.9%→14.9% 是同一件事的两面——batch
增大,routed 专家的命中并集扩张,而 lm_head 这类**与 batch 无关的固定读**被摊薄(vocab
151936 × hidden 2048 → 0.62 GB 权重,TP2 切分后每 rank 每 token 仍读 0.31 GB,EXP-014 §6);
AllReduce 恒 14–15% 是 TP2 的固定税,与讲义 01 的 allreduce 墙同源。**这张表防了哪些坑**:
①node 级 trace(§4 段 5)否则整表作废;②稳态窗(15 s 后开 20 s)否则混进 prefill;③9 类 +
防静默 other;④采集窗 kernel 时间合计 43.1 s / 36.3 s ≈ 2 GPU × 20 s 窗上限附近(含 node 级
tracing 开销),故**占比是窗内相对值,绝对吞吐一律以 sweep JSON 为准**(EXP-014 §7)。

配套曲线 `moe_perf/figures/d1_fig1_decode_scaling.png`(源数据 `derived/d1_scaling.csv`,x 轴并发
1→128、y 轴输出 tok/s):读图先找**交点**(bs≈8,0.97×),再看两侧斜率——左侧 MoE 陡(激活参数量
优势),右侧 dense 反超(读放大)。bs=1 的 roofline 对照:MoE 实测 221 / 理论 ~373(59%),dense
109 / ~142(77%)——MoE 达成率更低,量化解释了"MoE 的 bs=1 优势没有理论上那么大"(routing 4.9%
+ moe_align 4.1% 的额外开销)。

### 5.3 EXP-015 三张表:主证据、辅助证据与陷阱

**kernel A/B(主证据,µs,default→tuned)**:

| M | EP | Δ | 非EP | Δ |
|---|---|---|---|---|
| 1 | 38.2→34.9 | **-8.5%** | 24.4→23.4 | **-3.8%** |
| 8 | 389.4→389.0 | ~0 | 250.9→252.7 | ~0 |
| 32 | 563.4→564.1 | ~0 | 506.1→507.2 | ~0 |
| 64 | 578.2→573.1 | -0.9% | 568.2→566.9 | ~0 |
| 128 | 609.8→585.9 | **-3.9%** | 601.8→579.4 | **-3.7%** |
| 256 | 621.4→597.9 | **-3.8%** | 609.2→589.0 | **-3.3%** |

**读法**:收益的**形状**比幅度重要——两端显著、中段打平。为什么是这个形状?看上游默认
启发式(`vllm/model_executor/layers/fused_moe/fused_moe.py:1371-1395`,main@7aa248fc):
`block_m` 按 M 分四档(16/32/64/128)、`block_n` 按 M≤64 二分、`group_m` 只在
`tokens_per_expert > 128` 时才开 16、`num_warps` 按 M≤128 二分——一组**粗粒度阶梯**。中段
M 恰落在启发式调得较准的区间,搜索结果与它撞车(打平);M=1 的极端 decode 形状与 M≥128
的大 tile 区,阶梯分辨率不够,搜索才有空间。这条读法直接决定了 D3 的判定:**tile 空间在
中段已被启发式覆盖 → 不做无数据支撑的 kernel 改动**(EXP-015 §6)。

**A/B 的次序即协议**(`moe_perf/d2_ab.sh:50-73`):default 必须**先**测——config 一旦拷进
`fused_moe/configs/`,exact-match 文件名查找立刻生效,没有回头路;kernel 与 e2e 各自成对、不交叉,
让每对间隔尽可能短(热工况漂移是本机一阶噪声);EP 与非 EP 分别测,共享一次 serving 会互相污染。

**e2e(辅助证据)**:TPOT p50 c1 4.40→4.34 ms、c32 17.91→17.70、c128 28.78→28.47,**一致
+0.8~1.2%**;吞吐与 TTFT 在会话噪声内持平。**为什么只能当辅助**:跨会话漂移 ±5~8%
(EXP-015 §5 以 D1 同点为对照:本轮 default c32 1596 / c128 4233 tok/s vs D1 的 1691 /
3975),**效应量小于噪声量**,仓内措辞约定因此明确 **D2 的 e2e 不作 headline**;另外 e2e 未采
GPU 遥测(热工况不可证),吞吐只写"噪声内持平"、不作方向声明,TPOT p50 对热态不敏感予以
保留(EXP-015 §7)。

**陷阱 · Triton 首跑 JIT 伪影**:装入新 config 后**首个** c32 bench 的 TTFT p50 是 1021 ms
(default 176 ms),而后跑的 c128 正常(363 vs 359 ms)——新 tile 形状第一次被流量命中触发
现场编译,32 路并发同时阻塞。处理不是丢数据,而是 **warmup 后复测并两版并存**:c32 吞吐
1616 vs 1596、TPOT 17.76 vs 17.91;c128 吞吐 4178 vs 4233(噪声内)、TPOT 28.52 vs 28.78
(EXP-015 §5)。**读任何"换了实现之后第一次测"的数字,先问有没有 JIT/autotune 的一次性
成本混在里面。**

**交付物核对**:两个 JSON 各 **18 个 M 档** + 一个 `triton_version` 元键(仓内已勘正一处计数错误:
曾把元键计入,误报 19 档)。元键不是装饰——上游加载时会 `tuned_config.pop("triton_version", None)`
再转 int 键(fused_moe.py:1155-1157),所以它**必须存在且必须被排除在 M 档之外**。correctness:
`pytest tests/kernels/moe/test_moe.py::test_fused_moe` → **120 passed, 120 skipped, 0 failed**
(skipped 为异平台/异 dtype 参数化)。PR 材料齐备,**提交动作留给本人,未提交**。

## 6. 误区与边界

1. **"nsys 采下来就是 kernel 分解。"** 本仓第一次采的 bs=32 分解表**整张作废**:默认
   `--cuda-graph-trace=graph` 下 CUDA graph 内的 decode kernel 不单列,other 占 77%,fused_moe 只有
   384 个实例(实为 prefill 混样);node 级重采后 other 降到 1.2%(EXP-014 §7,graphlevel 文件保留
   作对照证据)。**一般形式:profiler 的默认聚合口径可能把你要找的东西整个藏起来,而结果表看上去
   完全正常。**
2. **"闭环误差 0.02% 说明测量很准。"** 错两次:①闭环误差衡量的是**分解的完整性**而非任何一段的
   精度——六段全偏移同一个常数也能闭环;②0.02% 小于 0.08% 主要因为分母大(2738 vs 218 ms),残差
   绝对值反而更大(0.515 vs 0.176 ms,本文现算)。正确用法是**把残差指认到具体代码段**,指认不了
   就不该声称闭环。
3. **"曾误计 19 个 M 档。"** 仓内已勘正:`triton_version` 是元键不是 M 档,实为 18 档。**交付物的
   规格数字要按消费方的解析逻辑数**(上游 loader 明确 pop 掉该键,fused_moe.py:1155-1157),不是
   按文件里有几个顶层 key 数。
4. **"0.27 GB/s 就是这条链路的带宽。"** 它是 **telemetry-derived effective throughput**:分子是
   NIXL 自报 totalBytes,分母是 NIXL 自报 xferDuration,里面含着 descriptor 碎片化的固定开销
   (每 16 KiB 花 ~62 µs,约为裸延迟 14.5–15.9 µs 的 4 倍)。它衡量的是**这套软件栈在这种访问模式
   下的有效速率,不是链路能力**;相关措辞约定:xferDuration 已含 posting,**不与 postDuration 相加**。
5. **"tuned config 全面胜出。"** 真实形状是两端显著、中段打平。把"M=1 −8.5%"讲成"MoE kernel 提速
   8.5%",既丢了 M 档定语也丢了 EP/非 EP 双口径(−8.5% / −3.8%);e2e 的 +0.8~1.2% 更不能当
   headline,它低于跨会话漂移。
6. **"PD 分离在这台机器上失败,说明 PD 分离是错的。"** 不是。收益项在并发 1 下恒为 0(无干扰可
   消除)、满负载下被 0.27 GB/s 的成本项压垮,这是**形态与互联能力错配**(REPORT §2.4);换方向
   (push)只挽回 8K TTFT −6.7%、吞吐 +10–13%,量级不变(EXP-011),问题不在实现方向。

**适用边界**:①**同机是本方法的硬前提**——四个进程共享一把墙钟,epoch 才能直接相减;跨节点
时 `t0_epoch`/`done_epoch` 的对齐需另设时钟同步方案(PTP/NTP 残差会直接进入分解),六段闭环
的 0.1% 量级不可外推。②**并发 1**——全部占比数字来自 attribution 模式(并发 1、每桶 12 请求、
max_tokens=32、排除首请求),并发上去后 kv_wait 会混入排队。③**pull 路径**——patch 只覆盖
pull,push 侧若要同样关联需仿做(EXP-013 §7)。④**平台**——单机 2×RTX 4090、P2P 驱动禁用、
vLLM 0.25.1(ENV-B)、Qwen2-7B-Instruct、1P1D(非 xPyD);MoE 线为 ENV-C(main@7aa248fc)、
Qwen1.5-MoE-A2.7B-Chat、TP2+EP。可外推的是**方法**,不是数字。

## 7. 连环追问

1. **Q:kv_wait 到底从哪一刻算到哪一刻?**
   从 D 端 connector 在 `start_load_kv` 首次看见该请求(含尚未建立时的握手等待)起,到该
   请求**全部** NIXL read handle 变为 DONE 止(patch:36-41 的 docstring 写死了这条口径)。
   perf_counter 计时长,两端另记 epoch 供跨进程对齐。
2. **Q:你凭什么说那 54% 是因果而不是相关?**
   ①分子分母同属一条请求、同一时钟域(analyze_ext1.py:102-108),不是跨臂替代;②kv_wait 与 NIXL
   自报的 xferDuration 只差 0.3–1.9 ms,窗口里装的确实是传输;③六段闭环误差 p50 <0.1%,残差被指认
   到 proxy 内部解析段。再加打 patch 前后 TTFT 噪声内一致的对照,排除"观测改变了被观测对象"。
3. **Q:16 行改动会不会把被测系统改慢了?**
   设了对照:打 patch 后 TTFT p50 218/727/2738 vs 未打 patch 的矩阵 219/719/2719(EXP-006/007),
   噪声内。实现侧也做了压制:惰性 `%s` 格式化、每请求只发一行、聚合只做整型累加。
4. **Q:0.27 GB/s 为什么远低于 EXP-002 的单向 D2D 0.60–0.91 GB/s?**
   访问模式不同。KV 通路是每 block 每层单发的 16 KiB 小拷贝,实测每 descriptor 61.5–65.8 µs(本文
   现算),约为 GPU 间裸延迟 14.5–15.9 µs 的 4 倍,固定开销主导;裸拷贝测试搬的是大块连续内存。
5. **Q:那要怎么把 KV 通路提上去?**
   按 §3.6 的判据,方向是**合并 descriptor**(层维度批量化、减少发起次数),不是换传输方向——方向
   已被 EXP-011 排除(8K TTFT −6.7%,量级不变)。本仓未做该改造,不外推收益。
6. **Q:MoE 的 fused_moe 占 56.4%,为什么调完 config 只快 1%?**
   折算式 $\Delta_\mathrm{e2e} \approx \Delta_\mathrm{kernel} \times$ 占比:kernel 端 serving 相关
   M 档改善 3.3–3.9%,乘 56.4% 得约 2% 的上限,实测 TPOT +0.8~1.2%,同量级。**但该幅度低于跨会话
   漂移(±5~8%),按措辞约定不作 headline,主证据是 kernel A/B。**
7. **Q:为什么 M=1 反而是收益最大的一档(−8.5%)?**
   M=1 是最极端的形状:每个专家只分到极少 token,默认启发式的粗阶梯
   (fused_moe.py:1371-1395:`block_m` 四档、`group_m` 只在 `tokens_per_expert > 128` 时才开)
   在这里分辨率最不够;搜索出的 tuple(BLOCK_SIZE_M=16 / N=64 / K=64 / GROUP_SIZE_M=32 /
   warps=4 / stages=4)与启发式给的不同,因而有空间。
8. **Q:120 passed 里有 120 skipped,是不是一半没测?为什么最后判"不做 kernel 改动"?**
   skipped 是异平台/异 dtype 的参数化(本机 Ada、BF16),不是被跳过的失败;correctness 的
   作用是**证伪"config 改动改变了数值结果"**,不是覆盖率声明。D3 的判定同样由数据给出:
   中段 M 的 tuned 与 default 打平,说明 Triton tile 空间在该形状已被启发式覆盖,config 就是
   最优杠杆(EXP-015 §6)。**负结论照常报告**——这比硬做一个没有数据支撑的"优化"诚实。
9. **压力问 Q:闭环误差 p50 <0.1% 会不会是自证循环——六段都由你自己定义,当然加得起来?**
   诚实答:部分是。六段中有五段的端点来自自己插的打点,望远镜求和在**代数上**必然只剩
   $t_\mathrm{p\_send}-t_\mathrm{recv}$ 一项,所以闭环本身**不能**证明任何一段的数值正确;
   它只能证明两件事:①没有整段被漏掉或重复计入;②TTFT(client 侧独立测量,与打点体系无关)
   与这套打点体系一致。真正给 kv_wait 定性的是**链 2**(与 NIXL 自报 xferDuration 独立吻合)
   与**链 1**(与 Prometheus counter 独立吻合)——这两条的另一端都不由我定义。这也是为什么
   必须是三条正交链,而不是"闭环误差很小"一条。
10. **压力问 Q:54–64% 换一台有 NVLink 的机器还剩多少?你的方法还能用吗?**
    数字不能外推,方法能——但要付一次代价。数字上:占比 ≈ $t_\mathrm{xfer}$ / TTFT,而
    $t_\mathrm{xfer} \propto 1/BW_\mathrm{eff}$;换到高速互联占比会塌到个位数甚至更低,PD 的收益项
    (消除干扰、独立扩缩)才有机会占上风——**本仓测的是该形态的下界条件,不是对形态的否定**
    (REPORT §2.4)。方法上:身份链(X-Request-Id 贯穿)与三重互证原样可用,**时钟域不行**——跨
    节点后 epoch 不再同源,必须先解决时钟同步,否则 0.1% 量级的闭环无从谈起;另外本仓只测了
    1P1D、并发 1、pull 路径,xPyD 下还要多一层"P/D 配对与排队"的段,六段分解要重新设计。

## 8. 工业对照与延伸

- **可观测性这一层**:上游有同方向的 draft PR #52859(NVIDIA,NIXL push/pull lifecycle
  tracing),本仓查重后按 fail-closed 原则把 EXT-1 定位为**本地测量 patch,不投上游**
  (`pd_disagg/ext1/DEDUP.md`)。生产系统会把这类关联做成一等公民(OpenTelemetry span:
  proxy → P → D → NIXL transfer 一条 trace),而不是一行 `logger.info` + 离线正则 join;
  代价是采样与传播开销,收益是跨节点天然带时钟同步语义。
- **身份这一层**:本仓靠 client 自定 `X-Request-Id` 贯穿三方(`ext1_proxy.py:165`、
  `serving.py:117 _base_request_id` 从 header 取 id);生产网关会强制注入 trace id 并逐跳透传。
  vLLM 的 NIXL 已把身份三层正交拆开(引擎身份 / 会话身份 / 内存寻址,theory/02 §2),这正是它对
  0.17.1 P2pNccl 那种隐式 rendezvous key 分叉免疫的原因(EXP-012 实机复现)。
- **KV 通路这一层**:生产 PD 分离跑在 NVLink / IB / RDMA NIC 上,KV 走 GPUDirect RDMA,带宽两个
  数量级于本机;descriptor 碎片化在那里被大得多的链路带宽掩盖,在本机则成了主导项——**同一份
  connector 代码,在不同互联上暴露的瓶颈完全不同。**
- **MoE config 这一层**:上游 config 是 exact-match 文件名查找
  (`E=<专家数>,N=<中间维>,device_name=<GPU>.json`),没有插值、没有邻近回退——所以"社区
  空缺"是一个**离散**的事实:文件在或不在。本仓补的两个 tuple(E=30,N=1408 / E=60,N=704)
  就是把两个具体格子填上,材料齐备但**未提交**。

延伸阅读(源码/文档锚):

1. `pd_disagg/ext1/nixl_req_telemetry_v0251.patch` 全文(4 处改动,逐行 `# EXT1`)+
   `ext1/orig/` 原件——自己 diff 一遍,确认"~16 行"这个定性。
2. `pd_disagg/analysis/nixl_token_accounting.md` 机理链——D 端从 `get_computed_blocks` 到
   `_apply_prefix_caching`(base_worker.py:2165-2189)的完整记账链,§3.5 链 1"零缺口"结论的对照面。
3. `docs/theory/02_pd_kv_path.md` §2/§5——NIXL 控制面/数据面速查(pull_scheduler.py:265-275、
   metadata.py:152-158、pull_worker.py:101-178)。
4. `vllm/model_executor/layers/fused_moe/fused_moe.py:1366-1412`(默认启发式全文)与 `:1145-1166`
   (config 文件查找与 `triton_version` 弹出)——§5.3"为什么两端有收益、中段打平"的源码依据。
5. `moe_perf/PR_DRAFT.md`——交付材料的六件套对照(含 hardening 项:补 ≥3 轮交叉 kernel A/B、
   rebase 后重验、e2e 只作 supporting)。
