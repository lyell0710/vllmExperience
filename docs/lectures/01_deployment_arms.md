---
status: complete
关联EXP: EXP-002, EXP-004, EXP-005, EXP-006, EXP-007, EXP-008
配套: docs/theory/02_pd_kv_path.md(速查版) · pd_disagg/REPORT.md(结论版) · 本文=逐段走读版
---

# 深度讲义 01 · 两张消费级 4090 该怎么用:四臂形态的资源账、互联墙与功率帽探案

## 1. 这一篇回答什么问题

只有两张 RTX 4090(无 NVLink、P2P 被驱动禁用),单卡混部、双副本、张量并行 TP2、
Prefill-Decode 分离四种形态该选哪个,以及为什么在这台机器上答案是唯一的。读完你应当能:
手推 decode 的带宽 roofline 和 TP2 的收益/代价账(算式到 ms 级);解释 1.78 GB/s、
0.26–0.27 GB/s、22.7 vs 0.6 GB/s 这三组互联数字各自代表哪条路径、怎么测出来;
面对"你的 SLO 阈值是不是挑出来的""replica2 凭什么不到 2× 也叫近线性"这类追问给出
带证据锚点的回答。

## 2. 直觉与第一性原理

**先想没有这个问题的世界。** 如果模型能塞进一张卡、且一张卡吞吐够用,部署没有选型问题:
一个进程,prefill 和 decode 混跑,这就是本仓的 colocate 基线(EXP-004)。选型问题诞生于
"多出一张卡"——多出来的算力、显存、带宽要通过某种**组织方式**变成吞吐或延迟,而每种
组织方式都要付一种代价。

**日常类比:两个厨师开餐馆。**
- replica2 = 开两家一模一样的店:菜单、灶台全复制,互不说话,客流对半分;
- tp2 = 两人同炒每一道菜:你切一半我切一半,但每道菜出锅前必须把两人的半成品合到一起;
- pd1p1d = 一人只备菜、一人只掌勺:备好的菜要整盘从一号灶端到二号灶。

类比的失效点必须点破:厨师之间"传菜"几乎免费,而 GPU 之间传数据在本机要走一条被禁用了
直连(P2P)的 PCIe 路径——**合菜(allreduce)与端菜(KV 传输)的成本在这台机器上不是
二阶小量,而是主项**。这是全篇的第一性原理:**部署形态的本质是"用通信换组织",通信有多贵,
形态就有多少自由度。**

三本资源账(每臂都要各记一遍,详细算式在 §3):

| 臂 | 显存(权重) | 算力组织 | 跨卡通信 |
|---|---|---|---|
| colocate | 1 卡 × 全量 14.2 GB | prefill/decode 混跑互相干扰 | 0 |
| replica2 | 2 卡 × 全量(复制) | 两条独立流水线 | **0** |
| tp2 | 每卡 1/2(7.1 GB) | 每个算子切一半、逐层合并 | 每层 allreduce |
| pd1p1d | 2 卡 × 全量(P、D 各一份) | 阶段专业化(P 计算受限/D 带宽受限) | 每请求整份 KV |

一眼可见:四臂中只有 tp2 和 pd1p1d 把跨卡通信放进了关键路径。所以在互联受限的平台上,
**先测互联,再谈形态**——这就是本仓把 EXP-002(硬件三数)放在一切实验之前的原因。

## 3. 完整推导/机制

### 3.1 互联三数:三条路径,三个数字,不可互换

- **P2P 能力**:`nvidia-smi topo -p2p r` 返回 GNS(GPU not supported)、
  p2pBandwidthLatencyTest 的 connectivity matrix 全 0(`pd_disagg/hw/topo.txt`、
  `hw/p2p_bandwidth_latency.txt`,EXP-002)——GeForce 驱动层禁用,不是拓扑问题。
- **裸拷贝路径**:单向 D2D 0.60–0.91 GB/s,双向 22.6–22.8 GB/s,GPU 间延迟
  14.5–15.9 µs,卡内 memcpy ~924 GB/s(同上)。单双向差约 25 倍是"无 P2P"的定量指纹:
  无 P2P 时 cudaMemcpyPeer 退化为经主机内存的分段中转,单向吃满中转开销;双向两个方向
  的分段流水互相填空,逼近 Gen4 x16 的双向极限。
- **collective 路径**:nccl-tests `all_reduce_perf -b 1M -e 512M -f 2 -g 2`,
  avg bus bw **1.78 GB/s**,256 MB 以上大消息 ~1.85 GB/s(`hw/all_reduce_perf.txt`,
  EXP-002)。NCCL 探测不到 P2P 后回退 SHM(共享内存中转)传输。
- **KV 通路**:NIXL 的有效吞吐 0.26–0.27 GB/s(telemetry-derived,EXP-006/007)。
  注意这是第三条路径,既不等于裸拷贝也不等于 collective——为什么它最慢,是讲义 02 的主题。

### 3.2 decode 为什么由带宽定价:TPOT 下限的推导

一步一理由:

1. decode 每生成 1 个 token 要做一次完整前向。——自回归定义:第 t+1 个 token 依赖前
   t 个 token 的 KV 与全部层权重,绕不开。
2. bs=1 时一次前向必须把全部权重从显存读一遍,且读进来的每个权重只做约 2 次浮点运算
   (乘、加)。——算术强度 ≈ 1 FLOP/byte,远低于 GPU 计算/带宽平衡点,处在 roofline
   的内存受限侧;缓存放不下 14.2 GB,复用可忽略。
3. 因此 $\mathrm{TPOT}_{\min} = \dfrac{W_{\text{bytes}}}{BW}$。——时间由搬运字节数
   除以带宽给出下界,这是内存受限段 roofline 的直接读法。
4. 代入本机实测:$14.2\,\mathrm{GB} / 924\,\mathrm{GB/s} \approx 15.4\,\mathrm{ms}$。
   ——14.2 GB 为 Qwen2-7B BF16 权重(仓内口径,`moe_perf/d1_analyze.py` roofline 注释);
   924 GB/s 用本机实测卡内 memcpy(EXP-002)而非标称 1008 GB/s,因为实测更贴近可达值
   (d1_analyze.py 用标称值,两口径都在仓内,引用时须注明用的哪个)。
5. 实测 TPOT p50 = 15.87–16.35 ms(EXP-004),达成率约 94–97%。——decode 确为权重
   带宽受限,模型的其余一切(算子效率、调度)只在这 ~5% 余量里活动。

### 3.3 TP2 的收益与代价:一半权重 + 一份通信税

**decode 侧(收益成立)**:每卡只持一半权重,步骤 3 的 $W$ 减半:
$7.1/0.924 \approx 7.7\,\mathrm{ms}$;再加每 token 的小消息 allreduce 实测代价
~1.3 ms(EXP-005 §6),合计 ≈ 9.0 ms;实测 9.26–9.48 ms(EXP-005/007)。账能闭合:
**-42% 的 decode 提速 = 权重带宽分摊 − 通信税**。

**prefill 侧(收益归零)**:prefill 是计算受限(8192 token 一批,GEMM 算术强度高),
计算减半本应省约一半时间;但每层输出要做一次大消息 allreduce:
$8192 \times 3584 \times 2\,\mathrm{B} = 58.7\,\mathrm{MB}$(hidden=3584,BF16),
28 层合计 ~1.64 GB 通信量。若按大消息实测 1.85 GB/s 完全串行传输要 ~0.89 s——已超过
实测 TTFT 693.7 ms(EXP-005),说明真实执行存在计算-通信重叠/分块调度,该算式只能做
**量级判断**(推断,不是逐毫秒预测):通信代价与计算减半的收益同量级,相互抵消。实测锚点:
tp2 8K TTFT 693.7 ms ≈ 单卡冷态 ~700 ms(EXP-005),**零加速**。另注:Megatron 式 TP
每层前向通常有注意力出投影、MLP 下投影两次 allreduce,仓内按每层一次做下界计数
(EXP-005 §6 "28 层 × 58.7MB"),取哪个计数不改变量级结论。

**汇总到吞吐**:饱和吞吐 tp2 相对 colocate 只有 +13~19%(512/2K/8K 桶:12.31/10.36、
4.16/3.63、1.02/0.90,EXP-007)——decode 的 -42% 在批量化后被稀释(大 batch 下 decode
逐渐转向计算/调度约束),prefill 的 allreduce 墙成为主导。

### 3.4 replica2:零通信的复制,近线性的扩展

不切模型、不传 KV,唯一代价是权重显存翻倍(两卡各持 14.2 GB)与外置轮询代理
(`pd_disagg/matrix/rr_proxy.py`)。因果链:零跨卡流量 → 互联质量与它无关 → 扩展效率
只受负载均衡与客户端限制。实测 2K/8K 桶扩展 1.93×/1.98×(7.00/3.63、1.78/0.90,
EXP-007);512 桶 1.50×(15.58/10.36)带欠饱和疑点(EXP-007 §7:SAT_CONC=64 或代理
上限,引用 512 扩展效率前须复测)。

### 3.5 pd1p1d:每请求一份 KV 的搬运账

Qwen2-7B 的 KV 每 token 字节数:
$28\,\text{层} \times 2\,(\mathrm{K,V}) \times 4\,\text{KV头} \times 128\,\text{维}
\times 2\,\mathrm{B} = 57344\,\mathrm{B}$——与 EXP-006 单请求探针 bytes=917504
= 16 token × 57344 B 完全吻合(block=16 取整)。8K 请求全量 KV ≈ 469.8 MB(EXP-011
push 臂实测全量);pull 臂经前缀缓存裁剪实拉 439.7 MB(EXP-006)。在 0.27 GB/s 的
有效吞吐下:$439.7\,\mathrm{MB} / 0.27\,\mathrm{GB/s} \approx 1.63\,\mathrm{s}$,
与实测 avg xfer 1602.7 ms(EXP-006)对上。容量上限:
$0.27\,\mathrm{GB/s} \div 470\,\mathrm{MB/req} \approx 0.57\,\mathrm{req/s}$,
实测饱和 0.54 req/s@8K(EXP-007)——**传输带宽即容量**,账在两端都闭合。

## 4. 代码逐段走读

单测量点的完整执行路径 = `run_point.sh`(起测)→ `vllm bench serve`(打点)→
`collect_point.py`(落账)。按执行顺序读六段。

**段 1:快照与遥测先行(`pd_disagg/scripts/run_point.sh:20-33`)**

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

角色:一次测量的三件套(before 快照、GPU 工况流水、统一 UTC 前缀)在 bench 之前就位。
为什么这么写:快照抓的是**引擎端口**(`ENGINE_PORTS` 与 bench 打的端口分离,pd 臂 bench
打代理、快照仍直抓 8100/8200)——传输是否真实发生要看引擎侧计数器增量,代理不可信;
遥测采样是功率帽事件(§5.4)之后加装的制度化产物,注释里写着动机。改错会怎样:若快照抓
代理端口,PD 臂 gate 全部拿不到 nixl 计数器,整点作废;若去掉遥测,臂间差异可能被功率
状态淹没且无法事后自证工况。

**段 2:三种打法与唯一 seed(`run_point.sh:35-52`)**

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

角色:同一个入口跑三种模式——attribution(并发 1,量"干净"的延迟基线)、saturation
(无限速率高并发,探最大吞吐,给 sweep 档位定标尺)、sweep(固定 offered rate,量
负载-goodput 曲线)。为什么这么写:`--save-detailed` 落每请求的 TTFT/ITL 数组,goodput
必须逐请求判定(段 5);`SEED` 从外部注入,协议 v2 下每个测量点唯一(§5.2 讲为什么);
`--ignore-eos` 保证输出长度恒定,否则各点输出长不可比。改错会怎样:漏 `--save-detailed`
则 goodput 无法计算;seed 写死 42 会踩前缀缓存污染(仓内实测 2048 桶 25% 命中虚高,
`pd_disagg/analysis/nixl_token_accounting.md`)。

**段 3:精确指标名取增量(`pd_disagg/scripts/collect_point.py:29-41`)**

```python
def exact_delta(deltas, metric_name, label_sub=None):
    """按精确指标名(跨引擎端口求和)取增量。名字来源: 2026-08-21 PD 探针实测,
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

角色:把 before/after 两份 Prometheus 快照的差值按**精确指标名**归并(P、D 两端口求和)。
为什么这么写:指标名全部来自 EXP-006 的探针实测(快照→单请求→快照→diff),而不是猜
子串——`_created` 系列是时间戳伪装成计数器,子串匹配会把它当增量收进来;传输计数只在
D 端(pull 语义,READ 发起方),跨端口求和才对。改错会怎样:早期版本用子串猜名,遇上
`pd1p1d_push` 臂(计数器移到 P 端)与 `_created` 就会算出错账。

**段 4:gate——数字与它的合格证同行(`collect_point.py:73-104`)**

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

角色:每个测量点的"合格证"。PD 臂要过五关:零失败请求、传输字节确有增量、**成功传输数
恰等于完成请求数**、零失败传输/通知、零过期。为什么这么写:PD 臂最阴险的失效模式是
"服务器照常返回、KV 其实没传"(fail policy 不设 fail 时静默回退本地重算)——只有引擎侧
计数器能拆穿;`startswith` 而非 `==` 是 EXP-011 的教训:push 臂名 `pd1p1d_push` 被
精确匹配漏掉,导致该臂两行的结构化 gate 字段为 None(原始计数幸存于 kv_deltas_raw,
EXP-011 §4 如实登记)。改错会怎样:没有 gate,PD 臂的"好看数字"可能根本没走传输路径。

**段 5:goodput 的定义(`collect_point.py:106-116`)**

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

角色:goodput = **同时满足 TTFT 与 TPOT 两条 SLO 的请求数 ÷ 墙钟时长**(req/s)。
为什么这么写:必须逐请求判定(detailed 数组),聚合分位数(如 p99≤SLO)只能给"整体
过/不过"的布尔量,画不出连续的 goodput 曲线;TPOT 用该请求 ITL 均值重算,不依赖 bench
的聚合口径。SLO 数值:TTFT ≤ 5× 无负载基线(512/2K/8K = 328/891/4626 ms)+
TPOT ≤ 50 ms,基线来自 EXP-004 并预注册锁定。改错会怎样:用均值 TTFT 判定会把"半数
请求超时"的点算成满分——goodput 曲线在过载段的陡降(fig1)正是逐请求判定才画得出来。

**段 6:遥测汇总——工况证据入行(`collect_point.py:138-149`)**

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

角色:把 2 s 采样流水压缩成每卡摘要(负载态最低/平均 SM 频率、峰值功率、见过的节流
原因位)随行写进 runs.jsonl。为什么这么写:`p_ > 100`(W)过滤空闲样本——空闲频率
210 MHz 会把均值拉没意义;节流原因保留原始位掩码集合,`0x4`(SW Power Cap)出现即
该点带功率帽工况。改错会怎样:不过滤空闲样本,"降频"永远存在;丢掉 reasons,就无法区分
热节流与功率帽(§5.4 的定案证据正是 reason 位)。

## 5. 实验数据怎么读

### 5.1 硬件三数原始文件怎么读

`hw/p2p_bandwidth_latency.txt` 节选(EXP-002):

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

读法:对角线是卡内 memcpy(~924 GB/s,decode roofline 的分母);非对角线才是跨卡。
"P2P=Enabled" 矩阵在本机与 Disabled 几乎相同——使能请求被驱动拒绝,回退同一条中转路径,
这本身就是 P2P 禁用的证据之一(连同 CANNOT Access Peer 行与 topo 的 GNS)。注意文件头
的 NOTE:CUDA sample 不是精密基准,单向矩阵各单元分散(0.60/0.69/0.91/4.36),所以
仓内引用一律用区间 0.60–0.91,不挑单值(EXP-002 §7)。

`hw/all_reduce_perf.txt` 读法:看 busbw 列而非 algbw——busbw 是按 allreduce 通信量
归一的口径(2 卡时两者相等),1 MB 到 512 MB 消息稳定在 1.67–1.87 GB/s,avg 1.78。
**平坦的带宽曲线**说明 SHM 回退路径没有大消息红利,这预言了 TP2 prefill 的大消息
allreduce 无处可逃(§3.3)。

### 5.2 84 个测量点是怎么设计的

网格(EXP-007 §2,协议 v2 共 84 个通过 gate 的测量点,`results/b1_matrix/runs.jsonl`,
汇总表 `derived/sweep_summary.csv`):

- **三个输入桶** 512/2048/8192(输出统一 128):短请求(调度开销敏感)、中等(混合)、
  长上下文(prefill/传输压力),一桶一个故事,不混算。
- **每臂每桶**:attribution(并发 1)+ saturation(SAT_CONC=64 探顶)+ sweep 4 个
  公共档 {0.5, 0.75, 0.9, 1.05}×colocate 饱和(512:{5.2,7.8,9.3,10.9} 等)+ 若干
  贴近自身饱和的档位。公共档保证四臂在**同一 offered load** 下可比;自身档保证每臂的
  拐点都被覆盖(否则 replica2 的高容量段全是外推)。
- **num_prompts 按桶递减**(320/192/56):保证每点时长量级相当,长桶不至于跑一小时。
- **每点唯一 seed,跨臂同点位同 seed**:唯一 seed 防前缀缓存污染——bench 的随机数据集
  同 seed 跨运行 prompt 逐 token 相同、且短桶 prompt 是长桶的精确前缀,实测造成 2048 桶
  25% 缓存命中、8192 桶 8.6%(`analysis/nixl_token_accounting.md`);跨臂同 seed 则保证
  臂间对比的工作负载逐字节相同。这一对设计合起来就是"协议 v2"。
- **防坑设计还有**:colocate 单卡基线作为所有双卡臂的"反例臂"(任何双卡方案先回答
  "比一张卡好多少");失败点保留(gate_pass=false 行不删、不进图表);~3% 的客户端瞬断
  同 seed 重跑(EXP-007 §4)。

### 5.3 三张主表逐行读

**饱和吞吐(req/s,EXP-007 §5)**:

| 桶 | colocate(1卡) | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 10.36 | 15.58* | 12.31 | 7.84 |
| 2048 | 3.63 | 7.00 | 4.16 | 2.12 |
| 8192 | 0.90 | 1.78 | 1.02 | 0.54 |

读法:先横比(同桶四臂),再算比值:replica2/colocate = 1.50*/1.93/1.98;tp2/colocate
= 1.19/1.15/1.13(即 +13~19%);pd1p1d/colocate = 0.76/0.58/0.60——**PD 用两张卡跑不过
一张卡**。带 * 的 512 桶 replica2 有欠饱和疑点(EXP-007 §7),所以"近线性 2×"只引
2K/8K 的 1.93/1.98×,这是限定语,不许丢。

**goodput 峰值(rps@档位,EXP-007 §5)**:colocate 8.57@9.3 / 2.41@2.7 / 0.43@0.68;
replica2 12.75@14 / 4.96@5.3 / 0.90@0.95;tp2 10.18@11.8 / 2.51@2.7 / 0.60@0.81;
pd1p1d 1.59@5.2 / 0.16@1.8 / 0.11@0.45。读法:goodput 峰值总小于饱和吞吐(饱和点上
延迟已炸,SLO 达标率崩塌);pd 的 1.59@5.2 意味着在 66% 饱和度时 goodput 已经只剩零头
——传输延迟(512 桶 ~114 ms 起步)直接吃掉 328 ms SLO 的三分之一。

**v2 归因 TTFT(并发 1、同热工况,p50 ms,EXP-007 §5)**:

| 桶 | colocate | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 512 | 65.4 | 66.2 | 64.0 | 219.3 |
| 2048 | 224.9 | 220.7 | 219.8 | 718.6 |
| 8192 | 925.2 | 902.6 | 881.3 | 2718.7 |

读法:非 PD 三臂几乎无差异(prefill 无并行收益,§3.3);pd 的溢价 154/494/1793 ms
全部来自 KV 通路——它的因果拆解(54.2/62.5/64.2%,512/2K/8K,p50,request 级因果
占比)在讲义 02。图表版:`pd_disagg/figures/fig7_saturation_overview.png`(总览)、
`fig1_goodput_curves.png`(曲线族,x=offered load,y=goodput,虚实线区分臂)——看
曲线族先看**过峰后的下降段**,那是各臂的失效方式:replica2 缓降,pd 贴地。

### 5.4 功率帽探案:隐藏变量是怎么被抓住的

这是本仓方法论含金量最高的一段(EXP-004 §7、EXP-005,日记 §9),完整链条:

1. **异常出现**:replica2@8K 归因 TTFT 714.6 ms,比 colocate 的 925.2 快 30%——并发 1
   下双副本不该有任何优势,这违反 §2 的资源账。
2. **拆分布而不是信均值**:colocate 8K 的 32 个请求 TTFT 呈**双段**:前 ~8 个 702–739 ms,
   之后 897–951 ms;replica2 全部 697–733 ms。→ 假设改写为"单卡持续负载下发生了劣化"。
3. **对照排除**:diag-1/2 直连两张卡分别测(901.2 / 892.7 ms)——排除"卡间个体差异";
   两卡都到 ~900,说明劣化与卡无关、与**持续负载**有关。
4. **遥测定案**:diag-3 持续 prefill 负载 + 1.5 s × 40 轮采样:温度 40→63°C、功率
   427–443 W(帽 450 W)、SM 频率 2820 ↔ 2460–2535 MHz、节流原因 **0x4 = SW Power Cap**
   (`records/data/EXP-005_throttle_trace.csv`)。63°C 排除热节流,定案功率帽。
5. **机理闭合**:replica2 轮询把请求对半分,每卡 50% 占空比,间歇期回 boost——所以它
   "快";colocate 的 925 是冷态(~700)与稳态(~905)的混合。降频约 12%(2820→2475 MHz)
   对应 TTFT +30%(700→905 ms),不完全成比例,残差疑与瞬时 boost/显存时钟相关——
   EXP-005 §6 如实写"未深究,非主线",这也是可学的:异常解释到能定方法论决策就停,
   不为完整感编故事。
6. **方法论落地**:headline 一律以 sweep 为准(满负载下各臂同为持续态,公平);
   run_point.sh 加装遥测(§4 段 1);归因表引用必须带工况标注。

防坑清单:消费卡基准必须声明功率工况;对比实验必须同热工况;冷启数字与稳态数字分开报。

## 6. 误区与边界

1. **"多卡跑 TP 理所当然更快。"** 本机 tp2 的 prefill 零加速(§3.3)、吞吐 +13~19%、
   per-GPU goodput 为四臂最差之一(REPORT §2.3 成本口径)。TP2 只在两个场景成立:模型
   单卡放不下(被迫),或要压 TPOT 到单卡达不到的水平(9.3 vs 16 ms)且愿付吞吐代价。
2. **"replica2@8K 快 30%,双副本有隐藏加速。"** 仓内被证伪的假设原案(§5.4):真相是
   功率帽让单卡基线变慢了。教训:**对比实验里"变快"与"对照变慢"不可区分,除非工况入账**。
3. **"benchmark 固定 seed 才科学。"** 仓内被证伪的第二案:固定 seed 让 vLLM 前缀缓存
   跨运行命中(2048 桶 25%),prefill 工作量凭空少四分之一,TTFT 虚低。可复现性靠
   "每点唯一且记录在案的 seed"达成,不靠全局同一个 seed。
4. **"PD 分离是大厂标配,至少不会更差。"** 它的价值主张(消除 prefill 对 decode 干扰、
   P/D 独立扩缩)在高速互联集群成立;本机 0.27 GB/s KV 通路下全负载段溃败,且换传输方向
   (push)只挽回 6.7% TTFT@8K(EXP-011)——形态与互联能力错配,不是形态本身错
   (REPORT §2.4 的公平陈述)。
5. **"饱和吞吐高就是好。"** tp2 512 桶饱和 12.31 高于 colocate 的 10.36,但两者 goodput
   峰值 10.18 vs 8.57 的差距被两张卡的成本除回去就是负收益;服务质量要用带 SLO 的 goodput
   与 per-GPU 口径双重核算。

**适用边界**:单机 2×RTX 4090、P2P 驱动禁用、Qwen2-7B BF16、随机负载三桶、输出 128、
1P1D(非 xPyD)。NVLink/数据中心平台三条互联数字全变,结论不可直接外推——可外推的是
**方法**:先测互联三数,再按 §2 资源账预测形态排序,最后矩阵实测验证。

## 7. 连环追问

1. **Q:TP2 的 TPOT 为什么是 9.3 ms 而不是 16.35/2 ≈ 8.2 ms?**
   A:decode 每 token 还要付一次小消息 allreduce,实测代价 ~1.3 ms/token(EXP-005 §6);
   7.7(半权重下限)+1.3 ≈ 9.0,实测 9.26–9.48 ms,账闭合。
2. **Q:1.78 GB/s 是怎么测的?**
   A:nccl-tests `all_reduce_perf -b 1M -e 512M -f 2 -g 2`,NCCL 探测不到 P2P 回退
   SHM 传输,取 avg busbw(`hw/all_reduce_perf.txt`,EXP-002)。它只代表 collective
   路径,不能代表 KV 通路(那是 NIXL 的 0.26–0.27 GB/s,telemetry-derived)。
3. **Q:单向 0.6 GB/s 与双向 22.7 GB/s 差 25 倍怎么解释?**
   A:无 P2P 时 cudaMemcpyPeer 走经主机的分段中转,单向暴露全部中转开销;双向两个方向
   的分段互相流水,逼近 Gen4 x16 双向极限。这个 25 倍差本身就是"无 P2P"的指纹(EXP-002 §6)。
4. **Q:goodput 为什么逐请求判而不用 p99 卡线?**
   A:要画连续曲线必须数出每个点的达标请求数(collect_point.py:174-184);p99 卡线只给
   布尔结果,且会把"51% 请求超时"与"1% 超时"判成同一种失败。
5. **Q:为什么 colocate 是"反例臂"?**
   A:所有双卡形态必须先回答"比一张卡好多少"——pd1p1d 三桶全输给单卡(0.58–0.76×),
   没有单卡臂这一事实根本暴露不出来。
6. **Q:8K 桶四臂 prefill 几乎无差异(925/903/881),为什么?**
   A:同热工况下功率帽把所有持续 prefill 的臂整平到同一频率(EXP-007 §6);TP2 的计算
   减半又被 allreduce 吃掉。物理约束一致,软件形态就分不出高下。
7. **Q:PD 的饱和 0.54 req/s 怎么从第一性原理预测?**
   A:每请求要传 ~470 MB KV,通路 0.27 GB/s,上限 0.27/0.47 ≈ 0.57 req/s,实测 0.54
   (EXP-007 §6)。当传输是关键路径,容量=带宽/单请求传输量。
8. **Q:replica2 需要什么额外组件,它会成为瓶颈吗?**
   A:一个轮询代理(rr_proxy.py)。512 桶饱和 15.58 存在代理/客户端并发上限疑点
   (EXP-007 §7 如实登记),这正是"零通信"形态把瓶颈推到入口层的表现;2K/8K 未见此效应。
9. **Q(压力):SLO 是拿被前缀缓存污染的基线锁定的,你的 goodput 结论会不会整个翻掉?**
   A:承认瑕疵:2048 桶 SLO 名义 5×,按 v2 干净基线实为 3.96×(891/225,EXP-007 §7)。
   防御是敏感性检验:阈值扫 0.5–4×,四臂 goodput **排序**全程稳定(fig6,REPORT 附录 A)。
   所以"replica2 最优、pd 溃败"的排序结论稳健,任何 goodput **绝对值**都必须连同阈值
   一起引用——这是口径边界,不辩解。
10. **Q(压力):84 个点看起来多,但每点只有一次 bench,多轮呢?**
    A:诚实回答:sweep 每点单次(n=32~320 请求内含分布,报 p50/p99),跨点趋势由 12–18
    点的曲线形状互相约束;瞬断点做过同 seed 重跑(3/~90 次)。但"同点跨会话重复"只在
    MoE 线做过(那里暴露出 ±5~8% 会话漂移,EXP-015)。若要引用单点绝对值到 ±5% 精度,
    应按仓内规范补 3 轮取 mean/std;引用排序与量级结论则现有数据足够。

## 8. 工业对照与延伸

- **PD 分离的本源语境**:DistServe/Splitwise 一类系统与 vLLM 上游 disaggregated serving
  示例假设的是跨节点池化 + NVLink/InfiniBand/RDMA NIC;那里 KV 传输走 GPUDirect RDMA,
  带宽两个数量级于本机,"消除干扰 + 独立扩缩"的收益才能覆盖传输成本。本仓测出的是该
  形态的**下界条件**,不是对形态的否定(REPORT §2.4)。
- **replica2 的生产形态**:多副本 + 专业负载均衡(K8s service、SGLang router 的
  cache-aware 路由)。本仓 rr_proxy 是最小实现,不做会话亲和;上游 router 会用前缀
  缓存感知调度拿到本仓拿不到的额外命中收益。
- **TP 的生产语境**:NVLink 平台上 allreduce 带宽 ~两个数量级高,TP2 prefill 不再零
  加速;vLLM 的 custom allreduce 小消息路径也依赖 P2P,本机被禁用后只剩 NCCL SHM——
  同一份代码在不同互联上走的是不同分支。
- **延伸阅读**:
  1. `pd_disagg/REPORT.md` §1–§2——本讲义的结论版与全部图;
  2. `pd_disagg/analysis/nixl_token_accounting.md`——前缀缓存污染的逐块定罪
     (bench test 请求 511 块,file:line 级);
  3. `docs/theory/02_pd_kv_path.md` §2——NIXL 控制面/数据面机制速查(含
     pull_scheduler.py:265-275、base_worker.py:2165-2189 锚点);
  4. `pd_disagg/hw/` 三个原始文件——建议亲手重读一遍表格,对照 §5.1 的读法;
  5. EXP-008——同一 colocate 臂跨 vLLM 版本的 system-version comparison(512 桶饱和
     +45%,计算受限桶零差异):部署形态之外,版本也是一个"免费"变量。
