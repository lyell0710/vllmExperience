# provenance: tool=ncu-nsys-analysis src="/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32_graphlevel.nsys-rep" date=2026-09-07T15:45Z gpu="2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU)" ncu=2026.1.0.0 nsys=2025.6.3.343 analyzed_on="cpod-1u8o0xv30sr6(2x RTX 4090)"

# vLLM MoE bs32 timeline 性能诊断(nsys graph 级 trace,占比表降级「参考」)

## 一、结论摘要

本文件经 META_DATA_CAPTURE 实查为 `--cuda-graph-trace` = **Graph** 模式:graph 内 kernel 全部不可见(kernel 表仅 41,858 行,同负载 node 版有 1,345,950 行),kernel 占比表整体降级「参考」,kernel 级结论以同组 node 版报告为准——这是本报告唯一的 P0。本文件的独立价值:graph 执行画像(每卡 890 次重放、128 张不同 graph、均值 13.41 ms/次,graph 段占真实 busy 的 91.4%)与 eager 段视角(logits AllGather 每次 603.0 us、合计 846.66 ms,约占真实 busy 3.2%,推断)。真实 GPU busy(含 graph)99.72%/99.77%,idle 近零;另注意 GPU 活动只覆盖 trace 后 13.09 s,头部 7.04 s 无任何 GPU 事件(bench 尚未打入,duration 口径 busy 只有 64.5%,两口径并报)。

| 项 | 值 |
|---|---|
| 被分析文件 | `/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32_graphlevel.nsys-rep`(sha256 前 12 位 `b843fecd2bc9`) |
| 采集环境 | 2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU);`--cuda-graph-trace` 模式 = **Graph,非 node**(META_DATA_CAPTURE 实查)→ 占比表降级「参考」;驱动 610.57.04(同目录 d1_sweep_manifest.txt,一句话引用) |
| 分析工具 | ncu 2026.1.0.0(本组无 ncu 文件,未使用)/ nsys 2025.6.3.343 / sqlite3 3.50.2(均现场 --version) |
| 硬件基线 | 128 SM / DRAM 峰值 1008.096 GB/s / L2 72 MiB(rep 导出 sqlite 的 TARGET_INFO_GPU,两卡同值) |

被测负载与 node 版相同(META_DATA_CAPTURE PROCESS_0:`vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --tensor-parallel-size 2 --enable-expert-parallel ...`;bench 并发 32,组内已核实)。

## 二、指标解读

下表时长数值均为 profiler 环境,仅作相对比较;标「参考」的行受 Graph 模式口径限制。

| 指标 | 数值 | 含义与影响 | 建议动作 | 来源 |
|---|---:|---|---|---|
| 采集口径 | CUDA_GRAPH_TRACE_OPTIONS:MODE = `Graph` | graph 内 kernel 不可见:kernel 表只剩 eager 段(41,858 行/45 种 vs node 版 1,345,950 行/50 种),「kernel 占比」只对 eager 段成立 | kernel 级结论以 node 版为准(P0-1) | META_DATA_CAPTURE;KERNEL 表 COUNT |
| trace 总时长 / GPU 事件窗口 | 20.249 s / 13.089 s(t=7.036→20.125 s) | 头部 7.04 s 无任何 GPU 事件:profiler 先启动、bench 后打入;busy 两口径:GPU 窗口内 99.7%,duration 口径 64.5%(13.05/20.25 s),引用时必须写明口径 | P2 留档(采集对齐) | ANALYSIS_DETAILS.duration;KERNEL/GRAPH_TRACE MIN/MAX |
| 仅按 kernel 表算的 busy | 8.56%/8.53%(**伪影**) | graph 执行不产生 kernel 行,直接套公式会得出「空闲 91.4%」的假结论,必须并入 GRAPH_TRACE 区间 | 无需动作(方法论提醒) | intervals.py(仅 KERNEL+MEMCPY+MEMSET) |
| 真实 GPU busy / idle | dev0 99.72%/0.28%,dev1 99.77%/0.23%(GPU 窗口口径) | 并发 32 打满,>1 ms 空隙每卡仅 2 个(合计 3.9/3.0 ms):结构面无优化空间 | 无需动作 | intervals.py 扩展合并(含 CUPTI_ACTIVITY_KIND_GRAPH_TRACE) |
| graph 执行画像 | 每卡 890 次、**128 张不同 graph**、合计 11.932/11.943 s,均值 13.41 ms、max 18.39/18.40 ms | graph 段占真实 busy 91.4%(23.875/26.111 s 两卡合计);128 张 graph 对应 vLLM 多 batch-size capture 池,窗口内混合命中;本表是 graph 模式独有产出 | 无需动作 | CUPTI_ACTIVITY_KIND_GRAPH_TRACE 聚合 |
| eager 段 kernel 合计 | 2,228.9 ms(两卡),占真实 busy 8.5% | graph 外可见部分:prefill 段 GEMM/MoE(eager 跑)+ logits 通信与采样链 | 见 P1-1 | cuda_gpu_kern_sum 总和 |
| eager 段 Top kernel(参考) | AllGather 38.0%(846.66 ms,1,404 实例,603.0 us);cutlass relu GEMM 22.6%(503.22 ms,328.9 us);AllReduce 12.7%(283.69 ms,392 实例,723.7 us);fused_moe 6.9%(153.56 ms,384 实例,399.9 us,prefill 段) | 占比仅对 eager 段(真实 busy 的 8.5%)成立;对全局的换算见 P1-1/P2 | 见 P1-1 | cuda_gpu_kern_sum(参考);GRAPH_TRACE |
| cudaGraphLaunch | 1,780 次,均值 259.4 us | graph 模式更接近真实 CPU 开销;node 版同 API 均值 1,080.1 us(约 4.2 倍伪影),解读 node 版时引用 | 无需动作 | cuda_api_sum;对照 d1_nsys_moe_bs32 同表 |
| cudaEventSynchronize | 702 次共 12.625 s,占 CPU API 跨度 62.6% | serve 型 CPU 等 GPU 正常形态,与真实 busy 99.7% 互证 | 无需动作 | cuda_api_sum;RUNTIME 跨度 20.168 s |
| memcpy 画像 | H2D Pinned 3,378 条共 1.70 MB;D2H 2,808 条共 533 KB;D2D 192 条共 1.245 GB、0.83 ms、聚合 1,497.63 GB/s | D2D 等效带宽超 DRAM 峰值 1008.096 GB/s,说明这批拷贝(条均 6.5 MB,L2 72 MiB 装得下)主要在 L2 内完成,数据不落显存——按算账规则这是健康信号不是异常;node 版的 44,978 条 D2D 大部分在 graph 内,此处不可见 | 无需动作 | memcpy_by_kind SQL;TARGET_INFO_GPU 基线 |
| 拷贝-计算重叠率 | dev0 14.49%,dev1 14.89% | memcpy 时间占比 <0.1%,按判读表记录即可 | 无需动作 | intervals.py 扫描线 |
| NVTX 阶段 | `execute_context_0(0)_generation_32(32)` 1,370 个 range 共 1,565.9 ms;另有 512–2,043 token 的 prefill context range | 约 685 步/卡的 32 并发 decode 与穿插 prefill,与 graph 重放 890 次/卡同量级互证 | 无需动作 | nvtx_pushpop_sum |

**数据缺口**(缺失 ≠ 实测为 0):

- graph 内 kernel 分解:本文件不可得(Graph 模式采集所致);同负载 node 版 `d1_nsys_moe_bs32.nsys-rep` 已补齐,无需重采。
- OS 运行时系统调用维度:未采到(osrt_sum `SKIPPED`,0 字节 csv);补采需 `-t cuda,nvtx,osrt`。
- 显存占用曲线:未采到(无 CUDA_GPU_MEMORY_USAGE_EVENTS 表);补采需 `--cuda-memory-usage=true`。
- kernel 级根因:本组无 ncu,ES 不可得。

## 三、问题清单(按优先级)

本次分档口径:占比表因 Graph 模式**降级「参考」**;kernel 级维度按退化规则 B 且仅对 eager 段成立;T 取值来源:cuda_gpu_kern_sum Time%(仅 graph 外 kernel)+ GRAPH_TRACE 合成的真实 busy(推断口径)。

### 🔴 P0(采集口径)

- **P0-1 非 node 模式采集,kernel 占比表不可作数**:MODE = `Graph`(META_DATA_CAPTURE 实查),graph 内 kernel/memcpy 全部不可见(kernel 行数 41,858 vs node 版 1,345,950)。按硬约束将「node 模式重采」列为 P0 动作——该动作已由同组 `d1_nsys_moe_bs32.nsys-rep`(实查 MODE=Node)满足,故落地动作为:kernel 级占比与热点结论一律以 `d1_nsys_moe_bs32_analysis.md` 为准(该版给出 fused_moe_kernel T=56.4% 的权威口径),本文件仅贡献 graph 粒度与 eager 段视角。G 不适用(口径问题,非时间收益)。

### 🟡 P1(3% ≤ G < 10%,推断口径)

- **P1-1 logits AllGather 的 eager 尾巴**:G ≈ 846.66 ms ÷ 26,111 ms(真实 busy,两卡)= **3.2%**(推断合成口径:分子来自 cuda_gpu_kern_sum,分母含 GRAPH_TRACE);加上 eager 段 AllReduce 283.69 ms 后通信 eager 合计约 4.3%。每次 603.0 us(1,404 实例)对应词表 logits 的 TP 聚合;在 node 版里它并入通信合计 15.0%。修复方向:并入 node 版 P1-1 的通信专项一起看(与 dense_tp2 对照、重叠余地),不单独立项;验证方式:通信优化后复采,eager AllGather 均值/占比下降。

### 🟢 P2(G < 3% 或纯记录)

- 采集窗口头部 7.04 s 无 GPU 事件(KERNEL/GRAPH_TRACE MIN(start)=7.036 s):不影响占比结论,但稀释 duration 口径 busy 至 64.5%;下次采集用 `--capture-range` 或延迟启动对齐 bench 起点,省 1/3 无效 trace 体积。
- eager 段 cutlass relu GEMM 503.22 ms(对真实 busy 约 1.9%)、prefill fused_moe 153.56 ms(约 0.6%)(cuda_gpu_kern_sum,参考):prefill 相关,权威占比见 node 版,留档。
- node 级采集把 cudaGraphLaunch CPU 均值从 259.4 us 抬到 1,080.1 us(约 4.2 倍,cuda_api_sum 两文件对照):伪影提醒,非本负载问题。
- D2D 等效带宽 1,497.63 GB/s 超 DRAM 峰值:L2 命中的健康信号,记录即可(memcpy_by_kind + TARGET_INFO_GPU)。

## 四、下一步行动

1. **kernel 级结论切换到 node 版**(P0-1):以 `d1_nsys_moe_bs32_analysis.md` 为 bs32 权威占比表(fused_moe_kernel 56.4% 为该版第一发现);本文件不再补采。
2. **通信 eager 尾巴并入通信专项**(P1-1 依据:G ≈ 3.2%,推断):与 node 版 15.0% 通信合计、bs1 版 13.8% 一并对照 dense_tp2,决定是否立项。
3. 下次同类采集对齐 bench 起点(P2 第 1 条),并在需要 graph 内分解时直接用 node 模式(两种模式各采一份的双轨做法本组已验证有效,保留)。

## 附录:复现命令

```bash
# 0) 版本与哈希
/usr/local/cuda/bin/nsys --version; /usr/local/cuda/bin/ncu --version; sqlite3 --version
sha256sum /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32_graphlevel.nsys-rep   # b843fecd2bc9...

# 1) 只读复制 + 导出(SCRATCH=/tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014)
cp /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32_graphlevel.nsys-rep "$SCRATCH/"
/usr/local/cuda/bin/nsys stats "$SCRATCH/d1_nsys_moe_bs32_graphlevel.nsys-rep" --sqlite "$SCRATCH/d1_nsys_moe_bs32_graphlevel.sqlite" \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,osrt_sum \
  --format csv --output "$SCRATCH/d1_nsys_moe_bs32_graphlevel" --force-overwrite=true 2>&1 | grep -E 'Exportation error|ERROR:|SKIPPED'
SQ="$SCRATCH/d1_nsys_moe_bs32_graphlevel.sqlite"

# 2) 采集验尸(P0 直接证据)+ 基线 + 规模 + 窗口
sqlite3 "$SQ" "SELECT value FROM META_DATA_CAPTURE WHERE name='CUDA_GRAPH_TRACE_OPTIONS:MODE'"          # Graph
sqlite3 -header "$SQ" "SELECT name,memoryBandwidth,l2CacheSize,smCount FROM TARGET_INFO_GPU"
sqlite3 "$SQ" "SELECT duration FROM ANALYSIS_DETAILS"                                                    # 20249151116
sqlite3 "$SQ" "SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"                               # 7036263568|20125376856
sqlite3 -header -csv "$SQ" "SELECT COUNT(*),COUNT(DISTINCT shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL"   # 41858,45

# 3) graph 执行画像(graph 模式独有表)
sqlite3 -header -csv "$SQ" "SELECT deviceId,COUNT(*),COUNT(DISTINCT graphId),SUM(end-start),ROUND(AVG(end-start)/1e3,1),ROUND(MIN(end-start)/1e3,1),ROUND(MAX(end-start)/1e3,1) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE GROUP BY deviceId"
# 890,128,11931929496,13406.7,110.8,18390.7 / 890,128,11943241584,13419.4,115.2,18401.2

# 4) 真实 busy/idle:python 扫描线(KERNEL+MEMCPY+MEMSET+GRAPH_TRACE 区间合并;kernel-only 口径 8.56% 是伪影)
python3 /tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014/intervals.py
# 扩展合并脚本与 bs1_graphlevel 版附录同款,替换 sqlite 文件名即可;输出 busy 99.72/99.77

# 5) eager 段占比 / API 对照 / memcpy(python csv 模块 + SQL)
python3 -c "import csv;rows=list(csv.DictReader(open('$SCRATCH/d1_nsys_moe_bs32_graphlevel_cuda_gpu_kern_sum.csv')));print(sum(int(r['Total Time (ns)']) for r in rows));[print(r['Time (%)'],r['Total Time (ns)'],r['Instances'],r['Avg (ns)'],r['Name'][:60]) for r in rows[:8]]"
sqlite3 -header -csv "$SQ" "SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(AVG(r.end-r.start),1) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY 1"   # 1780, avg 259413.7ns
sqlite3 -header -csv "$SQ" "SELECT ck.label,sk.label,dk.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start),ROUND(1.0*SUM(m.bytes)/SUM(m.end-m.start),2) FROM CUPTI_ACTIVITY_KIND_MEMCPY m LEFT JOIN ENUM_CUDA_MEMCPY_OPER ck ON ck.id=m.copyKind LEFT JOIN ENUM_CUDA_MEM_KIND sk ON sk.id=m.srcKind LEFT JOIN ENUM_CUDA_MEM_KIND dk ON dk.id=m.dstKind GROUP BY 1,2,3 ORDER BY 6 DESC"
```
