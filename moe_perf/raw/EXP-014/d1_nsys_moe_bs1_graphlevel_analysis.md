# provenance: tool=ncu-nsys-analysis src="/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1_graphlevel.nsys-rep" date=2026-09-07T15:45Z gpu="2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU)" ncu=2026.1.0.0 nsys=2025.6.3.343 analyzed_on="cpod-1u8o0xv30sr6(2x RTX 4090)"

# vLLM MoE bs1 decode timeline 性能诊断(nsys graph 级 trace,占比表降级「参考」)

## 一、结论摘要

本文件经 META_DATA_CAPTURE 实查为 `--cuda-graph-trace` = **Graph** 模式:graph 内 kernel 全部不可见(kernel 表仅 182,858 行,同负载 node 版有 5,177,228 行),kernel 占比表整体降级「参考」,kernel 级结论以同组 node 版报告为准——这是本报告唯一的 P0。本文件的独立价值在 graph 执行粒度:每卡 3,974 次 graph 重放、均值 3.789 ms/次,graph 段占真实 GPU busy 的 88.6%;graph 外 eager 段里 lm_head logits GEMV 每步 326.6 us、合计 2,470.23 ms,约占真实 busy 的 7.3%(推断),值得列入 ncu 采集清单。真实 GPU busy(kernel+graph 合并)98.54%/98.53%,结构面健康。

| 项 | 值 |
|---|---|
| 被分析文件 | `/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1_graphlevel.nsys-rep`(sha256 前 12 位 `82abb007d305`) |
| 采集环境 | 2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU);`--cuda-graph-trace` 模式 = **Graph,非 node**(META_DATA_CAPTURE 实查)→ 占比表降级「参考」;驱动 610.57.04(同目录 d1_sweep_manifest.txt,一句话引用) |
| 分析工具 | ncu 2026.1.0.0(本组无 ncu 文件,未使用)/ nsys 2025.6.3.343 / sqlite3 3.50.2(均现场 --version) |
| 硬件基线 | 128 SM / DRAM 峰值 1008.096 GB/s / L2 72 MiB(rep 导出 sqlite 的 TARGET_INFO_GPU,两卡同值) |

被测负载与 node 版相同(META_DATA_CAPTURE PROCESS_0:`vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --tensor-parallel-size 2 --enable-expert-parallel ...`;bench 并发 1,组内已核实)。

## 二、指标解读

下表时长数值均为 profiler 环境,仅作相对比较;标「参考」的行受 Graph 模式口径限制。

| 指标 | 数值 | 含义与影响 | 建议动作 | 来源 |
|---|---:|---|---|---|
| 采集口径 | CUDA_GRAPH_TRACE_OPTIONS:MODE = `Graph` | graph 内 kernel 不可见:kernel 表只剩 eager 段(182,858 行/20 种 vs node 版 5,177,228 行/39 种),任何「kernel 占比」只对 eager 段成立 | kernel 级结论以 node 版为准(P0-1) | META_DATA_CAPTURE;KERNEL 表 COUNT |
| trace 总时长 / GPU 事件跨度 | 20.212 s / 17.256 s | 首个 GPU 事件在 t=2.846 s;以下占比按 GPU 事件跨度口径 | 无需动作 | ANALYSIS_DETAILS.duration;KERNEL/GRAPH_TRACE MIN/MAX |
| 仅按 kernel 表算的 busy | 11.28%/11.27%(**伪影**) | graph 执行不产生 kernel 行,直接套 busy 公式会得出「GPU 空闲 88.7%」的假结论——graph 模式下必须并入 GRAPH_TRACE 区间 | 无需动作(方法论提醒) | intervals.py(仅 KERNEL+MEMCPY) |
| 真实 GPU busy / idle | dev0 98.54%/1.46%,dev1 98.53%/1.47% | kernel+memcpy+GRAPH_TRACE 区间合并后的口径:GPU 实际近乎打满,与 node 版 96.43%/96.46% 一致(node 版含更细空隙故略低) | 无需动作 | intervals.py 扩展合并(含 CUPTI_ACTIVITY_KIND_GRAPH_TRACE) |
| graph 执行画像 | 每卡 3,974 次、26 张不同 graph、合计 15.058 s/卡,均值 3.789 ms、min 110.8–118.1 us、max 8.76–9.04 ms | decode 每步一次 graph 重放,graph 段占真实 busy 的 88.6%(30.116 s / 34.007 s 两卡合计);本表是 graph 模式独有产出 | 无需动作 | CUPTI_ACTIVITY_KIND_GRAPH_TRACE 聚合 |
| eager 段 kernel 合计 | 3,862.2 ms(两卡),占真实 busy 11.4% | graph 外逐 kernel 可见:logits GEMV、AllGather、softmax、采样链 | 见 P1-1 | cuda_gpu_kern_sum 总和 |
| eager 段 Top kernel(参考) | gemvx 64.0%(2,470.23 ms,7,564 实例,KAvg 326.6 us);AllGather 11.6%(446.67 ms,59.05 us);softmax 9.0%(349.44 ms);TopP 采样 3.8%、RadixTopK 3.4% | 每 decode 步 eager 段约 0.49 ms/卡(1.93 s ÷ 3,974 步) vs graph 段 3.79 ms/卡,与 eager 占真实 busy 11.4% 互证;占比仅对 eager 段成立(参考) | 见 P1-1 | cuda_gpu_kern_sum(参考);GRAPH_TRACE |
| cudaGraphLaunch | 7,948 次,均值 263.1 us | graph 模式下的该值更接近真实 CPU 开销;node 版同 API 均值 1,110.1 us,说明 node 级采集把 launch CPU 开销抬高约 4.2 倍(伪影对照,分析 node 版时引用) | 无需动作 | cuda_api_sum;对照 d1_nsys_moe_bs1 同表 |
| cudaEventSynchronize | 3,782 次共 15.113 s,占 CPU API 跨度 75.16% | serve 型 CPU 等 GPU 正常形态,与真实 busy 98.5% 互证 | 无需动作 | cuda_api_sum;RUNTIME 跨度 20.108 s |
| 真实最大空隙 | 每卡 10 个 >1 ms,合计 71.9/75.4 ms,最大 10.95 ms | 与 node 版的 11 个 ~10 ms 步边界空隙同源同形态,占跨度 0.4%,无碍 | P2 留档 | intervals.py 扩展合并 |
| memcpy 画像 | H2D Pinned 15,666 条共 7.82 MB、20.17 ms;D2H 15,128 条共 90.8 KB、18.40 ms;**无 D2D 行** | node 版有 191,684 条 D2D——它们全部发生在 graph 内,graph 模式下连 memcpy 都不可见:这是「该维度部分不可见」的直接证据 | 无需动作 | memcpy_by_kind SQL 对照两文件 |
| 拷贝-计算重叠率 | dev0 16.78%,dev1 16.99% | memcpy 时间占比 ≤0.2%,按判读表记录即可 | 无需动作 | intervals.py 扫描线 |
| NVTX 阶段 | `execute_context_0(0)_generation_1(1)` 7,548 个 range 共 8,055.5 ms | 约 3,774 步/卡 decode 稳态,与 graph 重放次数 3,974 同量级互证 | 无需动作 | nvtx_pushpop_sum |

**数据缺口**(缺失 ≠ 实测为 0):

- graph 内 kernel 分解:本文件不可得(Graph 模式采集所致),不是负载没跑 kernel;同负载 node 版 `d1_nsys_moe_bs1.nsys-rep` 已补齐该维度,无需重采。
- OS 运行时系统调用维度:未采到(osrt_sum `SKIPPED`,0 字节 csv);补采需 `-t cuda,nvtx,osrt`。
- 显存占用曲线:未采到(无 CUDA_GPU_MEMORY_USAGE_EVENTS 表);补采需 `--cuda-memory-usage=true`。
- kernel 级根因:本组无 ncu,ES 不可得。

## 三、问题清单(按优先级)

本次分档口径:占比表因 Graph 模式**降级「参考」**;kernel 级维度按退化规则 B 且仅对 eager 段成立;T 取值来源:cuda_gpu_kern_sum Time%(仅 graph 外 kernel)+ GRAPH_TRACE 合成的真实 busy(推断口径)。

### 🔴 P0(采集口径)

- **P0-1 非 node 模式采集,kernel 占比表不可作数**:MODE = `Graph`(META_DATA_CAPTURE 实查),graph 内 kernel/memcpy 全部不可见(kernel 行数 182,858 vs node 版 5,177,228;D2D memcpy 0 条 vs 191,684 条)。按硬约束将「node 模式重采」列为 P0 动作——该动作已由同组 `d1_nsys_moe_bs1.nsys-rep`(实查 MODE=Node)满足,故落地动作为:kernel 级占比与热点结论一律以 `d1_nsys_moe_bs1_analysis.md` 为准,本文件仅贡献 graph 粒度与 eager 段视角。G 不适用(口径问题,非时间收益)。

### 🟡 P1(3% ≤ G < 10%,推断口径)

- **P1-1 lm_head logits GEMV 的 eager 尾巴每步 326.6 us**:G ≈ 2,470.23 ms ÷ 34,007 ms(真实 busy,两卡)= **7.3%**(推断合成口径:分子来自 cuda_gpu_kern_sum,分母含 GRAPH_TRACE)。它是 graph 外最大的单体 kernel(7,564 实例,KAvg 326.6 us,eager 段内占 64.0%,参考),对应词表投影 GEMV;在 node 版报告中它被并入 gemvx 族 32.3% 里,本文件把「logits 部分」单独量化了出来。修复方向:列入 ncu 采集清单(与 node 版 P0-1 同次采集,加 `-k regex:gemvx` 已覆盖);量化上看 W4A16/FP8 词表投影收益;验证方式:改后复采,eager 段该 kernel KAvg 下降。

### 🟢 P2(G < 3% 或纯记录)

- 采样/softmax eager 链:softmax 349.44 ms、TopP 145.64 ms、RadixTopK 132.28 ms(eager 段内 9.0%/3.8%/3.4%,参考;对真实 busy 均 ≈1% 级,来源:cuda_gpu_kern_sum + GRAPH_TRACE 合成):留档,若做采样融合可整体考虑。
- node 级采集把 cudaGraphLaunch CPU 均值从 263.1 us 抬到 1,110.1 us(约 4.2 倍,cuda_api_sum 两文件对照):伪影提醒,供解读 node 版数据时使用,非本负载问题。
- 真实空隙 >1 ms 每卡 10 个共 71.9/75.4 ms(0.4%,intervals.py):步边界停顿,与 node 版同源,留档。

## 四、下一步行动

1. **kernel 级结论切换到 node 版**(P0-1):阅读顺序上以 `d1_nsys_moe_bs1_analysis.md` 为 bs1 的权威占比表;本文件不再补采(node 版已存在)。
2. **把 logits GEMV 加进 ncu 采集清单**(P1-1 依据:G ≈ 7.3%,推断):node 版行动 1 的 `-k regex:gemvx` 已覆盖,采集时留意区分大 grid(logits,KAvg 326.6 us)与小 grid(每层投影,KAvg 15.5 us)两种配置,ncu 按 (kernel, grid) 分开判读;完成判据:两种配置各自拿到 ES。
3. 若后续要单独研究 graph 重放节奏(26 张 graph、每步 3.789 ms),本文件即是数据源,不必新采。

## 附录:复现命令

```bash
# 0) 版本与哈希
/usr/local/cuda/bin/nsys --version; /usr/local/cuda/bin/ncu --version; sqlite3 --version
sha256sum /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1_graphlevel.nsys-rep   # 82abb007d305...

# 1) 只读复制 + 导出(SCRATCH=/tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014)
cp /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1_graphlevel.nsys-rep "$SCRATCH/"
/usr/local/cuda/bin/nsys stats "$SCRATCH/d1_nsys_moe_bs1_graphlevel.nsys-rep" --sqlite "$SCRATCH/d1_nsys_moe_bs1_graphlevel.sqlite" \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,osrt_sum \
  --format csv --output "$SCRATCH/d1_nsys_moe_bs1_graphlevel" --force-overwrite=true 2>&1 | grep -E 'Exportation error|ERROR:|SKIPPED'
SQ="$SCRATCH/d1_nsys_moe_bs1_graphlevel.sqlite"

# 2) 采集验尸(本报告 P0 的直接证据)+ 基线 + 规模
sqlite3 "$SQ" "SELECT value FROM META_DATA_CAPTURE WHERE name='CUDA_GRAPH_TRACE_OPTIONS:MODE'"          # Graph
sqlite3 -header "$SQ" "SELECT name,memoryBandwidth,l2CacheSize,smCount FROM TARGET_INFO_GPU"
sqlite3 "$SQ" "SELECT duration FROM ANALYSIS_DETAILS"                                                    # 20212295393
sqlite3 -header -csv "$SQ" "SELECT COUNT(*),COUNT(DISTINCT shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL"   # 182858,20

# 3) graph 执行画像(graph 模式独有表)
sqlite3 "$SQ" "SELECT sql FROM sqlite_master WHERE name='CUPTI_ACTIVITY_KIND_GRAPH_TRACE'"
sqlite3 -header -csv "$SQ" "SELECT deviceId,COUNT(*),COUNT(DISTINCT graphId),SUM(end-start),ROUND(AVG(end-start)/1e3,1),ROUND(MIN(end-start)/1e3,1),ROUND(MAX(end-start)/1e3,1) FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE GROUP BY deviceId"
# 3974,26,15057802977,3789.1,118.1,9035.0 / 3974,26,15057904161,3789.1,110.8,8755.4

# 4) 真实 busy/idle 与空隙:python 扫描线,把 GRAPH_TRACE 并入区间合并
python3 /tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014/intervals.py           # kernel-only 口径(11.28% 伪影)
# 扩展合并(含 GRAPH_TRACE)一次性脚本,输出 busy 98.54/98.53、gaps>1ms 10 个 71.9/75.4ms:
python3 - <<'EOF'
import sqlite3
def merge(iv):
    iv.sort(); out=[]
    for s,e in iv:
        if out and s<=out[-1][1]:
            if e>out[-1][1]: out[-1][1]=e
        else: out.append([s,e])
    return out
c=sqlite3.connect("file:SCRATCH_PLACEHOLDER/d1_nsys_moe_bs1_graphlevel.sqlite?mode=ro",uri=True).cursor()
for d in (0,1):
    iv=[]
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_GRAPH_TRACE"):
        iv+=[[s,e] for s,e in c.execute(f"SELECT start,end FROM {t} WHERE deviceId=?",(d,))]
    m=merge(iv); busy=sum(e-s for s,e in m); wall=m[-1][1]-m[0][0]
    print(d, busy, wall, round(100*busy/wall,2))
EOF

# 5) eager 段占比与 API 对照(python csv 模块解析 kern_sum;graph launch 对照 node 版)
python3 -c "import csv;rows=list(csv.DictReader(open('$SCRATCH/d1_nsys_moe_bs1_graphlevel_cuda_gpu_kern_sum.csv')));print(sum(int(r['Total Time (ns)']) for r in rows));[print(r['Time (%)'],r['Total Time (ns)'],r['Instances'],r['Avg (ns)'],r['Name'][:60]) for r in rows[:8]]"
sqlite3 -header -csv "$SQ" "SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(AVG(r.end-r.start),1) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY 1"   # 7948, avg 263149.1ns
```
