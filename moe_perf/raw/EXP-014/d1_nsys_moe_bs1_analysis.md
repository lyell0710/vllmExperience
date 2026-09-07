# provenance: tool=ncu-nsys-analysis src="/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1.nsys-rep" date=2026-09-07T15:45Z gpu="2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU)" ncu=2026.1.0.0 nsys=2025.6.3.343 analyzed_on="cpod-1u8o0xv30sr6(2x RTX 4090)"

# vLLM MoE bs1 decode timeline 性能诊断(nsys node 级 trace)

## 一、结论摘要

被测负载是 vLLM serve Qwen1.5-MoE-A2.7B-Chat(TP2+EP,bench 并发 1、输入 128/输出 512,组内已核实),GPU 结构面健康:两卡 busy 96.43%/96.46%,空闲仅 3.57%/3.54%。时间去哪了非常集中:dense GEMV(gemvx 族)时间占比 T 32.3%、fused_moe_kernel 18.7%、NCCL 通信合计 13.8%,Top-3 kernel 合计 63.7%——bs1 decode 的主要矛盾在 kernel 内部效率而非调度结构。本组无 ncu 数据,按退化规则 B 不判 kernel 根因,第一动作是用 ncu 复采 gemvx(T=32.3%,P0 采集对象),预估收益待 ncu 的 ES 出来后按 G = ES × 32.3% 回填。

| 项 | 值 |
|---|---|
| 被分析文件 | `/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1.nsys-rep`(sha256 前 12 位 `ebffc130f8f5`) |
| 采集环境 | 2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU);`--cuda-graph-trace` 模式 = **Node**(META_DATA_CAPTURE 实查,graph 内 kernel 全量可见);驱动 610.57.04(同目录 d1_sweep_manifest.txt,一句话引用) |
| 分析工具 | ncu 2026.1.0.0(本组无 ncu 文件,未使用)/ nsys 2025.6.3.343 / sqlite3 3.50.2(均现场 --version) |
| 硬件基线 | 128 SM / DRAM 峰值 1008.096 GB/s / L2 72 MiB(rep 导出 sqlite 的 TARGET_INFO_GPU,两卡同值) |

被测进程命令行(sqlite `META_DATA_CAPTURE` PROCESS_0):`vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --port 8100 --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel --gpu-memory-utilization 0.88`。

## 二、指标解读

下表时长数值均为 profiler 环境,仅作相对比较,不作 benchmark 绝对值。

| 指标 | 数值 | 含义与影响 | 建议动作 | 来源 |
|---|---:|---|---|---|
| trace 总时长 / GPU 事件跨度 | 20.188 s / 17.956 s | 首个 GPU 事件在 t=2.141 s,头部约 2.2 s 是服务空转段;以下占比均按 GPU 事件跨度口径 | 无需动作 | ANALYSIS_DETAILS.duration;KERNEL 表 MIN(start)/MAX(end) |
| GPU busy / idle | dev0 96.43%/3.57%,dev1 96.46%/3.54% | 空闲 <5%,按判读表属健康:decode 流水线基本喂满,不是 CPU/launch 受限形态 | 无需动作 | intervals.py 区间合并(KERNEL+MEMCPY+MEMSET) |
| kernel 事件数 | 5,177,228 行 / 39 个不同 kernel | 20 s 内五百万级微 kernel(单卡均值约 8.3 us/个),靠 CUDA Graph 才可能喂满——node 模式把 graph 内每个 kernel 都录了下来,正是本文件的价值 | 无需动作 | KERNEL 表 COUNT;per_device_kernel_time |
| Top-1/Top-3 时间占比 T | 32.3% / 63.7% | 集中度高,单点优化目标明确,ncu 深挖收益可期 | 转 ncu(见问题清单) | cuda_gpu_kern_sum Time% |
| cudaGraphLaunch | 8,052 次,均值 1110.1 us | 每 decode 步每卡 1 次 graph 重放;注意该均值被 node 级采集显著抬高(同 workload graph 级 trace 里同一 API 均值 263.1 us,约 4.2 倍伪影),不能当真实 CPU 开销 | 无需动作(伪影提醒) | cuda_api_sum;对照 d1_nsys_moe_bs1_graphlevel 同表 |
| eager 段 launch 画像 | cudaLaunchKernel 123,456 次 + cuLaunchKernelEx 61,904 次,launch API 均值约 9.6 us,<5 us 短 kernel 占 63.2%(按 %LaunchKernel% 关联口径) | graph 外仍有约 18.5 万次逐个 launch(采样/logits 链);但 GPU 不空,launch 未成瓶颈 | 无需动作 | cuda_api_sum;KERNEL JOIN RUNTIME(correlationId) |
| 入队等待 QAvg vs 执行 KAvg | 头部 kernel QAvg 4.23–4.43 ms,KAvg 15.5–22.3 us,QCount≈Count(5,136,015/5,177,228) | 队列深、GPU 持续排队,是异步流水线健康的表现(好事);另注:graph 内 kernel 的 AAvg≈1.15 ms 实为整条 cudaGraphLaunch 的摊派,不是单 kernel 成本 | 无需动作 | cuda_kern_exec_sum |
| cudaEventSynchronize | 3,834 次共 16.104 s,占 CPU API 跨度 80.08% | serve 型负载 CPU 等 GPU 的正常形态(GPU busy 96%+ 与之互证),不按同步反模式报 | 无需动作 | cuda_api_sum;RUNTIME 表跨度 20.109 s |
| 最大 GPU 空隙 | 每卡 11 个 >1 ms 空隙,合计 83.19 ms(dev0)/76.01 ms(dev1),最大 11.90 ms,两侧都是 `_post_update_kernel → _apply_write_kernel` | 形态重复、都在步边界,指向偶发的调度/请求交接停顿,总量仅占跨度 0.46% | P2 留档 | top5_gaps SQL + intervals.py 直方图 |
| stream 并发 | 每卡 52 条 stream,重叠因子 1.25 | 计算与 NCCL 通信存在实质重叠(>1.0),多 stream 不是摆设 | 无需动作 | stream_overlap SQL |
| memcpy 画像 | D2D 191,684 条共 984.89 MB、163.19 ms(条均 5.1 KB/851 ns);H2D Pinned 15,880 条共 7.92 MB;D2H 15,336 条共 92.0 KB(条均 6 B) | 传输总时长约 202.5 ms,仅占 kernel 总时长 43.06 s 的约 0.5%,是延迟型微拷贝(采样输入/token 回读)而非带宽问题;D2D 条数巨大但时间无害 | P2 留档 | memcpy_by_kind SQL;cuda_gpu_mem_time_sum |
| 拷贝-计算重叠率 | dev0 7.05%,dev1 9.04%(memcpy 总时长 100.8/101.7 ms) | 重叠率低但 memcpy 时间占比 ≤0.5%,按判读表记录即可、不点名 | 无需动作 | intervals.py 扫描线 |
| NVTX 阶段 | `execute_context_0(0)_generation_1(1)` 7,652 个 range 共 14.949 s(占 NVTX 96.8%) | 窗口内约 3,826 步/卡 decode 稳态,无 JIT 首跑伪影嫌疑 | 无需动作 | nvtx_pushpop_sum |

**数据缺口**(缺失 ≠ 实测为 0):

- OS 运行时系统调用维度:未采到(osrt_sum 输出 `SKIPPED ... does not contain OS Runtime trace data`,0 字节 csv);补采需 `nsys profile -t cuda,nvtx,osrt`。
- 显存占用曲线/逐分配字节:未采到(sqlite 无 CUDA_GPU_MEMORY_USAGE_EVENTS 表);补采需 `--cuda-memory-usage=true`。
- kernel 级根因(SOL/occupancy/stall):本组无 ncu 文件,ES 不可得——正是问题清单采用退化规则 B 的原因;补采命令见「下一步行动」。

## 三、问题清单(按优先级)

本次分档口径:**退化规则 B**(只有 nsys,拿不到 ES):kernel 热点不判根因、按 T 分档为 ncu 采集对象(≥20% P0、5–20% P1、<5% P2),结构性问题照常按 G。T 取值来源:cuda_gpu_kern_sum Time%(两卡合并;node 模式,graph 内 kernel 全量计入)。同目录 derived/d1_kernel_share_bs1.csv 的分桶占比与本表同源同值(组内已核实,一句话引用)。

### 🔴 P0(采集对象:T ≥ 20%)

- **P0-1 dense GEMV(gemvx 族)是 bs1 decode 第一大户,根因待 ncu**:T = 32.3%(13,926.38 ms,742,260 实例,均值 18.76 us;来源:cuda_gpu_kern_sum)。该族覆盖注意力/共享专家投影与 lm_head 的 bf16 GEMV;bs1 GEMV 大概率带宽受限,但按规则 B 不在 nsys 侧下根因结论(推断,待证)。修复方向:先 `ncu` 复采该 kernel 拿 SOL 四象限与 Estimated Speedup,回填 G = ES × 32.3% 再定改法(若确认 DRAM 受限,方向是降权重字节数,如 W4A16/FP8,组内 d4 系列已有脚本);验证方式:改后复采 nsys,看该族 T 是否显著下降。

### 🟡 P1(采集对象:5% ≤ T < 20%;结构项 3% ≤ G < 10%)

- **P1-1 fused_moe_kernel(专家 grouped GEMM)**:T = 18.7%(8,054.96 ms,368,064 实例,均值 21.88 us;来源:cuda_gpu_kern_sum),离 P0 线一步之遥;bs32 版同名 kernel 占 56.4%(见同组 d1_nsys_moe_bs32_analysis.md,交叉引用),两个 batch 档共用一次 ncu 复采即可。动作:ncu 采集对象。
- **P1-2 TP2 通信 kernel 合计 T = 13.8%(上限口径)**:AllReduce 12.7%(5,477.09 ms,375,732 实例,均值 14.58 us)+ AllGather 1.1%(455.66 ms;均来源:cuda_gpu_kern_sum)。这是 TP2+EP 每层同步的结构成本,「可省比例」未知(通信不可能清零),证据不足按硬约束封顶 P1(G ≤ 13.8%,推断上限)。修复方向:与同组 dense_tp2 sweep 的通信占比对照定基线;工程上看通信-计算重叠(已有 1.25 重叠因子,仍有余地)或减少同步点;验证方式:复采后通信合计占比下降。
- **P1-3 flash_fwd_splitkv(decode 注意力)**:T = 6.2%(2,681.45 ms,183,648 实例,均值 14.60 us;来源:cuda_gpu_kern_sum)。动作:ncu 采集对象(顺带采,优先级低于 P0-1/P1-1)。
- **P1-4 GPU 空闲 3.6%(上限,健康区间内,行动价值低)**:G = 空隙合计 ÷ GPU 事件跨度 = 3.57%(dev0,推断上限;来源:intervals.py)。其中可具名的 >1 ms 空隙仅 11 个共 83.19 ms(0.46%),其余是亚毫秒微空隙,实际可回收远小于名义 G;判读表将 idle <5% 判为健康,故本条仅登记不催办。

### 🟢 P2(T < 3% 或 G < 3%)

- splitKreduce 4.2%、moe_align_block_size 4.1%、moeTopK 3.8%(以上按规则 B 属 5% 以下采集对象边缘,来源:cuda_gpu_kern_sum)——MoE 路由支路小 kernel 群,单个都不值一次专门 ncu,若做 kernel 融合可整体考虑。
- fp32 累加版 gemvx 3.3%、direct_copy 2.3%、act_and_mul 1.1%(cuda_gpu_kern_sum):留档。
- D2D 微拷贝碎片:191,684 条、条均 5.1 KB/851 ns、合计 163.19 ms 约占 kernel 时间 0.38%(memcpy_by_kind):典型碎片形态但时间无害,顺手才做。
- 步边界 ~10 ms 空隙 11 个(top5_gaps):与 P1-4 同源,留档观察。

## 四、下一步行动

1. **ncu 复采 gemvx 与 fused_moe_kernel**(P0-1、P1-1 依据:两者 T 合计 51.0%):在同负载下 `ncu --set full -k "regex:gemvx|fused_moe_kernel" -c 20` 采样,拿到 ES 后按 G = ES × T 回填本报告优先级;完成判据:两 kernel 均有 SOL/占用率/stall 数据,G 算式可复算。
2. **通信占比对照**(P1-2):用同组 dense_tp2 的 bench/gpu csv 与本报告 13.8% 对照,决定是否立通信优化专项;完成判据:得出「MoE 相对 dense 的通信增量」一个数。
3. 本文件口径已是 node 级,**无需重采 nsys**;graph 级伪影对照结论(cudaGraphLaunch 1110.1 vs 263.1 us)引自同组 graphlevel 报告,不必重复采集。

## 附录:复现命令

```bash
# 0) 版本与哈希
/usr/local/cuda/bin/nsys --version; /usr/local/cuda/bin/ncu --version; sqlite3 --version
sha256sum /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1.nsys-rep   # ebffc130f8f5...

# 1) 只读复制 + 导出(样本目录只读;SCRATCH=/tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014)
cp /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs1.nsys-rep "$SCRATCH/"
/usr/local/cuda/bin/nsys stats "$SCRATCH/d1_nsys_moe_bs1.nsys-rep" --sqlite "$SCRATCH/d1_nsys_moe_bs1.sqlite" \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,osrt_sum \
  --format csv --output "$SCRATCH/d1_nsys_moe_bs1" --force-overwrite=true 2>&1 | grep -E 'Exportation error|ERROR:|SKIPPED'
SQ="$SCRATCH/d1_nsys_moe_bs1.sqlite"

# 2) 采集验尸 / 硬件基线 / 跨度
sqlite3 "$SQ" "SELECT value FROM META_DATA_CAPTURE WHERE name='CUDA_GRAPH_TRACE_OPTIONS:MODE'"          # Node
sqlite3 "$SQ" "SELECT name,value FROM META_DATA_CAPTURE WHERE name LIKE 'PROCESS_0:%'"                  # 被测命令行
sqlite3 -header "$SQ" "SELECT name,memoryBandwidth,l2CacheSize,smCount FROM TARGET_INFO_GPU"
sqlite3 "$SQ" "SELECT duration FROM ANALYSIS_DETAILS"                                                    # 20187661752
sqlite3 "$SQ" "SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"                               # 2140934431|20096347557
sqlite3 -header -csv "$SQ" "SELECT COUNT(*),COUNT(DISTINCT shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL"   # 5177228,39

# 3) busy/idle、空隙、重叠率:python 扫描线(sqlite 无索引 JOIN 在 5.2M 行上不可行)
python3 /tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014/intervals.py
# 输出 $SCRATCH/d1_nsys_moe_bs1.intervals.txt:busy 96.43/96.46,gaps>1ms 11 个 83.19/76.01ms,overlap 7.05/9.04

# 4) launch / 同步 / stream / memcpy / 空隙两侧 kernel(SQL 同 skill references,逐条可执行)
sqlite3 -header -csv "$SQ" "SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(AVG(r.end-r.start),1) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY 1"
sqlite3 -header -csv "$SQ" "WITH w AS (SELECT MAX(end)-MIN(start) wall FROM CUPTI_ACTIVITY_KIND_RUNTIME) SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(100.0*SUM(r.end-r.start)/(SELECT wall FROM w),2) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%Synchronize%' OR s.value LIKE 'cudaMemcpy%' GROUP BY s.value ORDER BY 3 DESC LIMIT 6"
sqlite3 -header -csv "$SQ" "WITH k AS (SELECT streamId,deviceId,start,end FROM CUPTI_ACTIVITY_KIND_KERNEL), per_stream AS (SELECT deviceId,streamId,SUM(end-start) busy FROM k GROUP BY deviceId,streamId), uni AS (SELECT deviceId,SUM(e-s) union_busy FROM (SELECT deviceId,MIN(start) s,MAX(end) e FROM (SELECT deviceId,start,end,SUM(CASE WHEN pmax IS NULL OR start>pmax THEN 1 ELSE 0 END) OVER (PARTITION BY deviceId ORDER BY start) gid FROM (SELECT deviceId,start,end,MAX(end) OVER (PARTITION BY deviceId ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) pmax FROM k)) GROUP BY deviceId,gid) GROUP BY deviceId) SELECT p.deviceId,COUNT(DISTINCT p.streamId),SUM(p.busy),u.union_busy,ROUND(1.0*SUM(p.busy)/u.union_busy,3) FROM per_stream p JOIN uni u USING(deviceId) GROUP BY p.deviceId"
sqlite3 -header -csv "$SQ" "SELECT ck.label,sk.label,dk.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start),ROUND(1.0*SUM(m.bytes)/SUM(m.end-m.start),2) FROM CUPTI_ACTIVITY_KIND_MEMCPY m LEFT JOIN ENUM_CUDA_MEMCPY_OPER ck ON ck.id=m.copyKind LEFT JOIN ENUM_CUDA_MEM_KIND sk ON sk.id=m.srcKind LEFT JOIN ENUM_CUDA_MEM_KIND dk ON dk.id=m.dstKind GROUP BY 1,2,3 ORDER BY 6 DESC"
sqlite3 -header -csv "$SQ" "WITH k AS (SELECT k.deviceId,k.start,k.end,s.value AS name FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName), o AS (SELECT deviceId,start,end,name,MAX(end) OVER (PARTITION BY deviceId ORDER BY start ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prev_end FROM k) SELECT o.deviceId,(o.start-o.prev_end),ROUND((o.start-o.prev_end)/1e6,3),o.prev_end,(SELECT name FROM k WHERE k.deviceId=o.deviceId AND k.end=o.prev_end LIMIT 1),o.name FROM o WHERE o.prev_end IS NOT NULL AND o.start>o.prev_end ORDER BY 2 DESC LIMIT 5"

# 5) 报表 csv 解析(kern_sum Time% / kern_exec QAvg;kernel 名含逗号,必须 python csv 模块)
python3 -c "import csv;rows=list(csv.DictReader(open('$SCRATCH/d1_nsys_moe_bs1_cuda_gpu_kern_sum.csv')));print(sum(int(r['Total Time (ns)']) for r in rows));[print(r['Time (%)'],r['Total Time (ns)'],r['Instances'],r['Avg (ns)'],r['Name'][:60]) for r in rows[:12]]"
```
