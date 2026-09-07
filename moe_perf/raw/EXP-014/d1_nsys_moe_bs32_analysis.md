# provenance: tool=ncu-nsys-analysis src="/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32.nsys-rep" date=2026-09-07T15:45Z gpu="2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU)" ncu=2026.1.0.0 nsys=2025.6.3.343 analyzed_on="cpod-1u8o0xv30sr6(2x RTX 4090)"

# vLLM MoE bs32 timeline 性能诊断(nsys node 级 trace)

## 一、结论摘要

被测负载是 vLLM serve Qwen1.5-MoE-A2.7B-Chat(TP2+EP,bench 并发 32、384 条请求,组内已核实),GPU 几乎打满:两卡 busy 98.96%/99.06%,>1 ms 空隙每卡不超过 5 个、最大仅 2.888 ms,结构面无可挑剔。时间高度集中在单一 kernel:fused_moe_kernel 时间占比 T 56.4%(20,501.28 ms),远超第二名 NCCL AllReduce 11.9%,Top-3 合计 77.3%。本组无 ncu 数据,按退化规则 B 不判根因,第一动作是 ncu 复采 fused_moe_kernel(P0 采集对象),其后按 G = ES × 56.4% 回填全局预估收益——它是全系列杠杆最大的单点。

| 项 | 值 |
|---|---|
| 被分析文件 | `/root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32.nsys-rep`(sha256 前 12 位 `0274043de8e1`) |
| 采集环境 | 2x NVIDIA GeForce RTX 4090(rep 内 TARGET_INFO_GPU);`--cuda-graph-trace` 模式 = **Node**(META_DATA_CAPTURE 实查);驱动 610.57.04(同目录 d1_sweep_manifest.txt,一句话引用) |
| 分析工具 | ncu 2026.1.0.0(本组无 ncu 文件,未使用)/ nsys 2025.6.3.343 / sqlite3 3.50.2(均现场 --version) |
| 硬件基线 | 128 SM / DRAM 峰值 1008.096 GB/s / L2 72 MiB(rep 导出 sqlite 的 TARGET_INFO_GPU,两卡同值) |

被测进程命令行同 bs1 版(sqlite `META_DATA_CAPTURE` PROCESS_0):`vllm serve Qwen/Qwen1.5-MoE-A2.7B-Chat --tensor-parallel-size 2 --enable-expert-parallel ...`;bench 侧 max_concurrency=32、num_prompts=384(d1_nsys_moe_bs32_bench.log,一句话引用)。

## 二、指标解读

下表时长数值均为 profiler 环境,仅作相对比较,不作 benchmark 绝对值。

| 指标 | 数值 | 含义与影响 | 建议动作 | 来源 |
|---|---:|---|---|---|
| trace 总时长 / GPU 事件跨度 | 20.158 s / 16.647 s | 首个 GPU 事件在 t=3.460 s,头部约 3.5 s 为 bench 启动前空转;以下占比按 GPU 事件跨度口径 | 无需动作 | ANALYSIS_DETAILS.duration;KERNEL 表 MIN(start)/MAX(end) |
| GPU busy / idle | dev0 98.96%/1.04%,dev1 99.06%/0.94% | 并发 32 把流水线彻底喂满,空闲近零:不存在结构性优化空间,收益只能来自 kernel 本身 | 无需动作 | intervals.py 区间合并(KERNEL+MEMCPY+MEMSET) |
| kernel 事件数 | 1,345,950 行 / 50 个不同 kernel(单均约 27 us) | node 模式 graph 内 kernel 全量可见,本表即 bs32 占比的权威口径 | 无需动作 | KERNEL 表 COUNT |
| Top-1/Top-3 时间占比 T | 56.4% / 77.3% | 单点集中度极高(>80% 线附近),ncu 单 kernel 深挖是明确路线 | 转 ncu(见问题清单) | cuda_gpu_kern_sum Time% |
| 入队等待 QAvg vs 执行 KAvg | fused_moe QAvg 23.66–23.81 ms vs KAvg 233.0–239.7 us,QCount≈Count | 队列深约百倍于执行时长:GPU 在大批量下持续排队,异步流水线健康(好事) | 无需动作 | cuda_kern_exec_sum |
| cudaGraphLaunch | 2,028 次,均值 1080.1 us | node 级采集抬高该值(同 workload graph 级 trace 同一 API 均值 259.4 us,约 4.2 倍伪影),不作真实 CPU 开销引用 | 无需动作(伪影提醒) | cuda_api_sum;对照 d1_nsys_moe_bs32_graphlevel 同表 |
| cudaEventSynchronize | 900 次共 16.050 s,占 CPU API 跨度 80.0% | serve 型负载 CPU 等 GPU 的正常形态,与 busy 99% 互证,不按同步反模式报 | 无需动作 | cuda_api_sum;RUNTIME 表跨度 20.062 s |
| 最大 GPU 空隙 | dev0 5 个 >1 ms(合计 10.19 ms)、dev1 3 个(6.21 ms),最大 2.888 ms,集中在 t≈3.5 s 启动段 | 流水线几乎无断裂,空隙都在 bench 起步时 | 无需动作 | intervals.py 直方图 + top5_gaps SQL |
| stream 并发 | 每卡 178 条 stream,重叠因子 1.098/1.107 | 存在计算-通信重叠;大量 stream 来自 NCCL channel,属正常 | 无需动作 | stream_overlap SQL |
| memcpy 画像 | D2D 44,978 条共 6.571 GB、65.04 ms、聚合 101.03 GB/s;H2D Pinned 4,284 条共 2.16 MB;D2H 3,600 条共 686 KB | 传输合计约 76.2 ms,占 kernel 总时长 36.33 s 的约 0.2%,且 D2D 达百 GB/s 量级:传输维度无问题 | 无需动作 | memcpy_by_kind SQL;cuda_gpu_mem_time_sum |
| 拷贝-计算重叠率 | dev0 82.76%,dev1 75.60% | 大部分拷贝被计算藏住,与重叠因子 >1 互证,健康 | 无需动作 | intervals.py 扫描线 |
| NVTX 阶段 | `execute_context_0(0)_generation_32(32)` 1,770 个 range 共 3,680.1 ms;另见 512–2,040 token 的 prefill context range | 窗口内约 885 步/卡的 32 并发 decode 稳态,混有 prefill;无 JIT 首跑伪影嫌疑 | 无需动作 | nvtx_pushpop_sum |

**数据缺口**(缺失 ≠ 实测为 0):

- OS 运行时系统调用维度:未采到(osrt_sum 输出 `SKIPPED ... does not contain OS Runtime trace data`,0 字节 csv);补采需 `nsys profile -t cuda,nvtx,osrt`。
- 显存占用曲线/逐分配字节:未采到(sqlite 无 CUDA_GPU_MEMORY_USAGE_EVENTS 表);补采需 `--cuda-memory-usage=true`。
- kernel 级根因(SOL/occupancy/stall):本组无 ncu 文件,ES 不可得,分档走退化规则 B;补采命令见「下一步行动」。

## 三、问题清单(按优先级)

本次分档口径:**退化规则 B**(只有 nsys,拿不到 ES):kernel 热点按 T 分档为 ncu 采集对象(≥20% P0、5–20% P1、<5% P2),结构性问题照常按 G。T 取值来源:cuda_gpu_kern_sum Time%(两卡合并;node 模式全量)。同目录 derived/d1_kernel_share_bs32.csv 的分桶占比与本表同源同值(组内已核实,一句话引用)。

### 🔴 P0(采集对象:T ≥ 20%)

- **P0-1 fused_moe_kernel(专家 grouped GEMM)一家独大,根因待 ncu**:T = 56.4%(20,501.28 ms,86,400 实例,均值 237.28 us;来源:cuda_gpu_kern_sum)。并发 32 下专家 GEMM 成为绝对主体,但 nsys 无法区分它是带宽受限、occupancy 不足还是 tile 配置欠优(规则 B 不判根因)。修复方向:`ncu --set full` 复采该 kernel(bs32 负载下),按 ES 回填 G = ES × 56.4%;若 ncu 确认可调,方向依次是 Triton 配置调参(BLOCK/num_warps/num_stages)与专家负载均衡(组内 d5_eplb 脚本可承接);验证方式:改后复采 nsys,看 T 与端到端吞吐同向变化。

### 🟡 P1(采集对象:5% ≤ T < 20%)

- **P1-1 TP2 通信 kernel 合计 T = 15.0%(上限口径)**:AllReduce 11.9%(4,323.85 ms,88,200 实例,均值 49.02 us)+ AllGather 3.1%(1,118.34 ms,1,800 实例,均值 621.30 us;均来源:cuda_gpu_kern_sum)。结构性成本、可省比例未知,按硬约束封顶 P1(G ≤ 15.0%,推断上限)。修复方向:与 dense_tp2 对照定通信增量;看通信-计算重叠余地(当前重叠因子 1.098/1.107);验证方式:复采后通信合计占比下降。
- **P1-2 cutlass bf16 GEMM(s16816gemm_relu 64x64)**:T = 9.0%(3,252.50 ms,88,234 实例,均值 36.86 us;来源:cuda_gpu_kern_sum),dense 投影/共享专家 GEMM 主力。动作:ncu 采集对象(与 P0-1 同一次采集顺带)。
- **P1-3 flash_fwd_splitkv(注意力)**:T = 6.7%(2,418.02 ms,42,672 实例,均值 56.67 us;来源:cuda_gpu_kern_sum)。动作:ncu 采集对象,优先级最低。

### 🟢 P2(T < 5% 且无结构问题)

- ampere_bf16_s16816gemm 2.2%、cutlass wmma gemm 1.8%、moe_align_block_size 1.0%、moeTopK 0.7%、triton silu 0.6%(cuda_gpu_kern_sum):留档。
- GPU 空闲 G = 1.04%/0.94%(intervals.py):健康,无动作。
- memcpy/重叠维度全部健康(101.03 GB/s、重叠率 75.60–82.76%):无动作。

## 四、下一步行动

1. **ncu 复采 fused_moe_kernel(bs32 负载)**(P0-1 依据:T = 56.4%,全报告最大杠杆):`ncu --set full -k fused_moe_kernel -c 20` 挂到同参数 serve + 并发 32 bench 上;完成判据:拿到 SOL/occupancy/stall 与 ES,回填 G = ES × 56.4% 并按标准模式重排优先级。
2. **同一次 ncu 顺带抓 cutlass relu GEMM 与 flash_fwd_splitkv**(P1-2/P1-3,合计 T = 15.7%),避免二次搭环境。
3. **通信占比对照**(P1-1):汇总本报告 15.0% 与 bs1 版 13.8%(见 d1_nsys_moe_bs1_analysis.md),对照 dense_tp2 sweep,决定通信优化是否立项。
4. 本文件口径已是 node 级,**无需重采 nsys**。

## 附录:复现命令

```bash
# 0) 版本与哈希
/usr/local/cuda/bin/nsys --version; /usr/local/cuda/bin/ncu --version; sqlite3 --version
sha256sum /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32.nsys-rep   # 0274043de8e1...

# 1) 只读复制 + 导出(SCRATCH=/tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014)
cp /root/projects/vllm/experiments/moe_perf/raw/EXP-014/d1_nsys_moe_bs32.nsys-rep "$SCRATCH/"
/usr/local/cuda/bin/nsys stats "$SCRATCH/d1_nsys_moe_bs32.nsys-rep" --sqlite "$SCRATCH/d1_nsys_moe_bs32.sqlite" \
  --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,osrt_sum \
  --format csv --output "$SCRATCH/d1_nsys_moe_bs32" --force-overwrite=true 2>&1 | grep -E 'Exportation error|ERROR:|SKIPPED'
SQ="$SCRATCH/d1_nsys_moe_bs32.sqlite"

# 2) 采集验尸 / 硬件基线 / 跨度 / 规模
sqlite3 "$SQ" "SELECT value FROM META_DATA_CAPTURE WHERE name='CUDA_GRAPH_TRACE_OPTIONS:MODE'"          # Node
sqlite3 -header "$SQ" "SELECT name,memoryBandwidth,l2CacheSize,smCount FROM TARGET_INFO_GPU"
sqlite3 "$SQ" "SELECT duration FROM ANALYSIS_DETAILS"                                                    # 20157801998
sqlite3 "$SQ" "SELECT MIN(start),MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"                               # 3459977014|20107316940
sqlite3 -header -csv "$SQ" "SELECT COUNT(*),COUNT(DISTINCT shortName) FROM CUPTI_ACTIVITY_KIND_KERNEL"   # 1345950,50

# 3) busy/idle、空隙、重叠率:python 扫描线
python3 /tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_moe-exp014/intervals.py
# 输出 $SCRATCH/d1_nsys_moe_bs32.intervals.txt:busy 98.96/99.06,gaps>1ms 5/3 个,overlap 82.76/75.60

# 4) launch / 同步 / stream / memcpy(SQL 与 bs1 版附录同款,替换 $SQ 即可,逐条可执行)
sqlite3 -header -csv "$SQ" "SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(AVG(r.end-r.start),1) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%GraphLaunch%' GROUP BY 1"
sqlite3 -header -csv "$SQ" "WITH w AS (SELECT MAX(end)-MIN(start) wall FROM CUPTI_ACTIVITY_KIND_RUNTIME) SELECT s.value,COUNT(*),SUM(r.end-r.start),ROUND(100.0*SUM(r.end-r.start)/(SELECT wall FROM w),2) FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE s.value LIKE '%Synchronize%' OR s.value LIKE 'cudaMemcpy%' GROUP BY s.value ORDER BY 3 DESC LIMIT 6"
sqlite3 -header -csv "$SQ" "SELECT ck.label,sk.label,dk.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start),ROUND(1.0*SUM(m.bytes)/SUM(m.end-m.start),2) FROM CUPTI_ACTIVITY_KIND_MEMCPY m LEFT JOIN ENUM_CUDA_MEMCPY_OPER ck ON ck.id=m.copyKind LEFT JOIN ENUM_CUDA_MEM_KIND sk ON sk.id=m.srcKind LEFT JOIN ENUM_CUDA_MEM_KIND dk ON dk.id=m.dstKind GROUP BY 1,2,3 ORDER BY 6 DESC"

# 5) 报表 csv 解析(Time% 与 QAvg/KAvg;python csv 模块)
python3 -c "import csv;rows=list(csv.DictReader(open('$SCRATCH/d1_nsys_moe_bs32_cuda_gpu_kern_sum.csv')));print(sum(int(r['Total Time (ns)']) for r in rows));[print(r['Time (%)'],r['Total Time (ns)'],r['Instances'],r['Avg (ns)'],r['Name'][:60]) for r in rows[:12]]"
```
