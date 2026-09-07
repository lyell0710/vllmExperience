# provenance: tool=ncu-nsys-analysis src="/root/projects/vllm/experiments/pd_disagg/profiling/nsys_smoke.nsys-rep" date=2026-09-07T15:11:31Z gpu="NVIDIA GeForce RTX 4090 ×2(rep 内 TARGET_INFO_GPU,活动仅 deviceId 0)" ncu=2026.1.0.0 nsys=2025.6.3.343 analyzed_on="同机 2×RTX 4090 主机(与 rep 内 TARGET_INFO_GPU 一致)"

# nsys_smoke 冒烟 trace 性能诊断(nsys timeline)

## 一、结论摘要

该 rep 是 4096×4096 matmul ×10 的冒烟脚本 trace 而非 vLLM 服务,且为单次冷启动采样(用户已核实,本报告仅引用)。GPU 活动窗口内 busy 22.74%、idle 77.26%,而全部 idle 的 99.65% 由单个 94.063 ms 空隙贡献——该空隙落在 randn 初始化与首个 sgemm 之间,窗口内实测 48.278 ms(51.3%)是 cuLibraryLoadData 为首的懒加载/kernel 解析,属一次性冷启动开销,不是稳态问题。稳态段本身健康:10 个 cutlass sgemm 占 kernel 时间 99.6%、逐次波动仅 10.1 μs,QAvg 13.12 ms ≫ KAvg 2.77 ms 说明异步流水线已排满。按硬约束,单次含 JIT 首跑的采样整份占比表降级「参考」,第一动作是重采(P0-1,结构性 G = 94.063 ÷ 122.182 = 77.0%,预估可省上限,推断);第二动作是对 T = 99.6% 的 sgemm 按退化规则 B 转 ncu 复采(P0-2)。共 2 条 P0。

| 项 | 值 |
|---|---|
| 被分析文件 | `/root/projects/vllm/experiments/pd_disagg/profiling/nsys_smoke.nsys-rep`(sha256 前 12 位 `0219945116a8`) |
| 采集环境 | NVIDIA GeForce RTX 4090 ×2(TARGET_INFO_GPU 两行,CUPTI 事件仅 deviceId 0);CUDA_GRAPH_TRACE_OPTIONS:MODE=`Graph`(非 node),但本负载无 CUDA Graph(12 个 kernel 全部关联到 launch API),该口径本次不构成盲区;被测进程为 python -c matmul 冒烟脚本(META_DATA_CAPTURE PROCESS_0:COMMAND,已知事实引用) |
| 分析工具 | ncu 2026.1.0.0 / nsys 2025.6.3.343 / sqlite3 3.50.2(现场 --version) |
| 硬件基线 | 128 SM / DRAM 峰值 1008.096 GB/s / L2 72 MiB(TARGET_INFO_GPU.smCount/memoryBandwidth/l2CacheSize) |

说明:旁路 `nsys_smoke.sqlite` 比 rep 新 17 天、来历不明(已知事实),按红线本次未使用,全部数字来自本会话从 rep 重导的 sqlite 与 stats CSV。

## 二、指标解读

时长数值均为 profiler 环境,仅作相对比较。

| 指标 | 数值 | 含义与影响 | 建议动作 | 来源 |
|---|---:|---|---|---|
| 全 trace 跨度 | 4649.937 ms | 其中前 3331.933 ms(71.7%)无任何 GPU 事件,是 python/torch import 与 CUDA 初始化的纯 CPU 段;全 trace 口径 GPU busy 仅 0.60%。冒烟脚本一次性成本,不代表稳态 | 无需动作(留档) | ANALYSIS_DETAILS.duration;KERNEL 表 MIN(start) |
| GPU busy / idle(GPU 窗口口径) | 22.74% / 77.26% | 窗口 wall 122.182 ms、busy 27.785 ms(已知事实引用)。idle>20% 通常判 CPU/同步受限,但此处需配合 max_gap 定位后再定性 | 见 P0-1 | sqlite 区间合并查询(busy/idle) |
| 最大 GPU 空隙 max_gap | 94.063 ms | 位于 randn(distribution_elementwise_grid_stride_kernel)与首个 sgemm 之间;单条即占窗口全部 idle 94.397 ms 的 99.65%。窗口内实测 CUDA API 交叠:cuLibraryLoadData 4 条 43.977 ms + cuLibraryGetKernel 2.295 ms + cuKernelGetFunction 1.690 ms + cuGetProcAddress_v2 880 条 0.315 ms,合计 48.278 ms(占空隙 51.3%)= 懒加载 cutlass/cuBLAS 库与 kernel 解析;其余 45.786 ms 无 CPU 采样不能归因(见数据缺口)。OSRT 侧多线程交叠(poll 13 条交叠 265.519 ms > 窗口本身)不可用于归因 | 见 P0-1(warmup 后重采) | Top-5 gap 查询;gap 窗口 API 交叠查询 |
| launch 画像 | 12 次,均值 97.2 μs | 均值 97 μs 为 JIT 冷启动(已知事实引用)。拆分:cudaLaunchKernel 2 次(Max 985.262 μs,首次触发懒加载)、cuLaunchKernel 10 次均值 16.8 μs / Max 55.1 μs;短 kernel(<5 μs)0 条——不是 launch-bound,均值被首次 launch 拖高 | 无需动作(重采后自然消失) | launch 画像查询;cuda_api_sum |
| kernel 时间集中度 T | 99.6% | cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1> 10 实例合计 27.662 ms,均值 2.766 ms、StdDev 10.1 μs(逐次极稳);Top-1>80% → 单点目标明确,按退化规则 B 转 ncu 深挖。randn kernel 仅 0.4%(<5%,不出建议) | 见 P0-2 | cuda_gpu_kern_sum |
| sgemm 实现路径 | SIMT FP32 | kernel 名含 `simt_sgemm`(实测)= FP32 CUDA core 路径,未走 tensor core;折算 2×4096³ FLOP ÷ 2.766 ms ≈ 49.7 TFLOP/s(profiler 环境折算,仅定性,推断) | 见 P2-1 | cuda_gpu_kern_sum.Name;折算式 |
| 入队等待 QAvg vs 执行 KAvg | 13.12 ms vs 2.77 ms | sgemm 的 QCount 9/10、QAvg ≫ KAvg → CPU 快速提交、GPU 端深排队,是异步流水线健康形态,不是问题 | 无需动作 | cuda_kern_exec_sum |
| 同步 API 占比 | 7.43% | cudaDeviceSynchronize 仅 1 次、26.153 ms(CPU 侧 API 跨度 352.076 ms):脚本尾部一次性排空 10 个排队 matmul,预期行为;反直觉提醒:此值大不代表阻塞反模式,QCount>0 已证明启动是异步的 | 无需动作 | 同步 API 查询;cuda_api_sum |
| stream 重叠 | 单流,factor 1.000 | n_streams=1 →「单流执行,计算与传输无重叠机会」;本 trace 亦无 memcpy,无可重叠对象 | 无需动作 | stream 重叠因子查询 |
| memcpy | 0 条(实测为 0) | CUPTI_ACTIVITY_KIND_MEMCPY 表不存在;kernel 事件正常采到证明 CUDA trace 开启,故判「实测 0 条」而非「未采集」——脚本 randn 直接在 device 上生成,无 H2D | 无需动作 | sqlite_master 表探测 |
| memset | 10 次,7.522 μs,0.020 MB | 量级可忽略 | 无需动作 | cuda_gpu_mem_time_sum / mem_size_sum |
| 显存 API | cudaMalloc 7 次 1.154 ms | Max 仅 0.318 ms、cudaFree 1 次:无循环内分配抖动;字节数 RUNTIME 表拿不到(见数据缺口) | 无需动作 | 显存 API 画像查询 |
| OS 运行时热点 | poll 71.0% | poll 3178.113 ms/120 次 + pthread_cond_timedwait 1000.148 ms/2 次为空转等待底噪,属正常,不当发现报 | 无需动作 | osrt_sum |

**数据缺口**(「缺失」与「实测为 0」已在上表区分;memcpy 为实测 0 条,不列入缺口):

- NVTX 阶段归因:未采到(nvtx_pushpop_sum 输出 SKIPPED + 0 字节 csv);补采命令加 `-t nvtx`(torch 侧配合 torch.cuda.nvtx)。
- 空隙内 CPU 具体调用栈:未采到,META_DATA_CAPTURE `COLLECT_CPU_TRACE|false`,94.063 ms 空隙中 45.786 ms 无法归因;补采加 `--sample=cpu`(采样 + backtrace)。
- 显存占用曲线:未采到,`COLLECT_GPU_MEMORY_USAGE|false`;补采加 `--cuda-memory-usage=true`。
- cudaMalloc 分配字节数:CUPTI_ACTIVITY_KIND_RUNTIME 无字节列,仅有次数与 CPU 耗时,该维度本工具链拿不到。

## 三、问题清单(按优先级)

本次分档口径:退化规则 B(仅 nsys,无 ES;结构性问题按 G 分档,kernel 级热点不判根因、按 T 定采集优先级)。T 取值来源:本 rep `cuda_gpu_kern_sum` Time%。按硬约束 4,单次采样含 JIT 首跑伪影,整份占比表降级「参考」,重采列为 P0 动作。

### 🔴 P0(G ≥ 10%)

- **P0-1 单次冷启动采样,懒加载空隙支配 GPU 窗口,须重采**:G = 94.063 ms ÷ 122.182 ms = **77.0%**(结构性可省上限,GPU 窗口口径,推断,参考)。依据:max_gap = 94.063 ms,占窗口 idle 94.397 ms 的 99.65%(来源:Top-5 gap 查询 + busy/idle 查询,实测);空隙内懒加载/解析 API 交叠 48.278 ms、占 51.3%,其中 cuLibraryLoadData 4 条 43.977 ms(来源:gap 窗口 API 交叠查询,实测);归因「冷启动一次性开销」为推断(剩余 45.786 ms 无 CPU 采样佐证)。修复方向:这不是代码缺陷而是采集口径问题——(a)若目标是诊断 pd_disagg/vLLM 服务,对真实服务进程重采,必加 `--cuda-graph-trace=node`(本 rep MODE=`Graph`,服务负载下 graph 内 kernel 会整体不可见)、`-t nvtx`、`--cuda-memory-usage=true`;(b)若只为验证 matmul,脚本加 warmup(前 3~5 次迭代不计入)。验证方式:复采后 GPU 窗口 idle_pct < 5%、不再出现 >10 ms 单空隙。
- **P0-2 sgemm 热点 T = 99.6%,转 ncu 定根因**:规则 B 下 kernel 级热点不判根因,T = 99.6% ≥ 20% 为 P0 采集对象(来源:cuda_gpu_kern_sum Time%,参考口径)。依据:cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_align1> 10 实例合计 27.662 ms、均值 2.766 ms、StdDev 10.1 μs(实测)。修复方向:warmup 后用 ncu 采该 kernel(SOL 四象限 / roofline / pipe 利用率),再按标准模式(G = ES × T)重排优先级;是否值得深挖取决于 FP32 matmul 是否在真实业务热路径上——本 rep 为冒烟脚本,此条服务于「要深挖 matmul 本身」的场景。验证方式:拿到 ncu rep 后产出 ES,重算 G。

### 🟡 P1(3% ≤ G < 10%)

- 无。cudaDeviceSynchronize 26.153 ms(CPU 侧 7.43%)与 launch 均值 97.2 μs 均为冷启动/尾部排空的预期形态(依据见指标解读表),不立条目。

### 🟢 P2(G < 3%)

- **P2-1 sgemm 走 SIMT FP32 路径,未用 tensor core**:依据 kernel 名 `simt_sgemm`(实测)与折算 49.7 TFLOP/s(推断,profiler 环境折算)。若 FP32 matmul 属真实热路径,可开 TF32(torch.backends.cuda.matmul.allow_tf32=True)或改 BF16 走 tensor core;无 ES 且负载为冒烟脚本,G 不可算,留档观察,待 P0-2 的 ncu 数据验证。
- **P2-2 前 3331.933 ms(全 trace 71.7%)纯 CPU 启动段**:import/初始化一次性成本(来源:ANALYSIS_DETAILS + KERNEL MIN(start)),冒烟脚本预期,留档不动作。

## 四、下一步行动

1. 执行 P0-1 重采(G = 77.0%,预估上限):对真实 vLLM 服务进程采集(`--cuda-graph-trace=node -t nvtx --cuda-memory-usage=true --sample=cpu`),或至少给冒烟脚本加 warmup。完成判据:新 rep GPU 窗口 idle_pct < 5%,max_gap < 10× 平均 kernel 时长。
2. 执行 P0-2:warmup 后用 ncu 采 cutlass sgemm(T = 99.6%),拿 ES 后按标准模式重算 G;顺带验证 P2-1 的 TF32 假设(看 SOL 中 FMA/tensor pipe 利用率)。
3. 清理旁路 `nsys_smoke.sqlite`(比 rep 新 17 天、来历不明,本次未使用):删除或改名,避免未来 nsys 因 mtime 检查通过而静默采用旧数据。

## 附录:复现命令

```bash
# 环境与元信息
/usr/local/cuda/bin/ncu --version        # 2026.1.0.0
/usr/local/cuda/bin/nsys --version       # 2025.6.3.343
sqlite3 --version                        # 3.50.2
sha256sum /root/projects/vllm/experiments/pd_disagg/profiling/nsys_smoke.nsys-rep   # 0219945116a8...

# 提取管线(样本只读:先复制进 scratchpad;旁路 sqlite 不用,从 rep 重导)
S=/tmp/claude-0/-root/88ede59a-8180-499e-8272-830b701fec92/scratchpad/batch_pd-disagg
bash $S/extract.sh
# = cp rep 到 $S && /usr/local/cuda/bin/nsys stats $S/nsys_smoke.nsys-rep \
#     --sqlite $S/nsys_smoke_export.sqlite --force-export=true \
#     --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_kern_exec_sum,nvtx_pushpop_sum,osrt_sum \
#     --format csv --output $S/nsys_smoke_out --force-overwrite=true
# 成败判定:stdout+stderr 合流 grep 'Exportation error|ERROR:|SKIPPED' + csv 非空
# → 本次仅 nvtx_pushpop_sum SKIPPED(0 字节),其余 6 张 csv 正常

# sqlite 查询组(表探测/META_DATA_CAPTURE/TARGET_INFO_GPU/ANALYSIS_DETAILS/
#   busy-idle 区间合并/Top-5 gap/launch 画像/同步 API/stream 重叠/显存 API/执行序)
bash $S/queries.sh
# 其中:
#   [1] SELECT value FROM META_DATA_CAPTURE WHERE name='CUDA_GRAPH_TRACE_OPTIONS:MODE'  → Graph
#   [1b] PROCESS_0:COMMAND/ARGUMENT_*、COLLECT_CPU_TRACE=false、COLLECT_GPU_MEMORY_USAGE=false
#   [2] TARGET_INFO_GPU → 4090 ×2,1008096000000 B/s,75497472 B,128 SM
#   [3] ANALYSIS_DETAILS.duration → 4649936558 ns
#   [4] busy/idle → busy 27785368 ns / wall 122181943 ns / 22.74% / 77.26%
#   [5] Top gap → 94063472 ns @ gap_start 3332363951(randn → Kernel2)
#   [6] launch → 12 次,avg_launch 97203.8 ns,short_lt_5us=0
#   [7] 同步 API → cudaDeviceSynchronize 1 次 26153262 ns = 7.43%(RUNTIME 跨度 352076027 ns)
#   [8] stream → n_streams=1,overlap_factor=1.0
#   [9] MEMCPY 表不存在
#   [10] cudaMalloc 7 次 1154498 ns(Max 0.318 ms)、cudaFree 1 次
#   [12] KERNEL MIN(start)=3331933115,MAX(end)=3454115058

# gap 窗口归因 + 全 trace busy 占比(窗口 = [3332363951, 3426427423])
SQ=$S/nsys_smoke_export.sqlite
sqlite3 -header -csv $SQ "SELECT s.value api, COUNT(*) calls, SUM(MIN(r.end,3426427423)-MAX(r.start,3332363951)) in_gap_ns FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId WHERE r.end > 3332363951 AND r.start < 3426427423 GROUP BY s.value ORDER BY in_gap_ns DESC LIMIT 10"
#   → cuLibraryLoadData 43977478 / cuLibraryGetKernel 2295149 / cuKernelGetFunction 1690011 / cuGetProcAddress_v2 314890
sqlite3 -header -csv $SQ "SELECT s.value api, COUNT(*) calls, SUM(MIN(o.end,3426427423)-MAX(o.start,3332363951)) in_gap_ns FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId WHERE o.end > 3332363951 AND o.start < 3426427423 GROUP BY s.value ORDER BY in_gap_ns DESC LIMIT 8"
#   → poll 13 条交叠 265518553 ns(多线程,>窗口,不可归因)
python3 -c "print(round(100*27785368/4649936558,2))"   # 0.6  全 trace 口径 busy
python3 -c "print(round(100*94063472/122181943,1))"    # 77.0 P0-1 的 G
python3 -c "s=43977478+2295149+1690011+314890; print(round(s/1e6,3), round(100*s/94063472,1))"  # 48.278 ms / 51.3% 懒加载占空隙比
python3 -c "print(round(100*94063472/(122181943-27785368),2))"  # 99.65 gap 占 idle
python3 -c "print(round(100*3331933115/4649936558,1))" # 71.7 启动段占比
python3 -c "print(round(2*4096**3/2766187.3/1e3,1))"   # 49.7 TFLOP/s 折算

# stats CSV 解析(python csv 模块,kernel 名含逗号)
# cuda_gpu_kern_sum → Kernel2: 99.6% / 27661873 ns / 10 实例 / avg 2766187.3 / StdDev 10071.2
#                     randn: 0.4% / 115973 ns / 2 实例
# cuda_api_sum → cuLibraryLoadData 5 次 46208954 ns(58.3%);cudaDeviceSynchronize 26153262 ns(33.0%);
#                cudaLaunchKernel 2 次 998027 ns(Max 985262);cuLaunchKernel 10 次 168418 ns(avg 16841.8,Max 55096)
# cuda_kern_exec_sum → Kernel2: QCount 9, QAvg 13123346.3, KAvg 2766187.3
# cuda_gpu_mem_time_sum / mem_size_sum → memset 10 次 7522 ns / 0.020 MB
# osrt_sum → poll 3178112646 ns(71.0%)/ pthread_cond_timedwait 1000148494 ns(22.3%)
```
