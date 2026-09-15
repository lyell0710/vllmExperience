# EXP-020 · NCCL 旋钮矩阵复现 1.78 GB/s（EXP-019 §8 四步落地）

> **一句话结论**：**1.78 GB/s 复现成功，定因 = NCCL 走 Socket 网络路径（`NCCL_SHM_DISABLE=1`）**——该路径 16M–256M 平台 1.51–1.70 GB/s（5/5 档落入锁定窗 [1.5, 2.1]），与 EXP-002 的 1.77–1.85 同量级（低 ~8%）；而 SHM 路径今天 6.1–9.1 GB/s、`NCCL_P2P_LEVEL` 五种取值对传输选择零影响（P2P 驱动级禁用）、`NCCL_PROTO=LL/LL128` 2.9/4.4 GB/s 也不落窗。H1 的「Socket 路径」分支成立、「PCIe 未升 Gen4」分支不成立（全部档负载期 Gen4 x16 占比 ≥92%）；**为什么 8/21 那次 NCCL 选了 Socket** 因 EXP-002 无 NCCL_DEBUG 日志仍不可回溯。**附录 A（post-hoc）**：紧接的 EXP-021 抓到 SHM 路径自身 1/28 次塌陷到 2.1–2.3 GB/s、曲线形状比 Socket 档更像 EXP-002——**定因不唯一**，新增候选 H3「SHM 路径间歇塌陷态」，机理未知。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | 2×RTX 4090 无 P2P，driver 610.57.04，NCCL 2.28.9（v0.25.1 venv wheel，LD_LIBRARY_PATH 显式），nccl-tests tool_sha 717b683182（与 EXP-002/018 同一二进制） |
| 状态 | 完成（主矩阵判定 A；附录 A 登记 H3 替代候选，定因不唯一） |
| 关联清单项 | EXP-002《硬件三数》1.78 GB/s 复现；EXP-018《NCCL allreduce size 扫描》§7 / EXP-019《1.78 vs 6.2 GB/s 机制调查》§8 开放问题闭环 |

## 1. 目的与假设

EXP-019 把 1.78 vs 6.2 GB/s 判定为「真实环境差异」并留下两个候选：H1 = 传输路径/PCIe 状态（走 Socket 或 PCIe 未升 Gen4），H2 = 8/21 B1 战役并发负载。本实验按 EXP-019 §8 四步照做：抓完整 NCCL_DEBUG 日志 → 扫 NCCL 旋钮矩阵 → 每档同步采 PCIe 运行态 → 按锁定阈值判定。

**跑前锁定的判定阈值（跑完不改）：**

- 主矩阵 = `NCCL_SHM_DISABLE ∈ {0,1}` × `NCCL_P2P_LEVEL ∈ {默认, LOC, PIX, PHB, SYS}`（10 档）+ `NCCL_PROTO ∈ {Simple, LL, LL128}`（仅默认 SHM/P2P 下，3 档）；每档 `-b 1M -e 512M -f 2 -g 2 -n 20`（与 EXP-002/018 同参数），读 1M / 16M / 256M 三点 out-of-place busbw 与 Avg bus bandwidth。
- **判定 A（复现成功）**：某档在 1M–512M 区间的平台（取 16M–256M 均值）落在 **1.5–2.1 GB/s** → 「1.78 复现成功，定因 = 该档配置」；再核该档 NCCL 自报路径与 EXP-002 可推断条件是否吻合。
- **判定 B（不可复现）**：所有档平台都 ≥5 GB/s，或都明显偏离 1.78（不在 1.5–2.1 区间）→ 「1.78 不可由 NCCL 旋钮复现，H1（PCIe/Socket 路径）在本机当前态不成立，剩余候选 H2（8/21 并发负载）」。
- 额外候选档（预注册，不进主矩阵判定，只作旁证）：`NCCL_MAX_NCHANNELS=1`、`NCCL_SHM_USE_CUDA_MEMCPY=1`、`NCCL_ALGO=Tree`，同一阈值判读。
- PCIe 运行态判据：每档 500ms 轮询 `pcie.link.gen.current`；若某档测量期间 gen 未升到 4 而带宽落入 1.5–2.1，则 H1 的「PCIe 未升 Gen4」分支成立；若所有档测量期间都在 Gen4 且带宽 ≥5，则该分支在当前态无法触发。

## 2. 环境与配置

- 二进制 `/root/tools/nccl-tests/build/all_reduce_perf`（tool_sha 717b683182）；`LD_LIBRARY_PATH=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib`（libnccl.so.2.28.9，nccl-library=22809，与 EXP-002 raw 头部一致）。
- 每档统一附加 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV`（只在初始化期打印，不进计时区）以留下每档的 NCCL 自报路径；step1 的完整日志用 `INIT,NET,GRAPH,ENV`。
- 硬件占用：双卡（-g 2），跑前 `nvidia-smi --query-compute-apps` 确认无其他 compute 进程；系统空闲（load 见 §4 pcie.csv 的 SM clock 基线 210 MHz）。
- PCIe 采样：`nvidia-smi --query-gpu=index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm --format=csv -lms 500`，每档单独文件。
- 脚本：`scripts/nccl_knob_matrix.sh`（执行）+ `scripts/nccl_knob_matrix_analyze.py`（解析/判定，取不到数即报错中止）。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv   # 确认空闲
bash scripts/nccl_knob_matrix.sh                                # step1–3，STAMP 共用前缀
python3 scripts/nccl_knob_matrix_analyze.py <STAMP>            # step4，写 derived CSV + 判定
```

## 4. 原始数据

共用前缀 `20260915T0237`，全部在 `pd_disagg/hw/`，每个文本文件首行 provenance（含完整命令与该档旋钮）：

| 文件 | 内容 |
|---|---|
| `20260915T0237_nccl_debug_full.log`（+`_pcie.csv`） | step1：`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV` 完整 stdout+stderr（216 行），1M–512M n=20，avg 8.43 GB/s |
| `20260915T0237_nccl_knob_shm{0,1}_p2p{DEF,LOC,PIX,PHB,SYS}_protoDEF.txt` | 主矩阵 10 档，逐 size 表 + NCCL 自报 `Channel .. via ..` 行 + `# exit_code` |
| `20260915T0237_nccl_knob_shm0_p2pDEF_proto{Simple,LL,LL128}.txt` | PROTO 3 档 |
| `20260915T0237_nccl_knob_extra_{nchan1,shmcudamemcpy1,algoTree}.txt` | 预注册额外 3 档 |
| 每档同名 `_pcie.csv` | 500ms `nvidia-smi` 轮询：index,timestamp,pcie gen,width,SM clock |
| `pd_disagg/hw/derived/20260915T0237_nccl_knob_matrix.csv` | step4 汇总：每档 1M/16M/256M busbw、平台均值、avg、自报路径、PCIe Gen4 占比、SM 峰值、判定 |

脚本：`scripts/nccl_knob_matrix.sh`（采集）、`scripts/nccl_knob_matrix_analyze.py`（解析/判定，缺任一 size 行、Avg 行、via 行、PCIe 负载样本即中止）。对照数据：EXP-002 `pd_disagg/hw/all_reduce_perf.txt`（2026-08-21）、EXP-018 `pd_disagg/hw/20260829T104705_allreduce_size_scan_large.txt`。

## 5. 结果

**① 旋钮矩阵（out-of-place busbw，GB/s；平台 = 16M–256M 五点均值；判定按 §1 锁定阈值）**——`derived/20260915T0237_nccl_knob_matrix.csv`

| 档 | SHM_DISABLE | P2P_LEVEL | PROTO | 1M | 16M | 256M | 平台 | avg | NCCL 自报路径 | 负载期 Gen4 占比 | 判定 |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| shm0_p2pDEF | 0 | 默认 | 默认 | 7.09 | 9.15 | 9.07 | 9.07 | 8.82 | SHM/direct/direct | 92% | B（≥5） |
| shm0_p2pLOC | 0 | LOC | 默认 | 7.17 | 9.16 | 9.07 | 9.14 | 8.95 | SHM/direct/direct | 92% | B |
| shm0_p2pPIX | 0 | PIX | 默认 | 7.48 | 8.64 | 7.81 | 8.57 | 8.30 | SHM/direct/direct | 92% | B |
| shm0_p2pPHB | 0 | PHB | 默认 | 4.52 | 6.11 | 6.85 | 6.41 | 6.17 | SHM/direct/direct | 95% | B |
| shm0_p2pSYS | 0 | SYS | 默认 | 4.85 | 6.29 | 6.75 | 6.47 | 6.10 | SHM/direct/direct | 93% | B |
| **shm1_p2pDEF** | 1 | 默认 | 默认 | 1.22 | 1.63 | 1.69 | **1.67** | 1.58 | **NET/Socket/0** | 98% | **A（落窗）** |
| **shm1_p2pLOC** | 1 | LOC | 默认 | 1.25 | 1.66 | 1.72 | **1.70** | 1.60 | NET/Socket/0 | 98% | **A** |
| **shm1_p2pPIX** | 1 | PIX | 默认 | 1.17 | 1.48 | 1.52 | **1.51** | 1.43 | NET/Socket/0 | 98% | **A** |
| **shm1_p2pPHB** | 1 | PHB | 默认 | 1.21 | 1.68 | 1.71 | **1.69** | 1.58 | NET/Socket/0 | 98% | **A** |
| **shm1_p2pSYS** | 1 | SYS | 默认 | 1.22 | 1.65 | 1.71 | **1.68** | 1.59 | NET/Socket/0 | 98% | **A** |
| shm0_protoSimple | 0 | 默认 | Simple | 5.70 | 5.92 | 6.40 | 6.10 | 5.98 | SHM/direct/direct | 96% | B |
| shm0_protoLL | 0 | 默认 | LL | 2.94 | 2.96 | 2.96 | 2.97 | 2.91 | SHM/direct/direct | 96% | B（OTHER） |
| shm0_protoLL128 | 0 | 默认 | LL128 | 4.10 | 4.43 | 4.42 | 4.51 | 4.38 | SHM/direct/direct | 95% | B（OTHER） |
| extra_nchan1 | 0 | 默认 | 默认 +MAX_NCHANNELS=1 | 6.45 | 7.18 | 6.91 | 7.10 | 6.87 | SHM/direct/direct | 95% | B |
| extra_shmcudamemcpy1 | 0 | 默认 | 默认 +SHM_USE_CUDA_MEMCPY=1 | 6.13 | 8.52 | 8.32 | 8.45 | 7.94 | SHM/CE/direct | 95% | B |
| extra_algoTree | 0 | 默认 | 默认 +ALGO=Tree | 7.17 | 8.92 | 9.00 | 8.88 | 8.81 | SHM/direct/direct | 92% | B |

**② 复现档 vs EXP-002 逐 size 对照（out-of-place busbw，GB/s）**

| size | 1M | 2M | 4M | 8M | 16M | 32M | 64M | 128M | 256M | 512M | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-002（8/21，路径未知） | 1.75 | 1.67 | 1.83 | 1.78 | 1.77 | 1.79 | 1.79 | 1.83 | 1.85 | 1.84 | 1.78 |
| 本实验 shm1_p2pDEF（Socket） | 1.22 | 1.40 | 1.52 | 1.61 | 1.63 | 1.66 | 1.68 | 1.70 | 1.69 | 1.69 | 1.58 |
| 本实验 shm0_p2pDEF（SHM） | 7.09 | 7.86 | 9.12 | 9.18 | 9.15 | 9.03 | 9.06 | 9.04 | 9.07 | 9.07 | 8.82 |
| EXP-018（8/29，SHM） | 5.45 | 5.47 | 6.10 | 6.45 | 6.42 | 6.40 | 6.45 | 6.48 | 6.50 | 6.38 | 6.20 |

**③ step1 完整日志的 NCCL 自报关键事实**（`20260915T0237_nccl_debug_full.log`）：`NCCL version 2.28.9+cuda13.0`；`P2P is disabled between connected GPUs`；`cuMem host allocations do not appear to be working; falling back to /dev/shm`；拓扑 `GPU/0-c1000 ↔ GPU/0-e1000 (2/24.0/PHB)`、`NET/0-0 (3/1.2/PHB)`；默认路径 `Channel 00..03 : 0[0] -> 1[1] via SHM/direct/direct`，4 channels ring。

**④ PCIe 运行态**：16 档负载期（SM>1 GHz）样本 Gen4 占比 92–98%、width 全 16、SM 峰值 2805–2835 MHz；Gen2 样本仅出现在起停边沿（空闲态 Gen1/2 → 负载 Gen4）。

## 6. 分析与结论

**【实测】① 按锁定阈值判 A：1.78 GB/s 可由 NCCL 旋钮复现，复现档 = `NCCL_SHM_DISABLE=1`（Socket 路径）。** 5 个 Socket 档平台 1.51–1.70 全部落窗，`NCCL_P2P_LEVEL` 五种取值在 Socket 下差异 ≤0.2 GB/s；EXP-002 平台 1.77–1.85，本次 Socket 平台低 ~8%（在窗内）。NCCL 自报 `via NET/Socket/0` + 拓扑 `NET 1.2 GB/s`（eth0 10 GbE 的 1.25 GB/s 标称）与 1.6–1.7 的实测量级自洽（allreduce busbw 计 2×(n-1)/n=1 倍 algbw，ring 双向各占一半链路，1.7 GB/s busbw ≈ 双向合计 ~1.7 GB/s 通过 loopback socket）。

**【实测】② H1 的「PCIe 未升 Gen4」分支不成立。** 所有档负载期 Gen4 x16 占比 ≥92%，没有任何档在 Gen1/2 下测量；今天 SHM 默认档 9.07 GB/s 平台，PCIe 状态不是 1.78 的成因。

**【实测】③ `NCCL_P2P_LEVEL` 对本机零效果**：P2P 是驱动级禁用（`cudaDeviceCanAccessPeer=0`，EXP-002 topo GNS），NCCL 不论 LEVEL 取值都选 SHM；shm0 五档间 6.1–9.1 的分散（PHB/SYS 档偏低）在 via 路径、PCIe、SM 时钟上都无差异，属 SHM 路径的运行间抖动（见 §7），不是 LEVEL 的效应。

**【实测】④ 协议旋钮不产生 1.78**：LL 2.9（平台平坦，是 LL 8 字节 flag 协议的固定开销形状）、LL128 4.4、Simple 6.1（默认自动选择 ≈ Simple 但今天略高，见 §7）。

**【推断】⑤ 8/21 EXP-002 为何走了 Socket——不可回溯，两种候选机制：** (a) 当时 shell 环境里带了 `NCCL_SHM_DISABLE=1` 或等价（`NCCL_NET_DISABLE_INTRA` 反向）设置——EXP-002 provenance `env=n/a`，无法排除；(b) NCCL 拓扑检测把 GPU↔GPU 路径带宽估得低于 NET（`ncclTopoCheckNet` 逻辑：intra-node 路径 bw ≤ NET bw 时改走网络）——本机自报 24.0 vs 1.2，正常态不会触发。1M 点形状差异（EXP-002 从 1M 起就平坦 1.75，Socket 今天 1M 只有 1.22 且爬升到 8M 才平）说明 8/21 的路径与今天 Socket 档**同量级但不完全同形**，"定因 = Socket 路径"是按阈值的一级判定，不是逐点复刻。

**【推断】⑥ H2（8/21 并发负载）未被本实验检验**：矩阵在空闲机上跑，不能排除"并发负载 + SHM 路径也能掉到 1.78"；但既然 Socket 路径单独就能给出落窗数字，H2 从"唯一剩余候选"降为"次要候选"。去向见 §7。

**结论对 EXP-018/019 的回答**：1.78 与 6.2（今天 8.8）不是同一路径的数字——前者是 Socket 网络路径（NCCL 在无 P2P 时的第三顺位回退），后者是 SHM 中转（第二顺位）。对外引用 TP collective 带宽时必须注明路径；本机正常态（SHM）的大消息平台今天为 9.1 GB/s，8/29 为 6.5 GB/s，两次都远高于 1.78。

## 7. 异常、偏差与开放问题

- **SHM 路径自身第三次漂移**：1M 点 3.96（EXP-019 单点，8/30）/ 5.45（EXP-018，8/29）/ 7.09（本次）；平台 6.45（8/29）→ 9.07（本次）。三次都是 `via SHM/direct/direct`、Gen4 x16。本次矩阵内 shm0 五档也分散在 6.1–9.1。**SHM 路径的绝对数字不稳定（±30%）**，可能与 CPU 亲和/NUMA 调度、host 页缓存态有关（NCCL 自报 `Affinity for GPU 0 is 32-45,96-109`，本次未固定 CPU 亲和）。去向：若要给 SHM 路径一个对外数字，需固定 `taskset`/`NCCL_IGNORE_CPU_AFFINITY` 并 ≥3 轮取 stability——不在本实验范围，登记为 EXP-018 §7 抖动条目的加强版。
- **默认自动协议 vs 强制 Simple**：默认档 9.07 而强制 Simple 6.10，理论上大消息默认就是 Simple；差异同属上条抖动，本次未做同档重复，不下结论。
- **协议偏离**：无。矩阵、参数、阈值与 §1 预注册一致；额外 3 档为预注册旁证，均判 B。
- **8/21 Socket 选路成因不可回溯**（§6 ⑤）：EXP-002 无 NCCL_DEBUG、无环境变量记录。开放问题去向：不再追查（成本 > 收益），EXP-002 结论改写为"该次测量走了 Socket 路径"级别的追认，由主线程决定措辞。
- **H2 旁证（计划）**：T4（EXP-023）起 replica2 双实例时，机会性地在 vllm 负载下跑一次 shm0 默认档，看 SHM 路径是否会被并发负载压到 1.78 量级；若做了，结果以 `20260915T*_nccl_knob_h2_*` 前缀落盘并在本记录末尾追加「附录 A」（post-hoc，非预注册，只作旁证）。
- `NCCL_P2P_LEVEL` 的合法值按 NCCL 文档取 LOC/PIX/PHB/SYS（未含 NVL/PXB，本机无 NVLink、PXB 与 PHB 在本拓扑等价）。

## 8. 下游影响

- **EXP-002《硬件三数》的 1.78 GB/s 有了定性归属：Socket 网络路径数字**，不是 SHM 中转路径的带宽墙。EXP-018/019 的"两个数都停用"可以解除为"分路径引用"：Socket 路径 1.5–1.7（本实验 5 档），SHM 路径 ≥6（EXP-018 6.2 / 本次 8.8，抖动大，引用区间不引用单值）。
- **EXP-005 / RESUME_EVIDENCE 里"1.78 GB/s collective 墙预言 TP2 prefill 零加速"的因果链需重审**：vLLM TP2 实际走的是 SHM 路径（vLLM 进程无 `NCCL_SHM_DISABLE`），其带宽是 6–9 GB/s 而非 1.78；EXP-005 的"28 层 × 58.7MB allreduce 撞 1.78 墙"应按 SHM 路径数字重算（58.7MB/6.2 GB/s ≈ 9.5ms/层 vs /1.78 ≈ 33ms/层），零加速的定量解释强度下降——由主线程决定是否降级措辞。本记录不改任何已有文档。
- 工装：`scripts/nccl_knob_matrix.sh` + `nccl_knob_matrix_analyze.py` 可复用为"任意 NCCL 旋钮 × PCIe 运行态"采集器；provenance 行现在记录每档旋钮（弥补 EXP-002 教训）。
- 红线：对外写 TP collective 带宽必须带路径（SHM/Socket）+ 指针，单值 1.78 不再裸引。

## 附录 A（post-hoc，2026-09-15 03:00Z 追加）· SHM 路径间歇塌陷态——定因不唯一

**来源**：EXP-021《NCCL allreduce dtype 扫描》在本实验之后 10 分钟跑的 float 大消息第二轮（`pd_disagg/hw/20260915T0248_allreduce_size_scan_float_large_r2.txt`），NCCL 自报 `via SHM/direct/direct`（不是 Socket），却得到：

| size | 1M | 2M | 4M | 8M | 16M | 32M | 64M | 128M | 256M | 512M | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EXP-021 float r2（SHM，塌陷） | 1.91 | 1.70 | 1.91 | 2.07 | 2.05 | 2.09 | 2.13 | 2.21 | 2.29 | 2.29 | 2.07 |
| EXP-002（8/21，路径未知） | 1.75 | 1.67 | 1.83 | 1.78 | 1.77 | 1.79 | 1.79 | 1.83 | 1.85 | 1.84 | 1.78 |
| 本实验 shm1_p2pDEF（Socket） | 1.22 | 1.40 | 1.52 | 1.61 | 1.63 | 1.66 | 1.68 | 1.70 | 1.69 | 1.69 | 1.58 |

**【实测】** 塌陷态平台 2.15（16M–256M 均值）在锁定窗 [1.5, 2.1] 之外 2%，但**曲线形状（1M 起即平坦、缓升到 512M）与 EXP-002 逐点吻合**（逐 size 比值 1.02–1.24），而 Socket 档在 1M–8M 是爬升形（比值 0.70–0.90）。**【实测】** 跟进探针 `scripts/nccl_shm_collapse_probe.sh`（float/half 交替 5 轮，200ms PCIe 采样，raw `20260915T0252_nccl_shm_probe_*`）10 轮平台 5.85–8.73、负载期 Gen4 x16 占比 97–100%、无塌陷。今天 SHM 路径大消息扫描共 28 次，塌陷 1 次（EXP-021 未带 PCIe 采样，塌陷瞬间的链路状态未知）。

**【推断】对 §6 结论的修正**：主矩阵判定 A（Socket 路径落窗）按预注册阈值**维持**；但 1.78 的成因现在有两个都够格的候选，**不能唯一定因为 Socket**：
- H1-Socket：`NCCL_SHM_DISABLE=1` 或等价选路 → 1.5–1.7，量级吻合、形状不吻合；
- **H3-SHM 塌陷态**：SHM 路径在某种未知状态下 → 2.1–2.3，形状吻合、量级高 15–25%；出现频率今天 1/28；候选机理：PCIe 链路未升频（EXP-019 H1 的另一分支，本次探针 10 轮都在 Gen4 故未能验证）、主机侧页/THP 状态、其他未知。
- H2（并发负载）仍未检验，见 §7。

**去向**：定因需要"带 PCIe 采样的长跑（≥50 轮）抓塌陷态"，成本约 10 分钟 GPU，由用户决定是否投入；在此之前 EXP-002 的 1.78 措辞只能到"**非 SHM 正常态数字**（Socket 路径或 SHM 塌陷态），不代表本机 TP collective 的正常带宽"这一级。

## 附录 B（post-hoc，2026-09-15 03:40Z 追加）· H2「并发 vLLM 负载」旁证——两次尝试均因显存不足未得数据

借 EXP-023 的 replica2 双实例栈（两卡各占 23620/24564 MiB）做「负载发生器（`vllm bench serve` 8300，conc128，1200 请求，seed 7777，非记录点）并发时跑 SHM 路径 allreduce」的旁证，设计为 control_idle ×2 → under_load ×2 同命令对照：

| 尝试 | allreduce 配置 | 结果 | raw |
|---|---|---|---|
| 1 | `-b 1M -e 64M -n 20`，默认 4 channel | 4/4 `exit_code=3`：`allocator.cc:63 NCCL WARN Cuda failure 2 'out of memory'`（通信器初始化成功、NCCL 通道 buffer 分配失败） | `pd_disagg/hw/20260915T0335_nccl_h2_{control_idle,under_vllm_load}_r{1,2}.txt` + `_pcie.csv`；负载发生器 `20260915T0335_nccl_h2_loadgen_bench.log`（1200/1200 成功，21.46 req/s） |
| 2 | `NCCL_MAX_NCHANNELS=1 NCCL_BUFFSIZE=1048576 -b 1M -e 8M -n 20` | 同样 4/4 OOM | `20260915T0338_nccl_h2min_*`；负载发生器 21.65 req/s |

**结论**：H2 在"vLLM 常驻 + nccl-tests 并跑"的形态下**不可测**（CUDA 上下文 + NCCL kernel 加载后 ~900 MiB 余量不够）。要测 H2 需要把 vLLM 实例的 `--gpu-memory-utilization` 降到 ≤0.85 起栈——那已不是 EXP-007 的原配置，属于新实验；**H2 维持「未检验」**，§6⑥ 与附录 A 的候选排序不变。脚本 `scripts/nccl_h2_concurrent_load_probe{,_min}.sh` 保留（含就绪栈上的对照设计，换低显存栈即可复用）。

## 附录 C（预注册，2026-09-15 09:40Z 开跑前写入）· H3「SHM 路径间歇塌陷态」长跑探针

**动机**：附录 A 抓到 SHM 路径 1/28 次塌陷（平台 2.15 GB/s，曲线形状比 Socket 档更像 EXP-002），但那次没带 PCIe 采样，无法判断塌陷瞬间的链路状态。本探针把频率与链路状态一起抓。

**跑前锁定的判定阈值（跑完不改）**：

- 探针形态：SHM 路径（默认档，不设任何 NCCL 旋钮）大消息扫描 `-b 1M -e 512M -f 2 -g 2 -n 20`，float 与 half 交替，目标 60 轮；**抓到 ≥2 次塌陷即提前停**（已足够刻画，不必跑满）。每轮同步 200ms 采 `pcie.link.gen.current/width, clocks.sm/mem, pstate, power`。
- 塌陷定义：该轮大消息平台（16M–256M 均值）**< 3.0 GB/s**（正常态今天实测 5.9–9.1，塌陷态 2.1–2.3，3.0 是两者之间的空带）。
- **判定 A（H3 成立）**：抓到 ≥1 次塌陷，且塌陷轮次的负载期 PCIe **未升到 Gen4 x16**（Gen4 占比 < 50%）→ H3 机理 = PCIe 链路未升频，与 EXP-019 的 H1 分支合并；且能解释 8/21 的 1.78。
- **判定 B（H3 成立但机理另寻）**：抓到 ≥1 次塌陷，而塌陷轮次 PCIe 仍稳定 Gen4 x16 → 塌陷与链路速度无关，候选转向主机侧（页缓存/THP/NUMA 调度）或未知，H3 保留但归因推迟。
- **判定 C（H3 低频/不成立）**：60 轮零塌陷 → 塌陷频率 <1.7%，**1.78 归因于 Socket 路径的概率上升**（附录 A 的 H1-Socket 成为唯一单档可复现的解释）；但**不追认**——因为 1/28 与 1/60 的观测差异在统计上不足以排除低频塌陷（泊松下界）。
- 附带读数（不进判定）：dtype 与塌陷的相关性（float 是否更易塌）、sm/mem clock 与功率在塌陷轮的变化。

**结果（2026-09-15T09:36–09:46Z，60 轮，STAMP=20260915T0936）**

- 汇总 `pd_disagg/hw/derived/20260915T0936_nccl_h3_longrun.csv`（逐轮：平台/avg/Gen4占比/SM/mem/pstate/判定）；raw `pd_disagg/hw/20260915T0936_nccl_h3_{float,half}_r{1..60}.txt` + 同名 `_pcie.csv`；脚本 `scripts/nccl_h3_longrun_probe.sh`。
- **平台分布**：60 轮 min 3.23 / max 9.23 / mean 7.37 GB/s；float(30 轮) mean 7.25、half(30 轮) mean 7.48。
- **按锁定阈值（<3.0 判塌陷）：0 次塌陷 → 判定 C**——塌陷频率 <1.7%（泊松 1σ 上界约 5%）；60 轮负载期 PCIe Gen4 x16 占比 min 0.97、SM 2835 MHz、mem 10501 MHz、pstate P0/P2（个别轮含 P5）。
- **post-hoc 观察（不改判定，阈值不动）**：第 43 轮（float）平台 **3.23 GB/s**，未达塌陷线，但其**逐 size 曲线从 1M 起即平坦**（2.98 / 3.41 / 3.45 / 3.21 / 3.02 / 3.05 / 3.09 / 3.35 / 3.64 / 3.61），与 EXP-021 那次 2.15 的塌陷、以及 EXP-002 的 1.78 是**同一个曲线家族**（正常轮是 5.11→7.59 的爬升+平台形）。若改用"形状"判据而非"绝对值<3.0"，低平台态频率为 **1/60**；与 EXP-021 的 1/28 合并 = **2/88 ≈ 2.3%**，且**两次都在 Gen4 x16、时钟满血下发生** → H3 作为现象成立，但机理**不是** PCIe 未升频（判定 B 的机理分支），候选转向主机侧或未知。
- **对 1.78 归因的影响（不改 §6 结论强度）**：观测到的低平台态分位是 2.15 与 3.23，**尚未观测到 ≤2.3 以下**；而 Socket 档稳定复现 1.51–1.70。1.78 落在 Socket 档略微之上、低平台态之下——**证据天平向 Socket 路径倾斜，但仍不足以追认**（60+28 轮的样本量对"更低平台态存在"这一可能性给不出否定）。维持 §6 的"定因不唯一"与 R0-1 的"停用"。

**工具 bug 与作废前缀（登记）**：探针首跑（`STAMP=20260915T0933`）的内联解析代码把捕获组号写错（`group(4)` 应为 `group(2)`），逐轮判定全部落空、平台值打印为空，判据形同失效——本次随即终止该前缀，并删除其 0 数据行的 `derived/20260915T0933_nccl_h3_longrun.csv`（derived 可重算，非 raw）。该前缀下 4 轮 raw（`20260915T0933_nccl_h3_{float_r1,float_r3,half_r2,half_r4}.txt` + `_pcie.csv`）**本身是有效测量**，但为保持单前缀可比性未纳入汇总，**原地保留不删**（铁律 3）。修好组号后以新前缀 `20260915T0936` 重跑 60 轮，即上表数据。

**依据**：`scripts/nccl_shm_collapse_probe.sh`（EXP-021 §7 用过，10 轮无塌陷）扩到 60 轮 + 塌陷早停 + 每轮落 CSV。

