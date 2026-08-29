# EXP-019 · 1.78 vs 6.2 GB/s 机制调查（环境 diff，先于任何 bench）

> **一句话结论**：**计时口径干净，不是「计时区混入非传输开销」**——nccl-tests 用 CUDA event 计时、计时区内只有 ncclAllReduce 入队；真正差异是**传输路径/环境状态**：无 P2P 时 NCCL 走 CPU 中转（SHM）实测 3.96–6.2 GB/s，强制走网络（Socket）只剩 0.76 GB/s，EXP-002 的 1.78 落在「路径/PCIe 状态不同」的解释区间，非固定开销假象。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-30 |
| 环境 | 2×RTX 4090，无 P2P，driver 610.57.04，NCCL 2.28.9（v0.25.1 venv wheel，LD_LIBRARY_PATH 显式） |
| 状态 | 完成（判定升级为真实环境差异，另立假设） |
| 关联 | EXP-002《硬件三数》复测差异闭环；EXP-018《NCCL allreduce size 扫描》§7 开放问题 |

## 1. 目的与假设

EXP-018 发现 EXP-002 的「1.78 GB/s 带宽墙」复现不了（同二进制同 NCCL 同参数复测得 6.2 GB/s，差 3.5 倍）。本实验按「先 diff 环境、不先跑 bench」排查机制。

跑前锁定的判定阈值（跑完不改）：
- **若在计时区内找到非传输混入物（malloc/free/deviceProperties 等），且扣除后两数收敛到 <10% 差异 → 判定「计时口径问题」**；
- **否则升级为「真实环境差异」，另立假设并列出验证方法**。

## 2. 环境与配置

排查手段（均为只读/单点对照，不跑满矩阵）：
- `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH` 抓 NCCL **自报**的算法、拓扑、传输路径。
- 逐行读 nccl-tests 计时源码（`/root/tools/nccl-tests/src/common.cu` 的 `BenchTime`/`recordEvents`/`getElapsedTimes`）。
- 单点对照：默认（SHM 可用）vs `NCCL_SHM_DISABLE=1`（强制走网络 Socket）各 1 个 1M 点。

## 3. 步骤

```bash
# a. NCCL 自报
NCCL_DEBUG=INFO LD_LIBRARY_PATH=$VENV_NCCL all_reduce_perf -b 1M -e 8M -f 2 -g 2 -n 2
# b. 对照实验
#  A: 默认（SHM）1M 单点
#  B: NCCL_SHM_DISABLE=1 1M 单点
```

## 4. 原始数据

- NCCL_DEBUG 日志：本记录 §5 引用；完整输出未落盘（终端级证据，仅用于路径判断，不承载对外数字）。
- 对照实验输出：终端级证据（单点、非正式 bench），带宽值见 §5。

## 5. 结果

**① NCCL 自报的关键事实（NCCL_DEBUG=INFO）：**

| 项 | 自报值 | 含义 |
|---|---|---|
| P2P | `P2P is disabled between connected GPUs 1 and 0` | 无 P2P 确认 |
| 传输插件 | `Using network Socket` / `NET/Socket : Using eth0:10.42.91.42` | 初始化了 Socket 网络插件 |
| host 分配 | `cuMem host allocations do not appear to be working; falling back to /dev/shm` | host 内存分配回退 SHM |
| 拓扑 NET 路径 | `NET/0-0 (3/1.2/PHB)` | 网络路径带宽仅 **1.2** |
| 拓扑 GPU↔GPU | `GPU/0-e1000 (2/24.0/PHB)` | 经 CPU/PHB 中转 **24.0** |
| 算法 | `Connected all rings, use ring PXN 0 GDR 1` | ring 算法 |
| 通道 | `4 coll channels, 4 collnet channels, 0 nvls channels, 4 p2p channels` | — |

**② 计时代码核查（common.cu）：** `BenchTime` 计时循环内只调 `startColl`（即 `ncclAllReduce` 入队到 stream），配合 `recordEvents`（CUDA event 打点）；`completeColl` 做 stream sync。**计时区内无 cudaMalloc/cudaFree/cudaGetDeviceProperties/数据初始化**（`InitData` 在 `datacheck` 分支、计时循环之外）。默认 `blocking_coll=0`、`per_iter_timing=0` 时用 CPU 墙钟 `tim.elapsed()`，大消息下 GPU 传输主导、CPU 入队开销可忽略。

**③ 单点对照（1M 消息）：**

| 配置 | busbw | 说明 |
|---|---|---|
| 默认（SHM 中转） | 3.96 GB/s | 与 EXP-018 扫描的 4.73–5.45 同量级 |
| NCCL_SHM_DISABLE=1（强制 Socket） | **0.76 GB/s** | 掉 5.2 倍 |

## 6. 分析与结论

**① 计时口径排除（证伪「混入固定开销」假设）。** 按锁定阈值的第一分支：计时代码里找不到非传输混入物。EXP-002 的 1.78 不是「多测了一段 malloc/free」——这推翻了本轮交接单押注的「与 reduce 那次的 cudaMalloc 同型」假设。两个现象要分开看：llm-engine 的 88µs（D22 的 torch.distributed event 区间含等对端）确实是「多测了固定开销」；但 EXP-002 的 1.78 是 nccl-tests 的干净计时，性质不同。

**② 真正差异是传输路径/环境状态，方向收敛到两点：**

- **SHM vs Socket 差 5 倍**（3.96 vs 0.76 GB/s）。NCCL 拓扑自报 NET 路径仅 1.2 GB/s、CPU 中转 24 GB/s。1.78 GB/s 落在「Socket 之上、SHM 之下」——若 EXP-002 那次 NCCL 走了 Socket/网络路径（或混合），1.78 完全合理。
- **PCIe 链路状态**：EXP-018 已实测 allreduce 运行时 PCIe 从空闲 Gen1 升到 Gen4；EXP-002 的 provenance 未记录 PCIe 运行态。Gen1 的 SHM 中转带宽恰好是 Gen4 的约 1/4 量级。

**③ 结论：升级为真实环境差异，另立假设（按锁定阈值第二分支）。** 两个候选假设：

- **H1（首要）：EXP-002 测量时 PCIe 未升到 Gen4（或走了 Socket 网络路径）**，导致 SHM 中转带宽只有 1.78 量级。证据强度：对照实验证明路径切换可造成 5 倍差异；PCIe Gen 跳变量级（Gen1≈Gen4 的 1/4）与 3.5 倍吻合。
- **H2（次要）：8/21 B1 战役并发负载挤占 PCIe/内存带宽。** EXP-002 §7 自述测量时「B1 矩阵战役同日」，但无法回溯验证。

**关键负面结论（诚实标注）：** EXP-002 的 provenance 只记了 `env=n/a sha=n/a`，既没记 NCCL 环境变量、也没记 PCIe 运行态、更没记 NCCL_DEBUG 日志——**这次 3.5 倍差异无法定因，根子在 EXP-002 当时的 provenance 缺失**。这与 reduce 计时 bug 是同一类教训的第二种形态：那次是「计时区混入了东西」，这次是「测量时没记下足以复现的环境态」。

## 7. 异常、偏差与开放问题

- **无法回溯 EXP-002 当时的路径选择**：NCCL 算法/拓扑/传输路径是运行时决定的，EXP-002 没留 NCCL_DEBUG 日志，无法知道它当时走了 SHM 还是 Socket、PCIe 在 Gen1 还是 Gen4。
- 对照实验是单点、未重复，仅用于路径定性判断，不承载对外带宽数字。
- 未测 `NCCL_P2P_LEVEL` 不同取值、`NCCL_PROTO` 强制 Simple/LL/LL128 的带宽分档——这些是「复现 1.78」时该扫的旋钮。

## 8. 下游影响

- 判定**升级为真实环境差异**，不追认 EXP-002 为「测错」，也不追认 6.2 为权威——**在复现 1.78 之前，两个数都不作对外带宽墙引用**（与用户 2026-08-30 的「不追认、先停用」一致）。
- 复现实验设计（下次开查时直接照做，不再重新想）：
  1. `NCCL_DEBUG=INFO` 抓一次完整日志落盘（provenance 首行 + 日志全文）。
  2. 扫 `NCCL_P2P_LEVEL=0/1/...` 与 `NCCL_SHM_DISABLE=0/1` 的组合，找哪一档复现 1.78。
  3. 记录每档的 PCIe 运行态（`nvidia-smi` 轮询）+ NCCL 自报路径。
  4. 若某档复现 1.78 且与 EXP-002 当时的可推断条件吻合，则定因。
- 把「provenance 必须记录 NCCL 环境变量 + PCIe 运行态 + NCCL_DEBUG 日志」补进 EXP-002 的 §7 教训，供后续硬件测量复用。
