# vLLM 推理部署选型与 MoE 优化 · 实验证据仓库

在 **2×RTX 4090（无 NVLink、P2P 驱动禁用）** 上回答两个工程问题：**① 多出一张卡该怎么用**
（colocate / 双实例数据并行 / TP2 / PD 分离四臂选型，含瓶颈归因）；**② MoE 推理慢在哪、
还能快多少**（kernel 级分解 → config 调优 → 上游 PR 材料）。
B1 矩阵 84 个有效测量点（协议 v2，每点唯一 seed）、17 份八节实验记录、
全部数字首行 provenance 可溯源。

> 独立嵌套 git 仓库（外层 vllm 仓库通过 `.git/info/exclude` 忽略本目录，互不干扰）。
> **所有能进简历/报告的数字、图表、trace 的唯一权威存放地。**
> 简历句与证据的对应关系见 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)；
> 接手/交接从 [HANDOFF.md](HANDOFF.md) 读起。

## 🎯 Headline 结果

| 结论 | 关键数字 | 证据 |
|---|---|---|
| **两卡选型：数据并行完胜 TP2 与 PD 分离** | 饱和吞吐（req/s，2K 输入桶）：replica2 **7.00** · tp2 4.16 · pd1p1d 2.12 · colocate 单卡基线 3.63 | EXP-007 · `pd_disagg/results/b1_matrix/runs.jsonl` |
| **TP2 只赢 decode**：decode 提速 42%（权重带宽分摊），prefill 零加速 | allreduce 撞 NCCL bus bw **1.78 GB/s** 墙（P2P 禁用） | EXP-005 / EXP-002 · `pd_disagg/hw/all_reduce_perf.txt` |
| **PD 分离瓶颈定量**：KV 等待占 TTFT **54.2 / 62.5 / 64.2%**（512/2K/8K，p50，request 级因果占比） | 六段分解闭环误差 p50 <0.1%；bytes 与 Prometheus 对账分毫不差 | EXP-013 · `pd_disagg/ext1/derived/ext1_per_request.csv` |
| **MoE 的 decode 优势在 bs≈8 反转** | MoE/dense **2.03×**(bs=1) → 0.97×(bs=8) → **0.82×**(bs=128)；nsys node 级归因：fused_moe grouped GEMM 占 GPU 时间 **56.4%**(bs=32) | EXP-014 · `moe_perf/derived/d1_scaling.csv`、`moe_perf/derived/d1_kernel_share_bs32.csv` |
| **补齐两个社区空缺 MoE tuning config** | E=30,N=1408 / E=60,N=704（各 18 M 档）；kernel A/B：M=1 **-8.5% / -3.8%**，M≥128 -3.3~-3.9%；correctness 120 passed；PR 材料齐备（提交留用户） | EXP-015 · `moe_perf/PR_DRAFT.md` |
| **版本升级实测**（system-version comparison，v0.17.1→v0.25.1） | 512 桶饱和吞吐 **+45%**（7.14→10.36 req/s）；启动 308→58s；计算受限桶零差异 | EXP-008 |

## 📊 图表

![四臂饱和吞吐总览](pd_disagg/figures/fig7_saturation_overview.png)

> 两张卡怎么用：数据并行（replica2）饱和吞吐三个输入桶全部最高，PD 分离垫底。
> source: `pd_disagg/results/b1_matrix/runs.jsonl`（2026-08-21）· 脚本 `pd_disagg/scripts/make_fig7_overview.py`

![四臂 goodput 曲线](pd_disagg/figures/fig1_goodput_curves.png)

> SLO goodput 随 offered load：replica2 全场最高；PD 分离在全部负载段被传输延迟压垮。
> source: `pd_disagg/results/b1_matrix/runs.jsonl`（2026-08-21）· 脚本 `pd_disagg/scripts/make_figures.py`

![PD TTFT 分解](pd_disagg/figures/fig4_pd_ttft_decompose.png)

> PD 的 TTFT 分解：KV 传输占 54–64%，各分量与独立遥测对账吻合（request 级因果版见 EXP-013）。
> source: `pd_disagg/results/b1_matrix/runs.jsonl`（2026-08-21）

![MoE decode 反转点](moe_perf/figures/d1_fig1_decode_scaling.png)

> MoE 的 decode 优势在 bs≈8 反转：2.03×(bs=1) → 0.82×(bs=128)——小 batch 是激活参数量的胜利，大 batch 输给专家权重搬运。
> source: `moe_perf/raw/EXP-014/`（2026-08-23）· 脚本 `moe_perf/d1_analyze.py`

## 🔬 代码导览：~16 行把「KV 传输占 TTFT」从对账推断变成因果测量

四臂矩阵显示 PD 分离垫底，但"KV 传输占 TTFT 多少"最初只能靠分量对账（拿 colocate
无负载 TTFT 近似 P 段）间接推断。EXT-1 用 **~16 行本地可观测性改动**（打在 vLLM 0.25.1
NIXL connector，逐行 `# EXT1` 标记、原件备份可还原）把它升级为逐请求因果测量——核心节选：

```python
# pull_worker.start_load_kv —— D 端 connector 首见请求：记双时钟起点
for req_id, meta in metadata.reqs_to_recv.items():
    self._ext1_t0[req_id] = (time.perf_counter(), time.time())  # EXT1

# base_worker._pop_done_transfers —— 逐 handle 累加 NIXL telemetry
res = self.nixl_wrapper.get_xfer_telemetry(handle)
agg = self._ext1_agg.setdefault(req_id, [0, 0, 0, 0, 0])  # EXT1
agg[0] += res.totalBytes     # 与 Prometheus 计数器对账的 bytes
agg[1] += res.xferDuration   # 纯传输时间：与 kv_wait 仅差 0.3–1.9ms → 轮询开销可忽略

# 该请求全部 handle DONE 时——一行日志把三段身份与两段时钟钉在一起
logger.info(
    "EXT1_KV req_id=%s remote_request_id=%s kv_wait_ms=%.3f "
    "t0_epoch=%.6f done_epoch=%.6f bytes=%d ...",
    req_id,      # D 端 id（内嵌 client 自定的 X-Request-Id）
    remote_req,  # P 端 id —— PD 身份拆分的显式映射
    (time.perf_counter() - t0[0]) * 1e3,  # kv_wait：含 handshake 的完整等待窗口
    ...)
```

完整 patch：`pd_disagg/ext1/nixl_req_telemetry_v0251.patch`（原件备份 `ext1/orig/`）。
**三段关联思路**（EXP-013）：

1. **身份**：client 自定 `X-Request-Id` 原样贯穿 proxy→P→D，36/36 请求在 D 端
   req_id 与 remote_request_id 中均可见——跨进程 join 键；
2. **时钟**：同机 1P1D——时长用单调 perf_counter，跨进程对齐用同 host epoch；
3. **互证**：逐请求 bytes 求和与 Prometheus `nixl_bytes_transferred_sum` 分毫不差；
   六段分解闭环误差 p50 <0.1%；打 patch 前后 TTFT p50 218/727/2738 vs
   219/719/2719 ms——观测零扰动。

上游已有同方向 draft PR #52859，按 fail-closed 规则本 patch 定位为本地测量工具、
不投上游（查重记录 `pd_disagg/ext1/DEDUP.md`）。

## 🚀 复现 Quickstart

```bash
# 环境：vLLM 0.25.1（/root/venvs/v0.25.1）+ 2×RTX 4090；绘图 venv /root/venvs/kernel-opt

# 1) 不碰 GPU：从 raw 重算全部 B1 图表与 derived 表
cd pd_disagg
/root/venvs/kernel-opt/bin/python scripts/make_figures.py
/root/venvs/kernel-opt/bin/python scripts/make_fig7_overview.py

# 2) 复现 B1 单测量点（先起对应臂的 server；快照→bench→快照→追加 runs.jsonl）
scripts/run_point.sh colocate sweep 2048 128 2.7 8100 8100

# 3) EXT-1 全流程：1P1D 起栈 → 36 请求 → client×proxy×D 三方 join（需先打 ext1 patch）
bash pd_disagg/ext1/run_ext1.sh

# 4) MoE D1 曲线：MoE(TP2+EP) vs dense(TP2)，并发 1..128
bash moe_perf/d1_sweep.sh
```

## 目录结构

```
experiments/
├── README.md                  # 本文件：门面 + 约定 + 证据台账 + 红线状态
├── HANDOFF.md                 # 接手唯一入口（多 agent 接力）
├── LAB_JOURNAL.md             # 实验日记：每个工作段落的过程/决策/数字/产物（时间正序）
├── RESUME_EVIDENCE.md         # 简历句 ↔ 证据映射（最终写简历/面试用）
├── records/                   # 实验记录：每个实验一份 EXP-NNN（模板 TEMPLATE.md）
│   └── data/                  # 记录直属的小型原始数据（如节流采样 CSV）
├── pd_disagg/                 # B 线：推理部署选型战役
│   ├── DECISION.md            # 版本裁决（锁定 v0.25.1）+ 硬件基线数字
│   ├── EXPERIMENT_PLAN.md     # 计划 v2 + 17 条代码级核验表（面试深挖素材）
│   │                          #   ⚠ 其中"main=交付"段已被 DECISION.md 取代
│   ├── REPORT.md              # B4 报告 v2（一页结论 + 四章 + 附录）
│   ├── smoke/                 # R0-3：NIXL 1P1D smoke 双版本 PASS 证据
│   ├── hw/                    # R0-1：硬件三数（p2p / nccl / 拓扑）
│   ├── profiling/             # R0-5：torch profiler + nsys 验证与 trace
│   ├── scripts/               # run_point / collect_point / make_figures / provenance 等
│   ├── matrix/                # B1 工装（rr_proxy.py 等）
│   ├── results/               # B1+ 基准数据（schema 见 results/README.md）
│   ├── figures/               # 报告图 fig1-7（源数据必须可溯到 results/）
│   ├── ext1/                  # EXT-1：request 级 KV-wait 关联（patch + 工装 + 数据）
│   ├── p2pnccl_repro/         # R0-4：0.17.1 两 bug 动态复现栈
│   └── analysis/              # 源码机理分析（bug 链 / token 对账）
├── moe_perf/                  # D 线：MoE 性能战役（d1–d5 脚本 + raw/derived/figures + PR_DRAFT.md）
├── moe_configs/
│   └── DEDUP.md               # C2：config 查重判定
└── docs/
    ├── talk/TALK.md           # 现行面试讲稿（只留一份现行版，2026-08-24 建）
    └── theory/                # 原理笔记（五节制式，实证节指向自家 EXP 数字）
```

## 硬约定（所有新数据必须遵守）

1. **provenance 行**：每个结果文件首行来自 `scripts/provenance.sh` 的 `prov_line`
   （先 `prov_env A|B|C` 选环境）。sha 语义：wheel 环境记对应 tag 的 SHA，不是仓库 HEAD。
2. **命名**：`<UTCyyyymmddThhmm>_<对象>_<条件>.<ext>`，全小写下划线；
   一次测量的原始输出、metrics 快照、日志共用同一前缀。
3. **raw 与 derived 分离**：报告里的每张表/图必须能从 raw（`runs.jsonl` + 快照）重算；
   衍生表放 `derived/`，图放 `figures/`，图说明里写明源数据文件。
4. **单位入表头**：时延 ms、带宽 GB/s（10⁹）、吞吐 tok/s、KV 量 MB；不写裸数字。
5. **Gate 数据同行存储**：B1 每个测量点的 gate 字段与指标同存 `runs.jsonl` 一行；
   `gates.pass=false` 的行保留但**永不进 derived/ 与报告**。
6. **图表样式**：白底、单图单结论、标题写结论句而非变量名；
   四臂用固定配色贯穿全报告（colocate/replica2/tp2/pd1p1d 一色到底），图脚注 provenance。
7. **实验日记**：每个工作段落结束在 `LAB_JOURNAL.md` 追加一节
   （做了什么/为什么/关键数字/产物路径 + 下一步）；写简历时以
   日记（叙事）+ RESUME_EVIDENCE（句子）+ 本台账（状态）三件套为参照。
8. **实验记录**：每个实验一份 `records/EXP-NNN_<slug>.md`（按 TEMPLATE.md 八节写全：
   目的/配置/步骤/原始数据/结果/分析/异常/下游影响），实验结束当场写，不隔夜。
   **任何 GPU 跑——包括诊断跑、临时排障跑——一律存 raw**（bench 加 --save-result）；
   没存 raw 的数字降级为"终端级证据"，必须在记录 §4 注明证据等级。

## 实验记录索引

> 表头声明（8/24，依 CORE 规范核对）：本表不重复登记关键数字——单一事实源，
> 关键数字（带指针）统一见下方「证据台账」表。

| 编号 | 标题 | 日期 | 关联项 | 状态 |
|---|---|---|---|---|
| [EXP-001](records/EXP-001_nixl_smoke_version_verdict.md) | NIXL 1P1D smoke 与版本裁决 | 8/21 | R0-3 | 完成 |
| [EXP-002](records/EXP-002_hardware_baseline.md) | 硬件三数 | 8/21 | R0-1 | 完成 |
| [EXP-003](records/EXP-003_profiling_tooling.md) | profiling 工装验证 | 8/21 | R0-5 | 完成 |
| [EXP-004](records/EXP-004_b1_colocate_attribution_slo.md) | colocate 归因基线 + SLO 锁定 | 8/21 | B1 | 完成 |
| [EXP-005](records/EXP-005_replica2_tp2_powercap.md) | replica2/tp2 归因 + 功率帽调查 | 8/21 | B1 | 完成 |
| [EXP-006](records/EXP-006_pd1p1d_probe_attribution.md) | pd1p1d 探针 + 归因 + NIXL 大传输 | 8/21 | B1/R0-1 | 完成 |
| [EXP-007](records/EXP-007_b1_sweep_campaign.md) | B1 四臂 sweep 战役（协议 v2） | 8/21 | B1 | 完成 |
| [EXP-008](records/EXP-008_b3_version_compare.md) | B3 有限版本对照 | 8/21 | B3 | 完成 |
| [EXP-009](records/EXP-009_c1_moe_bringup.md) | C1 MoE 上卡 + C2 运行时证据 | 8/21 | C1/C2 | 完成 |
| [EXP-010](records/EXP-010_c3_w4a16_bringup.md) | C3 W4A16 Qwen3-30B-A3B 上卡 | 8/21 | C3 | 完成 |
| [EXP-011](records/EXP-011_ext2_nixl_push.md) | EXT-2 NixlPush 推方向单点对照 | 8/22 | EXT-2 | 完成 |
| [EXP-012](records/EXP-012_p2pnccl_dynamic_repro.md) | R0-4 P2pNccl 两 bug 动态复现 | 8/23 | R0-4 | 完成 |
| [EXP-013](records/EXP-013_ext1_request_level_kv_attribution.md) | EXT-1 request 级 KV-wait 关联 | 8/23 | EXT-1/B2 | 完成 |
| [EXP-014](records/EXP-014_d1_moe_kernel_decomposition.md) | D1 MoE decode 分解(曲线+nsys) | 8/23 | D1 | 完成 |
| [EXP-015](records/EXP-015_d2_moe_config_tuning.md) | D2 MoE config 调优+六件套验证 | 8/23 | D2/P1 | 完成 |
| [EXP-016](records/EXP-016_d4_fp8_vs_w4a16.md) | D4 FP8 vs W4A16(30B-A3B, Ada) | 8/23 | D4 | 完成 |
| [EXP-017](records/EXP-017_d5_eplb_gate.md) | D5 EPLB gate(W4A16拒/FP8重排+对照) | 8/23 | D5 | 完成 |

## 证据台账（勾一项 = 数据落盘 + 本表登记产物路径）

| 项 | 状态 | 关键数字 | 产物 |
|---|---|---|---|
| R0-1 硬件三数 | ✅ 8/21 | P2P=GNS 禁用；单向 D2D 0.60–0.91 GB/s，双向 22.7 GB/s，延迟 ~15µs；NCCL bus bw 1.78 GB/s | `pd_disagg/hw/{topo,p2p_bandwidth_latency,all_reduce_perf}.txt` |
| R0-2 三 venv + provenance | ✅ 8/21 | ~/venvs/{v0.17.1, v0.25.1, main} 均验证 import | `pd_disagg/scripts/provenance.sh`；setup 记录 `pd_disagg/setup_envs.log` |
| R0-3 NIXL 1P1D smoke | ✅ 8/21 | 双版本 3/3 PASS；avg xfer 14.1ms / 0.188MB / 13.3MB/s；裁决锁定 v0.25.1 | `pd_disagg/smoke/`、`pd_disagg/DECISION.md` |
| R0-4 0.17.1 课程基线 | ✅ 8/23 | 课程脚本非必需——官方 xPyD proxy+脚本在 v0.17.1 tag 内，自建 1P1D 复现栈 | `pd_disagg/p2pnccl_repro/launch_1p1d.sh`（从 tag 提取精简） |
| R0-5 profiling 工装 | ✅ 8/21 | torch profiler 直控 P/D 端口跑通（trace 落盘）；nsys 容器内可用 | `pd_disagg/profiling/r0_5_torch_profiler_check.txt`、`traces_smoke/`、`scripts/profile_ctl.sh` |
| R0-6 简历措辞排雷 | ◐ | 本地 tex 无违规表述（已核）；线上稿待改 | 仅用户可操作 |
| B1 四臂矩阵 | ✅ 8/21 | **全套完成（协议 v2，84 有效行）**。饱和 req/s（512/2K/8K）：colocate 10.36/3.63/0.90（单卡）· replica2 15.58/7.00/1.78 · tp2 12.31/4.16/1.02 · pd1p1d 7.84/2.12/0.54；goodput 峰值 replica2 全场最高；PD 全负载段被传输延迟压垮（512 桶 66% 饱和度 goodput 仅 1.59） | `results/b1_matrix/runs.jsonl` + `derived/sweep_summary.csv` + `figures/fig1-6` + EXP-007 |
| B2 归因层 | ✅ 8/23 | **收官**：EXT-1 request 级三段关联落地——KV 等待占 TTFT **54.2/62.5/64.2%**（512/2K/8K，p50，因果占比），闭环误差 p50 <0.1%（最差桶 0.084%，逐请求最大 0.11%），bytes 与 Prometheus 分毫不差；此前的分量对账（54–64%）被追认 | `figures/fig4,fig5`、`analysis/nixl_token_accounting.md`、EXP-013、`ext1/` |
| B3 版本对照 | ✅ 8/23 两维度定稿 | ①单实例 system-version：无负载延迟 Δ<1%、**512 桶饱和 +45%**（7.14→10.36）、计算受限桶零差异、启动 308→58s；②PD 可用性对照（EXP-012 闭环）：0.17.1 PD 默认配置正常请求即触发 D 挂死→不构成可用对照臂，结论即"不可用 vs 可用"，无吞吐对比可做。另录版本差异一例：profiler 接口 env var→CLI | EXP-008、EXP-012、REPORT v2 §3 |
| R0-4（降级路径） | ✅ 8/21 | 双 bug 源码机理分析完成（assert connector:433 / 分叉 input_processor.py:212 / 无超时 wait engine:317 / GET 静默乱码 / 四层 ID 链 / NIXL 身份拆分对照），全 file:line 核对 | `analysis/p2pnccl_bugs_id_chain.md` |
| R0-4 动态复现 | ✅ 8/23 | **实机 1P1D 坐实两 bug**：bug1 精确命中 `connector:433` AssertionError（addr串id+max_tokens>1）；bug2 D 整实例挂死（双请求 hang + 全线程 futex_wait + P /health 恒 200）；**实证修正**：裸直连先崩于 `connector:518` parse_request_id 早于 :433。py-spy 因容器 ptrace 限制未取栈帧（已诚实标注） | EXP-012；`pd_disagg/p2pnccl_repro/raw/EXP-012/` |
| B1 附带发现 | ✅ 8/21 | ① 功率帽节流：持续 prefill 降频 2820→2475MHz（SW Power Cap，非热），TTFT +30%；② NIXL 有效吞吐 0.26–0.27GB/s 恒定（descriptor ~16KB 碎片化）；③ TP2 decode 提速 42%（带宽分摊）但 prefill 零加速（allreduce 撞 1.78GB/s 墙） | runs.jsonl gpu_telemetry / gates 字段；`DECISION.md` 硬件基线 |
| B4 报告 | ✅ 8/23 v2 定稿 | 一页结论 + 四章 + 附录；v2 吸收 EXT-1 因果占比 / EXT-2 推拉对照 / EXP-012 动态复现与 B3 定稿 | `pd_disagg/REPORT.md` |
| C1 Qwen1.5-MoE 上卡 | ✅ 8/21 | TP2+EP 可用；未调优基线 TPOT 4.62ms（dense 7B TP2 的 2.0×）、饱和 11.50 req/s@512——D2 的 before 数字 | EXP-009 |
| C2 config 查重 | ✅ 8/21 | 三重闭环：本地判定 + 远端查重 + **运行时告警原文**（fused_moe.py:1106 点名 E=30,N=1408 缺失） | `moe_configs/DEDUP.md`、EXP-009 §5 |
| C3 W4A16 上卡 | ✅ 8/21 | **Qwen/Qwen3-30B-A3B-GPTQ-Int4**（W4A16，Marlin 路径确认）TP2+EP 上卡；TPOT 4.93ms / 饱和 10.02 req/s@512——与 2.7B BF16 相当（D1/D4 切入点） | EXP-010 |
| D1 MoE 分解 | ✅ 8/23 | **反转点发现**:MoE/dense 2.03×(bs=1)→0.97×(bs=8)→0.82×(bs=128);nsys node 级分解:fused_moe grouped GEMM 占 56.4%(bs=32)/ dense GEMV 40.9%(bs=1);D2/D3 目标由数据锁定 fused_moe | EXP-014、`moe_perf/`(figures+derived+raw) |
| D4 FP8 vs W4A16 | ✅ 8/23 | **W4A16(Marlin)decode 全 regime 胜 23–48%**(TPOT 4.91 vs 7.10ms@bs1),FP8 仅 c128 TTFT 反超(497 vs 613ms,prefill 计算受限)+ PPL 优 3.3% 相对(7.663 vs 7.922,同 31212 token);Ada 落地解释:oracle/fp8.py:103-122 capability 提升跳过 SM89 → TRITON block-scaled | EXP-016、`moe_perf/raw/EXP-016/` |
| D5 EPLB gate | ✅ 8/23 判定完成 | **W4A16 拒**(`routed_experts.py:151` NotImplementedError,上游 TODO 指认工程缺口);**FP8 臂 2 次真实重排**(balancedness 0.53–0.74 实测)+ **对照组**(无 EPLB 同负载逐字节一致)→ 输出分歧因果归属 EPLB(数值性定性);按清单 gate 规则**不上简历,白板级保留** | EXP-017、`moe_perf/raw/EXP-017/` |
| D2 config 调优 | ✅ 8/23 | **两个空缺 tuple JSON 交付**(E=30,N=1408 / E=60,N=704,各 18 M 档;8/24 勘正:曾误计 triton_version 元键为 19);kernel A/B 两端改善(M=1 **-8.5%/-3.8%**,M≥128 -3.3~-3.9%,中段持平);e2e TPOT **+0.8~1.2%** 一致(吞吐噪声内);correctness 120 passed;**PR 分支+六件套齐备,提交留用户** | EXP-015、`moe_perf/raw/EXP-015/`、PR_DRAFT.md、分支 `moe-config-4090-qwen15moe` |
| D3 kernel 优化 | ✅ 8/23 判定 | 依 D2 数据**转结论句**:tuned 与 default 在中段 M 打平 → Triton tile 空间已被启发式覆盖,config 即最优杠杆;不另做 kernel 改动(避免无数据支撑的"优化") | EXP-015 §6 |
| EXT-1 request 级关联 | ✅ 8/23 | 本地 patch（16 行，可还原）；KV 占 TTFT 54.2/62.5/64.2%；上游不投（#52859 在途，见 `ext1/DEDUP.md`） | EXP-013、`pd_disagg/ext1/` |
| EXT-2 NixlPush | ✅ 8/22 | 推方向 8K TTFT -6.7%、吞吐 +10–13%，量级不变（方向救不了 PD） | EXP-011 |

## 措辞红线状态（写简历/报告前查此表）

| 红线 | 当前 | 解锁条件 / 依据 |
|---|---|---|
| "P2P 受限" | ✅ 可用 | `hw/p2p_bandwidth_latency.txt`（connectivity=0）+ `hw/topo.txt`（GNS） |
| "社区空缺"（MoE config） | ✅ 可用 | 2026-08-21 远端复核完成：`moe_configs/DEDUP.md`（main 无 E=30；E=60,N=704 仅 MI300X；PR/issue 无冲突） |
| "KV 传输占 TTFT X%" | ✅ 可用 | **EXT-1 已解锁（EXP-013，2026-08-23）**：request 级三段关联（同身份同时钟域），KV 等待占 TTFT 54.2/62.5/64.2%（512/2K/8K），闭环误差 ≤0.08% |
| telemetry 带宽表述 | 限定 | 只能称 telemetry-derived effective throughput；xferDuration 不与 postDuration 相加 |
| 0.17 两 bug | 限定 | 只写"复现/定位/验证"，禁"发现/修复"；"吃透"→"梳理"。**动态复现已闭环（EXP-012）**：静态 file:line + 实机崩溃/挂死现场 + 实证修正，"复现/定位/验证"三词均有实测背书 |
| A/B 版本对照 | 限定 | 只称 system-version comparison，标注传输方向不同 |
| PR 状态 | 限定 | 未提交不写"提交"，未合并不写"合入" |
| D2 e2e +0.8~1.2% | 🚫 不作 headline | 低于跨会话漂移；主证据=kernel A/B（8/24 定档修正） |
| EXT-1 patch 定性 | 限定 | "~16 行本地可观测性改动"，不得表述为 NIXL/Connector 核心改造 |

### 方法论：诚实度文化

- **可溯源**：每个结果文件首行 provenance（env/sha/完整命令/GPU/驱动）；raw 不可变，
  表图一律由脚本从 raw 重算，坏数据移 archive 留痕而非原地改。
- **互证与对照**：关键数字带分布不带单点（p50/p90；EXP-013 每桶 n=11，
  六段闭环误差 p50 <0.1%，bytes 与 Prometheus 独立对账）；因果归属必设对照臂
  （D5 专设无 EPLB 对照组，输出逐字节一致才把分歧归因 EPLB）。
- **负结论与勘误留痕**：D3 依数据放弃 kernel 改动、D5 判"不上简历"、8/24 M 档
  19→18 勘正与 D2 e2e 降级出 headline，全部在台账原位标注，不删不藏。

## 相关仓库

- [vllmExperience](https://github.com/lyell0710/vllmExperience) —— 本仓（private）
- [Kernel_Optimazation](https://github.com/lyell0710/Kernel_Optimazation) —— CUDA kernel 优化实验仓
- [llm-engine](https://github.com/lyell0710/llm-engine) —— LLM 推理引擎仓

## 备份

- 本仓库 commit 即产物锚点；远程 = github.com/lyell0710/vllmExperience（private）。
- 大文件（trace、nsys-rep）入 git——本仓库就是证据箱，体积换可信度。
  **例外（8/23 起）**：单文件 >100MB 触 GitHub pre-receive 硬限，仅本地保存并在
  同目录 `LARGEFILES.md` 登记 sha256；`*.sqlite`（nsys 可再生衍生品）全局忽略。
  教训：push 成败必须看 `git status -sb`，管道 `| tail` 会吞掉真实返回码。
