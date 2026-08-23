# 实验证据仓库（vLLM 秋招项目）

独立嵌套 git 仓库（外层 vllm 仓库通过 `.git/info/exclude` 忽略本目录，互不干扰）。
**所有能进简历/报告的数字、图表、trace 的唯一权威存放地。**
简历句与证据的对应关系见 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)。

## 目录结构

```
experiments/
├── README.md                  # 本文件：约定 + 证据台账 + 红线状态
├── LAB_JOURNAL.md             # 实验日记：每个工作段落的过程/决策/数字/产物（时间正序）
├── RESUME_EVIDENCE.md         # 简历句 ↔ 证据映射（最终写简历/面试用）
├── records/                   # 实验记录：每个实验一份 EXP-NNN（模板 TEMPLATE.md）
│   └── data/                  # 记录直属的小型原始数据（如节流采样 CSV）
├── pd_disagg/
│   ├── DECISION.md            # 版本裁决（锁定 v0.25.1）+ 硬件基线数字
│   ├── EXPERIMENT_PLAN.md     # 计划 v2 + 17 条代码级核验表（面试深挖素材）
│   │                          #   ⚠ 其中"main=交付"段已被 DECISION.md 取代
│   ├── smoke/                 # R0-3：NIXL 1P1D smoke 双版本 PASS 证据
│   ├── hw/                    # R0-1：硬件三数（p2p / nccl / 拓扑）
│   ├── profiling/             # R0-5：torch profiler + nsys 验证与 trace
│   ├── scripts/               # provenance.sh / profile_ctl.sh / metrics_snapshot.sh
│   ├── matrix/                # B1 工装（rr_proxy.py 等）
│   ├── results/               # B1+ 基准数据（schema 见 results/README.md）
│   └── figures/               # 报告图（源数据必须可溯到 results/）
└── moe_configs/
    └── DEDUP.md               # C2：config 查重判定
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

| 编号 | 标题 | 关联项 | 状态 |
|---|---|---|---|
| [EXP-001](records/EXP-001_nixl_smoke_version_verdict.md) | NIXL 1P1D smoke 与版本裁决 | R0-3 | 完成 |
| [EXP-002](records/EXP-002_hardware_baseline.md) | 硬件三数 | R0-1 | 完成 |
| [EXP-003](records/EXP-003_profiling_tooling.md) | profiling 工装验证 | R0-5 | 完成 |
| [EXP-004](records/EXP-004_b1_colocate_attribution_slo.md) | colocate 归因基线 + SLO 锁定 | B1 | 完成 |
| [EXP-005](records/EXP-005_replica2_tp2_powercap.md) | replica2/tp2 归因 + 功率帽调查 | B1 | 完成 |
| [EXP-006](records/EXP-006_pd1p1d_probe_attribution.md) | pd1p1d 探针 + 归因 + NIXL 大传输 | B1/R0-1 | 完成 |
| [EXP-007](records/EXP-007_b1_sweep_campaign.md) | B1 四臂 sweep 战役（协议 v2） | B1 | 完成 |
| [EXP-008](records/EXP-008_b3_version_compare.md) | B3 有限版本对照 | B3 | 完成 |
| [EXP-009](records/EXP-009_c1_moe_bringup.md) | C1 MoE 上卡 + C2 运行时证据 | C1/C2 | 完成 |
| [EXP-010](records/EXP-010_c3_w4a16_bringup.md) | C3 W4A16 Qwen3-30B-A3B 上卡 | C3 | 完成 |
| [EXP-011](records/EXP-011_ext2_nixl_push.md) | EXT-2 NixlPush 推方向单点对照 | EXT-2 | 完成 |
| [EXP-012](records/EXP-012_p2pnccl_dynamic_repro.md) | R0-4 P2pNccl 两 bug 动态复现 | R0-4 | 完成 |
| [EXP-013](records/EXP-013_ext1_request_level_kv_attribution.md) | EXT-1 request 级 KV-wait 关联 | EXT-1/B2 | 完成 |
| [EXP-014](records/EXP-014_d1_moe_kernel_decomposition.md) | D1 MoE decode 分解(曲线+nsys) | D1 | 完成 |
| [EXP-016](records/EXP-016_d4_fp8_vs_w4a16.md) | D4 FP8 vs W4A16(30B-A3B, Ada) | D4 | 完成 |
| [EXP-017](records/EXP-017_d5_eplb_gate.md) | D5 EPLB gate(W4A16拒/FP8重排+对照) | D5 | 完成 |

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
| D2/D3 | 进行中 | D2 调优+AB 链上最终长跑(EXP-015);D3 依 D2 A/B 定 | — |
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

## 备份

- 本仓库 commit 即产物锚点；**远程备份待配置**（需一个私有 repo 地址）。
- 大文件（trace、nsys-rep）也入 git——本仓库就是证据箱，体积换可信度。
