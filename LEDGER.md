# LEDGER · 对内状态账本

> **本文件是状态与措辞的唯一权威；README 为对外版，措辞以本表为准。** 收纳自 README 门面化改造：实验记录索引（含日期/状态）、证据台账、措辞红线状态、硬约定、方法论、备份约定与待办。README 面向首次打开仓库的读者，只保留结论与方法；日期、状态、待办、措辞约束一律查本文件。

## 嵌套仓说明

独立嵌套 git 仓库（外层 vllm 源码仓通过 `.git/info/exclude` 忽略本目录，互不干扰）。**所有能进简历/报告的数字、图表、trace 的唯一权威存放地都在这里。**简历句与证据的对应关系见 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)；接手/交接从 [HANDOFF.md](HANDOFF.md) 读起。

## 硬约定（所有新数据必须遵守）

1. **provenance 行**：每个结果文件首行来自 `scripts/provenance.sh` 的 `prov_line`（先 `prov_env A|B|C` 选环境）。sha 语义：wheel 环境记对应 tag 的 SHA，不是仓库 HEAD。
2. **命名**：`<UTCyyyymmddThhmm>_<对象>_<条件>.<ext>`，全小写下划线；一次测量的原始输出、metrics 快照、日志共用同一前缀。
3. **raw 与 derived 分离**：报告里的每张表/图必须能从 raw（`runs.jsonl` + 快照）重算；衍生表放 `derived/`，图放 `figures/`，图说明里写明源数据文件。
4. **单位入表头**：时延 ms、带宽 GB/s（10⁹）、吞吐 tok/s、KV 量 MB；不写裸数字。
5. **Gate 数据同行存储**：B1 每个测量点的 gate 字段与指标同存 `runs.jsonl` 一行； `gates.pass=false` 的行保留但**永不进 derived/ 与报告**。
6. **图表样式**：白底、单图单结论、标题写结论句而非变量名；四臂用固定配色贯穿全报告（colocate/replica2/tp2/pd1p1d 一色到底），图脚注 provenance（**对外图脚注不带日期**：只写源数据文件 + 硬件 + 协议/轮数）。
7. **实验日记**：每个工作段落结束在 `LAB_JOURNAL.md` 追加一节（做了什么/为什么/关键数字/产物路径 + 下一步）；写简历时以日记（叙事）+ RESUME_EVIDENCE（句子）+ 本台账（状态）三件套为参照。
8. **实验记录**：每个实验一份 `records/EXP-NNN_<slug>.md`（按 TEMPLATE.md 八节写全：目的/配置/步骤/原始数据/结果/分析/异常/下游影响），实验结束当场写，不隔夜。 **任何 GPU 跑——包括诊断跑、临时排障跑——一律存 raw**（bench 加 --save-result）；没存 raw 的数字降级为"终端级证据"，必须在记录 §4 注明证据等级。

## 实验记录索引

> 表头声明（8/24，依 CORE 规范核对）：本表不重复登记关键数字——单一事实源，关键数字（带指针）统一见下方「证据台账」表。

| 编号 | 标题 | 日期 | 关联项 | 状态 |
|---|---|---|---|---|
| [EXP-001](records/EXP-001_nixl_smoke_version_verdict.md) | NIXL 1P1D smoke 与版本裁决 | 8/21 | R0-3 | 完成 |
| [EXP-002](records/EXP-002_hardware_baseline.md) | 硬件三数（R0-1 硬件画像） | 8/21 | R0-1 | 完成 |
| [EXP-003](records/EXP-003_profiling_tooling.md) | profiling 工装验证（torch profiler + nsys） | 8/21 | R0-5 | 完成 |
| [EXP-004](records/EXP-004_b1_colocate_attribution_slo.md) | B1 colocate 归因基线 + SLO 锁定 | 8/21 | B1 | 完成 |
| [EXP-005](records/EXP-005_replica2_tp2_powercap.md) | replica2/tp2 归因 + 功率帽节流调查 | 8/21 | B1 | 完成 |
| [EXP-006](records/EXP-006_pd1p1d_probe_attribution.md) | pd1p1d 指标探针 + 归因 + NIXL 大传输实测 | 8/21 | B1/R0-1 | 完成 |
| [EXP-007](records/EXP-007_b1_sweep_campaign.md) | B1 四臂 offered-load 扫描战役（协议 v2） | 8/21 | B1 | 完成 |
| [EXP-008](records/EXP-008_b3_version_compare.md) | B3 有限版本对照（v0.17.1 vs v0.25.1 单实例） | 8/21 | B3 | 完成 |
| [EXP-009](records/EXP-009_c1_moe_bringup.md) | C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据 | 8/21 | C1/C2 | 完成 |
| [EXP-010](records/EXP-010_c3_w4a16_bringup.md) | C3 Qwen3-30B-A3B W4A16 上卡 | 8/21 | C3 | 完成 |
| [EXP-011](records/EXP-011_ext2_nixl_push.md) | EXT-2 NixlPush 单点（推 vs 拉方向对照） | 8/22 | EXT-2 | 完成 |
| [EXP-012](records/EXP-012_p2pnccl_dynamic_repro.md) | R0-4 vLLM 0.17.1 P2pNccl 两缺陷动态复现（1P1D 实机） | 8/23 | R0-4 | 完成 |
| [EXP-013](records/EXP-013_ext1_request_level_kv_attribution.md) | EXT-1 request 级 KV-wait 关联（解锁"KV 占 TTFT%"红线） | 8/23 | EXT-1/B2 | 完成 |
| [EXP-014](records/EXP-014_d1_moe_kernel_decomposition.md) | D1 MoE decode 分解：吞吐-batch 曲线 + nsys kernel 占比 | 8/23 | D1 | 完成 |
| [EXP-015](records/EXP-015_d2_moe_config_tuning.md) | D2 MoE config 调优：4090 BF16 两个社区空缺 tuple + 六件套验证 | 8/23 | D2/P1 | 完成 |
| [EXP-016](records/EXP-016_d4_fp8_vs_w4a16.md) | D4 FP8 vs W4A16 同卡对比（Qwen3-30B-A3B,Ada SM89） | 8/23 | D4 | 完成 |
| [EXP-017](records/EXP-017_d5_eplb_gate.md) | D5 EPLB gate（W4A16 不支持 / FP8 真实重排 + 对照组归因） | 8/23 | D5 | 完成 |
| [EXP-018](records/EXP-018_nccl_allreduce_size_scan.md) | NCCL allreduce size 扫描（补 EXP-002 小消息缺口 + 复测大消息带宽） | 8/29 | R0-1 | 完成 |
| [EXP-019](records/EXP-019_nccl_bw_discrepancy_rootcause.md) | 1.78 vs 6.2 GB/s 机制调查（环境 diff，先于 bench） | 8/30 | R0-1 | 完成 |
| [EXP-020](records/EXP-020_nccl_knob_matrix_repro.md) | NCCL 旋钮矩阵复现 1.78 GB/s（EXP-019 §8 四步落地） | 9/15 | R0-1 | 完成（定因不唯一） |
| [EXP-021](records/EXP-021_nccl_allreduce_dtype_scan.md) | NCCL allreduce dtype 扫描（half/bfloat16 vs float，补 EXP-018 §7 缺口） | 9/15 | R0-1 | 完成（未决，噪声主导） |
| [EXP-022](records/EXP-022_d2_bigM_kernel_ab.md) | D2 大 M（512–4096）kernel A/B：tuned config 在 prefill 级 M 是否保持收益（补 EXP-015 §7 缺口） | 9/15 | D2 | 完成 |
| [EXP-023](records/EXP-023_replica2_512_saturation_conc128.md) | replica2@512 饱和复测（SAT_CONC=128）：EXP-007 的欠饱和疑点追认或修正 | 9/15 | B1 | 完成 |
| [EXP-024](records/EXP-024_512_bucket_conc128_parity.md) | 512 桶四臂统一到 conc128 口径（补 tp2 / pd1p1d 两臂，供 fig7 重算） | 9/15 | B1 | 完成 |
| [EXP-025](records/EXP-025_replica2_512_true_saturation.md) | replica2@512 真饱和点扫描（conc 128/192/256，同 N 同 seed） | 9/15 | B1 | 完成（判据 A：未封顶） |
| [EXP-026](records/EXP-026_nixl_descriptor_granularity.md) | NIXL descriptor 粒度实验：把「描述符碎片化」从推断变成实测 | 9/15 | B2 | 完成（H1 成立、H2 推翻） |
| [EXP-027](records/EXP-027_rr_proxy_overhead_split.md) | rr_proxy 开销拆分：replica2 的 1.63× 缺口里有多少是代理 | 9/15 | B1 | 完成（判定 B：≈2%，非主因） |
| [EXP-028](records/EXP-028_goodput_field_backfill.md) | 工装：`goodput_slo_rps` 在饱和模式下不再为空（SLO 缺省表进 collect_point） | 9/15 | B1 工装 | 完成 |
| [EXP-029](records/EXP-029_single_instance_capacity.md) | 1.63× 的最后一块：单实例能否承载双实例的负载（主机级干扰的定量拆分） | 9/15 | B1 | 完成（发现 1.63× 是口径不对称） |

## 证据台账（勾一项 = 数据落盘 + 本表登记产物路径）

| 项 | 状态 | 关键数字 | 产物 |
|---|---|---|---|
| R0-1 硬件三数 | ✅ 8/21（collective 带宽值停用待复核） | P2P=GNS 禁用；单向 D2D 0.60–0.91 GB/s，双向 22.7 GB/s，延迟 ~15µs；NCCL collective 带宽**旧值 1.78 GB/s 已停用**（EXP-018 复测得大消息平台 ~6.2 GB/s，机制未查明前两值均不作权威，见 EXP-018 §7） | `pd_disagg/hw/{topo,p2p_bandwidth_latency,all_reduce_perf}.txt` + `20260829T104705_allreduce_size_scan_*.txt` |
| R0-2 三 venv + provenance | ✅ 8/21 | ~/venvs/{v0.17.1, v0.25.1, main} 均验证 import | `pd_disagg/scripts/provenance.sh`；setup 记录 `pd_disagg/setup_envs.log` |
| R0-3 NIXL 1P1D smoke | ✅ 8/21 | 双版本 3/3 PASS；avg xfer 14.1ms / 0.188MB / 13.3MB/s；裁决锁定 v0.25.1 | `pd_disagg/smoke/`、`pd_disagg/DECISION.md` |
| R0-4 0.17.1 课程基线 | ✅ 8/23 | 课程脚本非必需——官方 xPyD proxy+脚本在 v0.17.1 tag 内，自建 1P1D 复现栈 | `pd_disagg/p2pnccl_repro/launch_1p1d.sh`（从 tag 提取精简） |
| R0-5 profiling 工装 | ✅ 8/21 | torch profiler 直控 P/D 端口跑通（trace 落盘）；nsys 容器内可用 | `pd_disagg/profiling/r0_5_torch_profiler_check.txt`、`traces_smoke/`、`scripts/profile_ctl.sh` |
| R0-6 简历措辞排雷 | ◐ | 本地 tex 无违规表述（已核）；线上稿待改 | 仅用户可操作 |
| B1 四臂矩阵 | ✅ 8/21 | **全套完成（协议 v2，84 有效行）**。饱和 req/s（512/2K/8K）：colocate **12.81**/3.63/0.90（单卡）· replica2 **20.87**/7.00/1.78 · tp2 **12.30**/4.16/1.02 · pd1p1d **8.15**/2.12/0.54〔9/15 勘注：**512 桶改为 conc128 口径**（EXP-023/024 四臂统一补测）；2K/8K 仍为 conc64（EXP-007）。旧 conc64 值 10.36/15.58/12.31/7.84 见 EXP-007 史料。512 扩展效率同口径 1.63×（原 1.50×）〕；goodput 峰值 replica2 全场最高；PD 全负载段被传输延迟压垮（512 桶 66% 饱和度 goodput 仅 1.59） | `results/b1_matrix/runs.jsonl` + `derived/sweep_summary.csv` + `figures/fig1-6` + EXP-007 |
| B2 归因层 | ✅ 8/23 | **收官**：EXT-1 request 级三段关联落地——KV 等待占 TTFT **54.2/62.5/64.2%**（512/2K/8K，p50，因果占比），闭环误差 p50 <0.1%（最差桶 0.084%，逐请求最大 0.11%），bytes 与 Prometheus 分毫不差；此前的分量对账（54–64%）被追认 | `figures/fig4,fig5`、`analysis/nixl_token_accounting.md`、EXP-013、`ext1/` |
| B3 版本对照 | ✅ 8/23 两维度定稿 | ①单实例 system-version：无负载延迟 Δ<1%、**512 桶饱和 +45%**（7.14→10.36）、计算受限桶零差异、启动 308→58s；②PD 可用性对照（EXP-012 闭环）：0.17.1 PD 默认配置正常请求即触发 D 挂死→不构成可用对照臂，结论即"不可用 vs 可用"，无吞吐对比可做。另录版本差异一例：profiler 接口 env var→CLI | EXP-008、EXP-012、REPORT v2 §3 |
| R0-4（降级路径） | ✅ 8/21 | 双 bug 源码机理分析完成（assert connector：433 / 分叉 input_processor.py：212 / 无超时 wait engine：317 / GET 静默乱码 / 四层 ID 链 / NIXL 身份拆分对照），全 file：line 核对 | `analysis/p2pnccl_bugs_id_chain.md` |
| R0-4 动态复现 | ✅ 8/23 | **实机 1P1D 坐实两 bug**：bug1 精确命中 `connector:433` AssertionError（addr 串 id+max_tokens>1）；bug2 D 整实例挂死（双请求 hang + 全线程 futex_wait + P /health 恒 200）；**实证修正**：裸直连先崩于 `connector:518` parse_request_id 早于：433。py-spy 因容器 ptrace 限制未取栈帧（已诚实标注） | EXP-012；`pd_disagg/p2pnccl_repro/raw/EXP-012/` |
| B1 附带发现 | ✅ 8/21 | ① 功率帽节流：持续 prefill 降频 2820→2475MHz（SW Power Cap，非热），TTFT +30%；② NIXL 有效吞吐 0.26–0.27GB/s 恒定（descriptor ~16KB 碎片化）；③ TP2 decode 提速 42%（带宽分摊）但 prefill 零加速（allreduce 受 collective 带宽约束，具体值待复核） | runs.jsonl gpu_telemetry / gates 字段；`DECISION.md` 硬件基线 |
| B4 报告 | ✅ 8/23 v2 定稿 | 一页结论 + 四章 + 附录；v2 吸收 EXT-1 因果占比 / EXT-2 推拉对照 / EXP-012 动态复现与 B3 定稿 | `pd_disagg/REPORT.md` |
| C1 Qwen1.5-MoE 上卡 | ✅ 8/21 | TP2+EP 可用；未调优基线 TPOT 4.62ms（dense 7B TP2 的 2.0×）、饱和 11.50 req/s@512——D2 的 before 数字 | EXP-009 |
| C2 config 查重 | ✅ 8/21 | 三重闭环：本地判定 + 远端查重 + **运行时告警原文**（fused_moe.py：1106 点名 E=30，N=1408 缺失） | `moe_configs/DEDUP.md`、EXP-009 §5 |
| C3 W4A16 上卡 | ✅ 8/21 | **Qwen/Qwen3-30B-A3B-GPTQ-Int4**（W4A16，Marlin 路径确认）TP2+EP 上卡；TPOT 4.93ms / 饱和 10.02 req/s@512——与 2.7B BF16 相当（D1/D4 切入点） | EXP-010 |
| D1 MoE 分解 | ✅ 8/23 | **反转点发现**：MoE/dense 2.03×(bs=1)→0.97×(bs=8)→0.82×(bs=128);nsys node 级分解：fused_moe grouped GEMM 占 56.4%(bs=32)/ dense GEMV 40.9%(bs=1);D2/D3 目标由数据锁定 fused_moe | EXP-014、`moe_perf/`(figures+derived+raw) |
| D4 FP8 vs W4A16 | ✅ 8/23 | **W4A16(Marlin)decode 全 regime 胜 23–48%**(TPOT 4.91 vs 7.10ms@bs1),FP8 仅 c128 TTFT 反超（497 vs 613ms,prefill 计算受限）+ PPL 优 3.3% 相对（7.663 vs 7.922，同 31212 token）;Ada 落地解释：oracle/fp8.py:103-122 capability 提升跳过 SM89 → TRITON block-scaled | EXP-016、`moe_perf/raw/EXP-016/` |
| D5 EPLB gate | ✅ 8/23 判定完成 | **W4A16 拒**（`routed_experts.py:151` NotImplementedError，上游 TODO 指认工程缺口）；**FP8 臂 2 次真实重排**（balancedness 0.53–0.74 实测）+ **对照组**（无 EPLB 同负载逐字节一致）→ 输出分歧因果归属 EPLB（数值性定性）；按清单 gate 规则**不上简历，白板级保留** | EXP-017、`moe_perf/raw/EXP-017/` |
| D2 config 调优 | ✅ 8/23 | **两个空缺 tuple JSON 交付**（E=30，N=1408 / E=60，N=704，各 18 M 档；8/24 勘正：曾误计 triton_version 元键为 19）；kernel A/B 两端改善（M=1 **-8.5%/-3.8%**，M≥128 -3.3~-3.9%，中段持平）；e2e TPOT **+0.8~1.2%** 一致（吞吐噪声内）；correctness 120 passed（`::test_fused_moe` 单函数；全文件子集 1041 passed / 127 skipped，EXP-015 §5.1）；**PR #54372 已提交（2026-08-29），OPEN 未合并，CI pre-run-check 待维护者加 `ready` label** | EXP-015、`moe_perf/raw/EXP-015/`、PR_DRAFT.md、分支 `moe-config-4090-qwen15moe` |
| D3 kernel 优化 | ✅ 8/23 判定 | 依 D2 数据**转结论句**：tuned 与 default 在中段 M 打平 → Triton tile 空间已被启发式覆盖，config 即最优杠杆；不另做 kernel 改动（避免无数据支撑的"优化"） | EXP-015 §6 |
| EXT-1 request 级关联 | ✅ 8/23 | 本地 patch（16 行，可还原）；KV 占 TTFT 54.2/62.5/64.2%；上游不投（#52859 在途，见 `ext1/DEDUP.md`） | EXP-013、`pd_disagg/ext1/` |
| EXT-2 NixlPush | ✅ 8/22 | 推方向 8K TTFT -6.7%、吞吐 +10–13%，量级不变（方向救不了 PD） | EXP-011 |
| EXP-020《NCCL 旋钮矩阵复现 1.78 GB/s》 | ✅ 9/15（定因不唯一） | Socket 路径 5 档平台 1.51–1.70 GB/s 复现落窗（5/5）；SHM 默认 9.07、P2P_LEVEL 五取值全走 SHM 6.1–9.1（零效应）；PROTO LL/LL128/Simple 2.97/4.51/6.10；负载期 PCIe Gen4 x16 占比 92–98%（"未升 Gen4"分支排除）；**SHM 亦见 1/28 次塌陷 2.1–2.3（形状更像 EXP-002）→ H3 候选，定因不唯一**；H2 并发负载两次 OOM 未检验。措辞：collective 带宽必须带路径（SHM ≥6 / Socket 1.5–1.7）+ 指针，1.78 不再裸引；R0-1 停用状态维持，追认与否待用户 | `pd_disagg/hw/20260915T0237_*`、`hw/derived/20260915T0237_nccl_knob_matrix.csv`、`hw/20260915T0252_nccl_shm_probe_*`、EXP-020 §5/附录 A/B |
| EXP-021《NCCL allreduce dtype 扫描》 | ✅ 9/15（未决） | 延迟地板 float/half/bf16 14.40/13.77/13.53 µs（差 <7%，成立）；平台 half 6.06、bf16 6.39 vs float 4.83（float 两轮 7.50/2.15，运行间差 111% > dtype 差 25–32%，不可判）；bf16 平台 6.0–6.9 GB/s、地板 13.4–13.8 µs 入画像。措辞：EXP-018 float 结论可对 bf16 沿用，带宽报区间不报单值 | `pd_disagg/hw/20260915T0248_allreduce_size_scan_*`（12 份）、`hw/derived/20260915T0248_nccl_dtype_table.csv` |
| EXP-022《D2 大 M kernel A/B》 | ✅ 9/15 | 3 轮交叉 mean±std（µs）：EP 512/1024/2048/4096 = 681.1→637.7 **−6.38%**、881.2→757.5 **−14.04%**、1381.0→1244.6 **−9.88%**、2432.4→2266.2 **−6.83%**；非 EP 630.0→612.4 **−2.79%**、708.6→666.2 **−5.99%**、912.9→805.2 **−11.80%**、1453.3→1303.1 **−10.34%**；8/8 \|Δ\| > 2×合并 std。措辞：PR #54372 大 M 档有独立复测；S3 可扩为「M=1 −8.2/−3.6%，M≥512 −2.8~−14.0%」；EXP-015 §6「config 之外空间有限」限定为 M≤256 | `moe_perf/raw/EXP-022/bigM_20260915T0304/`（12 份）、`moe_perf/derived/20260915T0304_exp022_bigM_ab.csv`；作废首跑 `bigM_20260915T0303/FAILED_NOTE.txt` |
| EXP-023《replica2@512 饱和复测（SAT_CONC=128）》 | ✅ 9/15 | replica2@512 conc128 **20.87 req/s**（conc64 15.58，+34% > 阈值 +5% → 欠饱和确认），TTFT p50 771 ms、TPOT p50 36.8 ms、gate 通过；colocate@512 conc128（fresh）12.81（原 10.36，TPOT p50 67 ms 已破 SLO）；512 桶扩展效率同口径 **1.63×**（原 1.50×）；conc128 仍未封顶，只写「≥20.87 @conc128」。措辞：EXP-007 §5 的 15.58\*/1.50× 引用时改为 ≥20.87 @conc128 / 1.63×；fig7 重算需排除污染行 `20260915T0330_colocate_512x128_saturation`（前缀命中 8.9%） | `pd_disagg/results/b1_matrix/raw/20260915T0329_replica2_*`、`20260915T0333_colocate_*`、`runs.jsonl` 行 125–127、服务日志 `20260915T0312_replica2_conc128_server_*` |
| EXP-024《512 桶四臂统一到 conc128 口径》 | ✅ 9/15 | 四臂同口径 conc128：replica2 **20.87** · colocate **12.81** · tp2 **12.30** · pd1p1d **8.15** req/s；512 扩展效率 **1.63×**（原 1.50×）。**两臂性质与 colocate/replica2 相反**：tp2 在 conc64 即饱和（12.31→12.30，−0.1%）、pd1p1d 在 conc64 即撞传输墙（`bytes/wall` 0.230→0.239 GB/s，上限模型 0.2393/0.02936 = 8.15 与实测 8.152 精确吻合）；只有 colocate/replica2 真欠饱和（+23.6%/+34.0%）。副作用：**tp2 是四臂唯一不触发功率帽**（功率峰 326 W，无 0x4），PD 的 TPOT p50 18.97 ms 四臂最好 / TTFT p50 12669 ms 四臂最差。方法学：**并发>1 时 `bytes/ΣxferDuration` 不再是吞吐**（它累加了并发重叠的每请求时长），聚合速率一律用 `bytes/wall` | `runs.jsonl` 行 128–129、`raw/20260915T0949_tp2_*`、`raw/20260915T0952_pd1p1d_*`、`figures/fig7`（已按白名单重算） |
| EXP-020 附录 C《H3 长跑探针》 | ✅ 9/15 | SHM 路径 60 轮大消息扫描（float/half 交替、200ms PCIe 采样）：**按锁定阈值（<3.0）零塌陷 → 判定 C**（频率 <1.7%）。post-hoc：第 43 轮平台 3.23 GB/s 未达线但**曲线从 1M 起平坦**，与 EXP-021 的 2.15、EXP-002 的 1.78 同族 → 低平台态频率 1/60，与 1/28 合并 2/88 ≈ 2.3%，**两次都在 Gen4 x16、时钟满血下**（机理不是 PCIe 未升频）。观测到的低平台态分位 2.15/3.23 未及 2.3 以下，而 Socket 档稳定 1.51–1.70 → **证据天平向 Socket 倾斜但仍不追认**，R0-1 维持"停用" | `pd_disagg/hw/20260915T0936_nccl_h3_*`（120 文件）、`hw/derived/20260915T0936_nccl_h3_longrun.csv`、`scripts/nccl_h3_longrun_probe.sh`、EXP-020 附录 C |
| EXP-025《replica2@512 真饱和点扫描》 | ✅ 9/15 | 同 N=1200 同 seed 三档：**22.118 / 24.343 / 25.297 req/s**（conc 128/192/256），conc192 vs 128 **+10.06%**（>5% → 判据 A：**未封顶**），conc256 再 +3.92%（>2%，仍未见平台）。**吞吐峰 ≠ goodput 峰**：TPOT p50 在 conc192 就破 SLO 50ms（53.03）、TTFT p50 在 conc256 涨到 2444ms → 按锁定 SLO 自算 goodput **5.16 / 0.47 / 0.00 req/s**（最优在 conc128、conc256 归零）。**N 效应实测确认**：conc128@N=1200 = 22.12 vs @N=400 = 20.87（**+6.0%**），故 20.87 是 N=400 口径。措辞：`20.87` 需补 `N=400`；`1.63×` 保持 conc128 口径且注明随 conc 变；goodput 绝对值因 SLO 错配**不进对外文本** |
| EXP-026《NIXL descriptor 粒度实验》 | ✅ 9/15 | 实测三点：① **粒度不是瓶颈**——合并开启下 descriptor 16 KiB→16 MiB（1000×）带宽仅 **−2.0%**（0.3832→0.3757 GB/s）；② **NIXL 默认合并描述符**——contiguous+16 KiB+默认参数 telemetry `descCount=1`（4096 合 1），关掉合并则 0.2673→**0.3883 GB/s（+45.3%）**；③ **天花板是 TCP 传输层且不可调**——UCX 自报 RMA 恒 `rma_am(tcp/eth0)`、GPU 缓冲 software emulation；10 档 `UCX_TLS` 中凡含 `shm`/`sm`/`cuda_ipc` 者后端初始化即失败（`NIXL_ERR_BACKEND`），能起来的组合仍选 TCP（0.361–0.386）。交叉验证：EXP-013 的 **61.5–65.8 µs/desc** 与 EXP-006 的 **0.26–0.27 GB/s** 均落在**未合并/散列**档（我复现 61.3/64.2 µs、0.2673 GB/s），与合并档（42.2 µs、0.3883）不符 → **vLLM 的实际布局没吃到合并**。α≈0、β≈0.375 GB/s。**EXP-020 附录 A「合并上界 2.4–3.4×」作废**（真实 +45%，天花板 0.38 非链路的 0.60–0.91）|
| EXP-027《rr_proxy 开销拆分》 | ✅ 9/15 | 两臂并发形态对齐后（A = 两直连客户端**同时**各打一实例、600/conc64；B = 经代理 1200/conc128）：直连合计 **22.280 req/s**（11.140 + 11.253，墙钟 53.9/53.3s）vs 经代理 **21.820**（复跑 **21.862**）→ 代理开销 **R = 2.06%（保守）/ 2.56%（乐观）**，两次 B 臂自身差 0.19%，**判定 B（<5% 阈值）→ 缺口不来自代理**。附带：直连两实例几乎对称（差 1.0%）；经代理臂 TTFT p50 更低（388 vs 594ms）而 TPOT 略高（41.0 vs 38.8ms），因并发分布不同。排除代理后，1.63× 的候选收窄为「两实例共享主机资源」+「colocate 分母本身是过饱和态」|
| EXP-029《单实例能否承载双实例的负载》 | ✅ 9/15 | 三臂（两实例都在跑，只改负载分布）：S1 单实例 600/conc64 = **11.135**（与 EXP-027 A1 的 11.140 差 **0.04%**，装置校验过）、S2 单实例 1200/conc128 = **13.049**、S3 单实例 1200/conc64 = 11.103。**H-A（单实例即可承载）被否，H-B 成立**（两实例合计 22.392 / 单实例 13.049 = 1.716×）。**但核心发现是「1.63×」口径不对称**：按**每 GPU 在飞量**对齐后 —— 每卡 64 在飞时 2 实例 22.392 vs 2×单实例 22.207 = **100.8%**（无主机级干扰）；每卡 128 在飞时 2 实例 25.297（EXP-025 实测）vs 预测 26.098 = **96.9%**。故 `1.63× = 20.87/12.807` 是拿"replica2 每卡 64 在飞"除"colocate 单卡 128 在飞"，两个不同操作点相除；**第二张卡实为 97–101% 线性加成**。附带：单实例 conc64→128 得 11.10→13.05（+17.5%）而 TPOT p50 39.0→69.6 ms（+79%，破 SLO）→ 单卡 conc64 是自然操作点 |

## 措辞红线状态（写简历/报告前查此表）

| 红线 | 当前 | 解锁条件 / 依据 |
|---|---|---|
| "P2P 受限" | ✅ 可用 | `hw/p2p_bandwidth_latency.txt`（connectivity=0）+ `hw/topo.txt`（GNS） |
| "社区空缺"（MoE config） | ✅ 可用 | 2026-08-21 远端复核 + 2026-08-29 对 cacc429f62 复验（rebase 后）：`moe_configs/DEDUP.md`（main 无 E=30；E=60，N=704 仅 MI300X；PR/issue 无冲突，空缺仍成立） |
| "KV 传输占 TTFT X%" | ✅ 可用 | **EXT-1 已解锁（EXP-013，2026-08-23）**：request 级三段关联（同身份同时钟域），KV 等待占 TTFT 54.2/62.5/64.2%（512/2K/8K），闭环误差 ≤0.08% |
| telemetry 带宽表述 | 限定 | 只能称 telemetry-derived effective throughput；xferDuration 不与 postDuration 相加 |
| 0.17 两 bug | 限定 | 只写"复现/定位/验证"，禁"发现/修复"；"吃透"→"梳理"。**动态复现已闭环（EXP-012）**：静态 file：line + 实机崩溃/挂死现场 + 实证修正，"复现/定位/验证"三词均有实测背书 |
| A/B 版本对照 | 限定 | 只称 system-version comparison，标注传输方向不同 |
| PR 状态 | 限定 | 已提交（#54372）可写"提交"；未合并不写"合入" |
| D2 e2e +0.8~1.2% | 🚫 不作 headline | 低于跨会话漂移；主证据=kernel A/B（8/24 定档修正） |
| EXT-1 patch 定性 | 限定 | "~16 行本地可观测性改动"，不得表述为 NIXL/Connector 核心改造 |

## 方法论：诚实度文化

- **可溯源**：每个结果文件首行 provenance（env/sha/完整命令/GPU/驱动）；raw 不可变，表图一律由脚本从 raw 重算，坏数据移 archive 留痕而非原地改。
- **互证与对照**：关键数字带分布不带单点（p50/p90；EXP-013 每桶 n=11，六段闭环误差 p50 <0.1%，bytes 与 Prometheus 独立对账）；因果归属必设对照臂（D5 专设无 EPLB 对照组，输出逐字节一致才把分歧归因 EPLB）。
- **负结论与勘误留痕**：D3 依数据放弃 kernel 改动、D5 判"不上简历"、8/24 M 档 19→18 勘正与 D2 e2e 降级出 headline，全部在台账原位标注，不删不藏。

## 待办（仅用户本人可执行）

1. **R0-6**：线上简历稿"发现/修复"→"复现/定位/验证"（待用户）。
2. **D2 PR #54372 跟进**：OPEN，pre-run-check 失败（缺 `ready` label / 作者 0 merged PR），需在 vLLM Slack #pr-reviews 求 reviewer 加 label（待用户）。
3. 可选（9 月池）：AutoGPTQMoEMethod 补 supports_eplb（EXP-017 §8，先查重）； B4 报告终稿通读；简历 9 月投递版成稿。

## 备份与 push 校验

- 本仓库 commit 即产物锚点；远程 = github.com/lyell0710/vllmExperience（private）。
- 大文件（trace、nsys-rep）入 git——本仓库就是证据箱，体积换可信度。 **例外（8/23 起）**：单文件 >100MB 触 GitHub pre-receive 硬限，仅本地保存并在同目录 `LARGEFILES.md` 登记 sha256；`*.sqlite`（nsys 可再生衍生品）全局忽略。教训：push 成败必须看 `git status -sb`，管道 `| tail` 会吞掉真实返回码。
