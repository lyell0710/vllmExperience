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

## 证据台账（勾一项 = 数据落盘 + 本表登记产物路径）

| 项 | 状态 | 关键数字 | 产物 |
|---|---|---|---|
| R0-1 硬件三数 | ✅ 8/21 | P2P=GNS 禁用；单向 D2D 0.60–0.91 GB/s，双向 22.7 GB/s，延迟 ~15µs；NCCL bus bw 1.78 GB/s | `pd_disagg/hw/{topo,p2p_bandwidth_latency,all_reduce_perf}.txt` |
| R0-2 三 venv + provenance | ✅ 8/21 | ~/venvs/{v0.17.1, v0.25.1, main} 均验证 import | `pd_disagg/scripts/provenance.sh`；setup 记录 `pd_disagg/setup_envs.log` |
| R0-3 NIXL 1P1D smoke | ✅ 8/21 | 双版本 3/3 PASS；avg xfer 14.1ms / 0.188MB / 13.3MB/s；裁决锁定 v0.25.1 | `pd_disagg/smoke/`、`pd_disagg/DECISION.md` |
| R0-4 0.17.1 课程基线 | ⬜ 阻塞 | — | 课程脚本不在本机，待提供；超时降级纯阅读 |
| R0-5 profiling 工装 | ✅ 8/21 | torch profiler 直控 P/D 端口跑通（trace 落盘）；nsys 容器内可用 | `pd_disagg/profiling/r0_5_torch_profiler_check.txt`、`traces_smoke/`、`scripts/profile_ctl.sh` |
| R0-6 简历措辞排雷 | ◐ | 本地 tex 无违规表述（已核）；线上稿待改 | 仅用户可操作 |
| B1 四臂矩阵 | ◐ | **attribution 12/12 全臂完成、gate 全 PASS**。TTFT p50@8K：colocate 925 / replica2 715 / tp2 694 / pd1p1d 2685ms；TPOT：tp2 9.3ms vs 其余 ~16ms；GPU·s/req@8K：2.95/5.58/3.79/9.24。SLO 已锁（328/891/4626ms + TPOT 50ms）。剩 sweep | `results/b1_matrix/runs.jsonl`（12 行）|
| B1 附带发现 | ✅ 8/21 | ① 功率帽节流：持续 prefill 降频 2820→2475MHz（SW Power Cap，非热），TTFT +30%；② NIXL 有效吞吐 0.26–0.27GB/s 恒定（descriptor ~16KB 碎片化）；③ TP2 decode 提速 42%（带宽分摊）但 prefill 零加速（allreduce 撞 1.78GB/s 墙） | runs.jsonl gpu_telemetry / gates 字段；`DECISION.md` 硬件基线 |
| B2 归因层 | ⬜ | — | — |
| B3 版本对照 | ⬜ | 已知差异一例：profiler 接口 env var→CLI（见 profiling 检查文件 note） | — |
| B4 报告 | ⬜ | — | — |
| C1 Qwen1.5-MoE 上卡 | ⬜ | — | — |
| C2 config 查重 | ✅ 8/21 | 本地+远端均确认空缺；#48309(4090D fp8, OPEN) 为相邻先例非重复 | `moe_configs/DEDUP.md`（含远端复核节） |
| C3 W4A16 上卡 | ⬜ | — | — |
| D1–D5 | ⬜ | — | — |
| EXT-1 / EXT-2 | ⚑ | 弹性，不阻塞主线 | — |

## 措辞红线状态（写简历/报告前查此表）

| 红线 | 当前 | 解锁条件 / 依据 |
|---|---|---|
| "P2P 受限" | ✅ 可用 | `hw/p2p_bandwidth_latency.txt`（connectivity=0）+ `hw/topo.txt`（GNS） |
| "社区空缺"（MoE config） | ✅ 可用 | 2026-08-21 远端复核完成：`moe_configs/DEDUP.md`（main 无 E=30；E=60,N=704 仅 MI300X；PR/issue 无冲突） |
| "KV 传输占 TTFT X%" | 🚫 禁用 | 待 EXT-1 request 级关联；此前只可写 telemetry 原生量 |
| telemetry 带宽表述 | 限定 | 只能称 telemetry-derived effective throughput；xferDuration 不与 postDuration 相加 |
| 0.17 两 bug | 限定 | 只写"复现/定位/验证"，禁"发现/修复"；"吃透"→"梳理" |
| A/B 版本对照 | 限定 | 只称 system-version comparison，标注传输方向不同 |
| PR 状态 | 限定 | 未提交不写"提交"，未合并不写"合入" |

## 备份

- 本仓库 commit 即产物锚点；**远程备份待配置**（需一个私有 repo 地址）。
- 大文件（trace、nsys-rep）也入 git——本仓库就是证据箱，体积换可信度。
