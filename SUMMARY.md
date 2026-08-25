# 项目汇总单（2026-08-21 · Day 0 全量收官）

> ⚠ **本文为 2026-08-21 Day 0 冻结快照**,其"任务完成度总表"与"还欠的事"均已过时。
> 最新状态以 [HANDOFF.md](HANDOFF.md) 与 [LEDGER.md](LEDGER.md) 证据台账为准
> (8/23 已闭环:R0-4 动态复现/EXT-1/EXT-2/B2/B3/B4 v2/D1;课程脚本已证非必需)。

> 一天内从环境体检到 M1 报告草稿的全部产出。阅读顺序：本文 →
> [STUDY_GUIDE.md](STUDY_GUIDE.md)（零基础教学）→ [REPORT](pd_disagg/REPORT.md) →
> [LAB_JOURNAL](LAB_JOURNAL.md)（过程叙事）→ [records/](records/)（逐实验细节）。

## 一、任务完成度总表

| 块 | 项 | 状态 | 一句话 |
|---|---|---|---|
| 地基 | R0-1 硬件三数 | ✅ | P2P 禁用（GNS）/ NCCL 1.78GB/s / NIXL 0.27GB/s |
| 地基 | R0-2 三 venv + provenance | ✅ | ~/venvs 三环境 + 全套卫生工装 |
| 地基 | R0-3 NIXL smoke + 裁决 | ✅ | 双版本 PASS，锁定 v0.25.1 主战场 |
| 地基 | R0-4 课程基线 | ✅ 降级完成 | 双 bug 源码机理全 file:line（动态复现待课程脚本） |
| 地基 | R0-5 profiling 工装 | ✅ | torch profiler 直控 + nsys 可用；发现接口版本演化 |
| 地基 | R0-6 简历排雷 | ◐ | 本地 tex 干净；**线上稿需你本人改** |
| 主线一 | B1 四臂矩阵 | ✅ | 60 扫描点 + 归因 + 饱和，goodput/成本边界全出 |
| 主线一 | B2 归因层 | ◐ | TTFT 分解（传输 54–64%）+ 记账溯源关闭；EXT-1 弹性未做 |
| 主线一 | B3 版本对照 | ◐ 有限版 | 512 桶饱和 +45%，延迟不变；PD-vs-PD 待课程脚本 |
| 主线一 | B4 报告 | ✅ v1 草稿 | `pd_disagg/REPORT.md`（一页结论 + 四章 + 附录） |
| MoE | C1 MoE 上卡 | ✅ | TP2+EP 可用，TPOT 4.62ms，D2 baseline 在案 |
| MoE | C2 config 查重 | ✅ | 三重闭环（本地+远端+运行时告警原文） |
| MoE | C3 W4A16 上卡 | ◐ | checkpoint 锁定 Qwen3-30B-A3B-GPTQ-Int4，下载/上卡中 |
| 9 月 | D1–D5 | ⬜ | 按计划 9 月执行（D2 的 before 基线今天已备好） |
| 弹性 | EXT-1/EXT-2 | ⬜ | 不阻塞主线 |

## 二、核心数字（简历/面试的弹药，全部有 provenance）

**硬件画像**：P2P=GNS 驱动禁用 · 单向 D2D 0.60–0.91 GB/s · 双向 22.7 GB/s ·
NCCL allreduce 1.78 GB/s · NIXL KV 有效 0.26–0.27 GB/s · 卡内 ~924 GB/s ·
450W 功率帽降频 2820→2475MHz（TTFT +30%）

**四臂矩阵**（512/2048/8192，SLO=TTFT≤328/891/4626ms+TPOT≤50ms）：

| | colocate(1卡) | replica2 | tp2 | pd1p1d |
|---|---|---|---|---|
| 饱和 req/s | 10.36/3.63/0.90 | 15.58/7.00/1.78 | 12.31/4.16/1.02 | 7.84/2.12/0.54 |
| goodput 峰值 | 8.57/2.41/0.43 | 12.75/4.96/0.90 | 10.18/2.51/0.60 | 1.59/0.16/0.11 |
| 无负载 TTFT ms | 65/225/925 | 66/221/903 | 64/220/881 | 219/719/2719 |
| TPOT ms | ~16 | ~16 | **9.3** | ~16 |

**版本对照**：0.17.1→0.25.1 延迟不变、512 桶饱和 +45%、启动 308→58s
**MoE**：A2.7B TP2+EP TPOT 4.62ms（dense TP2 的 2.0×）；E=30,N=1408 config 缺失
运行时告警在案

## 三、产物地图

```
experiments/（github.com/lyell0710/vllmExperience, private）
├── SUMMARY.md / STUDY_GUIDE.md      ← 你现在读的 + 明天的教材
├── README.md                        ← 对外门面（台账/红线/约定移至 LEDGER.md）
├── LAB_JOURNAL.md                   ← 复现级日记（Day 0 全程 §0–§13+）
├── RESUME_EVIDENCE.md               ← 简历句成稿候选（数字已填）
├── records/EXP-001~009 + TEMPLATE   ← 逐实验八节记录
├── pd_disagg/
│   ├── REPORT.md                    ← B4 报告 v1（M1 交付物）
│   ├── figures/fig1–6.png           ← 六张报告图（结论句标题）
│   ├── results/b1_matrix/           ← runs.jsonl(109行)+raw+snapshots+derived
│   ├── analysis/                    ← 双 bug 机理 + token 记账溯源
│   ├── hw/ smoke/ profiling/        ← 地基证据
│   └── scripts/                     ← 全套工装（可复现）
└── moe_configs/DEDUP.md             ← C2 三重闭环
```

## 四、还欠的事（按优先级）

1. **你本人**：线上简历稿措辞排雷（R0-6）；课程脚本传上来可解锁 R0-4 动态复现
   与 B3 完整版。
2. C3 上卡收尾（进行中）。
3. 9 月：D1 nsys 分解 → D2 config 调优 + 六件套 PR（baseline 已备）→ D3/D4/D5。
4. 弹性：EXT-1（解锁"KV 占 TTFT%"红线）、EXT-2（NixlPush 单点）。
5. B4 v2 定稿（8/31 前：吸收 EXT-1 或注记其缺席）。
