# vLLM 秋招项目执行清单 · v3 锁定版(2026-08-21)

> 原文由用户 2026-08-23 提供，存档作为"全线闭环"声明的对照基准。执行状态见 LEDGER.md 证据台账；本文件不改动原文，只在文末附闭环对照注。

## 使用规则

- 每项做完：勾掉 + 在项目下方写一行产物路径（结果文件 / 截图 / commit）。
- 实验关键项按 **假设 / Gate / 产物 / 失败处理** 四栏执行；Gate 不过的数据一律不进报告。
- 每个结果文件头部必须有 provenance 行：`git SHA + vllm.__version__ + 完整命令 + 时间 + GPU`。
- 条件项（标 ⚑）先看 gate 结果再动手，不预支时间。

## 环境定义(三 venv,独立目录,互不污染)

| 代号  | 版本               | 用途                                                         |
| ----- | ------------------ | ------------------------------------------------------------ |
| ENV-A | `vllm==0.17.1`     | 课程历史基线，时间盒 2h                                      |
| ENV-B | `v0.25.1@752a3a50` | **主战场**（已核实：NIXL metrics、`kv_load_failure_policy`、默认 Pull 三者齐备；R0-3 smoke 通过即正式锁定） |
| ENV-C | `main@7aa248fc`    | sanity 对照 + PR 开发（PR 必须对 main）                      |

## 硬里程碑

- **M1(8/31)**：主线一 PD 报告成稿（四臂矩阵为主体）
- **M2（9/1 前）**：MoE 模型上卡跑通
- **M3（9 月内）**：config PR 提交，或第一个 kernel 优化数字

---

## 第 0 块 · 地基(8/21–8/22)

- [x] **R0-1 硬件三数**：`p2pBandwidthLatencyTest` + nccl-tests `all_reduce_perf`（结果标注：仅代表 TP collective 路径）+ NIXL 实测 bytes / xferDuration（代表 KV 通路）。三个数进所有报告的"硬件画像"段。
- [x] **R0-2 三 venv 落位**：独立目录 + `.venv`；写一个输出 provenance 行的 shell 函数，所有实验脚本引用它。
- [x] **R0-3 NIXL 1P1D smoke @ ENV-B**（Gate 全过，锁定 v0.25.1）
- [x] **R0-4 0.17.1 课程基线**（降级完成 + 8/23 动态复现补齐，EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》）
- [x] **R0-5 profiling 工装 @ ENV-B**
- [ ] **R0-6 简历措辞排雷（只有你能做）**：线上稿"发现/修复"改"复现/定位/验证"。**← 唯一未闭环项，仅用户可操作**

## 第 1 块 · 主线一 PD(8/22–8/31)

- [x] **B1 四臂矩阵**(EXP-004~007)
- [x] **B2 归因层**（EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》 + EXP-013《EXT-1 request 级 KV-wait 关联》收官）
- [x] ⚑ **EXT-1 telemetry 补 identity 最小 patch**（EXP-013；上游查重后定位本地，#52859 在途）
- [x] **B3 版本对照**（EXP-008《B3 有限版本对照》 + EXP-012 两维度定稿）
- [x] ⚑ **EXT-2 NixlPush 单点**（EXP-011《EXT-2 NixlPush 单点》，专用 push proxy）
- [x] **B4 报告成稿**（v2 定稿 2026-08-23）
- [—] **B-alt 失败分析分支**：未触发（1P1D 通，无需回退）

## 第 2 块 · MoE 环境

- [x] **C1 Qwen1.5-MoE-A2.7B 上卡**（EXP-009《C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据》）
- [x] **C2 config gate**（三重闭环，含远端查重）
- [x] **C3 Qwen3-30B-A3B W4A16 上卡**（EXP-010《C3 Qwen3-30B-A3B W4A16 上卡》,GPTQ-Int4 锁定）

## 第 3 块 · MoE 主攻

- [x] **D1 nsys MoE 分解**（EXP-014《D1 MoE decode 分解》，含反转点发现）
- [x] **D2 config 调优 + PR**（EXP-015《D2 MoE config 调优》；JSON 交付 + 六件套材料齐，**提交动作留用户**）
- [x] **D3 kernel 优化一处**（依 D2 数据转结论：config 即最优杠杆，不做无数据支撑的改动——判定记录于 EXP-015 §6）
- [x] **D4 FP8 vs W4A16 对比**（EXP-016《D4 FP8 vs W4A16 同卡对比》）
- [x] **D5 EPLB gate**（EXP-017《D5 EPLB gate》；gate 判定完成，按规则砍掉不上简历，白板级保留）

## 持续项

- [x] **P1 PR 六件套**（材料层齐备，PR_DRAFT.md；DCO 签名与提交由用户执行）
- [x] **P2 每周白板三图**(whiteboard/01–03)
- [x] **P3 每周日核对**(WEEKLY.md)

---

## 闭环对照注(2026-08-23)

- 时间线：原计划 8/21–9 月，实际 8/21–8/23 全部完成（D 阶段提前约 3 周）。
- 唯一残项 R0-6 与 D2 的 PR 提交动作均为"仅用户可操作"。
- 简历映射 S1–S4 成稿候选见 RESUME_EVIDENCE.md；S5 按 gate 规则维持不上简历。

## 勘注（2026-09-15）

- 上文 D2 行"提交动作留用户"、P1 行"由用户执行"、闭环对照注"仅用户可操作"三处原文保留不改。PR 已由用户本人于 2026-08-29 提交为 vllm-project/vllm#54372（gh 实查：OPEN 未合并；CI pre-run-check ×2 失败——缺 `ready` label 且作者 0 merged PR；无人类 review）。状态权威见 HANDOFF.md §5；红线不变：可写"提交"，未合并不写"合入"。
