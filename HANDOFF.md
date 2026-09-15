# 交接文档(给接手的 agent)· 2026-09-15(第三次更新)

> 本文件是**唯一入口**。项目全貌读 [SUMMARY.md](SUMMARY.md)（注：其"还欠的事" 已过时，以本文 §5 为准）；状态/措辞速查读 [LEDGER.md](LEDGER.md)（README 已改为对外门面）； 过程叙事读 [LAB_JOURNAL.md](LAB_JOURNAL.md)（§17/§18 是最新）； 简历句读 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)；周核对读 [WEEKLY.md](WEEKLY.md)。

## 0. 这是什么项目

用户（李玉章，lyell0710@gmail.com）的 vLLM 秋招项目：在 **2×RTX 4090**（无 NVLink、 P2P 驱动禁用）上做推理部署选型 + MoE 性能优化，产出简历/面试素材。 **证据仓库** = `/root/projects/vllm/experiments/`（独立嵌套 git，远程 `github.com/lyell0710/vllmExperience` private，gh 已登录 lyell0710）。

## 1. 铁律(每次运作必守)

1. 每次运作结束在 `LAB_JOURNAL.md` 末尾追加一节（四问格式）。
2. 每个实验一份 `records/EXP-NNN_<slug>.md`（八节模板），当场写。
3. 任何 GPU 跑一律存 raw（bench 加 `--save-result`）；没存的降级"终端级证据"。
4. 结果文件首行 provenance；数据里程碑 commit + push。
5. 措辞红线查 LEDGER.md 表。**"KV 占 TTFT X%" 已解锁（EXP-013《EXT-1 request 级 KV-wait 关联》）**；0.17 两 bug 仍只写"复现/定位/验证"；PR #54372 已提交（2026-08-29）可写"提交"，未合并不写"合入"。
6. **杀进程用方括号技巧**(`pkill -f '[v]llm serve'`)——复合命令里普通 pattern 会匹配自身 shell 导致 exit 144（本会话又验证了两次）。

## 2. 环境(都已就位)

| 代号 | venv | 版本 | 用途 |
|---|---|---|---|
| ENV-A | `/root/venvs/v0.17.1` | 0.17.1(PyPI) | 历史基线（已完成使命） |
| ENV-B | `/root/venvs/v0.25.1` | 0.25.1(752a3a5044)| **主战场**；**注意：带 EXT-1 本地 patch**（16 行 `# EXT1` 标记，`pd_disagg/ext1/nixl_req_telemetry_v0251.patch`，原件 ext1/orig/） |
| ENV-C | `/root/venvs/main` + `/root/projects/vllm` editable | 0.26.1rc1.dev(main@7aa248fc)| MoE 开发/PR（ray 已装） |

- matplotlib 在 `/root/venvs/kernel-opt/bin/python`（ENV-B 没有）。
- 模型（HF cache）:Qwen2-7B、Qwen2.5-0.5B、Qwen1.5-MoE-A2.7B、 Qwen3-30B-A3B-GPTQ-Int4、**Qwen3-30B-A3B-FP8（8/23 新下，35G,D4 用）**。
- 磁盘 350G 余 ~180G；docker 跑不了容器；py-spy/gdb 不可用（ptrace 限制）。
- `uv` 装包：`/root/.local/bin/uv pip install --python <venv>/bin/python <pkg>`。

## 3. 工装地图

- PD 线：`pd_disagg/scripts/`(run_point/collect_point/metrics_snapshot/ provenance/profile_ctl/make_figures)+ `pd_disagg/ext1/`（EXT-1 全套： patch、instrumented proxy、client、run_ext1.sh、analyze_ext1.py）。
- MoE 线（8/23 新增）：`moe_perf/`——d1_sweep.sh / d1_nsys.sh（**必须 `--cuda-graph-trace=node`，默认 graph 级会把 CUDA graphs 内的 decode kernel 全部藏掉，踩坑记录在 EXP-014《D1 MoE decode 分解》 §7**）/ d1_analyze.py / d1_kernels.py / d2_tune.sh / d2_ab.sh / d4_fp8_w4a16.sh / d4_ppl.py / d5_eplb.sh / PR_DRAFT.md。
- nsys 精确控窗：`--profiler-config.profiler=cuda` + nsys `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown`, HTTP /start_profile /stop_profile 触发。

## 4. 已完成(✅ = 数据落盘 + 记录 + 推送;详见 LEDGER.md 台账)

R0-1~R0-5、B1、**B2（EXT-1 收官，KV 占 TTFT 54.2/62.5/64.2%）**、B3（两维度定稿）、**B4 报告 v2 定稿**、C1/C2/C3、EXT-2、R0-4 动态复现（EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》）、 **D1（EXP-014：MoE decode 反转点 2.03×→0.82×，fused_moe 56.4%@bs32）**、 **D4（EXP-016《D4 FP8 vs W4A16 同卡对比》：W4A16 decode 胜 23–48%，FP8 PPL 优 3.3%，Ada 路径 file：line）**、 **D5（EXP-017《D5 EPLB gate》：gate 判定完成，不上简历，白板素材）**、交付物对抗校验（20 条修复，commit cc473de）、**D2（EXP-015《D2 MoE config 调优》：两空缺 config 交付，kernel 两端 -3.3~-8.5%，e2e TPOT +0.8~1.2%，120 passed；D3 依数据转结论句）**。记录 EXP-001~023 全齐。M1/M2 提前达成，M3 材料齐备。

## 5. 状态:清单全线完成(2026-08-23 傍晚)

D0 地基、B1–B4、C1–C3、D1–D5、EXT-1/2、P1（材料层）/P2/P3 全部收官， 记录 EXP-001~023 齐（见 LEDGER.md 索引）。M1/M2 提前达成；M3 = 材料齐备。

### 唯余一项,只能由用户本人执行(9/7 勘:D2 PR 已提交)
1. **R0-6**：线上简历稿"发现/修复"→"复现/定位/验证"。
2. ~~D2 PR 提交~~ **已完成**：vllm-project/vllm#54372 已由用户本人于 2026-08-29 提交（gh 实查 OPEN 未合并；正文存档 `moe_perf/PR_BODY.txt`，commit 1252684）。红线不变：未合并不写"合入"。LEDGER D2 行/待办同句待同步（2026-09-07 批次改动范围外）。

### 可选(9 月池)
- AutoGPTQMoEMethod 补 supports_eplb（EXP-017 §8，上游 TODO 邀请，先查重）；
- B4 报告终稿通读；简历 9 月投递版成稿（S1–S4 全有数，见 RESUME_EVIDENCE）。

## 6. 已知坑(别重复踩)

- nsys 看 vLLM decode 必须 `--cuda-graph-trace=node`(EXP-014 §7)。
- 多模型 bench：`MODEL` 必须显式设，否则 404。
- tp2 gpu-memory-utilization 0.88(0.9 warmup OOM)。
- 前缀缓存污染：每点唯一 seed。
- 功率帽：持续 prefill 降频 ~12%，同热工况才可比。
- 大文件 push 慢（nsys rep 上百 MB），push 放后台跑。
- **（9/15）`pkill -f '[v]llm serve …'` 与含 `vllm serve …` 字面量的启动命令写在同一条复合命令里仍会 exit 144**——方括号只保护 pattern 本身，不保护同一命令行里的其它字面量；启动与清理分两条命令投递。
- **（9/15）`kill` 父脚本不会杀掉脱离的 `nvidia-smi -lms` 采样器**——它会成孤儿（ppid=1）继续写 raw 文件，且**不占 GPU 计算位**，`nvidia-smi --query-compute-apps` 查不出来。收尾复核必须遍历 `/proc/*/cmdline` 搜 `query-gpu`。
- **（9/15）冷 page cache 下 v0.25.1 起栈约 613 s，`benchmark_moe.py` 的 `ray.init()` 会超时**（EXP-022 首跑作废）——GPU 任务前先热身模型文件或把预算按 10 分钟设。

## 7. 当前状态快照(2026-09-15,总览文档 + 四实验补跑批次)

- git HEAD：本批次 commit 落地后与 origin/main 同步（以 `git status -sb` / `git log -1` 实时核对为准）。
- 硬件占用：双卡空闲（nvidia-smi 实查 0% util，无 compute 进程）——可跑 GPU。
- 下一步第一动作：**用户裁决** NCCL collective 带宽措辞——EXP-020《NCCL 旋钮矩阵复现 1.78 GB/s》给出 Socket 路径 1.51–1.70 落窗但 SHM 路径亦见 1/28 次塌陷 2.1–2.3（定因不唯一），决定 R0-1 是否从「停用」改为「带路径引用」并连带修 EXP-005 §6 / RESUME_EVIDENCE / README 约 20 处"待复核"。agent 侧可做的下一件：EXP-020 H3 长跑探针（≥50 轮带 PCIe 采样）；ncu ES 复采仍需采集主机（主力机 RmProfilingAdminOnly=1）。

## 8. 还开着的事（跨仓，按 ROI 排序）

1. **EXP-002「1.78 GB/s」vs EXP-018「6.2 GB/s」已闭环（EXP-019）**：判定为真实环境差异、非计时口径问题（nccl-tests 计时区内无 malloc 混入）；SHM 3.96 vs Socket 0.76 GB/s 差 5 倍指向路径/PCIe 状态差异。复现实验设计固化在 EXP-019 §8（扫 NCCL_P2P_LEVEL × NCCL_SHM_DISABLE 找复现档），不再阻塞任何对外主张。
2. **EXP-D22「88µs 反推」已重做（llm-engine f3c2c3d）**：实测拆分纯传输 14µs（2.1%）+ torch.distributed 派发/同步 55.7µs（8.1%），归因「亏在派发/同步、不在带宽」。
3. **md_reflow.py 已加 4 处语法保护**（callout / HTML 注释 / 引用块表格 / li 内嵌引用块）与内容守恒断言。注意守恒断言的盲区：它把全角半角视为等价，所以**语法字符必须走 protect() 挖空，不能指望断言拦下**。

