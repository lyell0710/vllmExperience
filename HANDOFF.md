# 交接文档(给接手的 agent)· 2026-08-23(第二次全面更新)

> 本文件是**唯一入口**。项目全貌读 [SUMMARY.md](SUMMARY.md)(注:其"还欠的事"
> 已过时,以本文 §5 为准);状态/措辞速查读 [LEDGER.md](LEDGER.md)(README 已改为对外门面);
> 过程叙事读 [LAB_JOURNAL.md](LAB_JOURNAL.md)(§17/§18 是最新);
> 简历句读 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md);周核对读 [WEEKLY.md](WEEKLY.md)。

## 0. 这是什么项目

用户(李玉章,lyell0710@gmail.com)的 vLLM 秋招项目:在 **2×RTX 4090**(无 NVLink、
P2P 驱动禁用)上做推理部署选型 + MoE 性能优化,产出简历/面试素材。
**证据仓库** = `/root/projects/vllm/experiments/`(独立嵌套 git,远程
`github.com/lyell0710/vllmExperience` private,gh 已登录 lyell0710)。

## 1. 铁律(每次运作必守)

1. 每次运作结束在 `LAB_JOURNAL.md` 末尾追加一节(四问格式)。
2. 每个实验一份 `records/EXP-NNN_<slug>.md`(八节模板),当场写。
3. 任何 GPU 跑一律存 raw(bench 加 `--save-result`);没存的降级"终端级证据"。
4. 结果文件首行 provenance;数据里程碑 commit + push。
5. 措辞红线查 LEDGER.md 表。**"KV 占 TTFT X%" 已解锁(EXP-013)**;0.17 两 bug
   仍只写"复现/定位/验证";PR 未提交不写"提交"。
6. **杀进程用方括号技巧**(`pkill -f '[v]llm serve'`)——复合命令里普通 pattern
   会匹配自身 shell 导致 exit 144(本会话又验证了两次)。

## 2. 环境(都已就位)

| 代号 | venv | 版本 | 用途 |
|---|---|---|---|
| ENV-A | `/root/venvs/v0.17.1` | 0.17.1(PyPI) | 历史基线(已完成使命) |
| ENV-B | `/root/venvs/v0.25.1` | 0.25.1(752a3a5044)| **主战场**;**注意:带 EXT-1 本地 patch**(16 行 `# EXT1` 标记,`pd_disagg/ext1/nixl_req_telemetry_v0251.patch`,原件 ext1/orig/) |
| ENV-C | `/root/venvs/main` + `/root/projects/vllm` editable | 0.26.1rc1.dev(main@7aa248fc)| MoE 开发/PR(ray 已装) |

- matplotlib 在 `/root/venvs/kernel-opt/bin/python`(ENV-B 没有)。
- 模型(HF cache):Qwen2-7B、Qwen2.5-0.5B、Qwen1.5-MoE-A2.7B、
  Qwen3-30B-A3B-GPTQ-Int4、**Qwen3-30B-A3B-FP8(8/23 新下,35G,D4 用)**。
- 磁盘 350G 余 ~180G;docker 跑不了容器;py-spy/gdb 不可用(ptrace 限制)。
- `uv` 装包:`/root/.local/bin/uv pip install --python <venv>/bin/python <pkg>`。

## 3. 工装地图

- PD 线:`pd_disagg/scripts/`(run_point/collect_point/metrics_snapshot/
  provenance/profile_ctl/make_figures)+ `pd_disagg/ext1/`(EXT-1 全套:
  patch、instrumented proxy、client、run_ext1.sh、analyze_ext1.py)。
- MoE 线(8/23 新增):`moe_perf/`——d1_sweep.sh / d1_nsys.sh(**必须
  `--cuda-graph-trace=node`,默认 graph 级会把 CUDA graphs 内的 decode kernel
  全部藏掉,踩坑记录在 EXP-014 §7**)/ d1_analyze.py / d1_kernels.py /
  d2_tune.sh / d2_ab.sh / d4_fp8_w4a16.sh / d4_ppl.py / d5_eplb.sh / PR_DRAFT.md。
- nsys 精确控窗:`--profiler-config.profiler=cuda` + nsys
  `--capture-range=cudaProfilerApi --capture-range-end=stop-shutdown`,
  HTTP /start_profile /stop_profile 触发。

## 4. 已完成(✅ = 数据落盘 + 记录 + 推送;详见 LEDGER.md 台账)

R0-1~R0-5、B1、**B2(EXT-1 收官,KV 占 TTFT 54.2/62.5/64.2%)**、B3(两维度
定稿)、**B4 报告 v2 定稿**、C1/C2/C3、EXT-2、R0-4 动态复现(EXP-012)、
**D1(EXP-014:MoE decode 反转点 2.03×→0.82×,fused_moe 56.4%@bs32)**、
**D4(EXP-016:W4A16 decode 胜 23–48%,FP8 PPL 优 3.3%,Ada 路径 file:line)**、
**D5(EXP-017:gate 判定完成,不上简历,白板素材)**、交付物对抗校验(20 条
修复,commit cc473de)、**D2(EXP-015:两空缺 config 交付,kernel 两端
-3.3~-8.5%,e2e TPOT +0.8~1.2%,120 passed;D3 依数据转结论句)**。
记录 EXP-001~017 全齐。M1/M2 提前达成,M3 材料齐备。

## 5. 状态:清单全线完成(2026-08-23 傍晚)

D0 地基、B1–B4、C1–C3、D1–D5、EXT-1/2、P1(材料层)/P2/P3 全部收官,
记录 EXP-001~017 齐(见 LEDGER.md 索引)。M1/M2 提前达成;M3 = 材料齐备。

### 唯余两项,均只能由用户本人执行
1. **R0-6**:线上简历稿"发现/修复"→"复现/定位/验证"。
2. **D2 PR 提交**:分支 `moe-config-4090-qwen15moe`(/root/projects/vllm,
   两 JSON 已暂存)→ 逐行 review → `git commit -s` → fork/push → 按
   `moe_perf/PR_DRAFT.md` 开 PR(六件套数字已全部回填)。

### 可选(9 月池)
- AutoGPTQMoEMethod 补 supports_eplb(EXP-017 §8,上游 TODO 邀请,先查重);
- B4 报告终稿通读;简历 9 月投递版成稿(S1–S4 全有数,见 RESUME_EVIDENCE)。

## 6. 已知坑(别重复踩)

- nsys 看 vLLM decode 必须 `--cuda-graph-trace=node`(EXP-014 §7)。
- 多模型 bench:`MODEL` 必须显式设,否则 404。
- tp2 gpu-memory-utilization 0.88(0.9 warmup OOM)。
- 前缀缓存污染:每点唯一 seed。
- 功率帽:持续 prefill 降频 ~12%,同热工况才可比。
- 大文件 push 慢(nsys rep 上百 MB),push 放后台跑。

## 7. 当前状态快照(2026-08-24,审计收尾批次)

- git HEAD:main 与 origin/main 同步(以 `git status -sb` / `git log -1` 实时核对为准;本批次 = 2026-08-24 审计收尾 commit)。
- 硬件占用:双卡正被另一实验占用——本仓一切 GPU 运行(bench/复测/profile)暂停。
- 下一步第一动作:用户本人执行 R0-6 线上简历排雷 + D2 PR review/`git commit -s`/提交(见 §5)。
