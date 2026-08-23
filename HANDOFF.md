# 交接文档(给接手的 agent)· 2026-08-23(第二次全面更新)

> 本文件是**唯一入口**。项目全貌读 [SUMMARY.md](SUMMARY.md)(注:其"还欠的事"
> 已过时,以本文 §5 为准);状态速查读 [README.md](README.md) 证据台账;
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
5. 措辞红线查 README 表。**"KV 占 TTFT X%" 已解锁(EXP-013)**;0.17 两 bug
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

## 4. 已完成(✅ = 数据落盘 + 记录 + 推送;详见 README 台账)

R0-1~R0-5、B1、**B2(EXT-1 收官,KV 占 TTFT 54.2/62.5/64.2%)**、B3(两维度
定稿)、**B4 报告 v2 定稿**、C1/C2/C3、EXT-2、R0-4 动态复现(EXP-012)、
**D1(EXP-014:MoE decode 反转点 2.03×→0.82×,fused_moe 56.4%@bs32)**、
**D4(EXP-016:W4A16 decode 胜 23–48%,FP8 PPL 优 3.3%,Ada 路径 file:line)**、
**D5(EXP-017:gate 判定完成,不上简历,白板素材)**、交付物对抗校验(20 条
修复,commit cc473de)。记录 EXP-001~014+016+017。M1/M2 均提前达成。

## 5. 进行中/未完成(按序接手)

1. **D2(EXP-015,链 5 最终长跑中,任务 b6l7lff67)**:`gpu_chain5.sh` =
   d2_tune(EP E=30,N=1408 → 非 EP E=60,N=704,全量 18 batch 档,~10–14h)
   → 自动接 `d2_ab.sh`(kernel A/B + e2e A/B + correctness)。完成后:
   写 EXP-015、回填 `PR_DRAFT.md` 数字、在 `/root/projects/vllm` 建分支
   `moe-config-4090-qwen15moe` 放两个 JSON(d2_ab 的 phase3 已把 JSON 拷进
   configs/,分支化即可)。**PR 由用户本人 review + `git commit -s` + 提交**。
2. **D3**:等 D2 A/B 数据定——若 tuned config 已贴 roofline,D3 改为
   "以数据说明 config 即最优杠杆";否则按 A/B 差距找 kernel 级机会。
3. 收尾:EXP-015 记录 + 台账/WEEKLY/日记 + commit/push;可选:更新两个
   artifact(凡跑必录 / 四臂实验手册)收录 8/23 全部新结果。
4. 可选上游素材(9 月池):AutoGPTQMoEMethod 补 supports_eplb(EXP-017 §8,
   上游 TODO 邀请,做前查重)。

### 用户本人负责(agent 干不了)
- **R0-6**:线上简历稿"发现/修复"→"复现/定位/验证"(从 8/21 挂起至今)。
- **D2 PR 的最终提交**(review 每一行 + DCO 签名 + 开 PR)。

## 6. 已知坑(别重复踩)

- nsys 看 vLLM decode 必须 `--cuda-graph-trace=node`(EXP-014 §7)。
- 多模型 bench:`MODEL` 必须显式设,否则 404。
- tp2 gpu-memory-utilization 0.88(0.9 warmup OOM)。
- 前缀缓存污染:每点唯一 seed。
- 功率帽:持续 prefill 降频 ~12%,同热工况才可比。
- 大文件 push 慢(nsys rep 上百 MB),push 放后台跑。

## 7. 当前状态快照(2026-08-23 ~10:45Z)

- 后台:**链 5**(任务 b6l7lff67)= d2 全量调优(EP→非EP)→ d2_ab,不再
  抢占;Monitor bdl4ab1ps 盯阶段标记。git push 慢爬中(大文件 ~13KB/s,
  落后若干 commit,本地为锚,勿并发第二个 push)。
- 本地 commit 到 07c8c94(EXP-016/017 + 对抗校验修复 + 台账全同步)。
- 下一步第一动作:收 STAGE_D2_TUNE_DONE / STAGE_D2_AB_DONE → 读
  raw/EXP-015(configs_{ep,noep} JSON + kernel_*.log + e2e_*.json +
  correctness_pytest.log)→ 写 EXP-015 → PR 分支 + PR_DRAFT 回填。
