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
**D1(EXP-014:MoE decode 反转点 2.03×→0.82×,fused_moe 56.4%@bs32)**。
记录 EXP-001~014。M1 提前达成;M2 已达成。

## 5. 进行中/未完成(按序接手)

1. **D2(EXP-015,后台跑批中)**:`d2_tune.sh`(EP E=30,N=1408 → 非 EP
   E=60,N=704)→ watcher 自动接 `d2_ab.sh`(kernel A/B + e2e A/B +
   correctness)。完成后:写 EXP-015、回填 `PR_DRAFT.md` 数字、在
   `/root/projects/vllm` 建分支 `moe-config-4090-qwen15moe` 放两个 JSON。
   **PR 由用户本人 review + `git commit -s` + 提交(AGENTS.md 禁纯 agent PR)**。
2. **D4(EXP-016)**:`d4_fp8_w4a16.sh`(FP8 上卡成败本身是数据;Ada SM89
   无 Hopper FP8 路径的落地解释写进记录)+ `d4_ppl.py` 两 checkpoint wikitext
   PPL。GPU 空了就跑。
3. **D5(EXP-017)**:`d5_eplb.sh`(窗口 50/间隔 100 逼真实重排;gate 三项,
   任一不过整条砍)。预研已做:qwen3_moe 支持 EPLB、qwen2_moe 不支持;
   证据锚点 `eplb_state.py:748 "Rearranging experts"`。
4. **D3**:目标已由 D1 锁定 = fused_moe 路径;等 D2 A/B 数据决定 config 之外
   还有没有 kernel 级机会(若 tuned config 已贴 roofline,D3 改为"以数据说明
   config 即最优杠杆")。
5. 收尾:EXP-015~017 记录 + README 台账 + WEEKLY + LAB_JOURNAL + commit/push;
   RESUME_EVIDENCE S3/S4/S5 按结果升级。

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

## 7. 当前状态快照(2026-08-23 ~09:00Z)

- 后台:d2 tune(EP 进行中,1.92k 配置×18 batch 档×2 tuple,数小时)→
  watcher 自动接 d2_ab;git push(EXP-014 大文件)后台中。
- 本地 commit 到 d5693e4(EXP-013/014 + B4 v2 + WEEKLY + whiteboard/3 图)。
- 下一步第一动作:等 d2_ab 出数 → EXP-015 + PR 分支;GPU 空隙插 D4/D5。
