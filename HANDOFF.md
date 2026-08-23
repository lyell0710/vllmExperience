# 交接文档（给接手的 agent）· 2026-08-22

> 本文件是**唯一入口**。读完这一篇即可无缝接手。
> 项目全貌读 [SUMMARY.md](SUMMARY.md)；状态速查读 [README.md](README.md) 证据台账；
> 过程叙事读 [LAB_JOURNAL.md](LAB_JOURNAL.md)；简历句读 [RESUME_EVIDENCE.md](RESUME_EVIDENCE.md)。

## 0. 这是什么项目

用户（李玉章，lyell0710@gmail.com）的 vLLM 秋招项目：在 **2×RTX 4090**（无 NVLink、
P2P 驱动禁用）上做推理部署选型 + MoE 性能优化，产出简历/面试素材。
**证据仓库** = `/root/projects/vllm/experiments/`（独立嵌套 git，远程
`github.com/lyell0710/vllmExperience` private，gh 已登录 lyell0710）。

## 1. 铁律（每次运作必守，已存入 Claude 长期记忆）

1. **每次运作结束在 `LAB_JOURNAL.md` 末尾追加一节**（同一文件顺写、时间正序，
   四问：做了什么/为什么/关键数字/产物路径 + 下一步）。
2. **每个实验一份 `records/EXP-NNN_<slug>.md`**（八节模板 `records/TEMPLATE.md`），
   实验结束当场写。
3. **任何 GPU 跑一律存 raw**（bench 加 `--save-result`）；没存的降级为"终端级证据"
   并在记录 §4 注明。
4. 结果文件首行 provenance；每个数据里程碑 **commit + push**（作者身份已配好，
   commit 尾注 `Co-Authored-By: Claude ...`）。
5. **措辞红线**（写简历/报告前查 README「措辞红线状态」表）：0.17 bug 只写
   "复现/定位/验证"禁"发现/修复"；PR 未提交不写"提交"；"KV 占 TTFT X%" 需
   EXT-1 才解锁（现仅可写 telemetry 原生量）。

## 2. 环境（都已就位，直接用）

| 代号 | venv | 版本 | 用途 |
|---|---|---|---|
| ENV-A | `/root/venvs/v0.17.1/bin/` | vLLM 0.17.1（PyPI） | 历史基线 / R0-4 复现 |
| ENV-B | `/root/venvs/v0.25.1/bin/` | vLLM 0.25.1（752a3a5044） | **主战场** |
| ENV-C | conda py312 + `/root/projects/vllm` editable | 0.26.1rc1.dev（main@7aa248fc） | MoE 开发 / PR |

- **禁止在 `/root/projects` 目录下裸跑 python**（vllm/ 目录名遮蔽 import）；
  bench 客户端用 ENV-B 的 `vllm bench serve` 即可。
- `uv` 在 `/root/.local/bin/uv`（装包：`uv pip install --python <venv>/bin/python <pkg>`；
  各 venv 无独立 pip）。
- **模型全在本地**（`~/.cache/huggingface`）：Qwen2-7B-Instruct、Qwen2.5-0.5B、
  Qwen1.5-MoE-A2.7B-Chat、Qwen3-30B-A3B-GPTQ-Int4。磁盘 350G 余 ~205G，够用。
- docker 在本机**跑不了容器**（无特权，平台限制）——别试。
- **杀进程用 `pkill -f '[v]llm serve'`**（方括号技巧，避免复合命令里误杀自身 shell；
  昨晚为此吃过两次 exit 144）。

## 3. 核心工装（`pd_disagg/scripts/`，全部可复现）

- `run_point.sh <arm> <mode> <in> <out> <rps|-> <bench_port> <engine_port...>`
  单测量点执行器（快照→bench→快照→collect_point→runs.jsonl）。
  - mode: `attribution`(并发1) / `saturation`(饱和探吞吐) / `sweep`(给 rps)
  - 环境变量：`MODEL`（多模型时**必须显式设**，否则 404）、`SEED`（协议 v2 每点唯一）、
    `NUM_PROMPTS`、`GPU_COUNT`、`SLO_TTFT_MS`/`SLO_TPOT_MS`（sweep 算 goodput）、
    `PROV_ENV_LABEL`/`PROV_SHA_OVR`（B3 跨版本标注）。
- `collect_point.py` 汇集成 runs.jsonl 一行（schema 见 `results/README.md`）。
  gate 判定：`is_pd = arm.startswith("pd1p1d")`。
- `metrics_snapshot.sh` /metrics 直抓+diff；`provenance.sh`；`profile_ctl.sh`；
  `make_figures.py`（六图一表，dataviz 规范，从 runs.jsonl 重算）。
- **SLO 已锁定**（`results/README.md`，不可回改）：TTFT ≤ 328/891/4626ms（512/2K/8K）
  + TPOT ≤ 50ms。

## 4. 已完成（✅ = 数据落盘 + 记录 + 已推送）

| 项 | 状态 | 关键产物 |
|---|---|---|
| R0-1 硬件三数 | ✅ | `hw/` · P2P=GNS禁用 / NCCL 1.78GB/s / NIXL 0.27GB/s |
| R0-2 三venv+provenance | ✅ | `scripts/provenance.sh` |
| R0-3 NIXL smoke+版本裁决 | ✅ | `smoke/` `DECISION.md`（锁 v0.25.1） |
| R0-5 profiling 工装 | ✅ | `profiling/` `scripts/profile_ctl.sh` |
| **B1 四臂矩阵** | ✅ | runs.jsonl(122行) + `figures/fig1-6` + `derived/` + EXP-004~007 |
| B2 归因层 | ◐ | PD TTFT 分解(传输54-64%) + `analysis/nixl_token_accounting.md`；**缺 EXT-1** |
| B3 版本对照 | ◐ | EXP-008(单实例：512桶饱和+45%)；**缺 PD-vs-PD** |
| **B4 报告** | ✅v1 | `pd_disagg/REPORT.md` |
| R0-4 降级分析 | ✅ | `analysis/p2pnccl_bugs_id_chain.md`(全 file:line) |
| C1 MoE 上卡 | ✅ | EXP-009 · A2.7B TP2+EP TPOT 4.62ms |
| C2 config 查重 | ✅ | `moe_configs/DEDUP.md`(三重闭环) |
| C3 W4A16 上卡 | ✅ | EXP-010 · Qwen3-30B-A3B-GPTQ-Int4 TPOT 4.93ms |
| **EXT-2 NixlPush** | ✅ | EXP-011 · 推方向 8K TTFT 2537 vs pull 2718；量级不变(方向救不了 PD) |
| 汇总+教学 | ✅ | SUMMARY.md · STUDY_GUIDE.md · 网页手册 artifact |

## 5. 未完成（按优先级，接手就干这些）

### ✅ P1 · R0-4 动态复现（2026-08-23 完成 → EXP-012）
两 bug 实机坐实：bug1 精确命中 `connector:433` AssertionError（手工 addr串id+max_tokens>1）；
bug2 D 整实例挂死（双请求 hang + 全线程 futex_wait + P /health 恒 200）。实证修正：裸直连 P
先崩于 `connector:518` parse_request_id，早于 :433。py-spy 因容器 ptrace 限制未取栈帧（诚实标注）。
raw 见 `pd_disagg/p2pnccl_repro/raw/EXP-012/`。**接续**：据此定 B3 完整版表述（0.17.1 PD 默认配置
正常请求即触发 D 挂死 = 不可用对照臂，vs 0.25.1 NIXL 可用），然后做下面 P2。

<details><summary>原 P1 说明（已完成，存档）</summary>
- **现状**：`pd_disagg/p2pnccl_repro/` 已备好 `launch_1p1d.sh`（0.17.1 P2pNccl 1P1D，
  Qwen2-7B 双卡，proxy http 10001 / zmq 30001，P:20003 D:20005）。昨晚起过一次，
  P/D 就绪、NCCL 握手成功（见 `repro_decode_tail.txt`），但一次经 proxy 的请求探测
  被环境中断（额度），**未得干净结论**。
- **要干**：`bash launch_1p1d.sh` → 等 P/D 就绪 → 经 proxy(10001) 发正常请求。
  **预期 bug 2**：D 端因 request_id 随机后缀分叉，在无超时 `Condition.wait`
  （p2p_nccl_engine.py:317）挂死 → 请求 hang。抓 D 端日志证实（找 recv_tensor 卡住
  / recv_store 里是 P 端 key）。
  **预期 bug 1**：直接打 P 端点(20003)发 `max_tokens>1` → `assert req_id in
  self.chunked_prefill`（p2p_nccl_connector.py:433）崩 EngineCore。
- **机理已静态定位**（`analysis/p2pnccl_bugs_id_chain.md`），动态复现是把"崩溃现场"
  截图/存日志坐实。产出 → EXP-012 + 解锁 B3 完整版(PD-vs-PD)。
- **注意**：quart 已装进 0.17.1 venv。proxy 会给两端传同一 request_id，
  分叉发生在各实例内部 InputProcessor（已确认默认 `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=False`）。
</details>

### ⬜ P2 · EXT-1 telemetry request 级关联（解锁最后一条红线，现为最高优先）
- 目标：给 NIXL telemetry 加 request 关联的**本地最小 patch**（ENV-C=main 上改，
  改前 `gh` 查重），解锁"D 等待远端 KV 对 TTFT 的关键路径贡献占比"。
- 解锁后可把 REPORT §2.2 的传输占比从"分量对账"升级为因果占比声明。

### ⬜ P3 · B4 v2 定稿（8/31 M1 截止）
- 吸收 EXT-2（§2.4 补推/拉对照，EXP-011 已有数据）、R0-4 动态复现结果；
- EXT-1 若做完则升级占比声明，否则注记其缺席。

### ⬜ P4 · 9 月 D 阶段（M2/M3）
- D1 nsys MoE 分解（baseline：EXP-009/010 已有）；D2 benchmark_moe.py 调优
  E=30,N=1408 的 4090 config + **六件套 PR**（C2 已确认社区空缺，切入点在案）；
  D3 kernel 优化；D4 FP8 vs W4A16（C3 已上卡）；D5 EPLB（默认不上简历）。

### 用户本人负责（agent 干不了）
- **R0-6**：线上简历稿把 vLLM 两 bug 的"发现/修复"改"复现/定位/验证"。
- 课程脚本：非必需（R0-4 已用 tag 里的官方脚本替代）。

## 6. 已知坑/偏差（别重复踩）
- tp2 起服务用 `--gpu-memory-utilization 0.88`（0.9 会 warmup OOM）。
- 前缀缓存污染：`vllm bench` 固定 seed 会命中缓存虚高数字；**协议 v2 每点唯一 seed**
  已解决，续跑务必带 `SEED=`。
- 功率帽：持续 prefill 降频（450W 帽，非热），TTFT +30%；同热工况才可比，
  run_point 已自动采 GPU 遥测入 runs.jsonl。
- ~3% 运行率的客户端 ServerDisconnected 瞬断：失败行保留(gate_pass=false)，同 seed 重跑。

## 7. 当前状态快照（2026-08-23 更新）
- 所有进程已停，双卡空闲。git 干净，最新 commit `28b4bd8`（已推送）。
- R0-4 动态复现（EXP-012）✅ 收官；py-spy 因容器 ptrace 限制装了但用不了（同 docker 平台限制）。
- 未完：P2 EXT-1（现最高优先）→ P3 B4 v2（8/31）→ P4 九月 D 阶段；B3 完整版表述据 EXP-012 定。
- 下一步第一动作：读 EXP-012 记录确认 B3 表述，然后开 P2（EXT-1，ENV-C=main 上改前先 gh 查重）。
