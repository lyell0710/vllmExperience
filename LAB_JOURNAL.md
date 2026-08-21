# 实验日记（Lab Journal）

> 规则：每个工作段落结束追加一节，按时间正序。每节固定四问：
> **做了什么 / 为什么（决策依据）/ 关键数字 / 产物路径**，末尾记下一步。
> 写简历时以本文件（过程与叙事）+ RESUME_EVIDENCE.md（句子装配）+
> README.md 台账（状态速查）三件套为准。数字一律以 provenance 文件为最终依据。

---

## 2026-08-21 · Day 0：地基全部落定 + B1 开跑

### 上午：三 venv + NIXL smoke + 版本裁决
- **做了什么**：搭 `~/venvs/{v0.17.1, v0.25.1, main}`；按官方 nixl_connector_usage
  同机示例跑 1P1D smoke（GPU0=P:5600 / GPU1=D:5601 / toy proxy），v0.25.1 与
  main 双版本各跑一遍；按预定规则裁决主战场版本。
- **为什么**：清单 R0-2/R0-3；裁决规则预先定死（三项检查、平手取 release）防拍脑袋。
- **关键数字**：双版本 3/3 PASS；v0.25.1 avg xfer 14.128ms / 0.188MB / 13.27MB/s，
  P90 xfer 23.7ms，descriptors=24；ERROR=0。
- **产物**：`pd_disagg/smoke/`、`pd_disagg/DECISION.md`（锁定 **v0.25.1=ENV-B 主战场**）。

### 下午 1：开发环境修复（会话另线完成，记录关键差异点）
- main 仓库（/root/projects/vllm@7aa248fc）切为 editable 安装 = ENV-C；
  注意事项：precompiled 模式下改 csrc/ 不生效；`/root/projects` 下裸跑 python
  会被目录名遮蔽 import。

### 下午 2：R0-1 硬件三数（简历"P2P 受限"定语的实测支撑）
- **做了什么**：cuda-samples p2pBandwidthLatencyTest + nccl-tests all_reduce_perf
  （-g 2，NCCL 2.19.7）+ 拓扑记录（`topo -m` 因容器 hwloc 限制不可用，
  改用 `topo -p2p r` + PCIe link）。
- **关键数字**：
  - P2P：connectivity=0，`topo -p2p r`=**GNS（驱动禁用）**
  - 单向 D2D **0.60–0.91 GB/s**（无 P2P 时 cudaMemcpyPeer 分段中转）；
    双向 **22.6–22.8 GB/s**；GPU 间延迟 14.5–15.9µs；本卡内 ~924 GB/s
  - NCCL all_reduce avg bus bw **1.78 GB/s**（大消息 1.85）——仅代表 TP collective 路径
  - 含义预估：Qwen2-7B 8K 输入 KV≈460MB，穿卡传输在秒级量级；TP=2 将重度受
    allreduce 约束 → 四臂矩阵预期区分度极大
- **产物**：`pd_disagg/hw/*.txt`（均带 provenance），DECISION.md 硬件基线节回填。

### 下午 3：R0-5 profiling 工装（发现一个版本演化点）
- **做了什么**：0.5B 起真实引擎验证 torch profiler 直控（start→请求→stop→trace
  落盘）；nsys 容器内冒烟通过。
- **发现（B3 素材）**：v0.25.1 **弃用 `VLLM_TORCH_PROFILER_DIR` 环境变量**
  （日志报 Unknown env var、端点 404），改为
  `--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=...`；
  0.17.1 仍是环境变量。代理不转发 profile 端点 → 一律直控引擎端口。
- **产物**：`pd_disagg/profiling/r0_5_torch_profiler_check.txt`、`traces_smoke/`、
  `scripts/profile_ctl.sh`。

### 下午 4：证据仓库定型 + GitHub 私有备份
- experiments/ 原被外层 exclude 且无版本控制 → 建独立嵌套 git 仓库，
  推送 `github.com/lyell0710/vllmExperience`（private）。
- 四层结构：README（约定+台账+红线）/ RESUME_EVIDENCE（句子装配）/
  results/README（B1 schema）/ scripts（provenance、metrics_snapshot、run_point、
  collect_point）。

### 下午 5：C2 远端查重收尾（"社区空缺"红线解锁）
- **做了什么**：gh 直连复核——上游 main configs 目录无任何 E=30 文件、
  E=60,N=704 仅 MI300X；全状态 PR/issue 搜索无 config 类冲突；
  相邻先例 #48309（4090D fp8）仍 OPEN。
- **结论**：目标 tuple 空缺确认，无重复，**"社区空缺"措辞解锁**（引用本节日期）。
- **产物**：`moe_configs/DEDUP.md` 远端复核节。

### 下午 6：SLO 方案锁定 + B1 colocate 归因跑（3/3 gate PASS）
- **做了什么**：SLO 采用 DistServe 式相对定义（TTFT≤5×无负载基线、TPOT≤50ms
  固定、附录做 SLO-scale 敏感性曲线）；colocate 臂（GPU0 单卡，
  `--max-model-len 16384` 其余默认）跑 512/2048/8192×128 并发=1 归因，
  全链路（快照→bench→快照→runs.jsonl）首跑验证通过。
- **关键数字**（p50，32 请求/点，seed=42，ignore-eos）：

  | 输入桶 | TTFT p50 (ms) | TPOT p50 (ms) | → TTFT SLO (5×) |
  |---|---|---|---|
  | 512  | 65.52  | 15.87 | 328 |
  | 2048 | 178.28 | 15.93 | 891 |
  | 8192 | 925.18 | 16.34 | 4626 |

  解读：bs=1 时 TPOT 恒定 ~16ms（≈63 tok/s，decode 带宽约束），TTFT 随输入
  长度近线性——prefill 计算主导，与 GDDR6X 预期一致。
- **产物**：`results/b1_matrix/runs.jsonl`（前 3 行）、`raw/`、`snapshots/`；
  SLO 表锁进 `results/README.md`。

### 下一步（8/22）
1. replica2 / tp2 / pd1p1d 三臂 attribution（PD 臂首跑：人工核对 NIXL 指标名
   → 固化 collect_point 的 gate 判定；顺手拿 8K 大传输的 NIXL 实测补硬件画像）。
2. sweep 网格设计（每桶 rps 档位由 attribution 吞吐推算），跑 colocate sweep。
3. 待办：R0-4 课程脚本（用户提供）、R0-6 线上稿（用户）。

## 2026-08-21 · Day 0（续）：四臂 attribution 全部完成 + 两个一手发现

### 晚 1：replica2 / tp2 / pd1p1d 三臂归因（12/12 gate 全 PASS）
- **做了什么**：三臂各跑 512/2048/8192×128、并发 1、32 请求/点。replica2 =
  双 TP1 副本 + rr_proxy(8300)；tp2 = TP=2 单实例；pd1p1d = NixlConnector
  P(GPU0:8100,side 5600)+D(GPU1:8200,side 5601)+toy proxy(8192)，
  **未用 enforce-eager（CUDA graphs 正常）**，failure_policy=fail。
- **四臂汇总（p50）**：

  | arm | TTFT@512 | @2048 | @8192 | TPOT | GPU·s/req@8K |
  |---|---|---|---|---|---|
  | colocate | 65.5 | 178.3 | 925.2* | 15.9–16.3 | 2.95 |
  | replica2 | 65.2 | 173.5 | 714.6 | 15.9–16.4 | 5.58 |
  | tp2      | 62.8 | 173.5 | 693.7 | **9.3–9.5** | 3.79 |
  | pd1p1d   | 214.4 | 554.6 | 2685.4 | 15.9–16.4 | 9.24 |

  *colocate@8K 是冷→稳态混合（见发现①），稳态约 905ms。

### 发现①：功率帽节流（先见异常 → 分布拆解 → 机理坐实）
- 异常：replica2@8K(715ms) 反而快过 colocate(925ms)，并发 1 下不应如此。
- 拆解：colocate 32 请求 TTFT 分布双段（前 ~8 个 702–739ms，其后 897–951ms）；
  replica2 全部 697–733ms。
- 机理（nvidia-smi 采样坐实）：持续 8K prefill 下 GPU 功率 427–443W 顶 450W 帽，
  SM 频率 2820→2460–2535MHz，节流原因位 **0x4 = SW Power Cap**（温度仅 63°C，
  非热因）。replica2 轮转 = 每卡 50% 占空比 → 维持 boost。
- **方法论决定**：attribution 数字代表各臂占空比下的真实工况，headline 以
  sweep（满负载，各臂同为持续态）为准；工装已加 GPU 遥测
  （run_point.sh 2s 采样 → runs.jsonl `gpu_telemetry`），此后每点自带工况证据。
  SLO 表维持已锁值（5× 余量远大于 30% 效应）。

### 发现②：TP2 的不对称收益 —— 硬件三数的因果闭环
- decode TPOT **16→9.3ms（-42%）**：bs=1 decode 是权重带宽约束，每卡半份权重；
  小消息 allreduce 走延迟路径，代价 ~1.3ms/token。
- 8K prefill **零加速**（694 vs 冷态单卡 ~700ms）：prefill allreduce 是大消息
  （28 层 × 58.7MB），正好受 1.78GB/s collective 带宽约束 → 计算减半被通信吃掉。

### 晚 2：PD 指标探针 → gate 判定固化 → NIXL 大传输实测
- 探针（单请求 before/after diff）确认 v0.25.1 指标体系：传输计数全在 **D 端**
  （Pull 语义）——`nixl_bytes_transferred_sum/count`、`nixl_xfer_time_seconds_sum`、
  `nixl_post_time_seconds_sum`、`nixl_num_descriptors_sum`；P 端仅 failed/expired；
  `_created` 是时间戳需排除；bonus:
  `prompt_tokens_by_source_total{source="external_kv_transfer"}` 逐 token 记账远端 KV。
- collect_point.py gate 判定换成精确指标名（跨端口求和），字段扩展
  （xfer/post 时间、descriptors、external_kv_tokens、failed_notifications）。
- **NIXL 大传输实测**（R0-1 收尾）：**0.26–0.27 GB/s 恒定**
  （29.4/88.1/439.7 MB/xfer；8K avg xfer 1602.7ms；descriptor ~16KB/个 =
  每 block 每层单发 → 碎片化小拷贝）。PD TTFT 各分量对账：
  2685 ≈ P prefill(~900) + xfer(1603) + 首步/代理。只可称
  telemetry-derived effective throughput。
- **开放问题（B2 归因）**：D 端实拉 7668 token/req（<8192），bytes 与
  external_kv_tokens 两计数器独立互证——疑与 block/前缀缓存记账相关，待查。

### 下一步（8/22）
1. sweep 网格设计与试跑（rps 档位由 attribution 吞吐推算；先 colocate 臂）。
2. B2：xfer 时间直方图桶分析 + "7668 token"记账问题溯源（读 D 端调度代码）。
3. 考虑给 attribution 加标准化冷却协议后补一轮（报告用哪版由 sweep 结果决定）。
