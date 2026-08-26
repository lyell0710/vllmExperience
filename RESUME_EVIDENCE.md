# 简历证据映射（最终写简历和面试用的唯一入口）

> 结构：**简历句（占位符版）→ 当前可填 → 缺口 → 支撑文件 → 面试防御**。
> 占位符 `[...]` 只能用 gate 通过的数据填；填入时在本文件登记来源文件。
> 措辞红线速查见 [LEDGER.md](LEDGER.md#措辞红线状态写简历报告前查此表)。

**项目标题**：vLLM 推理部署与 MoE 性能优化｜RTX 4090×2｜2026.08–2026.09

**时间语义(8/23 更新)**:9 月初投递版 = S1–S4 全部成稿(S3 按 PR 状态写
"材料备齐"或"已提交(附链接)");S5 维持不上简历;面试季随 PR review 进展升级 S3。

---

## S1 · 部署选型（四臂矩阵）——最能证明系统实验能力

> 在双 RTX 4090 消费级平台上评测 vLLM 四种部署方案：单卡混部、双副本、TP=2 与
> NIXL Prefill–Decode 分离；覆盖 512/2K/8K 输入及多档请求负载，以 TTFT、TPOT、
> SLO goodput 和 GPU-seconds/request 建立部署选型边界，在 `[场景]` 下 `[方案]`
> 相比 `[基线]` 实现 `[X%]` 的 `[指标]` 改善。

- **当前可填**：
  - "消费级/互联受限"定语：P2P 驱动级禁用（GNS）、单向 D2D 0.60–0.91 GB/s、
    NCCL bus bw 1.78 GB/s → `pd_disagg/hw/`
  - attribution 层四臂对比（并发 1，8K，协议 v2 同热工况）：TTFT
    925.2/902.6/881.3/2718.7ms（非 PD 三臂无实质差异，功率帽整平）、
    TPOT 16 vs tp2 9.3ms、GPU·s/req 2.94–9.59 → REPORT §2.2 / runs.jsonl v2 行
  - 归因子结论三条（可各自成半句）：① TP2 decode -42%（带宽分摊）但 prefill
    零加速（大消息 allreduce 撞 1.78GB/s collective 墙）；② NIXL KV 通路有效
    吞吐 0.26–0.27GB/s 恒定（telemetry-derived）（~16KB/descriptor 碎片化小拷贝）；③ 消费卡 450W
    功率帽使持续 prefill 降频 ~12%、TTFT +30%（SW Power Cap 遥测坐实）
- **✅ headline 数字已全部到位（8/21 夜 sweep 完成，60 个扫描点全 gate）**。
  **S1 成稿候选（数字已填，供压缩）**：
  > 在双 RTX 4090（无 NVLink、P2P 驱动级禁用）上系统评测 vLLM 四种部署形态
  > （单卡混部/双副本/TP=2/NIXL Prefill–Decode 分离）：512/2K/8K 输入 × 多档
  > 负载 60+ 测量点，以 TTFT/TPOT/SLO goodput/GPU·s-per-request 建立选型边界
  > ——双副本取得近线性 2× 吞吐扩展（2K 输入 7.00 vs 3.63 req/s）与全场最高
  > goodput；TP=2 的 decode 提速 42% 被 1.78GB/s allreduce 带宽墙抵消（吞吐仅
  > +13–19%）；以 request 级三段关联（同请求身份+同时钟域，~16 行本地可观测性
  > patch——非 connector 核心改造，如实定性）测得 KV 等待占 TTFT 54–64%（因果占比，闭环误差 p50 <0.1%（最差桶 0.084%）），定量证明
  > PD 分离在 0.27GB/s KV 有效吞吐（telemetry-derived）下全负载段不可取；
  > 推/拉两方向实测同贴互联墙（push 仅 -6.7% TTFT）。
- **支撑文件**：`pd_disagg/REPORT.md`（B4 v2 定稿，2026-08-23）、`figures/fig1–6`、
  `derived/sweep_summary.csv`、runs.jsonl（124 行，前 109 行为 8/21 B1 sweep
  快照）、EXP-007/013；
  面试叙事线：LAB_JOURNAL §9–§12（功率帽/污染修正/传输墙三个完整探案）
- **面试防御**："凭什么说传输真的发生了" → 每测量点 gate 字段
  （nixl bytes 增量、成功传输数=预期远端请求数、failed=0、expired=0、
  failure_policy=fail、/metrics 直抓引擎端口）随数据同行存于 runs.jsonl。
  "54–64% 怎么来的" → EXP-013《EXT-1 request 级 KV-wait 关联》三重互证：逐请求 bytes 和=Prometheus 分毫不差、
  kv_wait≈xferDuration（差 0.3–1.9ms）、六段分解闭环误差 p50 <0.1%（最差桶 0.084%）；观测无扰动
  （patch 前后 TTFT 噪声内）；36/36 请求身份双端匹配。

## S2 · 架构演化与 bug 链路——面试深挖主力，不做头号成果

> 复现并验证 vLLM v0.17 P2pNcclConnector 的跨实例请求标识缺陷，梳理
> Proxy→Endpoint→InputProcessor→Connector 的 ID 传播链；结合 P2pNccl 删除 PR
> 与 NIXL Pull 实现，说明新架构如何通过本地/远端身份拆分和显式映射避免
> rendezvous key 分叉。

- **当前可填**：演化链的代码级证据已钉行号（`EXPERIMENT_PLAN.md` 核验表
  #4 别名、#7 删除提交 5add018beb 属 v0.25.1 祖先、#8 0.17 的 request_id#layer
  key、#9 新版 remote_engine_id/remote_request_id/remote_block_ids 显式映射）；
  NIXL Pull 实测通路 → `smoke/`。
  另有版本演化实例：profiler 接口 env var → `--profiler-config.*` CLI
  （`profiling/r0_5_torch_profiler_check.txt` note）。
- **✅ 源码机理分析完成**（`analysis/p2pnccl_bugs_id_chain.md`，全 file:line
  双版本核对）：assert 崩溃点 connector:433 与崩溃链、随机后缀分叉点
  input_processor.py:212、PUT 模式 D 端无超时 Condition.wait 挂死
  （engine:317）+ 内存泄漏、GET 模式静默乱码、四层 ID 传播表、
  NIXL 三层身份拆分对照；文末含面试 2 分钟口径草稿。
  版本性能维度补充（EXP-008《B3 有限版本对照》，system-version comparison 非组件归因）：
  无负载 Δ<1%、512 桶饱和 +45%、启动 308→58s。
- **✅ 动态复现闭环（EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》，8/23）**：bug1 精确命中 connector:433 原生
  traceback（需地址串 id + max_tokens>1 两条件，实证修正静态分析——裸 id 先崩
  :518）；bug2 D 整实例挂死行为学+wchan 闭环；B3 完整版表述定稿
  （0.17.1 PD 默认配置不可用对照臂 vs 0.25.1 NIXL 可用）。
- **红线**：只写"复现/定位/验证/梳理"，禁"发现/修复/吃透"。

## S3 · MoE kernel 分解与上游贡献——上限最高，最后压轴

> 对 vLLM fused MoE 路径进行 kernel 级分解，为 Qwen1.5-MoE 在 RTX 4090 TP2+EP
> 场景调优 Triton block config；在 `[batch/shape]` 下将 kernel 延迟降低 `[X%]`、
> 端到端吞吐提高 `[Y%]`，通过 correctness、kernel A/B 与 serving benchmark 验证，
> 并向上游提交 PR `#[编号]`。（合并后升级为"已合入"。）

- **当前可填**："社区空缺"已解锁（三重闭环：本地判定 + 远端查重 + 运行时告警
  原文点名 E=30,N=1408 缺失，EXP-009《C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据》）；C1 上卡完成，**未调优基线在案**：
  TP2+EP TPOT 4.62ms（dense 7B TP2 的 2.0×）、饱和 11.50 req/s@512
  ——D2 调优 A/B 的 before 数字。
  **✅ D1 分解完成（EXP-014《D1 MoE decode 分解》，8/23）**：nsys node 级分解定位 fused_moe grouped
  GEMM 占 GPU 时间 **56.4%**（bs=32 serving batch）；MoE/dense decode 反转点
  2.03×（bs=1）→0.97×（bs=8）→0.82×（bs=128）——"为什么调这个 config"的
  数据答案 + 报告第一页图（d1_fig1_decode_scaling.png）。
- **✅ D2 完成（EXP-015《D2 MoE config 调优》，8/23）**，S3 成稿候选：
  > 以 nsys kernel 级分解定位 fused MoE grouped GEMM 占 serving batch GPU
  > 时间 56%，据此为社区空缺的 RTX 4090 BF16 config（E=30,N=1408 EP /
  > E=60,N=704 TP，本地+远端+运行时告警三重查重确认空缺）完成上游标准调优
  > （benchmark_moe.py，1920 配置×18 M 档×2 tuple，M 档数 8/24 勘正）：kernel 延迟两端改善
  > （decode M=1 -8.5%[EP 臂;非 EP -3.8%]，prefill M≥128 -3.3~-3.9%），
  > 经 correctness（120 passed）/kernel A/B/e2e bench 三级验证，PR 材料
  > 按仓库六件套标准备齐。（e2e TPOT +0.8~1.2% 仅作防御层数字——
  > 低于跨会话漂移，不进简历句，8/24 定档修正。）
  （PR 提交后把"材料备齐"升级为"已提交(附链接)";merge 后再升级。）
- **D3 判定**：依数据转结论句——中段 M 与默认启发式打平证明 config 即最优
  杠杆，不做无数据支撑的 kernel 改动（面试口径：知道什么时候不做优化）。
- **面试防御**：AGENTS.md 六件套（DCO/查重说明/AI 声明/测试命令+数据/e2e
  bench）；Triton 首跑 JIT 伪影的识别与 warmup 复测（EXP-015 §7）。

## S4 · 量化对比（✅ D4 完成 8/23，可上简历）

> 在 RTX 4090 上对比 `[具体FP8格式]` 与 `[具体W4A16 checkpoint/格式]`，量化吞吐、
> 显存与 PPL/任务精度变化，给出 Ada 平台在不同 batch 和上下文长度下的量化选型边界。

- **✅ 已完成（EXP-016《D4 FP8 vs W4A16 同卡对比》）**，成稿候选：
  > 在 2×RTX 4090 上对比 Qwen3-30B-A3B 官方 FP8（block FP8，Triton
  > block-scaled 路径）与 GPTQ-Int4（W4A16，Marlin）：W4A16 在 decode 全
  > regime 快 23–48%（TPOT 4.91 vs 7.10ms@bs1）且权重减半，FP8 仅在高并发
  > prefill 段 TTFT 反超（497 vs 613ms@c128）并以 wikitext PPL 占优 3.3%
  > 相对（同 31k 计分 token）；并从 vLLM backend 分派逻辑（capability 90/100
  > 快路径跳过 SM89）解释 Ada 为何只能走 Triton 而非 Hopper FP8 路径。
- **面试防御**：同 token 集 PPL 协议、regime 反转的机理（decode 带宽受限 vs
  prefill 计算受限 + Marlin 反量化开销）、oracle/fp8.py:103-122 行号。

## S5 · EPLB（✅ gate 已跑完 8/23，维持不上简历）

- **判定（EXP-017《D5 EPLB gate》）**：W4A16 兼容 gate 上游显式拒绝（`routed_experts.py:151`
  NotImplementedError，TODO 指认为工程缺口非根本不兼容）；FP8 臂 2 次真实
  重排 + balancedness 0.53–0.74 实测；一致性 gate FAIL 但经无-EPLB 对照组
  归因为重排引起的数值性分歧（浮点归约顺序）。按清单规则不上简历。
- **白板素材升级**：对照组方法学、量化×EPLB 支持矩阵 file:line、
  "bitwise 一致性不是 EP 布局变化下的合理判据"的反思。
- **潜在上游素材**（9 月可选池）：AutoGPTQMoEMethod 补 supports_eplb
  （上游 TODO 邀请参照 Fp8MoEMethod），做前查重。

---

## 面试"聊"的弹药库（不进简历正文）

| 话题 | 素材位置 |
|---|---|
| 硬件画像怎么测的、为何双向 22.7 单向 0.6 | `hw/p2p_bandwidth_latency.txt`（cudaMemcpyPeer 无 P2P 走分段中转 vs 双向流水）|
| smoke 三检查设计与版本裁决规则 | `DECISION.md` |
| 17 条代码级核验（行号可背） | `EXPERIMENT_PLAN.md` §0 |
| 为什么代理测不了 profile、怎么绕 | `scripts/profile_ctl.sh` 头注释 + 核验 #12 |
| provenance/gate 卫生体系 | `scripts/provenance.sh`、`results/README.md` |
| v1 调度链 / PD 请求流转 / MoE dispatch 三图 | P2 白板产出（每周补） |
