# 简历证据映射（最终写简历和面试用的唯一入口）

> 结构：**简历句（占位符版）→ 当前可填 → 缺口 → 支撑文件 → 面试防御**。
> 占位符 `[...]` 只能用 gate 通过的数据填；填入时在本文件登记来源文件。
> 措辞红线速查见 [README.md](README.md#措辞红线状态写简历报告前查此表)。

**项目标题**：vLLM 推理部署与 MoE 性能优化｜RTX 4090×2｜2026.08–2026.09

**时间语义**：9 月初投递版 = S1/S2 完成时 + S3 进行时；面试季随 D3/D4/D5 逐句升级。

---

## S1 · 部署选型（四臂矩阵）——最能证明系统实验能力

> 在双 RTX 4090 消费级平台上评测 vLLM 四种部署方案：单卡混部、双副本、TP=2 与
> NIXL Prefill–Decode 分离；覆盖 512/2K/8K 输入及多档请求负载，以 TTFT、TPOT、
> SLO goodput 和 GPU-seconds/request 建立部署选型边界，在 `[场景]` 下 `[方案]`
> 相比 `[基线]` 实现 `[X%]` 的 `[指标]` 改善。

- **当前可填**：
  - "消费级/互联受限"定语：P2P 驱动级禁用（GNS）、单向 D2D 0.60–0.91 GB/s、
    NCCL bus bw 1.78 GB/s → `pd_disagg/hw/`
  - attribution 层四臂对比（并发 1，8K）：TTFT 925/715/694/2685ms、
    TPOT 16 vs tp2 9.3ms、GPU·s/req 2.95–9.24 → `results/b1_matrix/runs.jsonl`
  - 归因子结论三条（可各自成半句）：① TP2 decode -42%（带宽分摊）但 prefill
    零加速（大消息 allreduce 撞 1.78GB/s collective 墙）；② NIXL KV 通路有效
    吞吐 0.26–0.27GB/s 恒定（~16KB/descriptor 碎片化小拷贝）；③ 消费卡 450W
    功率帽使持续 prefill 降频 ~12%、TTFT +30%（SW Power Cap 遥测坐实）
- **缺口**：sweep（offered-load 扫描 + goodput）——headline 数字来源
- **支撑文件**：`hw/*`、`results/b1_matrix/runs.jsonl`（12 行 attribution）、
  `figures/`（待）、LAB_JOURNAL 8/21 发现①②（面试叙事线）
- **面试防御**："凭什么说传输真的发生了" → 每测量点 gate 字段
  （nixl bytes 增量、成功传输数=预期远端请求数、failed=0、expired=0、
  failure_policy=fail、/metrics 直抓引擎端口）随数据同行存于 runs.jsonl。

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
- **缺口**：0.17.1 崩溃现场复现（R0-4，等课程脚本；降级方案=源码机理分析）
- **红线**：只写"复现/定位/验证/梳理"，禁"发现/修复/吃透"。

## S3 · MoE kernel 分解与上游贡献——上限最高，最后压轴

> 对 vLLM fused MoE 路径进行 kernel 级分解，为 Qwen1.5-MoE 在 RTX 4090 TP2+EP
> 场景调优 Triton block config；在 `[batch/shape]` 下将 kernel 延迟降低 `[X%]`、
> 端到端吞吐提高 `[Y%]`，通过 correctness、kernel A/B 与 serving benchmark 验证，
> 并向上游提交 PR `#[编号]`。（合并后升级为"已合入"。）

- **当前可填**：目标 tuple 本地缺失判定（E=30,N=1408 / E=60,N=704）→
  `moe_configs/DEDUP.md`。**"社区空缺"措辞在远端查重完成前禁用。**
- **缺口**：C1/C3 上卡、D1 分解、D2 调优+PR、（可选）D3 kernel 优化
- **面试防御**：AGENTS.md 六件套（DCO/查重说明/AI 声明/测试命令+数据/e2e bench）。

## S4 · 量化对比（可选句，D4 做完才上）

> 在 RTX 4090 上对比 `[具体FP8格式]` 与 `[具体W4A16 checkpoint/格式]`，量化吞吐、
> 显存与 PPL/任务精度变化，给出 Ada 平台在不同 batch 和上下文长度下的量化选型边界。

- **前置**：C3 锁定具体 checkpoint + 量化格式（AWQ/GPTQ/AutoRound 不许混称）。

## S5 · EPLB（默认不上简历）

- D5 全部 gate（真实重排 + 一致性 + W4A16 兼容）通过才升格为一个短句；
  否则仅作面试白板素材。

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
