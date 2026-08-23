# EXP-015 · D2 MoE config 调优:4090 BF16 两个社区空缺 tuple + 六件套验证

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23(10:16–15:30Z) |
| 环境 | ENV-C(main@7aa248fc, ray 双卡);A/B 与 e2e 同 ENV-C;Qwen1.5-MoE-A2.7B-Chat |
| 状态 | 完成(PR 提交留待用户本人) |
| 关联清单项 | D2;P1 六件套;S3 简历句;M3 |

## 1. 目的与假设
用上游标准工具(`benchmark_moe.py --tune`)为 C2 判定的两个社区空缺 tuple
产出 4090 BF16 config JSON,并按 AGENTS.md 标准完成 correctness + kernel A/B +
e2e bench 三级验证。假设:D1 已证 fused_moe 占 serving batch GPU 时间 56.4%,
config 调优应有可测收益。

## 2. 环境与配置
- 调优:`d2_tune.sh`——EP(E=30, shard N=2816→文件名 N=1408)与非 EP
  (E=60, N=704),搜索空间 1920 配置 × 19 个 M 档,ray 双卡分摊。
- A/B:`d2_ab.sh`——次序 default kernel → default e2e → 装 JSON → tuned
  kernel → tuned e2e → correctness;e2e 为 TP2+EP serving,c1/c32/c128。
- 补测:`d2_e2e_rerun.sh`——tuned e2e 带 warmup 复测(见 §7 JIT 伪影)。

## 3. 步骤
tune EP(8916s)→ tune 非 EP(4097s)→ kernel/e2e A/B → correctness
(main venv 补 pytest/tblib)→ warm 复测。全程一条自驱动链(gpu_chain5.sh)。

## 4. 原始数据
`moe_perf/raw/EXP-015/`:configs_{ep,noep}/ 两个 JSON(19 M 档全网格)、
tune_{ep,noep}.log、kernel_{ep,noep}_{default,tuned}.log、
e2e_{default,tuned}_c{1,32,128}.json+log、e2e_tuned_warm_c{32,128}.json、
correctness_tail.txt、manifest(provenance)。

## 5. 结果
**交付物**:`E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json` 与
`E=60,N=704,device_name=NVIDIA_GeForce_RTX_4090.json`(上游 exact-match
文件名,已进 PR 分支 `moe-config-4090-qwen15moe` 暂存区)。

**Kernel A/B**(us,default→tuned):

| M | EP | Δ | 非EP | Δ |
|---|---|---|---|---|
| 1 | 38.2→34.9 | **-8.5%** | 24.4→23.4 | **-3.8%** |
| 8 | 389.4→389.0 | ~0 | 250.9→252.7 | ~0 |
| 32 | 563.4→564.1 | ~0 | 506.1→507.2 | ~0 |
| 64 | 578.2→573.1 | -0.9% | 568.2→566.9 | ~0 |
| 128 | 609.8→585.9 | **-3.9%** | 601.8→579.4 | **-3.7%** |
| 256 | 621.4→597.9 | **-3.8%** | 609.2→589.0 | **-3.3%** |

**e2e A/B**(TP2+EP,in128/out256,TPOT p50):c1 4.40→4.34ms、
c32 17.91→17.70、c128 28.78→28.47——**一致 +1.1~1.2%**,与
"kernel 增益 × fused_moe 时间占比"的折算吻合(D1:56.4%@bs32)。
吞吐/TTFT 在会话噪声内持平(D1 同点参照:c32 1691、c128 3975 tok/s vs
本轮 default 1596/4233——会话间 ±5~8% 漂移,大于调优效应)。
warm 复测(warmup 后,c32 JIT 伪影消除:TTFT 1021→225ms):
c32 吞吐 1616 vs default 1596(+1.2%)、TPOT 17.76 vs 17.91;
c128 吞吐 4178 vs 4233(-1.3%,噪声内)、TPOT 28.52 vs 28.78。
**终判:TPOT +0.8~1.2% 一致成立;吞吐/TTFT 噪声内持平。**

**correctness**:`pytest tests/kernels/moe/test_moe.py::test_fused_moe` → **120 passed, 120 skipped, 0 failed**(139.7s,GPU0;skipped 为异平台/异 dtype 参数化)

## 6. 分析与结论
- 调优收益的真实形状:**两端显著(M=1 decode -8.5%/-3.8%;M≥128 -3.3~-3.9%),
  中段(M=8–64)与默认启发式打平**——不是"全面大胜",是对空缺 tuple 的
  规范补全 + 两端改善;简历/PR 措辞按此如实表述。
- e2e 放大验证:TPOT +1.1~1.2% ≈ kernel 增益 × 占比,机理自洽;
  serving 吞吐层面的会话噪声(功率帽/热态)大于该效应,不作吞吐声明。
- D3 判定(依数据):tuned config 与 default 在中段 M 打平说明 Triton tile
  空间在该形状已被启发式覆盖;**config 之外的 kernel 级优化空间有限**,
  D3 按计划改为"以数据说明 config 即最优杠杆"结论句,不另做 kernel 改动。

## 7. 异常、偏差与开放问题
- **Triton 首跑 JIT 伪影**:装入新 config 后首个 c32 bench 的 TTFT p50
  1021ms(default 176ms)、时长 +2s——新 tile 形状首次被流量命中触发现场
  编译,32 路并发同时阻塞;c128(后跑)TTFT 正常(363 vs 359ms)佐证一次性。
  处理:warmup 后复测(数字见 §5),PR 正文注明该现象。
- correctness 首跑两次失败:main venv 缺 pytest/tblib(依赖补装后通过收集
  1168 用例;正式跑 `::test_fused_moe` 核心网格)。
- e2e 未采 GPU 遥测(d2_ab 疏漏)——吞吐对比的热工况不可证,故吞吐结论
  仅"噪声内持平"不作方向声明(TPOT p50 对热态不敏感,保留)。
- 大 M(512–4096)kernel A/B 未单测(A/B 网格取 1–256);tuned JSON 含
  全 19 档,大 M 档的收益由 tune 内部测量支撑,独立复测留给上游 CI。

## 8. 下游影响
- **PR 六件套齐备**(PR_DRAFT.md 数字已回填):分支 + JSON 暂存 + 查重说明
  (含 #48309 相邻先例)+ 测试命令与数据 + AI 声明 + DCO 指引。
  **提交动作留给用户本人**(AGENTS.md:agent 不得代提)。
- S3 简历句可写:"为社区空缺的 E=30,N=1408 / E=60,N=704 调优 4090 BF16
  config(kernel 两端 -3.3~-8.5%,e2e TPOT +1.1~1.2%),按仓库标准完成
  correctness/kernel A/B/e2e 三级验证"。
- M3(config PR)达成至"材料齐备待提交";D3 转结论句。
