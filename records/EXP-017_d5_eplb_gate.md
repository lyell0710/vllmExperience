# EXP-017 · D5 EPLB gate(W4A16 不支持 / FP8 真实重排 + 对照组归因)

> **一句话结论**：gate 判定：**D5 不上简历**（W4A16 不支持 EPLB，FP8 下重排真实发生但 2-rank 收益空间受限）。负结果照登，方法学与支持矩阵的 file：line 证据完整保留。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23（09:58–10:40Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；Qwen3-30B-A3B（GPTQ-Int4 / FP8）；TP2+EP+EPLB |
| 状态 | 完成（gate 判定齐；按清单 D5 不上简历，白板级保留） |
| 关联清单项 | D5；S5（维持默认不上简历） |

## 1. 目的与假设
逼出一次**真实**专家重排（默认数千 step 才触发，"能启动"不算过）+ 重排前后输出一致性 + W4A16 权重搬运兼容性。EPLB 配置：`window_size=50, step_interval=100, log_balancedness=true`（默认 1000/3000 调小 20-30×）。

## 2. 环境与配置
- 服务：TP2+EP+`--enable-eplb`,max-model-len 4096,util 0.88。
- 探针：8 条固定 prompt，greedy（temperature=0，seed=7，max_tokens=48），重排前后各一轮；中间 256 请求负载（128/128，conc16）推进 engine steps。
- **对照组**(`d5_control.sh`)：同 FP8 服务器同负载，**EPLB 关闭**，同探针。

## 3. 步骤
w4a16 臂 → 失败取证 → fp8 臂（gate 1/2）→ 对照组（gate 2 归因）。

## 4. 原始数据
`moe_perf/raw/EXP-017/` 按臂列（8/23 审计修正，此前「各臂」统述不准确）：
- `fp8/`（全套 7 类）：server.log（含 2852 条 balancedness 逐 step 记录）、probe_{before，after}.txt、probe_diff.txt、rearrange_evidence.txt（注：摘录仅含 profile 行，两次真实重排行在 server.log 10:09:38/10:10:19）、load.log、manifest。
- `w4a16/`（启动即抛 NotImplementedError，无探针阶段）：server.log、manifest。
- `control_noeplb/`（probe 逐字节一致故无 diff；关 EPLB 故无重排证据）：server.log、probe_{before，after}.txt、load.log、manifest。

## 5. 结果
**GATE3（W4A16 兼容性）= 上游显式不支持**：
```
NotImplementedError: EPLB is not supported AutoGPTQMoEMethod.
```
抛点 `fused_moe/routed_experts.py:139-152`（`quant_method.supports_eplb` 门）， TODO 注释明言"其他量化方法无本质差异，参照 Fp8MoEMethod 实现"——**工程缺口而非根本不兼容**（潜在上游贡献点，做前需查重）。（注：该臂两次运行同点同错；首次曾被进程清理竞态误判，复跑排除干扰后同错。）

**GATE1（真实重排）= PASS（FP8 臂）**：`eplb_state.py:748` 原文—— 1 次 profile 重排（启动，0.15s）+ **2 次负载触发的真实重排**(10:09:38、 10:10:19)；逐 step balancedness 实测 0.53–0.74（avg/max tokens per expert 在案）——重排确有事可做，非空转。

**GATE2（输出一致性）= FAIL，且因果归属 EPLB（对照组）**：
| 组 | EPLB | 负载前后输出 |
|---|---|---|
| fp8 臂 | 开（2 次真实重排） | **分歧**（token 级，双侧均连贯，无损坏签名） |
| 对照组 | 关 | **逐字节一致** |

对照组证明本栈在该探针协议下确定（负载/批处理不引入分歧）→ fp8 臂的分歧 **由 EPLB 重排引起**；定性与"专家重摆改变浮点归约顺序 → greedy 在临界 token 翻转"一致（diff 全文均为合理延续文本）。

## 6. 分析与结论
- 按清单规则（任一 gate 不过 → 整条砍掉）：**D5 不上简历**（S5 默认维持）。
- 白板/面试素材反而完整：①真实重排 + balancedness 数据；②对照组方法学（先证明测量协议在无处理组时稳定，再归因）；③量化×EPLB 支持矩阵的 file：line（GPTQ 拒于 routed_experts.py：151，FP8 过 supports_eplb 门， eplb 通信组建立与 EPLB rank 分配日志在案）；④"输出一致性"作为 EPLB gate 的判据反思——EP 布局变化天然破坏 bitwise 复现，更合理的判据是 logprob 漂移幅度或专家权重校验和。
- 2-rank 下 EPLB 的适用性：重排能触发、能执行、开销可测（profile 0.15s），但收益空间受限（64 专家/rank，线性放置）——与「小 rank 数适用边界」预期一致。

## 7. 异常、偏差与开放问题
- 首次 w4a16 臂失败曾与我方进程清理竞态重叠（两日志字节数相同、同 traceback， 复跑排除干扰后确认同因）——事故经过与排除法记录于 LAB_JOURNAL §19。
- **开放问题**：分歧的"数值性 vs 权重搬运错误"区分未做到权重级（校验和比对/逐层 logprob 漂移）；当前"良性"定性基于文本连贯性 + 对照组 + balancedness 正常，非权重级证明。
- 真实重排的耗时行未在日志出现（仅 profile 有 "in 0.15 s"），async 模式下完成日志路径不同，未深挖。

## 8. 下游影响
- S5 维持不上简历；RESUME_EVIDENCE S5 更新为"gate 已跑完，白板素材"。
- 潜在上游素材：AutoGPTQMoEMethod 的 supports_eplb 扩展（参照 Fp8MoEMethod， 上游 TODO 邀请）；列入 9 月可选池，做前 gh 查重。
