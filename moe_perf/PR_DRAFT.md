# D2 PR 草稿(六件套对照)· 状态:待调优数字回填

> 目标分支:vllm-project/vllm main;本地分支 `moe-config-4090-qwen15moe`。
> **提交人必须是本人**(AGENTS.md:pure code-agent PR 不允许;需人工逐行
> review + 亲自开 PR)。agent 只准备分支、证据与本草稿。

## PR 标题(候选)

`[Kernel] Add fused MoE Triton configs for Qwen1.5-MoE on RTX 4090 (E=30,N=1408 EP / E=60,N=704 TP)`

## PR body 骨架

### Purpose
Add tuned Triton fused-MoE block configs for NVIDIA GeForce RTX 4090 (BF16):
- `E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json` — Qwen1.5-MoE-A2.7B, TP=2 + expert parallel
- `E=60,N=704,device_name=NVIDIA_GeForce_RTX_4090.json` — Qwen1.5-MoE-A2.7B, TP=2

Currently both shapes fall back to the default config with the runtime warning
`Using default MoE config. Performance might be sub-optimal!` (fused_moe.py).

### Not duplicating existing work(六件套 #6)
- Searched configs dir on main: no E=30 file exists for any device; E=60,N=704
  exists only for AMD_Instinct_MI300X.(复核日期 2026-08-21,记录在案)
- PR search: closest is #48309 (RTX 4090**D**, E=8, fp8) — different device_name
  (exact-match lookup, files not shared), different shapes, different dtype.
- Issue search: no open request; runtime warning reproduced locally.

### Test Plan(六件套 #4)
1. Correctness: `pytest tests/kernels/moe/test_moe.py`(相关子集)on 4090.
2. Kernel A/B: `benchmark_moe.py`(non-tune)default vs tuned per batch size.
3. e2e serving: `vllm bench serve`(in128/out256 + in512/out128,attribution
   + saturation)main 默认 vs 加 config,同 seed。

### Test Result(待回填)
- Kernel A/B 表:[TUNE 后回填:各 batch 档 us 对比 + 提升 %]
- e2e:[回填:TPOT/吞吐 before→after]
- correctness:[回填:pytest 输出]

### AI assistance(六件套 #5)
AI assistance (Claude) was used to run the tuning harness, prepare benchmarks,
and draft this description. I reviewed every changed line and ran the tests
myself.(提交前由本人确认此句属实)

### 其余件套
- #1 DCO:`git commit -s`(Signed-off-by 必须是本人身份)
- #2 pre-commit:JSON 文件也要过(检查末尾换行/格式)
- #3 issue 先行:config 类先例多为直接 PR(如 #48309 无 issue);沿例直接 PR,
  正文引用运行时告警作为动机。

## 回填检查清单
- [ ] configs_ep JSON 落盘 + 拷入 vllm/model_executor/layers/fused_moe/configs/
- [ ] configs_noep JSON 同上
- [ ] kernel A/B 数字(EXP-015 §5)
- [ ] e2e 数字(EXP-015 §5)
- [ ] correctness pytest 输出
- [ ] 分支 commit(-s,本人身份)——由用户执行
- [ ] 用户逐行 review 后自行 push + 开 PR
