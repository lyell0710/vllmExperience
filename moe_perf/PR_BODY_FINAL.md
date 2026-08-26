# 可直接粘贴的 PR 正文（2026-08-26 定稿）

提交步骤见本文件末尾。**Signed-off-by 与「Create pull request」必须由本人执行**——vLLM 的 AGENTS.md 不接受纯 agent 提交的 PR。

---

## 标题

```
[Kernel] Add fused MoE Triton configs for Qwen1.5-MoE on RTX 4090 (E=30,N=1408 EP / E=60,N=704 TP)
```

## 正文（以下整段粘进 PR description）

```markdown
## Purpose

Add tuned Triton fused-MoE block configs for NVIDIA GeForce RTX 4090 (BF16), for Qwen1.5-MoE-A2.7B:

- `E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json` — TP=2 + expert parallel
- `E=60,N=704,device_name=NVIDIA_GeForce_RTX_4090.json` — TP=2

Both shapes currently fall back to the default heuristic, emitting the runtime warning
`Using default MoE config. Performance might be sub-optimal!` (`fused_moe.py`).

## Not duplicating existing work

- No `E=30,*` config exists for any device on main.
- `E=60,N=704` exists only for `AMD_Instinct_MI300X`; config lookup is exact-match on `device_name`, so the file is not shared.
- The two existing RTX 4090 configs (`E=64,N=640`, `E=8,N=3584`) are `dtype=fp8_w8a8` — different dtype, different filenames.
- Closest open PR: #48309 targets RTX 4090**D**, E=8, fp8 — different device, shape and dtype.
- No open issue requests these shapes; the runtime warning is reproduced locally.

## Test Plan

1. Correctness: `pytest tests/kernels/moe/test_moe.py` (subset, excluding deepseek / fp8 / int8 / wna16).
2. Kernel A/B: `benchmark_moe.py` (non-tune mode), default heuristic vs tuned JSON, per batch size.
3. e2e serving: `vllm bench serve`, same seed, with and without the configs (supporting evidence only).

## Test Result

Hardware: 2×NVIDIA GeForce RTX 4090 (no NVLink, P2P disabled at driver level). vLLM main @ `7aa248fc`.

**Kernel A/B — 3 rounds with alternating arm order** (odd rounds run default first, even rounds tuned first, to cancel thermal drift and ordering effects). Values are mean±std over 3 rounds, in microseconds.

E=30,N=1408 (TP=2 + EP):

| M | default | tuned | Δ |
|---|---|---|---|
| 1 | 38.1±0.0 | 35.0±0.2 | **−8.2%** |
| 8 | 389.8±0.5 | 389.5±0.3 | −0.1% |
| 32 | 563.2±0.2 | 564.0±0.0 | +0.1% |
| 64 | 578.1±0.1 | 572.9±0.0 | −0.9% |
| 128 | 609.6±0.2 | 585.8±0.1 | **−3.9%** |
| 256 | 621.4±0.1 | 597.6±0.2 | **−3.8%** |

E=60,N=704 (TP=2):

| M | default | tuned | Δ |
|---|---|---|---|
| 1 | 24.4±0.1 | 23.5±0.1 | **−3.6%** |
| 8 | 250.7±0.3 | 252.4±0.5 | +0.7% |
| 32 | 506.1±0.1 | 507.5±0.1 | +0.3% |
| 64 | 568.4±0.1 | 566.8±0.1 | −0.3% |
| 128 | 601.8±0.1 | 579.4±0.0 | **−3.7%** |
| 256 | 609.4±0.1 | 588.8±0.1 | **−3.4%** |

**Gains are concentrated at the two ends (M=1 decode and M≥128 batched prefill); in the mid range the default heuristic is already near-optimal and the tuned config is on par.** Reporting this honestly rather than quoting only the best bucket — the round-to-round std (≤0.5 us) confirms the mid-range nulls are real, not noise.

Correctness: `pytest tests/kernels/moe/test_moe.py -k "not deepseek and not fp8 and not int8 and not wna16"` → **1041 passed, 127 skipped, 0 failed** (877.9s). This covers the whole `test_moe.py` file minus the deepseek / fp8 / int8 / wna16 parametrizations, i.e. a strict superset of the `test_fused_moe`-only run.

e2e serving (supporting only): TPOT p50 improves consistently by +0.8–1.2% across concurrency 1/32/128; throughput and TTFT stay within session noise, so no throughput claim is made. Note that the first request after adding a new config triggers a one-off Triton JIT compile for the new tile shapes — a cold benchmark without warm-up will attribute that compile time to the first TTFT samples.

## AI assistance

AI assistance (Claude) was used to run the tuning harness, prepare benchmarks, and draft this description. I reviewed every changed line and ran the tests myself.
```

---

## 提交步骤（你本人执行）

```bash
cd /root/projects/vllm
git status                     # 应只有两个 JSON 处于 staged 状态

# 1. DCO 签名提交（-s 生成你的 Signed-off-by，这是法律声明，必须你本人做）
git commit -s -m "[Kernel] Add fused MoE Triton configs for Qwen1.5-MoE on RTX 4090 (E=30,N=1408 EP / E=60,N=704 TP)"

# 2. 推到你自己的 fork（remote 已配好，名为 myfork）
git push myfork moe-config-4090-qwen15moe

# 3. 开 PR：浏览器打开下面的链接，标题与正文照上面粘贴
#    https://github.com/lyell0710/vllm/compare/main...moe-config-4090-qwen15moe
```

提交前逐项确认：

- [ ] 两个 JSON 的内容你已逐行看过（各 18 个 M 档，字段与上游一致）
- [ ] `AI assistance` 那句属实——你确实逐行 review 过、测试确实跑过
- [ ] Signed-off-by 是你本人的姓名与邮箱
- [ ] pre-commit 通过（JSON 末尾换行、格式）

## 证据位置

| 内容 | 路径 |
|---|---|
| 3 轮交叉 A/B 原始日志（12 份，各带 provenance 首行） | `experiments/moe_perf/raw/EXP-015/hardening_20260826T1115/` |
| correctness 全量日志 | 同上目录 `*_correctness_pytest.log` |
| 重测脚本 / 解析脚本 | `experiments/moe_perf/d2_hardening.sh` / `d2_hardening_analyze.py` |
| 实验记录 | `experiments/records/EXP-015_d2_moe_config_tuning.md` |
| 复核过程（上游漂移 / 重复性 / 格式先例） | `experiments/moe_perf/PR_DRAFT.md` §提交前复核 |
