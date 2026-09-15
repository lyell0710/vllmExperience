# D2 PR 草稿(六件套对照)· 状态：已提交为 vllm-project/vllm#54372（2026-08-29），本文保留为提交前底稿

> **提交前 hardening 清单（2026-08-24 增补，做完再提）**： 1. 核对 JSON：18 个 M 档 + triton_version 元键与上游既有 config 惯例一致； 2. correctness 重跑并保留**完整**日志（现仅存 tail 3 行）； 3. kernel A/B 补 ≥3 轮交叉（不同次序）取 mean±std； 4. `git fetch upstream && rebase` 到最新 main 后重验（7aa248fc 已滞后）； 5. e2e 数字在 PR 里只作 supporting（+0.8~1.2% < 会话漂移）， headline 用 kernel A/B 两端（M=1 -8.5%/-3.8%，M≥128 -3.3~-3.9%）。 **分支已就绪**：`/root/projects/vllm` 的 `moe-config-4090-qwen15moe` 分支， 两个 JSON 已 `git add` 暂存。你 review 后执行： `cd /root/projects/vllm && git commit -s -m "<下方标题>"`（-s 生成你的 Signed-off-by），然后 fork/push/开 PR。**不要用 agent 身份提交。**

> 目标分支：vllm-project/vllm main；本地分支 `moe-config-4090-qwen15moe`。 **提交人必须是本人**（AGENTS.md：pure code-agent PR 不允许；需人工逐行 review + 亲自开 PR）。agent 只准备分支、证据与本草稿。

## 提交前复核(2026-08-26)

三项独立复核，结论：**测量口径未失效，可按现有数字提交**。

### 复核一：上游是否动过被测代码路径

基点 `7aa248fcfe` 至今，上游 main 领先 550 个提交，但：

| 路径 | 自 2026-08-21 起的提交数 | 含义 |
|---|---|---|
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | **0** | config 查表逻辑与 Triton kernel 本体未变，A/B 数字仍成立 |
| `vllm/model_executor/layers/fused_moe/configs/` | 1（L40S fp8，#53819） | 与本 PR 两个形状无交集 |

结论：**不需要 rebase 后重测**。本 PR 只新增两个 JSON 数据文件，rebase 本身不可能产生冲突；重建 CUDA 扩展做重测没有信息增益。

### 复核二：重复性（六件套 #6 复验）

| 检查 | 结果 |
|---|---|
| 上游有无任何 `E=30,*` config | 无 |
| 上游有无 `E=60,N=704` | 仅 `AMD_Instinct_MI300X`（不同设备，exact-match 查表不共用） |
| 上游已有的 4090 config | `E=64,N=640` 与 `E=8,N=3584`，均为 `dtype=fp8_w8a8`（不同 dtype，文件名不同） |

### 复核三：格式与先例

| 项 | 本 PR | 上游惯例 |
|---|---|---|
| M 档数 | 18 | 18（抽样 200 份中 162 份为 18 档） |
| 元键 | `triton_version` | 200 份中 50 份带此键 |
| 字段名 | `BLOCK_SIZE_{M,N,K}` / `GROUP_SIZE_M` / `num_warps` / `num_stages` | 一致 |
| 缩进 / 末尾换行 | 4 空格 / 有 | 与 `E=60,N=704,device_name=AMD_Instinct_MI300X.json` 一致 |

**先例**：#53819「[Kernel][Perf] Tune fused_moe FP8 config for Qwen3.5 on L40S (+7%)」于 2026-08-26 合入，**无前置 issue、直接 PR**，与本 PR 同类。标题格式可对齐该 PR。

### 复核四：correctness 证据链（已修）

原 `raw/EXP-015/correctness_pytest.log` 内容仅一行 `No module named pytest`（那次调用失败），真实结果只留了 3 行 tail。按 CORE 铁律 6「主张有据」，PR 正文断言的 `120 passed` 当时缺完整日志支撑。已重跑并保留**带 provenance 首行的全量日志**：`raw/EXP-015/hardening_20260826T1115/20260826T1115_correctness_pytest.log`。

结果 **1041 passed / 127 skipped / 0 failed（877.9s）**。注意与旧 tail 的「120 passed / 120 skipped（139.7s）」不矛盾：旧数字来自 `test_moe.py::test_fused_moe` 单个函数（240 例），本次跑的是整个文件减去 deepseek / fp8 / int8 / wna16 参数化（1169 例），是严格超集。PR 正文按本次口径写。

### 复核五：kernel A/B 轮次（已补）

原数字为单轮。已按 hardening 清单 #3 补 **3 轮交叉次序**（奇数轮 default 先、偶数轮 tuned 先，排除热漂移与次序效应），取 mean±std。脚本 `d2_hardening.sh`，解析 `d2_hardening_analyze.py`。

---

## PR 标题(候选)

`[Kernel] Add fused MoE Triton configs for Qwen1.5-MoE on RTX 4090 (E=30,N=1408 EP / E=60,N=704 TP)`

## PR body 骨架

### Purpose
Add tuned Triton fused-MoE block configs for NVIDIA GeForce RTX 4090 (BF16):
- `E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json`— Qwen1.5-MoE-A2.7B, TP=2 + expert parallel
- `E=60,N=704,device_name=NVIDIA_GeForce_RTX_4090.json`— Qwen1.5-MoE-A2.7B, TP=2

Currently both shapes fall back to the default config with the runtime warning `Using default MoE config. Performance might be sub-optimal!` (fused_moe.py).

### Not duplicating existing work(六件套 #6)
- Searched configs dir on main: no E=30 file exists for any device; E=60,N=704 exists only for AMD_Instinct_MI300X.（复核日期 2026-08-21，记录在案）
- PR search: closest is #48309 (RTX 4090**D**, E=8, fp8)— different device_name (exact-match lookup, files not shared), different shapes, different dtype.
- Issue search: no open request; runtime warning reproduced locally.

### Test Plan(六件套 #4)
1. Correctness： `pytest tests/kernels/moe/test_moe.py`（相关子集）on 4090.
2. Kernel A/B: `benchmark_moe.py`(non-tune)default vs tuned per batch size.
3. e2e serving: `vllm bench serve`(in128/out256 + in512/out128,attribution
   + saturation)main 默认 vs 加 config，同 seed。

### Test Result(2026-08-23 实测,2×RTX 4090,vLLM main@7aa248fc)

Kernel A/B（`benchmark_moe.py` 非 tune 模式，default config vs tuned JSON）:

| M | EP default→tuned (us) | Δ | 非 EP default→tuned (us) | Δ |
|---|---|---|---|---|
| 1 | 38.2→34.9 | **-8.5%** | 24.4→23.4 | **-3.8%** |
| 8 | 389.4→389.0 | ~0 | 250.9→252.7 | ~0 |
| 32 | 563.4→564.1 | ~0 | 506.1→507.2 | ~0 |
| 64 | 578.2→573.1 | -0.9% | 568.2→566.9 | ~0 |
| 128 | 609.8→585.9 | **-3.9%** | 601.8→579.4 | **-3.7%** |
| 256 | 621.4→597.9 | **-3.8%** | 609.2→589.0 | **-3.3%** |

（收益集中在 M=1 与 M≥128 两端；默认启发式在中段 M 已接近最优——如实陈述。）

e2e serving(`vllm bench serve`，Qwen1.5-MoE-A2.7B TP2+EP，in128/out256)： TPOT p50 一致改善 +1.1–1.2%(c1: 4.40→4.34ms；c32: 17.91→17.70； c128: 28.78→28.47)；吞吐与 TTFT 在会话噪声内持平（warmup 复测：c32 1596→1616 tok/s、c128 4233→4178，均噪声内；TPOT 终判 +0.8~1.2%）。注：首次带新 config 的流量会触发 Triton JIT 编译新 tile 形状（一次性， 秒级）——冷启 bench 若不预热会把首波 TTFT 计入编译时间。

correctness:`pytest tests/kernels/moe/test_moe.py::test_fused_moe` → 120 passed, 120 skipped, 0 failed (139.7s)

### AI assistance(六件套 #5)
AI assistance (Claude) was used to run the tuning harness, prepare benchmarks, and draft this description. I reviewed every changed line and ran the tests myself.（提交前由本人确认此句属实）

### 其余件套
- #1 DCO：`git commit -s`（Signed-off-by 必须是本人身份）
- #2 pre-commit：JSON 文件也要过（检查末尾换行/格式）
- #3 issue 先行：config 类先例多为直接 PR（如 #48309 无 issue）；沿例直接 PR， 正文引用运行时告警作为动机。

## 回填检查清单
- [x] configs_ep JSON 落盘 + 拷入 vllm/model_executor/layers/fused_moe/configs/
- [x] configs_noep JSON 同上
- [x] kernel A/B 数字（EXP-015《D2 MoE config 调优》 §5）
- [x] e2e 数字（EXP-015 §5）
- [x] correctness pytest 输出
- [x] 分支 commit（-s，本人身份）——由用户执行
- [x] 用户逐行 review 后自行 push + 开 PR
- [ ] 维护者加 `ready` label / 人类 review（待外部）
