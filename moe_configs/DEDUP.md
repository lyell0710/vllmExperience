# MoE config PR 查重记录 (2026-08-21)

目标：NVIDIA_GeForce_RTX_4090 的 Qwen1.5-MoE-A2.7B 配置
- E=30，N=1408 (TP2+EP， BF16)— 全库无任何 E=30 文件
- E=60,N=704（TP2 非 EP, BF16）— 仅 MI300X 有

搜索（GitHub API, is:pr is:open）:
- "fused_moe 4090"      → 1 条，无关（moe_wna16 bugfix #44563）
- "Qwen1.5-MoE"         → 6 条，全是 bugfix/XPU/ROCm，无 config 类
- "moe config RTX 4090" → 9 条，唯一相关： #48309

#48309 (2026-07-11, open): 给 RTX 4090**D** 加 E=8,N=3584/7168 fp8 配置。
不构成重复：① 4090D 是独立 device_name，vLLM 按精确名查表，两者文件不通用（该 PR 正文自己确认）；② shape 无交集（E=8 vs E=30/60）；③ dtype 不同（fp8 vs BF16）。PR 正文将引用 #48309 作为「相邻先例」，并说明以上区别。

结论：无重复，可开工。

## 远端复核（2026-08-21 ~15:45Z，gh 直连，最终）

- 上游 main configs 目录：**无任何 E=30 文件**；E=60，N=704 仅 `AMD_Instinct_MI300X` → 4090 目标 tuple 空缺仍成立
- PR 全状态搜索（"E=30 N=1408" / "E=60 N=704"）：命中均为无关项（#52651 GPTQ bugfix / #24700 默认 config 分析（CLOSED） / #41834 DeepSeek SM12x）
- issue 搜索 "Qwen1.5-MoE 4090 config"：仅 #15561（旧的模型加载提问）
- #48309（4090D fp8 config）仍 **OPEN 未合并**，继续作相邻先例引用

**最终结论：无重复，"社区空缺"措辞解锁（引用本节日期）。**

## SGLang 侧同构空缺(2026-08-24 侦察,未动工)

- SGLang 全库 363 个 fused MoE config（按 Triton 版本分目录），`NVIDIA_GeForce_RTX_4090` 仅 2 个旧 fp8 文件（与 vLLM 同源搬运）；**E=30，N=1408 与 E=60，N=704 全 Triton 版本目录均缺失**。
- 远端查重（gh api，三组关键词）：无 NVIDIA 4090 BF16 MoE config 类 PR/issue（仅 AMD 消费卡请求 #30245/#30599，不冲突）。
- 判定：**第二 PR 机会开放**。其 fused_moe_triton 与 vLLM 同源，EXP-015《D2 MoE config 调优》已调优的两个 JSON 大概率直接可用（须在 sglang 运行时 A/B 验证后再提）。源码已 clone 至 /root/repos/sglang（shallow）；sglang venv 未安装（用户暂停，待指示）。

## 复验（rebase 后，2026-08-29，针对 cacc429f62）

本地 fork rebase 到 upstream `cacc429f62`（08-28 快照）后，对「社区空缺」判定复验。

方法：`git ls-tree -r --name-only cacc429f62 vllm/model_executor/layers/fused_moe/configs/` 精确查上游目录（排除本分支已暂存的 2 个 JSON 干扰）。

| 主张 | 08-21 判定 | cacc429f62 复验 |
|---|---|---|
| 全库无 `E=30,*` | 成立 | **仍成立**（上游 configs 无任何 E=30 文件） |
| `E=60,N=704` 仅 MI300X | 成立 | **仍成立**（上游仅 `AMD_Instinct_MI300X`，无 4090） |
| 4090 现有 config 均为 fp8 | E=64,N=640、E=8,N=3584 | 同上，无新增 4090 BF16 |

附带核查（PR_DRAFT 复核一曾断言「fused_moe.py 0 提交」）：自 `7aa248fcfe` 起 676 提交内 `fused_moe.py` 有 6 行插入，位于 `invoke_fused_moe_triton_kernel`——`A_scale` 为 0-D tensor 时 reshape 成 1-D（量化路径 bugfix）。仅在 `A_scale is not None` 生效，**不触及 BF16 config 查表与 kernel 路径**，故「rebase 后无需重测」结论仍成立。

**复验结论：社区空缺主张继续成立，PR 无需改动。**

## 复验（2026-08-30，针对 fe755c8899）

rebase 到 upstream `fe755c8899`（08-30 拉取的最新 HEAD）后，第三次复验「社区空缺」判定。

方法：`git ls-tree -r --name-only upstream/main vllm/model_executor/layers/fused_moe/configs/` 精确查上游目录（共 332 个文件）。

| 主张 | 08-21 判定 | fe755c8899 复验 |
|---|---|---|
| 全库无 `E=30,*` | 成立 | **仍成立**（上游 configs 无任何 E=30 文件） |
| `E=60,N=704` 仅 MI300X | 成立 | **仍成立**（上游仅 `AMD_Instinct_MI300X`，无 4090） |
| 4090 现有 config 均为 fp8 | E=64,N=640、E=8,N=3584 | 仍仅这 2 个 fp8（`E=64,N=640` / `E=8,N=3584`，均 fp8_w8a8），无新增 4090 BF16 |

**复验结论：社区空缺主张继续成立，PR 以「新增」提交，无需改动。**
