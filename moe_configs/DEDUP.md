# MoE config PR 查重记录 (2026-08-21)

目标: NVIDIA_GeForce_RTX_4090 的 Qwen1.5-MoE-A2.7B 配置
- E=30,N=1408 (TP2+EP, BF16) — 全库无任何 E=30 文件
- E=60,N=704  (TP2 非 EP, BF16) — 仅 MI300X 有

搜索(GitHub API, is:pr is:open):
- "fused_moe 4090"      → 1 条,无关(moe_wna16 bugfix #44563)
- "Qwen1.5-MoE"         → 6 条,全是 bugfix/XPU/ROCm,无 config 类
- "moe config RTX 4090" → 9 条,唯一相关: #48309

#48309 (2026-07-11, open): 给 RTX 4090**D** 加 E=8,N=3584/7168 fp8 配置。
不构成重复: ① 4090D 是独立 device_name,vLLM 按精确名查表,两者文件不通用
(该 PR 正文自己确认); ② shape 无交集(E=8 vs E=30/60); ③ dtype 不同(fp8 vs BF16)。
PR 正文将引用 #48309 作为"相邻先例",并说明以上区别。

结论: 无重复,可开工。
