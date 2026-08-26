# 白板图 3 · MoE dispatch → GEMM → combine(vLLM fused_moe,TP2+EP)

> P2 持续项。以 Qwen1.5-MoE-A2.7B（E=60 专家，top-4，moe_intermediate N=1408） 在本机 TP2+EP 的真实形状标注；D1（EXP-014《D1 MoE decode 分解》）nsys 的 kernel 归类即按此图分段。

## 图(白板版,单层 MoE block 的一个 decode step)

```
hidden_states [T, 2048]           T = batch 内 token 数
   │
   ├─ gate(router)Linear [2048, 60] → logits
   │     fused_topk → topk_ids [T,4], topk_weights [T,4]
   ▼
【dispatch 段】
   moe_align_block_size:把 (token,expert) 对按 expert 分桶、
   pad 到 BLOCK_SIZE_M 对齐 → sorted_token_ids, expert_ids
   (EP 时:本 rank 只处理落在本地 30 个专家上的 token)
   ▼
【grouped GEMM 段】(fused_moe.py triton kernel × 2)
   GEMM1: x @ w13[30, 2×1408, 2048] → SiLU(a1)*a3  → [*, 1408]
   GEMM2: h @ w2 [30, 2048, 1408]                  → [*, 2048]
   (config JSON 按 (E, N, dtype, M) 选 tile:E=30,N=1408 ← 本机缺失=D2 目标)
   ▼
【combine 段】
   按 topk_weights 加权散射回 token 顺序(moe_sum / index_add)
   ▼
   TP/EP 归并通信:all_reduce(TP2 下两 rank 各算一半专家的部分和)
   [T, 2048] → 下一层
```

## 两种并行形态(C2 的 tuple 由来,白板必画)

| 形态 | 每 rank 专家 | 每专家 N | config tuple | 通信 |
|---|---|---|---|---|
| TP2+EP(`--enable-expert-parallel`) | 30（按专家切） | 1408（完整） | **E=30，N=1408** | 部分和 all_reduce |
| TP2 非 EP | 60（全量） | 704（按 N 切） | E=60,N=704 | 标准 TP all_reduce |

两个 tuple 的 4090 BF16 config JSON 上游均缺失（C2 三重闭环）→ 运行时 `WARNING fused_moe.py:1106 Using default MoE config`（EXP-009《C1 Qwen1.5-MoE-A2.7B 上卡（TP2+EP）+ C2 运行时证据》原文在案）。

## 为什么 decode 快、饱和不快(EXP-009 数字的机理)

- bs=1 decode：激活参数 2.7B ≪ dense 7B → 权重读取量小 → TPOT 4.62 vs 9.26ms(2.0×)。GDDR6X ~1008GB/s 是这条路的物理上限（D1 理论线）。
- 饱和（大 batch）：11.50 vs 12.31 req/s——dispatch/align/散射的固定开销
  + 专家负载不均 + 通信，把权重优势吃回去（D1 nsys 分解就是量化这一句）。

## MoE 概念三段式

- **专家路由的执行方式**：朴素实现=每专家一次小 GEMM（kernel 启动 60 次）； vLLM=moe_align 分桶后一个 grouped GEMM kernel 吃掉全部专家，tile 内按 expert_ids 换权重指针；为什么：decode 的 T 小，60 次 launch + 小矩阵打不满 SM，分桶合并是唯一能贴近 roofline 的形状。
- **EP vs TP 切法**：TP 切 N（每专家半个）= 每 token 两 rank 都算、通信是标准 allreduce；EP 切专家 = token 只去本地专家、部分和归并；为什么：EP 让单专家 GEMM 形状完整（N=1408 非 704），tile 效率高，代价是负载不均衡暴露（→ D5 EPLB 的存在理由）。
- **config JSON**：Triton kernel 的 tile(BLOCK_M/N/K， num_warps...)按（E，N，dtype，M） 查表；缺表 = 默认参数，官方注释"Performance might be sub-optimal"；为什么：MoE 的 M（每专家实际 token 数）随 batch/路由漂移， 静态启发式很难对所有形状最优 → 上游用 benchmark_moe.py 离线调优落 JSON。
