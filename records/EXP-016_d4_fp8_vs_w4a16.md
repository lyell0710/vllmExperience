# EXP-016 · D4 FP8 vs W4A16 同卡对比(Qwen3-30B-A3B,Ada SM89)

> **一句话结论**：同卡对比给出分工结论：**decode 全 regime W4A16 胜（23–48%）**，因为 decode 受权重带宽约束、4-bit 读取量减半；**TTFT 在高并发反转**（c128：FP8 497 vs 613ms），因为 prefill 受计算约束时 Marlin 要先反量化回 BF16。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23（09:27–10:20Z） |
| 环境 | ENV-B（752a3a5044, vllm 0.25.1）；TP2+EP；2×RTX 4090 |
| 状态 | 完成 |
| 关联清单项 | D4；S4 简历句解锁；C3 后续 |

## 1. 目的与假设
同模型（Qwen3-30B-A3B，激活 3B）两个官方量化 checkpoint 的吞吐-精度对比 + "Ada 为何走不了 Hopper FP8 路径"的落地解释。格式锁定（C3 红线：不混称）：
- **FP8**:`Qwen/Qwen3-30B-A3B-FP8`(fine-grained block FP8,31G)
- **W4A16**:`Qwen/Qwen3-30B-A3B-GPTQ-Int4`（GPTQ,16G,Marlin 路径=EXP-010《C3 Qwen3-30B-A3B W4A16 上卡》确认）

## 2. 环境与配置
`d4_fp8_w4a16.sh`：两臂同参数（TP2+EP、max-model-len 8192、util 0.88、每点唯一 seed、--save-result）；负载 attr512(512/128,conc1)+ c1/c32/c128(128/256)。PPL：`d4_ppl.py`——wikitext-2-raw test，窗 2048/步 1536（前 512 token 只作条件不计分），两臂**完全相同的计分 token 集**（同 Qwen3 tokenizer,31212 token），vLLM offline prompt_logprobs=1。

## 3. 步骤
fp8 臂 4 点 → w4a16 臂 4 点（首跑被脚本 10 分钟健康检查窗误杀，见 §7）→ PPL fp8 → PPL w4a16（首跑 OOM，见 §7）。

## 4. 原始数据
`moe_perf/raw/EXP-016/`:8 个 bench JSON+log、两臂 server.log、GPU 遥测 CSV、 ppl_{fp8,w4a16}.{json,log}、kernel path 摘录、manifest(provenance)。

## 5. 结果
**吞吐/延迟（p50）**：

| 点 | FP8 tok/s | W4A16 tok/s | Δ | FP8 TPOT | W4A16 TPOT | FP8 TTFT | W4A16 TTFT |
|---|---|---|---|---|---|---|---|
| attr512(conc1) | 132.6 | 186.4 | **+41%** | 7.10 | **4.91** | 57.3 | 58.3 |
| c1 | 139.3 | 201.3 | **+44%** | 7.12 | **4.91** | 21.2 | 16.6 |
| c32 | 1455.3 | 2160.0 | **+48%** | 19.93 | **13.77** | 258.3 | 241.1 |
| c128 | 3574.0 | 4393.9 | **+23%** | 33.81 | **26.76** | **497.1** | 612.8 |

**精度（wikitext-2 PPL，同 31212 计分 token）**：FP8 **7.663** vs W4A16 **7.922**（FP8 优 3.3% 相对）。权重体积：31G vs 16G。

**kernel 路径（SM89 实选，日志原文在案）**：
- FP8:linear = `TritonFp8BlockScaledMMKernel`;MoE = `Using TRITON Fp8 MoE backend out of potential backends: ['AITER','FLASHINFER_TRTLLM', 'FLASHINFER_CUTLASS','DEEPGEMM','TRITON','MARLIN',...]`； 另 `symm_mem.py:66 Device capability 8.9 not supported`。
- W4A16：GPTQ-Int4 → `MarlinLinearKernel` + `'MARLIN' WNA16 MoE backend`（本实验 ppl_w4a16.log 原文，已补录 w4a16_kernel_path.txt；serve 臂 server.log 因超时误杀重启仅含幸存进程输出——8/23 审计勘正；与 EXP-010 一致）。

## 6. 分析与结论
- **decode 全 regime W4A16 胜（23–48%）**：decode 是权重带宽受限（D1/EXP-014《D1 MoE decode 分解》机理），4-bit 权重读取量是 8-bit 的一半——GDDR6X 上直接换算成吞吐；30B-A3B 激活 3B 的小激活形态放大了权重读取占比，收益比 dense 更陡。
- **TTFT 在高并发反转（c128：FP8 497 vs 613ms）**：prefill 计算受限， FP8 的 Triton block-scaled GEMM 用 FP8 张量核吞吐，而 Marlin 需先反量化到 BF16 再算，大 M 下反量化开销显形——量化选型按 regime 分化，不存在全域胜者。
- **Ada 落地解释（file：line 级）**：vLLM 的 FP8 MoE backend 提升逻辑（`fused_moe/oracle/fp8.py:103-122`）只对 `capability_family(100)`(Blackwell， FLASHINFER_TRTLLM)与 `capability(90)`（Hopper，FLASHINFER_CUTLASS/TRITON 优先序）做快路径提升；SM89 两者都不命中 → 落到通用优先序里首个支持 SM89 的 TRITON。DEEPGEMM/FLASHINFER CUTLASS 内核依赖 SM90+ 的 TMA/WGMMA； Ada 的 FP8 张量核只能经 Triton/内联 PTX 路径使用——"能用 FP8，但用不上 Hopper 的高效路径"。
- **选型结论（消费级 Ada 双卡）**：decode 主导的 serving 选 **W4A16(GPTQ+ Marlin)**——快 23–48%、权重省一半（KV 余量更大）、代价 PPL +3.4% 相对； FP8 仅在 prefill 重/长输入 TTFT 敏感场景与精度敏感场景占优。

## 7. 异常、偏差与开放问题
- **w4a16 臂首跑被误杀**：GPTQ-Int4 加载 ~11 分钟，脚本健康检查窗仅 10 分钟， server 实际已 init 完成（log 在案）却被判 FAIL 杀掉——**超时窗必须按最慢臂设置**（已修为 18 分钟），错误结论"启动失败"未进任何记录。
- **PPL 首跑 OOM**：窗 3584 的 prompt_logprobs 需一次性 all_gather 全词表 logits(3584×151936×2B≈1.09GB)，30B 模型 util 0.88 下仅剩 1GB → 改窗 2048/步 1536 + util 0.80 通过。教训：prompt_logprobs 的显存峰值 ∝ 窗长×词表。
- **进程管理事故（方法学，不影响数据）**：旧 GPU 链因 wrapper cmdline 含 heredoc 文本被自家 `pkill -f` 连坐杀死后，其子进程成为孤儿继续跑，与新链短暂并发（D5 服务器 vs PPL 加载）——按 PID 精确清理后重跑；此期间无数据落盘， 两臂 bench 与 PPL 数据均产生于无争用窗口（GPU 遥测 CSV 可证）。
- PPL 绝对值不与文献可比（窗/步/语料拼接协议自定义）；仅用于两臂相对比较。
- 未扫长输入桶（2K/8K）的 TTFT 边界；当前结论限 128/512 输入，标注适用范围。

## 8. 下游影响
- S4 简历句解锁（RESUME_EVIDENCE 待升级）：具体格式已锁定，数字齐备。
- D5(EPLB × W4A16)接续用 GPTQ-Int4 checkpoint（链条已自动开跑）。
- 面试弹药：oracle/fp8.py 的 capability 分派逻辑 + TTFT 反转的 regime 分析。
