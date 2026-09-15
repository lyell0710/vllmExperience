# EXP-022 · D2 大 M（512–4096）kernel A/B：tuned config 在 prefill 级 M 是否保持收益（补 EXP-015 §7 缺口）

> **一句话结论**：**大 M 档 tuned config 不仅保持、而且放大了收益**——EP 臂 M=512/1024/2048/4096 为 −6.4/−14.0/−9.9/−6.8%，非 EP 臂 −2.8/−6.0/−11.8/−10.3%，8/8 档显著（|Δ| ≫ 2×合并 std，3 轮交叉次序 std ≤1.4%）。EXP-015 §6「两端显著、中段打平」的收益形状在 prefill 级 M 延伸为「M≥512 全段显著，1024–2048 峰值 −10~−14%」，PR #54372 的大 M 档由此有了独立复测支撑。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | ENV-C（`/root/venvs/main`，vllm 0.26.1rc1.dev682+g7aa248fcf；editable 仓 `/root/projects/vllm` 在分支 `moe-config-4090-qwen15moe` @ 3805e40e17 = PR #54372 提交态，configs/ 内含两份 4090 JSON）；Qwen1.5-MoE-A2.7B-Chat；2×RTX 4090（ray 双卡分摊 M 档） |
| 状态 | 完成（首次尝试 ray.init 超时作废，重跑完成，见 §7） |
| 关联清单项 | D2 / EXP-015《D2 MoE config 调优》§7「大 M(512–4096) kernel A/B 未单测」；PR #54372 支撑数据 |

## 1. 目的与假设

EXP-015 的 kernel A/B 网格只到 M=256，大 M 档（512–4096，prefill 级）的收益只有 tune 内部测量支撑。本实验用同一工具、同一 3 轮交叉次序协议，对 M ∈ {512, 1024, 2048, 4096} 做 default vs tuned 的独立复测，EP 与非 EP 两臂。

**假设（可证伪）**：大 M 档 tuned 保持 −3% 量级收益（EXP-015 M=128/256 的 −3.3~−3.9% 的延伸）。

**跑前锁定的判定阈值（跑完不改）**：
- 每 (臂, M) 各 3 轮，报 mean±std（样本 std）；Δ = (tuned − default)/default。
- 合并 std = sqrt(std_default² + std_tuned²)；**|tuned_mean − default_mean| > 2 × 合并 std 才算显著**。
- 显著且 Δ ≤ −2% → 「保持收益」；显著且 −2% < Δ < 0 → 「收益缩水」；不显著 → 「打平」；显著且 Δ > 0 → 「回退（tuned 更慢）」。
- 假设成立的标准：EP 与非 EP 两臂各 4 个 M 档中 ≥6/8 为「保持收益」；否则不成立，按实际形状改写 EXP-015 §6 的"收益形状"结论。
- 交叉次序：奇数轮 default 先、偶数轮 tuned 先（抵消热漂移）；每臂装载后双重断言 config 来源（文件存在性 + 日志 `Using default MoE config` / `Using configuration from`），断言失败即中止、该臂作废。

## 2. 环境与配置

- 命令（每轮每臂）：`CUDA_VISIBLE_DEVICES=0,1 /root/venvs/main/bin/python /root/projects/vllm/benchmarks/kernels/benchmark_moe.py --model Qwen/Qwen1.5-MoE-A2.7B-Chat -tp 2 [--enable-expert-parallel] --seed 0 --batch-size 512 1024 2048 4096`（与 EXP-015 §5.1 hardening 完全一致，仅 BS_LIST 换成大 M）。
- 臂切换：default 臂把 `E=30,N=1408,…4090.json` 与 `E=60,N=704,…4090.json` 移出 `configs/` 到 `moe_perf/raw/EXP-015/.held`，tuned 臂移回；trap 保证退出恢复 tuned 就位（仓库树不留脏）。
- 硬件占用：双卡（benchmark_moe.py 的 ray worker 每卡一个，M 档轮转分配——与 EXP-015 同）。跑前 `nvidia-smi --query-compute-apps` 确认空闲。
- 脚本：`moe_perf/d2_bigM_ab.sh`（由 `d2_hardening.sh` 派生，去掉 correctness 段）；解析 `moe_perf/d2_bigM_analyze.py`（在 `d2_hardening_analyze.py` 基础上加显著性判定与 CSV 落盘，取不到 `Kernel time` 即报错）。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
STAMP_OVERRIDE=<STAMP> bash moe_perf/d2_bigM_ab.sh 1   # 轮 1（default 先）
STAMP_OVERRIDE=<STAMP> bash moe_perf/d2_bigM_ab.sh 2   # 轮 2（tuned 先）
STAMP_OVERRIDE=<STAMP> bash moe_perf/d2_bigM_ab.sh 3   # 轮 3（default 先）
python3 moe_perf/d2_bigM_analyze.py moe_perf/raw/EXP-022/bigM_<STAMP>   # mean±std + 判定 → moe_perf/derived/<STAMP>_exp022_bigM_ab.csv
```

## 4. 原始数据

- `moe_perf/raw/EXP-022/bigM_20260915T0304/20260915T0304_kernel_{ep,noep}_{default,tuned}_r{1,2,3}.log`（12 份，首行 provenance 含完整命令/臂/轮次/bs_list；每份含 config 来源行——default 臂 `Using default MoE config`、tuned 臂 `Using configuration from …4090.json`——与 4 个 M 档的 `Batch size / config / Kernel time`）。
- 汇总：`moe_perf/derived/20260915T0304_exp022_bigM_ab.csv`（mean/std/Δ/合并 std/显著/判定 + 三轮原始值列）。
- 作废尝试：`moe_perf/raw/EXP-022/bigM_20260915T0303/`（仅 1 份含 ray.init 超时 traceback 的日志 + `FAILED_NOTE.txt`，无测量值，原地保留）。
- 脚本 `moe_perf/d2_bigM_ab.sh`、`moe_perf/d2_bigM_analyze.py`。

## 5. 结果

**Kernel A/B，3 轮交叉次序（r1/r3 default 先，r2 tuned 先），单位 us，mean±std（样本 std）；Δ=(tuned−default)/default；显著 = |Δ_abs| > 2×合并 std**

**EP（E=30,N=1408，`-tp 2 --enable-expert-parallel`）**

| M | default | tuned | Δ | 合并 std | 2×合并 std | 判定 | 三轮 default | 三轮 tuned |
|---|---:|---:|---:|---:|---:|---|---|---|
| 512 | 681.1±0.8 | 637.7±0.5 | **−6.38%** | 0.98 | 1.96 | 保持收益 | 682.0/681.1/680.3 | 637.2/637.8/638.2 |
| 1024 | 881.2±1.1 | 757.5±4.4 | **−14.04%** | 4.58 | 9.15 | 保持收益 | 882.3/880.0/881.3 | 762.6/754.6/755.2 |
| 2048 | 1381.0±6.9 | 1244.6±10.2 | **−9.88%** | 12.35 | 24.70 | 保持收益 | 1383.5/1386.4/1373.2 | 1232.8/1249.8/1251.2 |
| 4096 | 2432.4±16.6 | 2266.2±11.8 | **−6.83%** | 20.35 | 40.70 | 保持收益 | 2419.4/2426.7/2451.0 | 2279.6/2257.5/2261.4 |

**非 EP（E=60,N=704，`-tp 2`）**

| M | default | tuned | Δ | 合并 std | 2×合并 std | 判定 | 三轮 default | 三轮 tuned |
|---|---:|---:|---:|---:|---:|---|---|---|
| 512 | 630.0±0.6 | 612.4±0.1 | **−2.79%** | 0.58 | 1.16 | 保持收益 | 629.9/630.5/629.4 | 612.4/612.3/612.5 |
| 1024 | 708.6±0.7 | 666.2±0.6 | **−5.99%** | 0.96 | 1.92 | 保持收益 | 709.2/707.8/708.8 | 666.8/665.5/666.3 |
| 2048 | 912.9±4.1 | 805.2±4.1 | **−11.80%** | 5.79 | 11.58 | 保持收益 | 913.0/908.8/916.9 | 800.5/806.9/808.1 |
| 4096 | 1453.3±11.9 | 1303.1±11.3 | **−10.34%** | 16.44 | 32.88 | 保持收益 | 1444.9/1467.0/1448.1 | 1316.1/1297.1/1296.0 |

判定：「保持收益」**8/8**（阈值 ≥6/8）→ 假设成立。

（三轮原始值两列由脚本从 CSV 回填，与 `derived/20260915T0304_exp022_bigM_ab.csv` 逐字一致；mean±std 由 `d2_bigM_analyze.py` 计算。）

## 6. 分析与结论

**【实测】① 假设成立且被超出**：预期"保持 −3% 量级"，实得 −2.8%~−14.0%，8/8 档显著；最小的一档（非 EP M=512，−2.8%）其 |Δ|=17.6 us 也是 2×合并 std（1.2 us）的 15 倍。轮间 std 相对 mean ≤1.4%（M=4096 处绝对 std 最大 16.6 us，相对 0.7%），交叉次序下无热漂移方向性（r1/r3 default 先与 r2 tuned 先的数值交错无系统偏移）。

**【实测】② 收益形状补全**：结合 EXP-015 §5.1（M=1 −8.2/−3.6%，M=8–64 打平，M=128/256 −3.4~−3.9%），完整曲线为：decode 单 token 有收益 → 小批 decode（8–64）打平 → M≥128 收益随 M 增长，**1024–2048 达峰 −10~−14%** → 4096 回落到 −7~−10%。【推断】峰值区对应 tuned 网格切到更大 BLOCK_SIZE_M/更高 num_stages 的档位（tuned 日志中 M=1024/2048 的 config 与 default 启发式差异最大——见 raw 中 `config:` 行），4096 处回落是两者都进入大 tile 区、启发式已接近最优。

**【推断】③ 对 e2e 的含义**：prefill 阶段 fused_moe 占比未在 D1 单测（D1 只分解 decode），不能直接折算 TTFT 收益；按 kernel 单点 −10% 与 MoE 层在 prefill 中的一般占比（未测，不给数）只能说"方向为正、幅度待 e2e 验证"。**本记录不做 e2e 主张**。

**对 EXP-015 §6 的修正**：「config 之外的 kernel 级空间有限」的判断是基于中段打平；大 M 段 tuned 相对 default 启发式还有两位数百分比差距，说明**默认启发式在大 M 的 tile 选择远非最优**——这一条应作为 PR 正文的补充证据（大 M 独立复测），并且是 D3 结论句的一个反例区间（config 杠杆在大 M 更大，而非更小）。

## 7. 异常、偏差与开放问题

- **首次尝试作废**（`bigM_20260915T0303/`）：round1 default/ep 臂在 `ray.init()` 超时（raylet 冷启动 + vllm 冷导入期间 IO 压力 avg10=42%），无测量值；脚本断言（缺「Using default MoE config」）正确中止并复位 JSON。处理：`ray stop --force` + 单独 `ray.init()` 热身 3.2 s 后重跑，12 臂全部 rc=0、断言全过。**命名偏差**：作废目录前缀 0303 是启动前预设、比实际启动（02:58:23Z）早 5 分钟；重跑目录前缀 0304 = 实际启动分钟（03:04Z），合规。
- 协议与 EXP-015 §5.1 一致，无偏离；GPU 分配沿用 benchmark_moe.py 的 ray 双卡轮转（M 档在两卡间交替，与 EXP-015 同）——两卡个体差异未单独量化（EXP-015 也未），对 A/B 差值无影响（同 M 两臂落同一卡序）。
- 开放：大 M 的 e2e（prefill/TTFT）放大验证未做，去向 = 用户决定是否补一次 in2048/out128 的 TP2+EP serving A/B（EXP-015 的 d2_ab.sh 可复用）。
- 本仓 vllm editable 处于 PR 分支 3805e40e17（非 EXP-015 记录的 main@7aa248fc）；两者 fused_moe kernel 无差异（该分支只加 JSON），但记录如实标注。

## 8. 下游影响

- **PR #54372 支撑数据升级**：大 M 档从"tune 内部测量"升级为"独立 3 轮交叉复测，8/8 显著"，可作为 PR 评论补充（措辞：kernel-level −2.8%~−14.0% at M=512–4096，指针 `moe_perf/derived/20260915T0304_exp022_bigM_ab.csv`）。红线不变：PR 未合并不写"合入"。
- **S3 简历句可扩展**：原「kernel 两端 −3.3~−8.5%」可改为「M=1 −8.2%/−3.6%，M≥512 −2.8~−14.0%（峰值 M=1024–2048）」，指针 EXP-015 §5.1 + EXP-022 §5；仍不写 e2e 吞吐主张。
- EXP-015 §6/§8 的「D3：config 之外 kernel 级空间有限」结论句需加限定「（M≤256 区间）」——由主线程整合，本记录不改旧 records。
- 工装：`d2_bigM_ab.sh` 证明 d2_hardening 的臂切换+双断言模板可直接换 BS_LIST 复用；新增教训：**benchmark_moe.py 前先 `ray.init()` 热身或检查 IO 压力**，否则冷机首臂可能在 ray 启动处超时。
