# EXP-021 · NCCL allreduce dtype 扫描（half/bfloat16 vs float，补 EXP-018 §7 缺口）

> **一句话结论**：**dtype 不敏感假设「未决」——本机 SHM 路径的运行间噪声（同 dtype 两轮平台 7.50 vs 2.15 GB/s，探针 10 轮 5.9–8.7）淹没了任何 <10% 的 dtype 效应**：延迟地板（16B–8K）三 dtype 13.5–14.4µs 差 <7%（成立）；大消息平台 half/bf16 6.06/6.39 vs float 4.83（含一次塌陷）——差 25–32% 但小于 float 自身两轮差 111%，按锁定规则归「不可分辨」；32K–1M 过渡区 float 两轮均比 half/bf16 慢 2.5×（1M：450 vs 171/161µs），按锁定阈值判「不成立」，但同 dtype 的大消息文件 1M 点（float 182µs）与之矛盾——是状态效应不是 dtype 效应的嫌疑更大，未能干净证伪。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | 2×RTX 4090 无 P2P（NCCL 走 SHM），driver 610.57.04，NCCL 2.28.9（v0.25.1 venv wheel，LD_LIBRARY_PATH 显式），nccl-tests tool_sha 717b683182 |
| 状态 | 完成（判定级别：未决，噪声主导；跟进探针见 §7） |
| 关联清单项 | EXP-018《NCCL allreduce size 扫描》§7「只测了 float 未测 half」缺口；decode 实际 dtype 为 bf16 |

## 1. 目的与假设

EXP-018 只测了 `float/sum`，而 vLLM TP allreduce 实际是 bf16。补 `-d half` 与 `-d bfloat16`（二进制 `strings` 确认两者均支持）两条 size 扫描，与 EXP-018 的 float 并排。

**假设（可证伪）**：allreduce 对 dtype 不敏感——同字节数下 busbw 相同（NCCL 的 SHM/网络路径搬的是字节，reduce 算子在 4090 上远非瓶颈）。

**跑前锁定的判定阈值（跑完不改）**：
- 比较对象：同一会话内 float vs half vs bfloat16，同字节数（nccl-tests 的 size 列即字节数，count 列随 dtype 变）。
- 大消息区间（1M–512M，n=20）：取 16M–256M 平台均值，`|busbw_dtype − busbw_float| / busbw_float < 10%` → 假设成立；≥10% → 不成立，dtype 敏感。
- 小消息区间（8B–1M，n=100）：取 8 KiB（decode 级）与 1 MiB 两点的 time，同样 <10% 判不敏感；小消息延迟地板（16B–8K 区间 time 均值）三 dtype 也应在 ±10% 内。
- 会话内抖动参照：每 dtype 跑两轮（r1 正序 / r2 反序），两轮平台差作为噪声尺度；若 dtype 间差 < 两轮差，则 dtype 效应不可分辨（归入成立）。
- **与 EXP-018 的 float 结果并排只作跨会话参照**，不参与判定（EXP-020 §7 已证 SHM 路径跨会话漂移 ±30%）。

## 2. 环境与配置

- 命令与 EXP-018 完全一致，仅加 `-d <dtype>`：小消息 `-b 8 -e 1M -f 2 -g 2 -n 100`，大消息 `-b 1M -e 512M -f 2 -g 2 -n 20`。附加 `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT` 留自报路径（初始化期打印，不进计时区）。
- 硬件占用：双卡；跑前 `nvidia-smi --query-compute-apps` 确认无其他 compute 进程。
- 脚本 `scripts/nccl_size_scan_dtype.sh`；共用 STAMP 前缀，文件名 `<STAMP>_allreduce_size_scan_<dtype>_<small|large>_r<round>.txt`。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
bash scripts/nccl_size_scan_dtype.sh      # 2 轮 × 3 dtype × 2 区间 = 12 个文件
python3 scripts/nccl_dtype_scan_analyze.py <STAMP>   # 并排表 + 判定 → derived/<STAMP>_nccl_dtype_table.csv
```

## 4. 原始数据

`pd_disagg/hw/20260915T0248_allreduce_size_scan_<dtype>_<small|large>_r<1|2>.txt`（12 个文件，dtype ∈ float/half/bfloat16；每文件首行 provenance 含完整命令、dtype、轮次；末行 `# exit_code=0`；含 NCCL 自报 `via SHM/direct/direct`）。
汇总：`pd_disagg/hw/derived/20260915T0248_nccl_dtype_table.csv`（summary 段 = 判定用四指标；per_size 段 = 逐 size 两轮均值 time/busbw 三 dtype 并排）。脚本 `scripts/nccl_size_scan_dtype.sh` / `scripts/nccl_dtype_scan_analyze.py`。
跟进探针（post-hoc，§7）：`pd_disagg/hw/20260915T0252_nccl_shm_probe_<float|half>_r<1..5>.txt` + 同名 `_pcie.csv`（200ms PCIe gen/width/SM/mem clock/pstate/power），脚本 `scripts/nccl_shm_collapse_probe.sh`。
跨会话参照：EXP-018 `pd_disagg/hw/20260829T104705_allreduce_size_scan_{small,large}.txt`（float）。

## 5. 结果

**① 判定用四指标（两轮均值；Δ 相对 float；阈值 |Δ|<10%）**——`derived/20260915T0248_nccl_dtype_table.csv` summary 段

| dtype | 平台 r1 | 平台 r2 | 平台均值 (GB/s) | Δ平台 | 两轮差 | t(8 KiB) µs | Δ | t(1 MiB, small 文件) µs | Δ | 地板 16B–8K µs | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float | 7.50 | **2.15** | 4.83 | ref | 111% | 13.84 | ref | 450.1 | ref | 14.40 | ref |
| half | 6.06 | 6.05 | 6.06 | +25.5% | 0.2% | 14.29 | +3.3% | 171.3 | −61.9% | 13.77 | −4.4% |
| bfloat16 | 6.05 | 6.74 | 6.39 | +32.5% | 10.8% | 13.39 | −3.3% | 161.4 | −64.1% | 13.53 | −6.1% |

**② 大消息逐 size busbw（GB/s，out-of-place）**

| size | 1M | 2M | 4M | 8M | 16M | 32M | 64M | 128M | 256M | 512M | avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float r1 | 5.77 | 7.51 | 7.73 | 7.56 | 7.06 | 7.46 | 7.58 | 7.82 | 7.57 | 7.87 | 7.35 |
| float r2（塌陷） | 1.91 | 1.70 | 1.91 | 2.07 | 2.05 | 2.09 | 2.13 | 2.21 | 2.29 | 2.29 | 2.07 |
| half r1 | 5.38 | 5.10 | 5.95 | 5.87 | 5.95 | 6.00 | 5.96 | 5.89 | 6.51 | 6.87 | 5.99 |
| half r2 | 5.51 | 5.20 | 6.01 | 5.67 | 5.76 | 5.85 | 6.04 | 6.13 | 6.48 | 6.83 | 5.94 |
| bfloat16 r1 | 4.95 | 4.96 | 5.93 | 5.79 | 5.88 | 5.91 | 6.03 | 6.04 | 6.38 | 6.40 | 5.81 |
| bfloat16 r2 | 5.30 | 5.74 | 6.74 | 6.82 | 6.65 | 6.67 | 6.62 | 6.84 | 6.91 | 6.92 | 6.57 |
| EXP-018 float（8/29 参照） | 5.45 | 5.47 | 6.10 | 6.45 | 6.42 | 6.40 | 6.45 | 6.48 | 6.50 | 6.38 | 6.20 |

**③ 小消息逐 size time（µs，n=100；两轮各列）**

| size | 16B | 256B | 8K | 16K | 32K | 64K | 128K | 256K | 512K | 1M |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| float r1/r2 | 14.7/13.9 | 17.6/13.6 | 13.9/13.8 | 14.1/23.3 | 30.5/31.2 | 54.2/57.6 | 109.8/97.8 | 228.9/193.6 | 222.3/246.4 | **431.6/468.6** |
| half r1/r2 | 13.6/13.5 | 15.5/13.5 | 13.5/15.1 | 13.5/13.3 | 17.8/19.4 | 27.5/32.0 | 49.5/55.5 | 92.9/92.8 | 88.4/106.6 | **169.2/173.4** |
| bfloat16 r1/r2 | 13.5/13.6 | 13.5/13.4 | 13.4/13.4 | 13.3/13.3 | 19.1/13.9 | 32.2/20.3 | 57.9/36.8 | 106.4/68.2 | 97.0/75.7 | **183.3/139.4** |
| EXP-018 float | 13.9 | 13.8 | 13.8 | 13.6 | 18.9 | 28.6 | 49.0 | 86.4 | 116.7 | 221.7 |

**④ 跟进探针（post-hoc，float/half 交替 5 轮大消息 + 200ms PCIe 采样）**：float 平台 8.73 / 5.89 / 7.01 / 6.15 / 5.94，half 平台 5.87 / 5.85 / 6.21 / 6.53 / 6.27（GB/s）；10 轮负载期 PCIe Gen4 x16 占比 97–100%，SM 2835 MHz、显存 10501 MHz，pstate P0/P2；**未再出现塌陷**。

## 6. 分析与结论

**【实测】① 延迟地板与 decode 级 8 KiB 点：三 dtype 在 ±7% 内（13.4–14.4µs）→ 假设在延迟主导区成立。** 这是 decode 真正关心的点（EXP-018 §6①），bf16 与 float 无差别。

**【实测】② 大消息平台：按锁定规则「dtype 间差 < 同 dtype 两轮差 → 不可分辨，归入成立」。** float 两轮 7.50/2.15（差 111%），half/bf16 四轮 6.05–6.74；探针再给 float 5.9–8.7、half 5.9–6.5，两 dtype 的分布重叠。dtype 效应若存在，小于 SHM 路径 ±30% 的运行间噪声——**无法在本机证实 <10% 的等价，也没有证据说它们不等价**。

**【实测】③ 32K–1M 过渡区：float 两轮都比 half/bf16 慢 2.3–2.8×（1M：432/469 vs 169/173/183/139µs），按锁定阈值判「不成立」。** 但【实测】同一 dtype 的大消息文件里 float 1M = 182µs（r1）≈ half 195/190 ≈ bf16 212/198，且 EXP-018 的 float 1M = 222µs——同 dtype 同 size 跨进程差 2.5×，说明这个"float 慢"更像每个进程初始化后落入的**状态**（与 §7 的塌陷同源嫌疑），不是 reduce 算子的 dtype 成本（float 与 half 的 reduce 在 4090 上都远非瓶颈——推断）。**按预注册阈值的字面结论是"过渡区不成立"，按证据权重是"未能证伪"**——两者都写下，不选择性报告。

**【推断】④ 对下游的实际含义**：vLLM decode 的 allreduce（bf16，≤ 几十 KiB）落在延迟主导区，dtype 无影响；prefill（4 MiB 量级）落在平台区，bf16 的实测 6.0–6.9 GB/s 与 float 的"正常态" 6–9 同量级。EXP-018 用 float 测出的地板/平台结论对 bf16 可沿用，但引用带宽时必须报区间不报单值。

## 7. 异常、偏差与开放问题

- **SHM 路径塌陷态（核心异常）**：float r2 大消息 SHM 路径塌到 2.07 GB/s avg（1M 起就平坦 1.7–2.3），NCCL 自报仍是 `via SHM/direct/direct`，用时 24s（正常 10s）。本实验未带 PCIe 采样，无法判断当时链路是否在 Gen1/2。**post-hoc 探针**（`scripts/nccl_shm_collapse_probe.sh`，float/half 交替 5 轮 + 200ms PCIe 采样）10 轮全在 Gen4、无塌陷。今天 SHM 路径大消息扫描共 28 次（EXP-020 12 + 本实验 6 + 探针 10），塌陷 1 次；曲线形状（1M 起平坦）与 EXP-002 的 1.78 一致，比 EXP-020 的 Socket 档（1M 1.22 爬升到 8M 才平）更像。**去向：写入 EXP-020 附录 A 作为 H3「SHM 路径间歇塌陷态」候选**；机理未知（候选：PCIe 链路未升频 / 主机侧页缓存或 THP 状态 / 未知），需带 PCIe 采样长跑 ≥50 轮才能抓到并定因——留待用户决定是否投入。
- **协议偏离**：无（命令、轮次、阈值与 §1 一致）。探针为预注册之外的跟进，只作旁证不进判定。
- 小消息区间的 float 两轮在 32K–1M 都慢（§6③），可能是同一塌陷态的部分表现（该区间的 time 由 SHM 带宽主导）；两个 float small 进程分别是会话第一个和最后一个，不支持"冷启动"解释。
- 8 B 点噪声（bf16 r2 28.5µs、half r1 17.2µs）——与 EXP-018 的 8 B 17µs 同类，极小子消息的测量噪声，不参与判定（地板取 16B–8K）。

## 8. 下游影响

- EXP-018 §7 的"未测 half"缺口关闭为：**延迟地板对 dtype 不敏感（实测）；平台带宽的 dtype 等价在本机噪声下不可判**。EXP-018 的 float 地板/平台结论可对 bf16 沿用，措辞加"区间"。
- 硬件画像新增：bf16 allreduce 平台 6.0–6.9 GB/s（4 轮）、地板 13.4–13.8µs（`pd_disagg/hw/20260915T0248_*bfloat16*`）。
- 工装：`nccl_size_scan_dtype.sh` / `nccl_dtype_scan_analyze.py`；**教训回写 EXP-020 §8 工装条**：任何 NCCL 带宽测量必须同步采 PCIe 运行态（本实验没采，塌陷态因此无法定因）——`nccl_shm_collapse_probe.sh` 的采样模板可复用。
- 对 EXP-020 判定的影响：见 EXP-020 附录 A（定因不唯一）。
