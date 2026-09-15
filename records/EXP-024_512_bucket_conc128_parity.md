# EXP-024 · 512 桶四臂统一到 conc128 口径（补 tp2 / pd1p1d 两臂，供 fig7 重算）

> **一句话结论**：补测后 512 桶四臂同口径（conc128）= **replica2 20.87 · colocate 12.81 · tp2 12.30 · pd1p1d 8.15 req/s**，扩展效率 **1.63×**；两臂的性质完全不同——**tp2 在 conc64 就已饱和**（12.31→12.30，−0.1%），**pd1p1d 在 conc64 就已撞传输墙**（`bytes/wall = 0.230 → 0.239 GB/s`，上限模型 `0.2393/0.02936 = 8.15` 与实测 8.152 精确吻合），只有 colocate/replica2 两臂真的欠饱和（+23.6% / +34.0%）。副作用发现：**tp2 在 conc128 下不触发功率帽**（功率峰 326 W，无 0x4），是四臂里唯一不上 450 W 的（负载摊到两卡）；而 PD 的 TPOT p50 18.97 ms 是四臂最好（D 只做 decode），代价是 TTFT p50 12669 ms。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | ENV-B（`/root/venvs/v0.25.1`，vllm 0.25.1 sha 752a3a5044，带 EXT-1 本地 patch）；Qwen2-7B-Instruct；2×RTX 4090；driver 610.57.04 |
| 状态 | 完成（两臂均一次成功，未触发回退规则；seed/口径一致性检查通过） |
| 关联清单项 | B1 / EXP-007《B1 四臂 offered-load 扫描战役》§5 脚注；EXP-023《replica2@512 饱和复测（SAT_CONC=128）》§8 下游影响 |

## 1. 目的与假设

EXP-023 已把 512 桶的 replica2 与 colocate 复测到 conc128（20.87 / 12.81 req/s），但 tp2 与 pd1p1d 两臂的 512 桶仍是 conc64（12.31 / 7.84）。`make_fig7_overview.py` 的选择逻辑是"后来者覆盖"，若不补齐，重算 fig7 会让**同一个桶里两个臂 conc128、两个臂 conc64**——柱子之间不可比（EXP-023 §7 登记）。

本实验补齐 tp2@512 与 pd1p1d@512 的 conc128 点，使 512 桶四臂同口径。

**这不是假设检验型实验，是口径补齐**，故不设"效应量"阈值；跑前锁定的成立条件与回退规则（跑完不改）：

- 两个点各自要求：`failed_requests == 0`、gate 通过、N=400 跑完；GPU 遥测（2s 采样）落盘。
- 记录 TTFT/TPOT p50/p90/p99 与饱和吞吐，供报告说明 conc128 下各臂的 SLO 位置（不参与判定）。
- **回退规则**：任一臂起栈失败/OOM（各 2 次尝试）或 bench 失败 → 停、写 §7，fig7 退回"冻结 conc64"方案并在 LEDGER/README 注明"512 桶 conc128 仅两臂可用，未进图"。**未触发**。
- 时间盒两臂合计 ≤60 分钟；**实际用 5 分钟**（tp2 起栈 110s + 单点 33s；pd1p1d 起栈 76s + 单点 49s）。
- **口径一致性检查（通过）**：seed 1099、`NUM_PROMPTS=400`、random 512×128、`--ignore-eos`、`--request-rate inf`、`SAT_CONC=128`，与 EXP-023 两点逐项一致。

## 2. 环境与配置

配置与 EXP-005/006/007 同，唯一变量是客户端并发 64→128。

- **tp2 臂**：`CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2-7B-Instruct --port 8100 --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.88`（0.88 沿用 EXP-007 的登记偏差）；脚本 `pd_disagg/scripts/tp2_stack.sh up 20260915T0947`。就绪 +110s，双卡各 23168 MiB。bench 打 8100，/metrics 直抓 8100，`GPU_COUNT=2`。
- **pd1p1d 臂**：P = GPU0:8100 / side 5600 / kv_producer；D = GPU1:8200 / side 5601 / kv_consumer；都用 `--kv-transfer-config` 带 `kv_load_failure_policy:"fail"`；proxy = `smoke/toy_proxy_v0251.py --port 8192 --prefiller-ports 8100 --decoder-ports 8200`；脚本 `pd_disagg/scripts/pd_stack.sh up 20260915T0950`。就绪 +71s/+76s，双卡各 23618 MiB。bench 打 8192，/metrics 直抓 8100+8200，`GPU_COUNT=2`。
- fresh server、冷缓存；seed 1099 每实例只服务一次。**前缀缓存复查（通过）**：pd1p1d 的 P/D 两侧 `prefix_cache_hits_total` 均为 0；D 侧 `external_prefix_cache_hits_total = 204800` = 全部 400×512 token 均来自远端（符合 pull 语义，非污染）。
- 新增脚本：`pd_disagg/scripts/tp2_stack.sh`、`pd_disagg/scripts/pd_stack.sh`（起/拆栈；down 走 `/proc` 定位 + SIGTERM，**不用 pkill 字面量**——EXP-023 §7 的自杀教训）。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv
P=20260915T0947; bash pd_disagg/scripts/tp2_stack.sh up $P
cd pd_disagg && SEED=1099 NUM_PROMPTS=400 SAT_CONC=128 GPU_COUNT=2 \
  scripts/run_point.sh tp2 saturation 512 128 - 8100 8100
cd .. && bash pd_disagg/scripts/tp2_stack.sh down

P=20260915T0950; bash pd_disagg/scripts/pd_stack.sh up $P
cd pd_disagg && SEED=1099 NUM_PROMPTS=400 SAT_CONC=128 GPU_COUNT=2 \
  scripts/run_point.sh pd1p1d saturation 512 128 - 8192 8100 8200
cd .. && bash pd_disagg/scripts/pd_stack.sh down
```

## 4. 原始数据

全部在 `pd_disagg/results/b1_matrix/`：

| 测量点 | runs.jsonl run_id | raw / snapshots |
|---|---|---|
| **tp2@512 conc128** | `20260915T0949_tp2_512x128_saturation`（第 128 行，gate_pass=true） | `raw/20260915T0949_tp2_512x128_saturation_{bench.json,bench.log,gpu.csv}`、`snapshots/20260915T0949_..._8100_{before,after}.prom`、服务日志 `raw/20260915T0947_tp2_conc128_server_8100.log` |
| **pd1p1d@512 conc128** | `20260915T0952_pd1p1d_512x128_saturation`（第 129 行，gate_pass=true） | `raw/20260915T0952_pd1p1d_512x128_saturation_*`、`snapshots/20260915T0952_..._{8100,8200}_{before,after}.prom`、服务日志 `raw/20260915T0950_pd1p1d_conc128_{P_8100,D_8200,proxy}.log` |

对照点（conc64，EXP-007）：`20260821T1917_tp2_512x128_saturation`、`20260821T1957_pd1p1d_512x128_saturation`。同口径另两臂（EXP-023）：`20260915T0329_replica2_…`、`20260915T0333_colocate_…`。

## 5. 结果

**512 桶四臂，conc64 vs conc128（N=400，seed 1099，p50 除注明）**

| 臂 | conc64 req/s | **conc128 req/s** | Δ | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | GPU·s/req | conc128 下 SLO（TTFT≤328 / TPOT≤50） | 功率峰 W | 节流原因 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| colocate | 10.36 | **12.81** | +23.6% | 835 | 6466 | 67.1 | 0.078 | 双破 | 450.1 | 0x1+**0x4** |
| replica2 | 15.58 | **20.87** | +34.0% | 771 | 3257 | 36.8 | 0.096 | TTFT 破 / TPOT 过 | 450.3 | 0x1+**0x4** |
| tp2 | 12.31 | **12.30** | −0.1% | 1046 | 7086 | 70.3 | 0.163 | 双破 | **326.3** | 仅 0x1 |
| pd1p1d | 7.84 | **8.15** | +4.0% | **12669** | 23438 | **18.97** | 0.245 | 双破 | 453.5 | 0x1+**0x4** |

**扩展效率（512 桶，conc128 同口径）**：replica2 / colocate = 20.869 / 12.807 = **1.630×**（conc64 口径为 15.58/10.36 = 1.50×）。

**pd1p1d 的 KV 通路核算（两个并发档对照，同 seed 同 N）**

| 档 | KV bytes 总 | bytes/req | desc/req | Σ xfer_time | **bytes / Σxfer** | **bytes / wall** | 上限模型 `bytes/wall ÷ 29.36MB` | 实测 req/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| conc64 | 5.872 GB | 29.36 MB | 1792 | 864.6 s | 0.0068 GB/s | **0.2302 GB/s** | 0.2302/0.02936 = **7.84** | **7.842** |
| conc128 | 11.744 GB | 29.36 MB | 1792 | 4487.9 s | 0.0026 GB/s | **0.2393 GB/s** | 0.2393/0.02936 = **8.15** | **8.152** |

（`bytes/req` 与 `desc/req` 两档完全相同；`desc/req = 1792 = 56 × 32 = 56 × ⌈512/16⌉`，与 EXP-006 的 descriptor 模型一致。gate 全通过：xfers=400=completed、failed/expired=0。）

## 6. 分析与结论

**【实测】① 两臂根本不是"欠饱和"，性质与 colocate/replica2 相反。**

- **tp2**：12.31 → 12.30（−0.1%，在噪声内）→ **conc64 时已饱和**。TPOT p50 从 conc64 的 33.93 ms 涨到 conc128 的 70.29 ms（TTFT p50 765→1046 ms），说明并发翻倍只换来排队变长、吞吐不变。**tp2 的 512 桶天花板在 conc64 处就已经到顶。**
- **pd1p1d**：7.84 → 8.15（+4%）→ **conc64 时已撞传输墙**。证据是 上限模型在两个档各自精确命中（7.84 与 8.15，见 §5 表）。两档的 `bytes/wall` 都 ≈ 0.23–0.24 GB/s，与 EXP-006 的 0.26–0.27 GB/s 同量级。
- 只有 colocate（+23.6%）与 replica2（+34.0%）真的欠饱和。

**【实测】② tp2 是四臂里唯一不触发功率帽的**：conc128 下功率峰 326 W / 300 W（双卡各），节流原因只有 0x1（idle），SM 2822 / 2807 MHz 满血；其余三臂都出现 0x4（SW Power Cap）且逼近 450 W。机制：TP2 把 prefill 计算摊到两卡，单卡功耗减半，绕开了消费卡 450 W 帽——这是 EXP-005 功率帽发现的一个正面对照（该发现此前只用于解释 colocate 的降频）。

**【实测】③ PD 的 TTFT/TPOT 双面性在 conc128 下被放大**：TPOT p50 18.97 ms 是四臂最好（D 只做 decode，权重带宽不受 prefill 干扰，正是 PD 的价值主张①"消除 prefill 对 decode 的干扰"）；但 TTFT p50 12669 ms、p99 23438 ms，是四臂最差——价值主张①的收益远小于"KV 必须搬家"的代价。

**【方法学·重要】④ 并发 >1 时 `bytes / ΣxferDuration` 不再是吞吐**：conc64 时它给 0.0068 GB/s、conc128 时 0.0026 GB/s——比真实聚合速率低两个数量级，因为它把**并发重叠的每请求时长累加**了。EXP-006 在 conc1 下 `Σxfer ≈ wall`，所以当时 `bytes/Σxfer`（0.26–0.27 GB/s）恰好等于聚合速率；**这个等式只在串行成立**，与 EXP-002 的"两卡时 busbw 恰等于 algbw，这个巧合只在 2 卡成立"是同一类陷阱。**KPI 口径**：聚合速率一律用 `bytes / wall_time`；`ΣxferDuration` 只用于"每请求平均等待时长"（conc128 下 11.22 s/req，conc64 下 4.32 s/req，随并发近似线性——共享同一条墙）。红线不变：只称 telemetry-derived，且 `xferDuration 不与 postDuration 相加`。

**【推断】⑤ 512 桶的选型结论不变、更硬**：replica2 仍是最高（1.63×）；tp2 的 512 桶天花板已测到（12.3 req/s，与 colocate 12.81 几乎相同而占两张卡 → per-GPU 视角负收益）；PD 仍在传输墙上（8.15 req/s，容量由 0.24 GB/s 决定，与并发无关）。

**【推断】⑥ 扩展效率 1.63× 的剩余缺口**：conc128 下 replica2 的 TPOT p50 36.8 ms 仍 < SLO 50 ms，即**仍未封顶**；rr_proxy 单进程转发在 20+ req/s、128 路流式下的开销未拆分。两项都未测，去向见 §7。

## 7. 异常、偏差与开放问题

- **pd_stack.sh 的 proxy 就绪探测误报**：`WARN: proxy :8192 未探到 /health 或 /v1/models`——`toy_proxy_v0251.py` 不暴露这两个端点（只有 `/v1/completions` 等转发路径）。实际 bench 400/400 成功，说明代理正常；该 WARN 是探测方式错误，脚本已按"WARN 后照常继续"设计，未影响数据。**待办**：把就绪探测改为直接 POST 一个 1-token 请求，或读 uvicorn 的启动行。
- **未触发回退规则**：两臂各一次成功，无 OOM、无端口冲突、无 bench 失败。时间盒 5/60 分钟。
- **口径一致性**：通过（§1 逐项核对）。唯一未复制的是 EXP-023 的 `replica2@512` 用了 rr_proxy 而 tp2/pd1p1d 各自用自己的服务拓扑——这是各臂的定义差异，不是偏差。
- **开放**：① replica2 的 conc≥192 真饱和点；② rr_proxy 开销拆分（直连 8100+8200 各 conc64 vs 经代理 conc128）；③ tp2 的功率帽免疫是否能把 conc 继续推高（当前 TPOT 70 ms 已破 SLO，意义有限）；④ PD 在 512 桶的 TTFT 12.7 s 远超 SLO——本桶的 PD 只应作为"传输墙"证据，不应作为可用配置。
- **本记录不改任何已有 records**；EXP-007 的 conc64 数字按史料保留，口径修正以勘注 + LEDGER/README/fig7 为准。

## 8. 下游影响

- **fig7 重算（直接动因）**：`make_fig7_overview.py` 改为**显式声明每个 (arm, bucket) 用哪一行**（禁止"后来者覆盖"），512 桶四臂全部指向 conc128 行，2048/8192 仍为 conc64 行；图注注明两桶口径不同。`runs.jsonl` 不记录 SAT_CONC（`run_point.sh` 未写该字段），同桶多行时必须靠白名单——这是 EXP-023 §7 那个坑的根治办法。
- **三处表格需同步 512 列**：`LEDGER.md` B1 行、`pd_disagg/REPORT.md` §四臂表、`docs/TECH_DOC_vllm_engineering.md` §3.1 与 C11（1.50×→1.63×）。
- **B3 版本对照不受影响**：`7.14→10.36`（+45%）两侧都是 conc64，口径自洽，保留原值并加口径注（不外推到 conc128——v0.17.1 的 conc128 未测）。
- **红线**：512 桶的 conc128 数字是**单次测量**（与 EXP-007 每点单次同规格），引用排序与量级足够，引用到 ±5% 精度需补 3 轮；fig7 只作排序展示。
- 工装新增 `tp2_stack.sh` / `pd_stack.sh`（起栈≈2 min，含 OOM/端口检测与就绪轮询），可复用于 tp2/pd1p1d 的任何单点或 sweep。
