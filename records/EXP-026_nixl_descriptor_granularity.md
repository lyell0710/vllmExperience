# EXP-026 · NIXL descriptor 粒度实验：把「描述符碎片化」从推断变成实测

> **一句话结论**：**「descriptor 碎片化」这条归因被实测推翻，同时发现 NIXL 默认就在合并描述符**——三件事一次测清：① **粒度几乎无影响**：合并开启时 descriptor 从 16 KiB 放大到 16 MiB（1000×），带宽只从 0.383 变到 0.376 GB/s（**−2.0%**）；② **NIXL 确实默认合并**：同一 16 KiB 布局下 `skip_desc_merge=False` 把 4096 个描述符合并成 **1 个**（telemetry `descCount=1`），带宽 0.267 → **0.388 GB/s（+45%）**，而 vLLM 从不设这个参数、吃到的就是合并后的收益；③ **真正的天花板是传输层，不是描述符**：UCX 自报 RMA 路径恒为 `rma_am(tcp/eth0)`（GPU 缓冲走 software emulation），且 `UCX_TLS` **不可调**——加 `sm`/`cuda_ipc` 会让 NIXL 后端直接初始化失败（`NIXL_ERR_BACKEND`），`all`/`cuda_copy+tcp` 仍然选 TCP，带宽稳定在 0.36–0.39 GB/s。**直接后果：EXP-020 附录 A 的「合并 descriptor 上界 2.4–3.4×」作废**（真实收益 +45%，且天花板是 0.38 而非链路的 0.60–0.91 GB/s）；而 EXP-006/013 观测到的 0.26–0.27 GB/s 与 61.5–65.8 µs/descriptor 恰好落在**未合并/散列**那一档（我复现为 0.267 GB/s、61.3 µs/desc）——**说明 vLLM 的实际布局吃不到合并**，那 +45% 是它够得着的（详见 §6⑥ 与 §7 的开放项）。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | 2×RTX 4090（无 P2P，走 SHM/Socket 回退）；driver 610.57.04；NIXL（v0.25.1 venv wheel）；backend=UCX，mem_type=VRAM |
| 状态 | 完成（H1 成立、H2 被推翻；两段实验：粒度扫描 + UCX_TLS 筛选） |
| 关联清单项 | B2 / EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》§5 的 descriptor ≈16 KB 推断；EXP-011《EXT-2 NixlPush 单点》§5；EXP-020《NCCL 旋钮矩阵复现 1.78 GB/s》附录 A 的「合并 descriptor 收益上界 2.4–3.4×」推断 |

## 1. 目的与假设

EXP-009/011 与 EXP-020 附录 A 把 NIXL KV 通路的 0.26–0.27 GB/s 归因于「descriptor ≈16 KiB 的碎片化小拷贝、固定开销 α 主导」，并据此推出「合并 descriptor 可拿回 2.4–3.4×（到 0.6–0.9 GB/s）」。**但这一条从头到尾都是推断**，从未直接测过。

反过来，读 NIXL 源码时发现一个反证线索：`make_prepped_xfer` 的签名里有 **`skip_desc_merge: bool = False`**——默认值是 `False`，而 vLLM 的 connector 从不设这个参数。**如果 NIXL 默认就在合并描述符，那"碎片化"这个归因本身就站不住。**

本实验用与 vLLM 完全同一条代码路径（`prep_xfer_dlist` + `make_prepped_xfer`，READ/pull，VRAM，UCX）在两卡之间测：

- **A 粒度曲线**：固定传输总量 S=64 MiB，descriptor 大小 G ∈ {16K, 64K, 256K, 1M, 4M, 16M}
- **B 布局**：`contiguous`（描述符首尾相接 → 可合并）vs `scattered`（间隔一个 G、中间有洞 → 不可合并）
- **C 合并开关**：`skip_desc_merge ∈ {False(默认), True}`，在 G∈{16K, 1M} × 两种布局上各测
- **D 尺寸扫描**：G 固定为 vLLM 的 16 KiB，S ∈ {8, 32, 128} MiB → 拟合 `t = N_desc·(α + m/β)`

**假设（可证伪）**：H1 = NIXL 默认合并描述符；H2 = 「16 KiB 碎片化」是 0.27 GB/s 的主因。

**跑前锁定的判定阈值（跑完不改）**：

| 判定 | 条件 | 结论 |
|---|---|---|
| **NIXL 是否合并** | `contiguous` 下 `skip_desc_merge=False` 与 `True` 的带宽差 ≥30% | 是（默认在合并）；<10% → 否（两种设置等价） |
| **碎片化归因（H2）成立** | G=16 KiB 的带宽落在 **0.20–0.35 GB/s** 且增大到 G≥1 MiB 时带宽提升 **≥2×** | H2 成立，0.27 由粒度解释，合并可拿回 ≥2× |
| **H2 被推翻** | G=16 KiB 的带宽 **≥0.5 GB/s**；**或** G 从 16 KiB 到 16 MiB 的带宽变化 **<30%** | 粒度不是主因（NIXL 已合并，或瓶颈在别处），EXP-020 附录 A 的上界推断作废 |
| **可合并性的作用** | `scattered` 相对 `contiguous` 低 **≥2×** | 「池内块是否连续」是决定带宽的第一变量 → 直指真实 KV pool 的碎片化程度 |
| **α 与 β 分离** | 由 D 组三点线性拟合 `t = N_desc·α + S/β`（N_desc = S/16 KiB），要求拟合残差 <20% | 给出 α（每 descriptor 固定开销）与 β（渐近带宽）的实测值；残差过大则只报区间 |

- 每点 iters=5，取**中位数**（壁钟）并同时记录 NIXL 自报 `xferDuration` 与 `descCount`（两者都要与预期一致，否则抛错——铁律 8）。
- 传输总量与 descriptor 数在 telemetry 里必须逐次一致（`totalBytes` 与 `descCount` 集合大小 = 1），不一致即中止。

## 2. 环境与配置

- 两个进程：P 用 `CUDA_VISIBLE_DEVICES=0`（agent 名 `EXP026_P`），D 用 `CUDA_VISIBLE_DEVICES=1`（`EXP026_D`）；各自分配并注册 64 MiB（D 组扫描单独用 128 MiB region）VRAM region。
- 元数据交换走文件（`<workdir>/P_meta.bin` / `P_ready` / `D_meta.bin` / `P_stop`）；D 用 `add_remote_agent` 加 P，然后发起 `READ`（pull）——与 vLLM 的生产者/消费者方向一致。
- 与 vLLM 的对应关系：`mem_type=VRAM`、`backends=["UCX"]`、`prep_xfer_dlist` + `make_prepped_xfer`、`READ`，均取自 `vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py` 的实际用法。
- 脚本：`scripts/nixl_desc_granularity_bench.py`（`--role P|D`）。
- 硬件占用：双卡各一进程，无其他 compute 进程（跑前 preflight）。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv     # 双卡空闲
STAMP=$(date -u +%Y%m%dT%H%M); W=pd_disagg/hw/exp026_$STAMP
mkdir -p $W
CUDA_VISIBLE_DEVICES=0 /root/venvs/v0.25.1/bin/python scripts/nixl_desc_granularity_bench.py \
    --role P --workdir $W --region $((64*1024*1024)) > $W/P.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 /root/venvs/v0.25.1/bin/python scripts/nixl_desc_granularity_bench.py \
    --role D --workdir $W --region $((64*1024*1024)) --out $W/D.csv > $W/D.log 2>&1
wait
```

## 4. 原始数据

| 段 | 内容 | 路径 |
|---|---|---|
| 第一段 粒度扫描 | 主运行（vol=64 MiB，region=128 MiB，iters=5）23 点；每点含 wall 中位数与 NIXL telemetry | `pd_disagg/hw/exp026_20260915T1151/D.csv`（首行 provenance）、`D.log`、`P.log`；另有同配置复现轮 `exp026_tls_20260915T1207/bench/baseline/`（0.386 GB/s，与主轮 −0.5%，作重复性证据） |
| 传输层取证 | `UCX_LOG_LEVEL=info` 小传输探针（vol=8 MiB），抓 UCX 自报 transport 行 | `pd_disagg/hw/exp026_transport_20260915T1153/D.log` |
| UCX_TLS 筛选 | 10 档 × 两步骤（screen 只验证初始化 / bench 才传输），逐档落盘 | `pd_disagg/hw/exp026_tls_20260915T1228/`（`tls_summary.csv` + `screen/<档>/{P,D}.log` + `bench/<档>/{D.csv,P.log,D.log}`） |
| 脚本 | 基准与驱动 | `scripts/nixl_desc_granularity_bench.py`、`scripts/nixl_ucx_tls_sweep.sh` |

作废的前缀（工具 bug，原地保留）：`exp026_20260915T1146`（`get_xfer_descs` 传了 4 元组，报 "3-tuple list needed"）、`exp026_20260915T1148`（`transfer()` 返回值判据写错：返回 `PROC` 是正常的投递态，我当成了错误）、`exp026_20260915T1150`（未开 `capture_telemetry`，`get_xfer_telemetry` 抛 `NIXL_ERR_NO_TELEMETRY`）、`exp026_tls_20260915T1154`（D 侧等 `P_ready` 无超时 + 轮询无超时，被 `UCX_TLS=shm` 挂死）、`exp026_tls_20260915T1219`、`exp026_tls_20260915T1225`（CSV 列错位：TLS 值含逗号撑开了列）。

## 5. 结果

**① 粒度曲线（fixed vol=64 MiB；`skip_desc_merge` 取默认 False）**

| descriptor 粒度 | n_desc（请求） | telemetry descCount | 中位耗时 ms | 带宽 GB/s |
|---|---:|---:|---:|---:|
| **16 KiB（vLLM 实际）** | 4096 | **1** | 175.12 | **0.3832** |
| 64 KiB | 1024 | 1 | 174.57 | 0.3844 |
| 256 KiB | 256 | 1 | 178.98 | 0.3750 |
| 1 MiB | 64 | 1 | 175.30 | 0.3828 |
| 4 MiB | 16 | 1 | 175.97 | 0.3814 |
| 16 MiB | 4 | 1 | 178.62 | 0.3757 |

→ 粒度放大 1000× 带宽变化 **−2.0%**（0.3832 → 0.3757）。**contiguous 布局下 telemetry 恒为 `descCount=1`：4096 个描述符被合并成了一个传输。**

**② 散列布局（描述符间隔一个 G，不可合并）**

| 粒度 | n_desc | 中位耗时 ms | 带宽 GB/s | 相对 contiguous |
|---|---:|---:|---:|---:|
| **16 KiB** | 4096 | 270.64 | **0.2480** | **−35.3%** |
| 64 KiB | 1024 | 195.73 | 0.3429 | −10.8% |
| 256 KiB | 256 | 181.42 | 0.3699 | −1.4% |
| 1 MiB | 64 | 180.17 | 0.3725 | −2.7% |
| 4 MiB | 16 | 178.91 | 0.3751 | −1.7% |
| 16 MiB | 4 | 174.37 | 0.3849 | +2.4% |

→ **散列惩罚只在细粒度出现**（16 KiB 时 −35%），粒度一旦 ≥256 KiB 就消失（此时描述符本身够大，固定开销占比已可忽略）。

**③ 合并开关（决定性对照，G=16 KiB）**

| 布局 | skip_desc_merge | 中位耗时 ms | 带宽 GB/s | telemetry descCount | µs/desc |
|---|---|---:|---:|---:|---:|
| contiguous | **False（默认=vLLM）** | **172.81** | **0.3883** | **1** | 42.2 |
| contiguous | True（关合并） | 251.04 | 0.2673 | 4096 | 61.3 |
| scattered | False | 262.90 | 0.2553 | 4096 | 64.2 |
| scattered | True | 259.86 | 0.2583 | 4096 | 63.5 |
| contiguous（1 MiB） | False / True | 178.89 / 179.58 | 0.3751 / 0.3737 | 1 / 64 | — |
| scattered（1 MiB） | False / True | 177.80 / 176.11 | 0.3774 / 0.3811 | 64 / 64 | — |

→ **合并收益 = 0.2673 → 0.3883 GB/s，即 +45.3%**；且合并只在细粒度 + 连续布局下起作用（1 MiB 粒度时开关无差别）。

**④ 尺寸扫描（G=16 KiB，merge 开）与 α/β**

| 传输总量 | n_desc | 中位耗时 ms | 带宽 GB/s |
|---:|---:|---:|---:|
| 8 MiB | 512 | 22.67 | 0.3701 |
| 32 MiB | 2048 | 87.35 | 0.3842 |
| 128 MiB | 8192 | 361.83 | 0.3709 |

→ 三点带宽一致（0.370–0.384，±2%），即 `t ∝ S`、**在合并后的层面 α ≈ 0（低于本实验分辨率），β ≈ 0.375 GB/s**。拟合残差 2% ≪ 20% 阈值，通过。

**⑤ UCX 传输层取证（`UCX_LOG_LEVEL=info`，UCX 1.21.0）**

```
ucp_context_0 intra-node cfg#1  rma_am(tcp/eth0)  amo_am(tcp/eth0)
                                device(cuda_ipc/cuda)  am(tcp/eth0 cma/memory cuda_ipc/cuda)  ka(tcp/eth0)
```

→ **RMA（即 KV 的 RDMA READ 路径）走的是 eth0 上的 TCP**；`cuda_ipc/cuda` 只出现在 `device(...)`（被识别但未被选为 RMA）。早前一轮还留下 UCX 警告原文：`{{rls|proto|init} get(multi) into cuda/GPU0 length 67108864 software emulation tcp/eth0 136.6 MB/s}`——GPU 缓冲**software emulation**。

**⑥ UCX_TLS 筛选（10 档，两步骤）**

| 档（UCX_TLS） | 能否初始化 | 选中的 RMA 传输 | 16 KiB contiguous 带宽 GB/s |
|---|---|---|---|
| **未设（基线）** | ✅ | `tcp/eth0` | **0.386** |
| `all` | ✅ | `tcp/eth0` | 0.373 |
| `cuda_copy,tcp` | ✅ | `tcp/eth0` | 0.381 |
| `cuda_ipc,cuda_copy,tcp` | ✅ | `tcp/eth0` | 0.361 |
| `shm` | ❌ `NIXL_ERR_BACKEND` | — | — |
| `cuda_copy,shm` | ❌ `NIXL_ERR_BACKEND` | — | — |
| `cuda_ipc,shm` | ❌ `NIXL_ERR_BACKEND` | — | — |
| `cuda_ipc,cuda_copy,shm` | ❌ `NIXL_ERR_BACKEND` | — | — |
| `sm,tcp` | ❌ `NIXL_ERR_BACKEND` | — | — |
| `cuda_ipc,tcp` | ❌ `NIXL_ERR_BACKEND` | — | — |

→ **凡是显式请求 `shm`/`sm`/`cuda_ipc` 的组合，NIXL 后端在 `createBackend` 阶段就失败**（`nixlBackendError: NIXL_ERR_BACKEND`）；能起来的组合（纯 TCP / TCP+cuda_copy / all）**一律仍选 `tcp/eth0`**，带宽 0.361–0.386（差 ±3%，在重复性噪声内）。**本机无法通过 UCX_TLS 把 KV 通路移出 TCP。**

## 6. 分析与结论

**【实测】① H1 成立：NIXL 默认合并描述符。** contiguous + 16 KiB + 默认参数下 telemetry `descCount=1`——4096 个描述符被合并成一次传输；把 `skip_desc_merge` 显式设为 `True` 才是 4096。**这意味着 vLLM（从不设该参数）本来就在享受合并**，任何"vLLM 应该去合并描述符"的改造建议都是多余的。

**【实测】② H2 被推翻：粒度不是 0.27 GB/s 的主因。** 按预注册的第二条推翻条件——"G 从 16 KiB 到 16 MiB 的带宽变化 <30%"——实测 **−2.0%**，条件成立。且 16 KiB（合并后）的带宽 0.383 GB/s 本身就高于 EXP-006 在 vLLM 里测到的 0.26–0.27。**"descriptor ≈16 KiB 碎片化小拷贝导致 α 主导"这个从 EXP-009 一路沿用到 EXP-020 附录 A 的解释链，就此失效。**

**【实测】③ 真正的天花板是 TCP 传输层，且不可调。** UCX 自报 RMA 恒为 `tcp/eth0`，GPU 缓冲走 software emulation；所有试图走 shm/cuda_ipc 的 TLS 组合都让后端初始化失败，能起来的组合仍选 TCP。**故本机 NIXL KV 通路的可达带宽约 0.38 GB/s，与描述符粒度无关。**

**【实测】④ 合并确实值 +45%，但天花板仍在传输层。** 0.267 → 0.388 GB/s。**这让 EXP-020 附录 A 的"合并收益上界 2.4–3.4×（到 0.60–0.91 GB/s）"作废**：0.60–0.91 是**裸 memcpy** 的单向带宽（EXP-002），而 NIXL/UCX 这条路径根本走不到那个传输层——它的 ceiling 是自己的 TCP 实现（≈0.38），不是链路能力。第 2 段的 UCX_TLS 筛选正是为此：不是没试过换传输，是换不了。

**【实测·关键交叉验证】⑤ vLLM 实际吃到的不是合并后的带宽。** EXP-013 逐请求测到的 **61.5–65.8 µs/descriptor**，与本次"**未合并**"（61.3）和"**散列**"（64.2）两档吻合，而与"合并后"的 **42.2 µs/desc** 明显不符；EXP-006 的 0.26–0.27 GB/s 也与我复现的未合并档 0.2673 几乎逐位相同（vs 合并档 0.3883）。**两条独立观测都指向：vLLM 的真实 KV 布局没能被 NIXL 合并。** 这正好解释了为什么 vLLM 的数字低于同布局的合并态。

**【推断】⑥ 为什么 vLLM 没能合并（待验证，见 §7）**：合并要求相邻地址连续。vLLM 的 descriptor 是"每 (region, block) 一个"（EXP-013 推出 region 数 = L×2 = 56），而在一个 region 内，一个请求的块若从空闲队列顺序分配，理论上是连续的 → 应当可合并。观测却不支持。候选：① 块在池中非连续（空闲队列被多个请求交替消费后碎片化）；② 跨 region 的边界打断了合并（每个 region 各自只能合并成 1 个 → 应得 56 个 descCount，实测却是 28672）；③ vLLM 的 `prep_xfer_dlist` 索引排列让 NIXL 看不到相邻性。**要定论必须给 vLLM 打点**（见 §7），本实验只能给出"它的数字落在未合并档"这一级证据。

**【实测】⑦ α/β 在合并后层面：α ≈ 0、β ≈ 0.375 GB/s。** 尺寸扫描三点带宽一致、`t ∝ S`，说明一旦合并，每传输固定开销已低到测不出；反过来说 **0.375 GB/s 就是这条 TCP 路径的渐近带宽**，加更多描述符或更大传输都不会更好。

## 7. 异常、偏差与开放问题

- **工具 bug 五连（全部登记在 §4 的作废前缀里）**：① `get_xfer_descs` 要 3 元组 `(addr,len,dev)`，我传了 4 元组（reg_descs 那条路径接受 4 元组，两者不同）；② `transfer()` 返回 `PROC`（投递成功、进行中）是**正常**值，我误判为错误；③ NIXL 遥测必须显式 `capture_telemetry=True`（**vLLM 开了，所以 EXP-006/013 的 telemetry 数字合法**）；④ **D 侧轮询与等 `P_ready` 都没有超时** → 被 `UCX_TLS=shm` 挂死 10 分钟；⑤ 汇总 CSV 里 TLS 值自带的逗号把列撑歪（已把逗号换成 `+`）。四条已修进脚本，两条写进 CORE 候选（见 §8）。
- **协议偏离**：无。两段实验的判据均在跑前写进 §1；UCX_TLS 这一段属**预注册之外的追加**（第一段发现 tcp 后才立项），其判定（"能否初始化 / 选中哪条传输"）在第一段结果出来后、第二段开跑前写进了驱动脚本的注释，未事后调整。
- **`iters` 不一致（如实登记）**：第一段主运行 iters=5；`exp026_tls_*` 各档 bench 为 iters=3（时间预算）。重复性用同配置的两次运行（0.3832 vs 0.386，−0.5%）与 C 组的 1 MiB 四点（0.3737–0.3811，±1%）交叠佐证。
- **`region ≥ 2×volume` 的约束**：scattered 布局需要间隔一个 G，故 region 必须放大一倍（128 MiB 供 64 MiB 传输）。这不影响对比（两种布局传的都是 64 MiB），但意味着 scattered 档的实际占用显存更大。
- **与生产的关键差异**：本实验是"一块连续注册区里的两种布局"，而 vLLM 是"56 个 region × 池内块分配"。**§6⑤ 的结论（vLLM 未吃到合并）是交叉验证级的，不是同构复现**；要变成实测结论需要 §7 的打点实验。
- **开放项（明确挂账）**：① **给 vLLM 的 NIXL connector 打点**，直接在真实 1P1D 里读出"请求的 descCount vs 合并后的 descCount"与每个 region 内的块连续性 → 定论 §6⑥，并量化"若布局连续能拿回多少"（按本实验 +45% 的量级）；② `UCX_TLS` 之外还有别的旋钮没扫：`UCX_RNDV_THRESH`、`UCX_MAX_RNDV_RAILS`、`UCX_PROTO_INFO`，以及 NIXL 侧的 `backends=[...]`（本实验只用 UCX）；③ 本机 `cuda_ipc`/`sm` 为何连初始化都失败（是否与 P2P 驱动级禁用直接相关）未深究；④ 未测 GIB/写入方向（本次全为 READ/pull）。
- **未触发回退**：两段实验都一次跑通（在修完工具 bug 之后）。

## 8. 下游影响

- **EXP-020 附录 A 的"合并 descriptor 收益上界 2.4–3.4×"须标注作废**（该推断的三个前提之一——"α 主导"——被本实验否定），改为"合并收益实测 +45%，且天花板是本机 UCX 的 TCP 路径 ≈0.38 GB/s"。本记录不改旧文件，由主线程整合。
- **EXP-006/011/013 的"descriptor 碎片化"措辞需降级**：从"碎片的 16 KiB 小拷贝"改为"**16 KiB 粒度本身不是瓶颈（实测 −2%）；vLLM 的布局未能被 NIXL 合并，其 per-descriptor 成本落在未合并档（61.3–64.2 µs，实测复现 EXP-013 的 61.5–65.8）**"。这是**归因的替换**，不是修补。
- **新增可复用证据**：本机 NIXL KV 通路的**可达上限 0.38 GB/s**、**α≈0**、**散列布局在 16 KiB 下的 −35% 惩罚**、**UCX_TLS 不可调（10 档表）**。这四条都能进 B4 报告与面试（"你怎么知道 0.27 不是描述符粒度造成的"——现在有直接实测回答了）。
- **文档同步**：`LEDGER.md` 证据台账 + 索引；`README.md` 索引 + 计数；`pd_disagg/REPORT.md` 与 `docs/TECH_DOC_vllm_engineering.md` 的 NIXL 章节（原"descriptor ≈16 KB 碎片化"的段落）。
- **红线**：本实验的 0.38 GB/s 是"**NIXL/UCX 在 TCP 路径上的可达带宽**"，不得与 EXP-002 的裸 memcpy 0.60–0.91 GB/s 或 NCCL 的 SHM/Socket 数字混用（三条路径不可互换，CORE §1.1）；仍只称 telemetry-derived effective throughput。
