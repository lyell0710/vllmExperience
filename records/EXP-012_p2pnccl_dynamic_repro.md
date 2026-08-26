# EXP-012 · vLLM 0.17.1 P2pNccl 两缺陷动态复现（1P1D 实机）

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-23 |
| 环境 | ENV-A（/root/venvs/v0.17.1，vLLM 0.17.1 g95c0f928c）+ quart 0.22.0 |
| 状态 | 完成（bug1 精确命中：433；bug2 行为学+wchan 闭环，py-spy 栈帧因容器 ptrace 限制未取，已诚实标注） |
| 关联清单项 | R0-4（动态复现，升级 [analysis/p2pnccl_bugs_id_chain.md](../pd_disagg/analysis/p2pnccl_bugs_id_chain.md) 的静态定位）；解锁 B3 完整版前置；S2 简历句 |

## 1. 目的与假设
把 [EXP-012 静态源码分析](../pd_disagg/analysis/p2pnccl_bugs_id_chain.md) 定位的两个缺陷从"复现级机理" 升级为**实机动态复现**，坐实崩溃/挂死现场。可证伪假设：
- **H1（缺陷 1）**：P 实例上出现 `max_tokens>1` 的请求 → `p2p_nccl_connector.py:433` `assert req_id in self.chunked_prefill` 失败 → EngineCore 崩溃（整实例）。
- **H2（缺陷 2）**：PUT_ASYNC 默认模式下，经 proxy 的正常请求因 P/D 各自 InputProcessor 追加不同随机后缀 → connector key 分叉 → D 在 `p2p_nccl_engine.py:317` 无超时 `recv_store_cv.wait()` 死等 → D 整实例挂死、不自愈，而 P 无恙。

## 2. 环境与配置
- 硬件：2×RTX 4090（cuda：0=P，cuda：1=D），无 NVLink、P2P 驱动禁用。
- 栈：`pd_disagg/p2pnccl_repro/launch_1p1d.sh`（从 v0.17.1 tag 提取官方 xPyD proxy+脚本精简）。
  - proxy：`disagg_proxy_p2p_nccl_xpyd.py`（quart，http 10001 / zmq ROUTER 30001）。
  - P：Qwen2-7B-Instruct fp16，http 20003 / kv zmq 21001，`kv_role=kv_producer`， `kv_buffer_size=1e1`，`send_type=PUT_ASYNC`，enforce-eager，chunked prefill 默认开。
  - D：同模型，http 20005 / kv zmq 22001，`kv_role=kv_consumer`，`kv_buffer_size=8e9`，PUT_ASYNC。
- `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION` 未设（默认 False）→ 随机后缀默认开启（H2 前置条件成立）。

## 3. 步骤
1. `bash launch_1p1d.sh`；轮询 `/health` 至 P、D 均 200（首次 24s，重启后 54s）。
2. **bug2**：经 proxy `POST localhost:10001/v1/completions`（max_tokens=16，temperature=0）两次，每次带超时；观察客户端返回、D 端日志增量、D 线程 wchan、GPU util、P `/health`。
3. **bug1 路径 A（裸直连）**：`POST localhost:20003/v1/completions`（max_tokens=16，裸 request_id）。
4. **bug1 路径 B（精确命中）**：重启干净栈后，`POST localhost:20003` 带手工 `X-Request-Id=___prefill_addr_10.42.91.42:21001___decode_addr_10.42.91.42:22001_bug1craft001`
   + max_tokens=16 + 9-token prompt（单步 prefill）。

## 4. 原始数据
raw 主体在 `pd_disagg/p2pnccl_repro/raw/EXP-012/`；provenance 登记以目录级 `raw/EXP-012/manifest.txt` 为权威（8/24 补建；6 文件中 5 个自带首行 provenance， `20260823T024038Z_bug2_curl.txt` 无首行 provenance——由 manifest 统一登记，raw 本体不改）：
- `20260822T110420Z_preflight_state.txt`— 8/22 复现环境搭建期 preflight 快照。
- `20260823T023833Z_live_preflight.txt`— 复现前 GPU/端口状态。
- `20260823T024038Z_bug2_curl.txt`— 首次 bug2 复现尝试的 curl 退出码记录（内容仅 CURL_EXIT=0；provenance 见 manifest）。
- `20260823T062037Z_bug2_evidence.txt`— bug2 症状链 + wchan 全线程 futex_wait 快照 + 取证限制。
- `20260823T062224Z_bug1_Pdirect_crash.txt`— 路径 A 完整 traceback（connector:518 ValueError）。
- `20260823T062538Z_bug1_L433_assert.txt`— 路径 B 完整 traceback（connector:433 AssertionError）。
- 服务端全量日志 `repro_prefill.log` / `repro_decode.log`（在 `pd_disagg/p2pnccl_repro/` 根目录，非 raw/ 内——历史落位如实登记；NCCL 握手、崩溃/挂起原文）。
- **证据等级说明**：bug1 两条均有 EngineCore 原生 traceback（一级证据）；bug2 为行为学（双请求挂死、零 decode 日志、D 存活 util 0）+ 内核 wchan（全线程 futex_wait_queue）； py-spy 精确 Python 栈帧**未取**——容器 `ptrace_scope=1` 且 `/proc` 只读、无 CAP_SYS_PTRACE， gdb 未装。故 bug2 的：317 定位由"wchan + 行为学 + 静态 file：line"三方闭环，非直接栈帧。

## 5. 结果
| 触发路径 | 崩溃/挂起点 | 现象 | 实例结局 |
|---|---|---|---|
| 经 proxy 正常请求（bug2） | `p2p_nccl_engine.py:313-317` recv_store_cv.wait 无超时 | curl 空/挂死；D 零 decode 日志；D 全线程 futex_wait、util 0 | **D 挂死不自愈；P /health 始终 200** |
| 直连 P 裸 id（bug1 路径 A） | `p2p_nccl_connector.py:518` parse_request_id `ValueError` | 首步 prefill save_kv_layer 解析 peer 地址失败 | P EngineCore 崩溃，HTTP 500，/health 503 |
| 直连 P + addr 串 id + max_tokens>1（bug1 路径 B） | `p2p_nccl_connector.py:433` `AssertionError` | decode 步 build_connector_meta producer 分支 assert 失败 | P EngineCore 崩溃，HTTP 500，/health 503 |

## 6. 分析与结论
- **H1 成立**（路径 B）：与源码逐行吻合——9-token prompt 单步 prefill 走 `add_request`（L408）发送、不入 `chunked_prefill` dict；max_tokens>1 使下一步以 decode 进 `scheduled_cached_reqs` → L433 `assert req_id in self.chunked_prefill` 失败，EngineCore 主循环未捕获 → 整实例死。
- **H2 成立**（bug2）：PUT_ASYNC 下传输层握手成功（NCCL InitRank OK）、P 端 200 完成 prefill，但 D 侧 recv key 分叉 → D 引擎全线程 futex 停摆、util 0、连续两请求均挂死、P 始终健康，完全符合"单边 D 挂死、不自愈"的预测签名。
- **实证修正静态分析（新增价值）**：静态文档称"直接压测 P + max_tokens>1 →：433"。实测发现 **裸直连会先崩于 connector：518**（parse_request_id 需要 id 内嵌 peer 地址，裸 `cmpl-` id 没有），早于任何 decode 步，**根本走不到：433**。要精确命中：433 必须同时满足：id 含 `___prefill_addr..._decode_addr...___` 地址串 + max_tokens>1。三条路径（A 崩：518 / B 崩：433 / C 经 proxy 则 P 不崩、触发 D 侧 bug2）共同构成该连接器脆弱性的完整触发图景。

## 7. 异常、偏差与开放问题
- **协议偏离（已记录）**：bug1 路径 B 用手工 `X-Request-Id` 注入地址串——这是必要手段，因为官方 proxy 会把 P 的 max_tokens clamp 成 1（disagg_proxy L131），正常链路下 P 永不进 decode 步；手工 id 忠实模拟了"proxy 未 clamp / 探活请求携带合法 id"的场景。
- **取证限制**：py-spy 精确栈帧未取（容器 ptrace 限制），见 §4。若日后拿到特权环境可补一帧 `recv_store_cv.wait` 的 Python 栈作锦上添花，但不影响结论闭环。
- 开放问题：bug2 附带的 recv_store 内存单调泄漏（静态分析 engine：554-565）本次未量化，非必需，留待有需要时单独测。

## 8. 下游影响
- **措辞红线**：0.17 两 bug 现已达"**复现/定位/验证**"全链闭环（静态定位 + 动态崩溃/挂死现场 + 实证修正）。仍**禁**"发现/修复"（未提 PR、且该连接器 2026-06 已被 #44854 整体移除）。
- **解锁 B3 完整版（PD-vs-PD）前置**：0.17.1 P2pNccl 链路本次证明在默认配置下正常请求即触发 D 挂死 → 0.17.1 PD 无法作为可用对照臂；B3 跨版本 PD-vs-PD 结论应据此表述（0.17.1 PD 不可用 vs 0.25.1 NIXL 可用），而非跑吞吐对比。
- **S2 简历句/B4 报告**：可引用三条一级/闭环证据，面试口径（analysis 文档 §面试 2 分钟）得实测背书。
