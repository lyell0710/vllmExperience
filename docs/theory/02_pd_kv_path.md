---
status: draft
date: 2026-08-24
关联EXP: EXP-002, EXP-006, EXP-007, EXP-011, EXP-012, EXP-013
---

# PD 分离 KV 通路(vLLM NIXL pull,v0.25.1)

## 1. 一句话结论

NIXL pull 的 KV 通路 = 控制面显式身份三元组交接 + 数据面 descriptor RDMA READ;
在无 P2P 的 4090×2 上有效吞吐恒 0.26–0.27GB/s(EXP-007《B1 四臂 offered-load 扫描战役》,telemetry-derived),
D 端等待远端 KV 占 TTFT 54–64%(EXP-013《EXT-1 request 级 KV-wait 关联》因果占比)——该互联下 PD 全负载段不可取。

## 2. 机制(自己的话)

**控制面**:P 端 request_finished 时把 remote_block_ids / remote_engine_id /
remote_request_id / host / port / remote_num_tokens 作为 kv_transfer_params 显式
交出(pull_scheduler.py:265-275),随 P 响应回 proxy,proxy 原样塞进发给 D 的
请求;D 校验要素齐备后装进 RemoteMeta(metadata.py:152-158),请求进
WAITING_FOR_REMOTE_KVS。

**数据面**:一次性 handshake 交换 NixlAgentMetadata(engine_id、
kv_caches_base_addr、num_blocks 等,metadata.py:46-58);D 按 remote_block_ids
生成 descriptor 直接 RDMA READ(pull_worker.py:101-178);remote_request_id 只作
释放通知 key。

**记账**:D 调度先扣本地 prefix cache 命中,connector 只拉未命中块(远端块列表
尾对齐裁剪,base_worker.py:2165-2189);bytes 在 _pop_done_transfers 按 NIXL
telemetry totalBytes 录入(base_worker.py:2027-2072),失败不计字节。

**对照(旧架构为什么死)**:0.17.1 P2pNccl 依赖"两端独立推导同一
request_id#layer key"的隐式契约——InputProcessor 随机后缀
(input_processor.py:212)即 key 分叉:PUT 模式 D 无超时死等(engine:317),
GET 模式静默乱码;chunked prefill 靠 connector:433 assert 焊死。NIXL 的
remote_request_id 是 P randomize 之后亲口告知的真实 id,对分叉天然免疫
(EXP-012《vLLM 0.17.1 P2pNccl 两缺陷动态复现》实机坐实两 bug)。

## 3. 本项目实证(必须指自家 EXP 数字)

- **EXP-013**(request 级三段关联,~16 行本地可观测性 patch):KV 等待占 TTFT
  **54.2/62.5/64.2%**(512/2K/8K,p50);六段分解闭环误差 p50 <0.1%(最差桶
  0.084%);kv_wait−xferDuration = 0.3–1.9ms(瓶颈=传输本身,非轮询);逐请求
  bytes 求和 = Prometheus 分毫不差;36/36 身份双端匹配;首请求 kv_wait
  +292ms@512 = handshake 一次性成本的直接观测。
- **EXP-007**:NIXL 有效吞吐 0.26–0.27GB/s 恒定(~16KB/descriptor 碎片化);
  pd1p1d 饱和 req/s 仅为单卡 colocate 的 0.58–0.76 倍(7.84/2.12/0.54 vs
  10.36/3.63/0.90)。
- **EXP-011《EXT-2 NixlPush 单点》**(推方向对照):NixlPush 8K TTFT -6.7%、吞吐 +10–13%——方向
  改变不了量级,墙在互联(EXP-002《硬件三数》:单向 D2D 0.60–0.91GB/s,P2P 驱动级禁用)。
- **EXP-006《pd1p1d 指标探针 + 归因 + NIXL 大传输实测》 / analysis**(记账溯源):bytes 反解 245,344 token 与 ext_kv 计数器
  分毫不差;缺口 16,800 token 全为 D 端本地 prefix cache 命中,其中 511 块源码
  定罪于 bench 的 test 请求——触发 sweep 协议 v2(每点唯一 seed)。
- **EXP-012**:0.17.1 两 bug 实机复现(bug1 connector:433 原生 traceback,
  且实证修正裸 id 先崩 :518;bug2 D 整实例挂死行为学 + wchan 闭环)。

## 4. 面试追问 Q&A

- **Q:KV 占比为什么饱和在 ~64%?**
  A:P 段(prefill)与传输同为 O(输入长),比值趋常数;短输入时 ~40ms 固定开销
  (HTTP/调度)稀释占比(EXP-013 §6)。
- **Q:0.27GB/s 为什么远低于单向 D2D 0.6–0.91GB/s?**
  A:per-descriptor ~16KB 碎片化小拷贝,launch/同步开销主导;且这是
  telemetry-derived 有效吞吐,不能当链路物理带宽讲(红线表限定)。
- **Q:换推方向能救吗?**
  A:实测不能——EXP-011:8K TTFT -6.7%,量级不变。
- **Q:观测本身有扰动吗?**
  A:patch 前后 TTFT 218/727/2738 vs 219/719/2719ms,噪声内(EXP-013 无扰动证明)。
- **Q:prefix cache 会不会污染记账?**
  A:会——D 端命中块被扣除不传;EXP-006 已逐块归因缺口,协议 v2 每点唯一 seed。

## 5. 延伸(源码/论文,file:line)

- 身份显式交接:nixl/pull_scheduler.py:265-275;RemoteMeta 五字段
  metadata.py:152-158;协议注释 metadata.py:39("Add remote_request_id to
  kv_transfer_params")。
- RDMA READ 与释放通知:pull_worker.py:101-178、:183、:247。
- prefix-cache 裁剪 / bytes 记账:base_worker.py:2165-2189、2027-2072。
- 旧架构缺陷点(0.17.1):connector:433/:518、input_processor.py:212、
  engine:317——全 file:line 双版本核对见
  `pd_disagg/analysis/p2pnccl_bugs_id_chain.md`。
- 数据:`pd_disagg/ext1/derived/ext1_per_request.csv`、
  `pd_disagg/results/b1_matrix/runs.jsonl`;记录 EXP-013 §5。
