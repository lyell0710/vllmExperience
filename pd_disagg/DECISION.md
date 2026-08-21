# 版本裁决(锁死后不再重开)

**问题**:四臂矩阵在 vLLM v0.25.1(release)还是 main@7aa248fc 上跑?

**规则**(定于 2026-08-21):两边各 ≤30 分钟 smoke,取三项齐活者;平手取 v0.25.1。
1. NIXL 1P1D 示例跑通(GPU0=P:5600, GPU1=D:5601, toy proxy)
2. 日志出现 `KV Transfer metrics:` 行(NIXL 计数器工作)
3. `kv_load_failure_policy=fail` 被接受(记录各版本默认值)

## smoke 结果(2026-08-21,原始记录 smoke/smoke_{v0.25.1,main}_result.txt)

| 检查项 | v0.25.1 | main@7aa248fc |
|---|---|---|
| 1P1D 跑通 | PASS (2/2 请求) | PASS (2/2 请求) |
| KV Transfer metrics | PASS (avg xfer 14.1ms) | PASS (avg xfer 14.7ms) |
| failure_policy=fail | PASS (且默认即 fail) | PASS (且默认即 fail) |
| 日志 ERROR 数 | 0 | 0 |

## 裁决

- **结果**:**v0.25.1 (release)**
- **理由**:三项平手,按规则平手取 release——报告可复现、读者有版本锚点;
  main 无独占测量件(metrics/failure_policy 两版行为一致)。
- **时间**:2026-08-21T09:15Z
- **锁定后规则**:矩阵全程用 ~/venvs/v0.25.1,provenance 行必须与之一致;
  main 环境保留用于 MoE tuning 开发,不参与矩阵;0.17.1 仅作 system-version 对照。

## provenance 行模板(所有结果文件第一行)

```
# provenance: sha=<git rev-parse HEAD 或 pypi 版本> version=<vllm.__version__> cmd="<完整命令>" kv_load_failure_policy=fail date=<ISO8601>
```

## 硬件基线(2026-08-21 实测,原始文件在 hw/,均带 provenance 行)

- nccl-tests all_reduce_perf -g 2(TP collective 路径参考,不代表 KV 通路):
  **avg bus bw 1.78 GB/s**,大消息(256M+)~1.85 GB/s → `hw/all_reduce_perf.txt`
- p2pBandwidthLatencyTest: P2P connectivity=0(GeForce 禁用,`topo -p2p r`=GNS);
  单向 D2D 0.60–0.91 GB/s(cudaMemcpyPeer 无 P2P 分段中转),**双向 22.6–22.8 GB/s**,
  GPU 间延迟 14.5–15.9 µs,本卡内 memcpy ~924 GB/s → `hw/p2p_bandwidth_latency.txt`
- PCIe: 双卡均 Gen4 x16(空闲降 Gen1),bus C1/E1;`topo -m` 在本容器不可用
  (hwloc 无 PU 信息),以 `topo -p2p r`+PCIe link 替代 → `hw/topo.txt`
- NIXL KV 通路初值(来自 R0-3 smoke,小传输、延迟主导,B1 时补大传输样本):
  avg 0.188 MB/transfer,avg xfer 14.1 ms,13.3 MB/s → `smoke/smoke_v0.25.1_result.txt`
