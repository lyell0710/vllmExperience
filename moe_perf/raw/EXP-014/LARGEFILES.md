# 本目录仅存本地的大文件(GitHub 单文件 100MB 硬限,pre-receive 拒收)

| 文件 | 大小 | sha256 | 说明 |
|---|---|---|---|
| d1_nsys_moe_bs1.nsys-rep | 159MB | 见 SHA256SUMS(哈希后台计算中,算完回填) | 原始采集(node 级);唯一不可再生项,仅本机保存 |
| d1_nsys_moe_bs1.sqlite | 565MB | 同上 | 衍生品:`nsys stats --force-export` 可从 .nsys-rep 再生 |
| d1_nsys_moe_bs32.sqlite | 148MB | 同上 | 同上(bs32 的 .nsys-rep 44MB 已入库) |

结论数据不受影响:kernel 占比表已固化在 `../../derived/d1_kernel_share_bs{1,32}.csv`。
