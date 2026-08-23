# 本目录仅存本地的大文件(GitHub 单文件 100MB 硬限,pre-receive 拒收)

| 文件 | 大小 | sha256 | 说明 |
|---|---|---|---|
| d1_nsys_moe_bs1.nsys-rep | 159MB | ebffc130f8f51316e3b3b73389e6ff8fa0fb667911de7e91add42f579c1e5693 | 原始采集(node 级);唯一不可再生项,仅本机保存 |
| d1_nsys_moe_bs1.sqlite | 565MB | 5475810d907f07811d45debe686b96b0e5cf42facc9628319f00957ab0b38fe2 | 衍生品:`nsys stats --force-export` 可从 .nsys-rep 再生 |
| d1_nsys_moe_bs32.sqlite | 148MB | 891ff2f64037c771588010a62d2316029c7f3ebae7ae57e40548636071d9b7fe | 同上(bs32 的 .nsys-rep 44MB 已入库) |

结论数据不受影响:kernel 占比表已固化在 `../../derived/d1_kernel_share_bs{1,32}.csv`。
