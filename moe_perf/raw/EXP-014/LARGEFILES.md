# 本目录的大文件处置（GitHub 单文件 100MB 硬限）

| 文件 | 大小 | sha256 | 说明 | 处置 |
|---|---|---|---|---|
| `d1_nsys_moe_bs1.nsys-rep` | 159 MiB (167,170,554 B) | ebffc130f8f51316e3b3b73389e6ff8fa0fb667911de7e91add42f579c1e5693 | 原始采集（node 级，20s 稳态窗，bs=1）；**唯一不可再生项** | 原始本体仅本机（gitignore）；**已 xz 压缩入库**为 `d1_nsys_moe_bs1.nsys-rep.xz`（66.9 MB，比率 0.400，`xz -t` 通过），随仓保全 |
| `d1_nsys_moe_bs1.sqlite` | 565 MiB | 5475810d907f07811d45debe686b96b0e5cf42facc9628319f00957ab0b38fe2 | 衍生品：`nsys stats --force-export` 可从 .nsys-rep 再生 | 仅本机（gitignore），无需保全 |
| `d1_nsys_moe_bs32.sqlite` | 148 MiB | 891ff2f64037c771588010a62d2316029c7f3ebae7ae57e40548636071d9b7fe | 同上（bs32 的 .nsys-rep 44MB 已直接入库） | 仅本机（gitignore），无需保全 |
| `d1_nsys_moe_bs1_graphlevel.sqlite` | 67 MiB | — | 同上（graphlevel 变体；其 .nsys-rep 25MB 已入库） | 仅本机（gitignore），无需保全 |
| `d1_nsys_moe_bs32_graphlevel.sqlite` | 15 MiB | — | 同上 | 仅本机（gitignore），无需保全 |

## 解压还原（从入库的 .xz 拿回可用的 .nsys-rep）

```bash
cd moe_perf/raw/EXP-014
xz -dk d1_nsys_moe_bs1.nsys-rep.xz          # 得到 d1_nsys_moe_bs1.nsys-rep（167 MB）
sha256sum d1_nsys_moe_bs1.nsys-rep          # 应为 ebffc130f8f5…5693
# 重新导出 sqlite（若需要）：
nsys stats --force-export --report cuda_gpu_kern_sum d1_nsys_moe_bs1.nsys-rep
```

## 结论数据不受影响

kernel 占比表已固化为 `../../derived/d1_kernel_share_bs{1,32}.csv`，图见 `../../figures/d1_fig1_decode_scaling.png`；
原始 trace 只用于复核与再导出，不参与任何数字的日常引用。

## 保全策略注（2026-09-15）

因运行主机即将到期，这里的原则是：**不可再生的 artifact 一定要进 git（可 push 到远端），可再生的衍生品只留本机**。
`d1_nsys_moe_bs1.nsys-rep` 原先因 159 MB 超 GitHub 单文件 100 MB 硬限而只留本机；本次以 `xz -6`
压到 66.9 MB（比率 0.400）后入库，消除该不可逆风险。压缩档比在库内的 25 MB 常规上限大，属**刻意偏离**
（CORE 铁律 5 的 ≤25 MB 是防仓膨胀的常规值，此处以"保全唯一不可再生证据"为先）；如需回收体积，
删掉 `*.xz` 即可，其余均不受影响。
