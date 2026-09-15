# EXP-023 · replica2@512 饱和复测（SAT_CONC=128）：EXP-007 的欠饱和疑点追认或修正

> **一句话结论**：**EXP-007 的 replica2@512 = 15.58 req/s 确系欠饱和**：同栈、同 seed（1099）、同 N=400，SAT_CONC 64→128 后饱和吞吐 **20.87 req/s**（+34%，远超锁定阈值 +5%），gate 通过、无失败请求。补测的 colocate@512 conc128（fresh 单实例，0 前缀命中）= **12.81 req/s**（原 conc64 10.36，+24%），**512 桶扩展效率按同并发口径修正为 20.87/12.81 = 1.63×**（原 1.50×；2048/8192 桶的 1.93/1.98× 未复测，不改）。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | ENV-B（`/root/venvs/v0.25.1`，vllm 0.25.1，sha 752a3a5044，带 EXT-1 本地 patch——非 PD 臂不触发）；Qwen2-7B-Instruct；2×RTX 4090 |
| 状态 | 完成（含 1 个预注册外补充点 + 1 个污染点登记，见 §7） |
| 关联清单项 | B1 / EXP-007《B1 四臂 offered-load 扫描战役》§5 脚注「512 replica2 疑受 SAT_CONC=64 上限影响」与 §7 开放项 |

## 1. 目的与假设

EXP-007 的 replica2@512 饱和吞吐 15.58 req/s（`20260821T1834_replica2_512x128_saturation`，SAT_CONC=64，N=400，seed 1099）被怀疑欠饱和：2048/8192 桶扩展效率 1.93/1.98×，512 桶只有 1.50×（15.58/10.36）。本实验按协议 v2 原样起 replica2 栈，只把客户端并发提到 128 复测同一点。

**假设（可证伪）**：SAT_CONC=64 时 replica2@512 欠饱和。

**跑前锁定的判定阈值（跑完不改）**：
- 指标：饱和吞吐 = completed / wall_time_s（与 EXP-007 §5 及 `make_fig7_overview.py` 同口径），gate（failed_requests=0）必须通过。
- **若 SAT_CONC=128 的饱和吞吐 > 15.58 × 1.05 = 16.36 req/s → 「原值确系欠饱和」**，512 桶扩展效率按新值报（新值/10.36）；
- **否则（≤16.36）→ 「原值追认」**，EXP-007 脚注改为"已复测，非欠饱和"。
- 附加读数（不参与判定）：TTFT/TPOT p50/p99 与并发 64 时的对比、GPU 遥测（功率帽/时钟）用于说明热工况可比性。
- 单次测量（与原点同为单次）；若 gate 失败（ServerDisconnected 等）同 seed 重跑一次，失败行保留在 runs.jsonl。
- 时间盒：起栈 + 跑完预计 ≤10 分钟；若超 45 分钟或 OOM/端口冲突 ≥2 次，停止并把现状写进 §7。

## 2. 环境与配置

- 服务（与 EXP-004/005/007 同配置）：
  - `CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2-7B-Instruct --port 8100 --max-model-len 16384`
  - `CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2-7B-Instruct --port 8200 --max-model-len 16384`
  - `python matrix/rr_proxy.py --port 8300 --backends 127.0.0.1:8100 127.0.0.1:8200`
- 客户端：`SEED=1099 NUM_PROMPTS=400 SAT_CONC=128 scripts/run_point.sh replica2 saturation 512 128 - 8300 8100 8200`（bench 打 8300，/metrics 直抓 8100/8200；随机数据集 512×128，ignore-eos，--save-result --save-detailed；GPU 2s 遥测）。seed 沿用 512 桶 saturation 的 x099 规则 = 1099（fresh server，无跨运行前缀缓存污染）。
- 硬件占用：双卡各一实例（fresh 栈，冷缓存；bench 前无 warmup，与原点同）。
- 落盘：`pd_disagg/results/b1_matrix/raw/<UTC>_replica2_512x128_saturation_{bench.json,bench.log,gpu.csv}`、`snapshots/<UTC>_..._{8100,8200}_{before,after}.prom`、`runs.jsonl` 追加一行（不改旧行）；服务/代理日志 `raw/<UTC>_replica2_conc128_server_{8100,8200}.log`、`raw/<UTC>_replica2_conc128_proxy.log`。
- 脚本：`pd_disagg/scripts/replica2_stack.sh`（起栈/等就绪/拆栈，本实验新增）。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv        # 双卡空闲
bash pd_disagg/scripts/replica2_stack.sh up <UTC>                    # 起 8100/8200/8300，等 /health 与 /v1/models 就绪
cd pd_disagg && SEED=1099 NUM_PROMPTS=400 SAT_CONC=128 scripts/run_point.sh replica2 saturation 512 128 - 8300 8100 8200
bash pd_disagg/scripts/replica2_stack.sh down                        # pkill -f '[v]llm serve' + 代理；/proc/*/exe 复核
```

## 4. 原始数据

全部在 `pd_disagg/results/b1_matrix/`：

| 测量点 | runs.jsonl run_id | raw / snapshots |
|---|---|---|
| **主点** replica2@512 conc128 | `20260915T0329_replica2_512x128_saturation`（第 125 行，gate_pass=true） | `raw/20260915T0329_replica2_512x128_saturation_{bench.json,bench.log,gpu.csv}`，`snapshots/20260915T0329_..._{8100,8200}_{before,after}.prom` |
| 污染点（不进 derived）colocate@512 conc128 于已服务过 seed 1099 的 8100 | `20260915T0330_colocate_512x128_saturation`（第 126 行，gate_pass=true 但**前缀缓存命中 18320/204800 tok = 8.9%**，见 `snapshots/20260915T0330_..._8100_{before,after}.prom` 的 `vllm:prefix_cache_hits_total` 0→18320） | `raw/20260915T0330_colocate_512x128_saturation_*` |
| **补充点** colocate@512 conc128 于 fresh 8100（0 命中） | `20260915T0333_colocate_512x128_saturation`（第 127 行，gate_pass=true，`prefix_cache_hits_total`=0） | `raw/20260915T0333_colocate_512x128_saturation_*`，`snapshots/20260915T0333_..._8100_{before,after}.prom` |
| 服务/代理日志 | — | `raw/20260915T0312_replica2_conc128_server_{8100,8200}.log`、`raw/20260915T0312_replica2_conc128_proxy.log`、`raw/20260915T0332_colocate_conc128_server_8100.log`（8100 重启实例） |

对照旧点：`20260821T1834_replica2_512x128_saturation`（conc64，15.58）、`20260821T1801_colocate_512x128_saturation`（conc64，seed 1099，10.36）。脚本：`pd_disagg/scripts/replica2_stack.sh`（新增）、`scripts/run_point.sh` / `collect_point.py`（未改）。

## 5. 结果

**饱和吞吐（req/s = completed / wall_time_s；512×128，N=400，seed 1099，request-rate inf）**

| 臂 | SAT_CONC | run_id | 吞吐 req/s | 输出 tok/s | TTFT p50/p90/p99 ms | TPOT p50/p99 ms | GPU 遥测（功率峰 W / 负载均值 SM MHz / 温度峰 °C / 节流原因） |
|---|---:|---|---:|---:|---|---|---|
| replica2 | 64（EXP-007 原点） | 20260821T1834 | 15.58 | 1994 | 611 / 887 / 1638 | 25.4 / 29.4 | 466 / 2631–2672 / 60–61 / SW Power Cap |
| **replica2** | **128** | **20260915T0329** | **20.87** | 2671 | 771 / 2327 / 3257 | 36.8 / 42.2 | 450 / 2598–2623 / 55 / SW Power Cap |
| colocate | 64（EXP-007 原点） | 20260821T1801 | 10.36 | — | 666 / — / — | 38.6 / — | — |
| colocate（污染，仅登记） | 128 | 20260915T0330 | (13.55) | (1735) | (1071 / 2881 / 4726) | (55.9 / 71.8) | 450 / 2444 / 63 / SW Power Cap |
| **colocate（fresh）** | **128** | **20260915T0333** | **12.81** | 1639 | 835 / 4626 / 6466 | 67.1 / 72.8 | 450 / 2461 / 63 / SW Power Cap |

派生：replica2 conc128 / conc64 = **+34.0%**；colocate conc128 / conc64 = **+23.6%**；**512 桶扩展效率（同 conc128 口径）= 20.87 / 12.81 = 1.63×**（原 conc64 口径 15.58/10.36 = 1.50×）。gate：三点 failed_requests=0。

## 6. 分析与结论

**【实测】① 判定：原值确系欠饱和**（20.87 > 16.36 阈值）。SAT_CONC=64 时 replica2 每实例只有 32 路并发，单实例 4090 上 Qwen2-7B 在 512×128 负载下 32 路远未填满 batch（原点 TPOT p50 25.4ms、TTFT p50 611ms 都显著低于 conc128 时的 36.8/771）——并发翻倍后吞吐 +34%，TPOT 只涨到 36.8ms（仍在 50ms SLO 内），说明 conc128 仍未触到 decode 天花板。

**【实测】② colocate 同样欠饱和**：conc64→128 吞吐 +24%，但 TPOT p50 已到 67ms（超 SLO 50ms），TTFT p90 4.6s——单实例在 conc128 是"吞吐更高但 SLO 全破"的过饱和态；而 replica2 在 conc128 仍 SLO 内。这正是复制臂的价值：同一客户端并发下每实例只承担一半。

**【推断】③ 扩展效率的正确口径**：EXP-007 §5 脚注怀疑的是 replica2 单点，实际两臂都欠饱和，且欠饱和程度不同（+34% vs +24%），所以扩展效率从 1.50× 升到 1.63× 而不是 2×。**1.63× 仍低于 2048/8192 桶的 1.93/1.98×**，剩余缺口的候选：(a) conc128 对 replica2 仍未饱和（TPOT 36.8ms < 50ms 有余量），真饱和需 conc≥192；(b) rr_proxy 单进程 uvicorn 转发在 20+ req/s、128 路流式下的开销（Peak concurrent requests 自报 182 > 128，提示客户端/代理侧排队）。两者都未在本实验区分——去向见 §7。

**【推断】④ 对 EXP-007 结论的影响**：「replica2 是吞吐王」方向不变、更强；「512 桶 1.50× 扩展」这一具体数字应替换为「≥1.63×（conc128，两臂同口径）」并注明 conc128 下 replica2 仍未封顶。

## 7. 异常、偏差与开放问题

- **起栈耗时 613s**（原 EXP-005 记录 ~70s）：v0.25.1 venv 与模型权重冷 page cache（机器长时间跑 ENV-C），两实例并行冷导入 IO 压力 avg10 37–42%；`replica2_stack.sh up` 的 300s 就绪上限先超时退出，进程 nohup 存活，改用 until 轮询等到就绪。热缓存后重启 8100 只需 60s。时间盒：03:12:49 起栈 → 03:37 全部跑完，25 分钟，未超 45 分钟。
- **预注册外的补充点**：colocate@512 conc128 不在 §1 计划内，为给扩展效率一个同口径分母而加测（fresh 实例、同 seed 1099、同 N）。作为补充数据标注，不改变 §1 判定（判定只用 replica2 20.87 vs 15.58）。
- **污染点登记**：`20260915T0330_colocate_512x128_saturation` 跑在已服务过 seed 1099 一半请求的 8100 上，`POST /reset_prefix_cache` 返回 404（该 API server 未暴露），前缀命中 8.9%（18320 tok）；行保留在 runs.jsonl（gate 字段本身为 true，因 gate 不检查前缀命中），**主线程整合 derived/figures 时需按 run_id 排除**。这是 EXP-007 §2 协议 v2「每点唯一 seed」教训的再现：同 seed 复用实例必须重启。
- **pkill 三重坑新形态**：`pkill -f '[v]llm serve ... --port 8100'` 与后续 `nohup vllm serve ... --port 8100` 写在同一条复合命令里，pattern 匹配到自身 shell 的 cmdline 字面量 → exit 144（方括号技巧只保护 pattern 本身，保护不了同一命令里的其他字面量）。规则：pkill 与含目标字面量的启动命令必须分两条命令。
- **H2 旁证尝试失败**：想借本栈做 EXP-020 的「并发负载压 SHM allreduce」旁证，nccl-tests 在两卡各余 ~900 MiB 下 exit_code=3（详见 EXP-020 附录 B），未得数据。
- 开放：conc≥192 的 replica2 真饱和点与 rr_proxy 开销的拆分（直连 8100+8200 各 conc64 vs 经代理 conc128）——去向：用户决定是否为 B4 报告补一次；本实验不再扩。

## 8. 下游影响

- **EXP-007 §5 表的 512 replica2 行**：15.58* 应改写为 20.87（指针 `runs.jsonl` run_id `20260915T0329_replica2_512x128_saturation`），脚注改为「conc64 欠饱和，conc128 复测 +34%；colocate conc128 12.81（run_id 20260915T0333）；扩展效率 1.63×（同口径）」——由主线程整合，本记录不改旧 records。
- **README / B4 / RESUME_EVIDENCE 中引用 512 桶扩展效率或 replica2 饱和吞吐的句子**需同步；2048/8192 桶数字不受影响。措辞红线：不能把 20.87 写成"饱和上限"（conc128 仍未封顶），只能写"≥20.87 req/s @conc128"。
- **fig7（四臂饱和吞吐总览）由 `make_fig7_overview.py` 从 runs.jsonl 重算**：它按 arm+桶取行，新增的三行（含污染行）会被纳入——主线程需在脚本里按 run_id 排除 `20260915T0330_*` 并决定 512 桶取 conc128 行还是保留 conc64 口径（建议改图注为 conc128）。
- 工装：新增 `pd_disagg/scripts/replica2_stack.sh`（起/拆栈，含端口占用与 OOM 检测、就绪轮询）；教训两条进 HANDOFF §6 候选：冷 venv 起栈按 10 分钟预算；pkill 与启动命令分条。
