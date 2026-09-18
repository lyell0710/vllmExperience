# EXP-025 · replica2@512 真饱和点扫描（conc 128 / 192 / 256，同 N 同 seed）

> **一句话结论**：**判据 A 成立——conc128 未封顶，20.87 不是终值**：同 N=1200 同 seed 下 22.12 / 24.34 / 25.30 req/s（conc 128/192/256），conc192 比 conc128 高 **+10.06%**（> 5% 阈值），conc256 再高 +3.92%（> 2% 饱和阈值，**仍未见平台**）。但**吞吐峰 ≠ goodput 峰**：TPOT p50 在 conc192 就破 SLO 50 ms（53.03），TTFT p50 在 conc256 涨到 2444 ms，按锁定 SLO（TTFT≤328 / TPOT≤50）自算 goodput 为 **5.16 / 0.47 / 0.00 req/s**——**goodput 最优在 conc128、conc256 归零**。所以"512 桶 replica2 的饱和点"必须带口径回答：**吞吐口径 ≥ conc256（仍未封顶），goodput 口径 conc128 已过最优**。另：N 效应被实测确认（conc128@N=1200 = 22.12 vs @N=400 = 20.87，**+6.0%**），故 EXP-023 的 20.87 是 N=400 口径、不可直接与新值并比。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | ENV-B（`/root/venvs/v0.25.1`，vllm 0.25.1 sha 752a3a5044）；Qwen2-7B-Instruct；2×RTX 4090；driver 610.57.04 |
| 状态 | 完成（三档均一次成功；判据 A 成立；goodput 口径另行计算） |
| 关联清单项 | B1；EXP-023《replica2@512 饱和复测（SAT_CONC=128）》§7 开放项；EXP-024《512 桶四臂统一到 conc128 口径》§6⑥ |

## 1. 目的与假设

EXP-023 在 conc128 测得 replica2@512 = 20.87 req/s，但同档 TPOT p50 只有 36.8 ms（< SLO 50 ms），说明**可能仍未封顶**（EXP-024 §6⑥ 登记）。本实验向上扫并发，找真饱和点。

**假设（可证伪）**：replica2@512 的真饱和点 > conc128，即 20.87 不是终值。

**跑前锁定的判定阈值（跑完不改）**：

- 三个档：conc ∈ {128, 192, 256}，**同 N=1200、同 seed 1099、同臂同配置**，每档独立起栈（fresh，见下"防污染"）。
- **为什么必须重跑 conc128**：EXP-023 的 20.87 是 **N=400**（3.1 波）测的，而 N=1200 是 9.4 波。长跑的 ramp/尾效应占比更小、速率天然偏高，**直接拿 conc192@N=1200 比 20.87@N=400 会把 N 的效应误读成并发效应**。故本实验内的 conc128@N=1200 是唯一合法对照。
- **判饱和**：相邻档吞吐相对变化 ≤2% 即视为已进平台。
- **判定 A（仍欠饱和）**：conc192 相对本实验 conc128 提升 **>5%** → 未封顶，以 conc256 结果决定新饱和点，512 桶扩展效率按新值重报。
- **判定 B（conc128 即饱和）**：三档两两互差 **≤5%** 且 conc256 ≥ conc192 − 2% → 平台在 ≤128，`20.87` 追认为终值（在 N=1200 复核后），1.63× 为终值；剩余缺口归 rr_proxy 或客户端（去向 = 拆代理开销）。
- **判定 C（过载倒挂）**：conc256 < conc192 − 5% → 峰值在 conc192 附近，报告峰值档而非最大档。
- 每档要求：`failed_requests == 0`、gate 通过；**且必须复核 `prefix_cache_hits_total` 增量 == 0**（见下）。
- **防污染（本实验的硬约束）**：protocol v2 规定 512 桶 saturation 用 seed 1099，三点若共用同一实例会因**同一批 prompt** 命中前缀缓存（EXP-007 §2 的原始教训），吞吐虚高。故**每一档点前必须重启两个引擎实例**，起栈后核对 `prefix_cache_hits_total` 与 `prefix_cache_queries_total` 均为 0 才开跑；跑完再核对。任一档增量 >0 → 该档作废重跑。
- 时间盒 ≤45 分钟（三次起栈 + 三点；冷 page cache 时单次起栈可能 ~10 分钟）。超时或两次 OOM/端口冲突 → 停，写 §7。
- 附加读数（不进判定）：TTFT/TPOT p50/p99、GPU 遥测（功率帽/时钟）、扩展效率相对 colocate@512 的位置。

## 2. 环境与配置

- 栈（同 EXP-005/007/023）：GPU0:8100 + GPU1:8200 两个单卡实例（`--max-model-len 16384` 其余默认）+ `matrix/rr_proxy.py --port 8300 --backends 127.0.0.1:8100 127.0.0.1:8200`；起栈脚本 `pd_disagg/scripts/replica2_stack.sh up <前缀>`（EXP-023 新增）。
- 客户端：`SEED=1099 NUM_PROMPTS=1200 SAT_CONC=<128|192|256> scripts/run_point.sh replica2 saturation 512 128 - 8300 8100 8200`（bench 打 8300，/metrics 直抓 8100/8200，2s GPU 遥测）。
- **每档独立起栈**：起栈 → 跑点 → 拆栈 → 下一档。起栈日志各带前缀 provenance；不覆盖任何已有文件。
- 落盘：`results/b1_matrix/raw/<UTC>_replica2_512x128_saturation_*` + `snapshots/` + `runs.jsonl` 追加（不改旧行）；服务/代理日志 `<UTC>_replica2_conc<C>_server_{8100,8200}.log`、`_proxy.log`。

## 3. 步骤

```bash
nvidia-smi --query-compute-apps=pid,process_name --format=csv     # 双卡空闲
for C in 128 192 256; do
  P=$(date -u +%Y%m%dT%H%M)
  bash pd_disagg/scripts/replica2_stack.sh up "$P"
  cd pd_disagg && SEED=1099 NUM_PROMPTS=1200 SAT_CONC=$C \
    scripts/run_point.sh replica2 saturation 512 128 - 8300 8100 8200
  curl -s localhost:8100/metrics | grep '^vllm:prefix_cache_(queries|hits)_total'   # 必须为 0
  cd .. && bash pd_disagg/scripts/replica2_stack.sh down
done
```

## 4. 原始数据

全部在 `pd_disagg/results/b1_matrix/`，三档均 N=1200 / seed 1099 / gate_pass=true：

| 档 | runs.jsonl run_id | raw / snapshots |
|---|---|---|
| conc128 | `20260915T1031_replica2_512x128_saturation` | `raw/20260915T1031_replica2_512x128_saturation_{bench.json,bench.log,gpu.csv}`、`snapshots/20260915T1031_..._{8100,8200}_{before,after}.prom`、服务日志 `raw/20260915T1030_replica2_conc128_server_{8100,8200}.log` + `_proxy.log` |
| conc192 | `20260915T1034_replica2_512x128_saturation` | 同上，前缀 `20260915T1033`（服务）/`20260915T1034`（测量） |
| conc256 | `20260915T1036_replica2_512x128_saturation` | 同上，前缀 `20260915T1035`（服务）/`20260915T1036`（测量） |

goodput 不是手抄的：本记录直接由 `raw/<run_id>_..._bench.json` 的逐请求数组（`ttfts` + `itls`）现场重算，脚本口径写在 §5 表下。
**前缀缓存复核**：三档起栈后 `prefix_cache_queries_total` 与 `hits_total` 均为 **0**（每档独立起栈），跑完 `hits` 仍为 0（EXP-007 §2 的同 seed 污染在本实验中被逐档重启规避）。

## 5. 结果

**三档并发扫描（N=1200，seed 1099，`--request-rate inf`，512×128）**

| conc | 吞吐 req/s | 相对前一档 | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | GPU·s/req | wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | **22.118** | — | 2831.1 | 479.3 | 3151.9 | 39.37 | 48.84 | 0.090 | 54.26 |
| 192 | **24.343** | **+10.06%** | 3115.9 | 853.4 | 4802.9 | 53.03 | 60.35 | 0.082 | 49.30 |
| 256 | **25.297** | +3.92% | 3238.0 | 2443.7 | 8993.9 | 54.28 | 59.87 | 0.079 | 47.44 |

（单位 ms；吞吐 = completed / wall_time_s；三档 completed=1200 / failed=0 / gate 全通过。GPU 遥测：三档都触发 SW Power Cap 0x4，功率峰 456.6 / 450.6 / 450.6 W，负载均值 SM 2521–2432 MHz。）

**goodput（按锁定 SLO：TTFT ≤ 328 ms、TPOT ≤ 50 ms；从 bench.json 的 `ttfts` + `itls` 逐请求重算）**

| conc | goodput req/s | 达标率 | TTFT 超时数 | TPOT 超时数 |
|---:|---:|---:|---:|---:|
| 128 | **5.161** | 23.3% | 915 / 1200 | 5 / 1200 |
| 192 | 0.467 | 1.9% | 1124 / 1200 | **736 / 1200** |
| 256 | **0.000** | 0.0% | 1198 / 1200 | 1050 / 1200 |

口径说明：TPOT 逐请求取该请求 `itls` 的均值；两条件同时满足才计入 goodput；分母 = wall_time。**这三档都是 `--request-rate inf`（饱和档），不是 goodput 的搜寻网格**——本表 goodput 只能说明"固定饱和投放时并发越高 SLO 越差"，**不能**与 EXP-007 sweep 的 goodput 峰值（8.57–12.75 req/s）直接比较，那是在不同 offered load 下扫出来的。

## 6. 分析与结论

**【实测】① 判据 A 成立：conc128 未封顶。** conc192 比 conc128 高 10.06%（>5% 阈值），conc256 再高 3.92%。相邻档增幅从 +10.06% 收到 +3.92%，**趋势指向平台在 conc256 附近或略高，但按跑前锁定的 ≤2% 判饱和标准，三档都未进平台**。故 EXP-023/024 的 20.87 是"conc128@N=400"的值，**不是 replica2 的 512 桶上限**；`1.63×` 扩展效率也是 conc128 口径的值，不是上限。

**【实测】② N 效应被实测确认（这是本实验顺带抓到的方法学要点）**：conc128 在 N=1200 下是 22.12 req/s，而 EXP-023 的 conc128@N=400 是 20.87——**同并发、同 seed，仅 N 不同就差 +6.0%**（长跑 ramp/尾效应占比更小）。这验证了我在 §1 里预判的陷阱：**若拿 conc192@N=1200 直接比 20.87@N=400，会把 N 的效应误算成并发效应**。今后报该臂数字必须同时带 conc 与 N。

**【实测】③ 吞吐峰与 goodput 峰分离（本实验最重要的定性结论）。** 三个档都在饱和投放（`request-rate inf`）下：
- 吞吐单调升到 25.30 req/s（+14.4% vs conc128）；
- TPOT SLO 违规数从 5 → **736** → 1050（conc192 起崩），TTFT 违规从 915 → 1124 → 1198；
- goodput 因此从 5.16 → 0.47 → **0.00 req/s**。

即：**继续加并发只是把同一份算力换成更长的排队，吞吐的边际增益（+3.92%）全部由 SLO 合规性偿付。** 这与项目一贯的 SLO-goodput 口径一致（EXP-007 §4），也再次说明"饱和吞吐"与"可用吞吐"是两个不同的量——**对 512 桶 + 这套 SLO，replica2 的有效工作点不高于 conc128**。

**【推断】④ 对 SLO 本身的一个提醒（不是本实验的结论，供报告裁决）**：512 桶的 TTFT SLO = 328 ms 是按 **并发 1** 的 colocate 基线 × 5 换算的（EXP-004），在**饱和投放**下这个绝对阈值几乎必然被突破（conc128 起 TTFT p50 已 479 ms，超 1.46×）。所以上表低 goodput 主要是"SLO 为低并发标定、却在高并发下判分"的口径错配，而**不是** replica2 退化。**结论只取相对关系**（并发越高 SLO 越差、goodput 峰值在 ≤conc128），不对外引用绝对 goodput 值。

**【实测】⑤ 功率帽在全部三档都触发**（0x4，450+ W），与 EXP-023/024 的 replica2 记录一致；无新异常。

**【对 EXP-023/024 的影响】**：两记录里"conc128 仍未封顶"的推断被坐实并定量（至少到 conc256）；`1.63×` 与 `≥20.87 @conc128` 的措辞**保持**，但应补一句"@N=400"以标明口径，且知其上界未定。

## 7. 异常、偏差与开放问题

- **口径已锁、无协议偏离**：三档 N/seed/命令逐项一致（N=1200、seed 1099、`--request-rate inf`、`SAT_CONC` 分别 128/192/256），每档独立起栈并复核 prefix cache = 0。判定严格按 §1 预注册阈值（A/B/C 三条 + ≤2% 判饱和）执行，未事后调整。
- **我自己的一个方法学补记**：`SAT_CONC=256` 已超过客户端侧"波数"直觉（N=1200/256 ≈ 4.7 波），但 bench 的 `Peak concurrent requests` 实际未达 256（受服务端吞吐限制），所以真实并发是"投放上限"而非"实际并发"。这不影响相对趋势，但**"conc256"应理解为投放参数**，不是实际同时在飞的请求数。
- **未做（明确挂账）**：① conc384/512 是否还有增益（按 +3.92% 的收敛趋势，预期平台就在附近，未测）；② **colocate@512 的同 conc 对照**（要报 conc256 口径的扩展效率就必须补测，本次未做，故 §6① 只重申 conc128 口径的 1.63×）；③ goodput 的正规搜寻应由 offered-load sweep 做（EXP-007 已覆盖），本实验的 goodput 只作并发方向的相对读数。
- **`goodput_slo_rps` 字段为 null**：`run_point.sh` 在 saturation 模式不填该字段（EXP-007 是 sweep 模式才有）。本次由 bench.json 逐请求数组现场重算，脚本口径已写进 §5 表下；若今后要在 runs.jsonl 里直接读 goodput，需要给 `collect_point.py` 加 saturation 模式的计算分支（工装待办）。
- 无 OOM、无端口冲突、无 ServerDisconnected；时间盒实际 ~7 分钟（3 次起栈各 ~61 s + 三个点各 ~50 s），远低于 45 分钟上限。

## 8. 下游影响

- **EXP-023 / EXP-024 的两句措辞要补口径**：`≥20.87 @conc128` → `≥20.87 @conc128, N=400`；"conc128 仍未封顶"→"未封顶，至少到 conc256（25.30 req/s，N=1200）"。`1.63×` 保持（conc128 口径），并注明"该比值随 conc 变，不是 replica2 的上限"。
- **PR/报告层面可用的新论点**：**「吞吐峰 ≠ goodput 峰」**第一次有了同轴数据（同一臂、同 N、同 seed，只变 conc）：吞吐 +14.4% 的同时 goodput 归零。这比单纯报一个饱和吞吐数字更能体现 SLO 口径的必要性，适合放进 B4 报告的「测量方法论」一节与面试的 C11/H2 类追问。
- **fig7 不受影响**：该图用 512 桶 conc128 行（EXP-023/024 白名单），本实验的三行是同臂更高 conc 的**新增**行，不进图也不改白名单（白名单是唯一命中设计，多出的行不会自动被选中——这正是 EXP-024 §1 根治"后者覆盖"要达到的效果，本次是它的第一次实战验证）。
- **工装待办**：`collect_point.py` 增加 saturation 模式的 goodput 计算（需 SLO 绝对值 + 逐请求数组），使 `goodput_slo_rps` 不再为 null。
- 红线：本实验的 goodput 绝对值**不进任何对外文本**（口径错配，见 §6④）；对外只可用"并发升高时 SLO 违规数 5→736→1050"与"吞吐边际增益 +3.92% 被合规性偿付"这类相对陈述。
