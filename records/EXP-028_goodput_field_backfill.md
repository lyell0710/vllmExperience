# EXP-028 · 工装：`goodput_slo_rps` 在饱和模式下不再为空（SLO 缺省表进 collect_point）

> **一句话结论**：`results/b1_matrix/runs.jsonl` 的 `goodput_slo_rps` 在**饱和模式下一直是 `null`**（EXP-025 §7 登记），根因是 `run_point.sh` 只在显式设了 `SLO_TTFT_MS`/`SLO_TPOT_MS` 环境变量时才把参数透给 `collect_point.py`；本次把 **SLO 锁定表搬进 `collect_point.py` 并按输入桶自动缺省**（显式传参仍优先），同时把 `make_figures.py` 里重复的那份表改为 import（单一事实源）。**验证：用新缺省重算 EXP-025 三点，goodput = 5.1609 / 0.4666 / 0.0000，与当时独立手算逐位相同**（达标数 280/23/0、总数 1200 亦相同）。行为变更只影响**此后**的写入；已有行不动（raw 不可变，其 goodput 随时可从 `bench.json` 重算）。

| 字段 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 环境 | ENV-B（`/root/venvs/v0.25.1`）；**本记录为纯重算，未占用 GPU** |
| 状态 | 完成（含与 EXP-025 的逐位对照验证） |
| 关联清单项 | B1 工装；EXP-025《replica2@512 真饱和点扫描》§7「`goodput_slo_rps` 字段为 null」；EXP-007《B1 四臂 offered-load 扫描战役》§4（SLO 锁定表） |

## 1. 目的与假设

**问题**：`runs.jsonl` 是 B1 的权威数据，但它的 `goodput_slo_rps` 只在 sweep 模式有值；饱和/归因点全是 `null`。这使"读者从 runs.jsonl 一句 SQL 就能拿到 goodput"做不到——每次都要回到 `bench.json` 手算（EXP-025 §5 就是这么做的）。

**改动（可验证的假设）**：把 SLO 表按输入桶做缺省后，`collect_point.py` 对同样的 `bench.json` 应给出**与独立手算完全相同**的 goodput。

## 2. 环境与配置

- 纯 Python 重算，无 GPU、无服务。
- 改动文件：`pd_disagg/scripts/collect_point.py`（SLO 缺省表 + `compute_goodput()` 抽出 + `goodput_detail` 字段 + `--dry-run`）、`pd_disagg/scripts/make_figures.py`（改为 import 同一张表）。

## 3. 步骤

```bash
# 用新缺省重算 EXP-025 三点（不传 --slo-*，与 run_point.sh 的实际行为一致）
python3 scripts/collect_point.py --prefix <前缀>_replica2_512x128_saturation \
    --arm replica2 --mode saturation --input-len 512 --output-len 128 --rps - \
    --gpu-count 2 --engine-ports 8100 8200 --gpu-csv <同前缀>_gpu.csv --seed 1099 --dry-run
```

## 4. 原始数据

- 验证输出：`pd_disagg/results/b1_matrix/raw/20260915T1252_exp028_goodput_backfill_verify.txt`（首行 provenance；含 EXP-025 §5 的手算值作为对照）。
- 被重算的输入：`raw/20260915T1031|1034|1036_replica2_512x128_saturation_bench.json`（EXP-025 的 raw）。

## 5. 结果

| 点（EXP-025） | 手算 goodput（§5） | 手算达标数 | `collect_point` 新缺省 goodput | 达标数/总数 | 一致 |
|---|---:|---:|---:|---|---|
| conc128（`20260915T1031`） | 5.1609 | 280/1200 | **5.1609** | 280/1200 | ✅ |
| conc192（`20260915T1034`） | 0.4666 | 23/1200 | **0.4666** | 23/1200 | ✅ |
| conc256（`20260915T1036`） | 0.0000 | 0/1200 | **0.0000** | 0/1200 | ✅ |

生效的 SLO（自动缺省）：`slo_ttft_ms=328`（512 桶）、`slo_tpot_ms=50.0`——与 `results/README.md` 锁定表一致。

## 6. 分析与结论

**【实测】三点逐位吻合**，说明新缺省路径与 EXP-025 当时的手算口径是同一个（TTFT 逐请求判、TPOT 取该请求 `itls` 均值、两条件同时满足才计入、分母 = wall_time）。**这条验证之所以重要**：EXP-025 的 goodput 是**手写脚本**算的，而从此以后由**入库工具**算——两者若不一致，就会变成"同一个指标两套数"的事故源；现在两者对齐，EXP-025 的数字可以放心引用。

**【实测】单一事实源已收拢**：SLO 表原本只在 `make_figures.py` 里（绘图用），现在移入 `collect_point.py`，`make_figures.py` 改为 `from collect_point import SLO_TTFT_MS_BY_BUCKET`。改动后 `make_figures.py` 的模块级导入实查通过（`SLO_TTFT = {512: 328, 2048: 891, 8192: 4626}`）。

**【实测】新增 `goodput_detail`** 字段（`slo_ttft_ms` / `slo_tpot_ms` / `ok` / `total` / `frac`），使每行的 goodput 都自带口径与达标率，不必回查锁定表。

**【推断】对既有数据的影响为零**：三行旧记录的 `goodput_slo_rps` 仍是 `null`，但 `goodput_detail` 的引入是**新增字段**（schema 扩展，非语义修改）；旧行缺该字段，读取方按"字段不存在"处理即可。**不回溯改写 `runs.jsonl`**——raw 不可变，且其值可从 `bench.json` 逐位重算（本记录即证明）。

## 7. 异常、偏差与开放问题

- **未做回溯填充（有意）**：改写已提交的 raw 行会违反铁律 3；若报告需要这些点的 goodput，用 `--dry-run` 重算（本记录给出命令）或按 run_id 从 `bench.json` 重算。
- **`--dry-run` 是新增的能力**，顺带解决了"验证工装改动不得不污染数据"的问题。
- **`run_point.sh` 未改**：它照旧只在显式设了环境变量时透传；缺省逻辑放在 `collect_point.py` 里，这样无论从 `run_point.sh` 还是手工调用都能拿到 goodput。
- **开放项**：① `goodput_detail` 需同步进 `results/README.md` 的 schema 说明（本记录 §8 列为待办）；② 若今后要按"多阈值敏感性"批量重算 goodput（fig6 那套），可以让 `compute_goodput` 复用同一 `frac` 口径，避免第三份实现。

## 8. 下游影响

- **`results/README.md` 的 schema 段需补 `goodput_detail`** 与"SLO 按桶自动缺省、显式传参优先"一句（**必做**，否则 schema 与产物不一致）。
- **EXP-025 §7 的工装待办关闭**（"若今后要在 runs.jsonl 里直接读 goodput，需要给 `collect_point.py` 加 saturation 模式的计算分支"）。
- **此后所有 B1 测量点**（含未来的复测）都会带 goodput 与口径，`fig7` 之外的按桶 goodput 分析不必再手算。
- **面试可用**：一个"指标两套实现 → 收拢成单一事实源并用历史独立计算做交叉验证"的小例子，正好对应 CORE 铁律 1（单一事实源）与"主张有据"。
