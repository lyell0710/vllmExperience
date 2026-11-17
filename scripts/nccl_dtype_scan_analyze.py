#!/usr/bin/env python3
"""EXP-021：解析 dtype 扫描 raw → derived/<STAMP>_nccl_dtype_table.csv + 按 §1 锁定阈值判定。
铁律 8：缺 size 行 / Avg 行 / exit_code 即 raise。用法: python3 nccl_dtype_scan_analyze.py <STAMP>
"""
import csv, glob, os, re, sys
from datetime import datetime, timezone
from statistics import mean

HW = "/root/projects/vllm/experiments/pd_disagg/hw"
STAMP = sys.argv[1]
THR = 0.10
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+(\w+)\s+sum\s+-1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+", re.M)
DTYPES = ["float", "half", "bfloat16"]
PLATEAU = [s << 20 for s in (16, 32, 64, 128, 256)]

def parse(path):
    txt = open(path, errors="replace").read()
    if "# exit_code=0" not in txt:
        raise SystemExit(f"FATAL: {path} exit_code 非 0/缺失")
    if not re.search(r"# Avg bus bandwidth\s*:", txt):
        raise SystemExit(f"FATAL: {path} 缺 Avg 行")
    pts = {}
    for m in ROW.finditer(txt):
        pts[int(m.group(1))] = (float(m.group(4)), float(m.group(6)))   # (time_us, busbw)
    if not pts:
        raise SystemExit(f"FATAL: {path} 无数据行")
    return pts

data = {}   # (dtype, range, round) -> pts
for f in sorted(glob.glob(f"{HW}/{STAMP}_allreduce_size_scan_*_r*.txt")):
    m = re.search(r"_scan_(\w+)_(small|large)_r(\d)\.txt$", f)
    if not m:
        raise SystemExit(f"FATAL: 文件名不合规 {f}")
    data[(m.group(1), m.group(2), int(m.group(3)))] = parse(f)
for dt in DTYPES:
    for rg in ("small", "large"):
        for rd in (1, 2):
            if (dt, rg, rd) not in data:
                raise SystemExit(f"FATAL: 缺 {dt}/{rg}/r{rd}")

def need(pts, s, path_hint):
    if s not in pts:
        raise SystemExit(f"FATAL: {path_hint} 缺 size={s}")
    return pts[s]

rows = []
# 大消息：平台均值（16M–256M）逐轮 + 两轮均值
plat = {}
for dt in DTYPES:
    per_round = []
    for rd in (1, 2):
        pts = data[(dt, "large", rd)]
        per_round.append(mean(need(pts, s, f"{dt}/large/r{rd}")[1] for s in PLATEAU))
    plat[dt] = (per_round[0], per_round[1], mean(per_round))
# 小消息：8K time、1M time、地板（16B–8K time 均值）
small = {}
for dt in DTYPES:
    per = []
    for rd in (1, 2):
        pts = data[(dt, "small", rd)]
        floor = mean(need(pts, s, f"{dt}/small/r{rd}")[0] for s in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192))
        per.append((need(pts, 8192, dt)[0], need(pts, 1 << 20, dt)[0], floor))
    small[dt] = tuple(mean(x[i] for x in per) for i in range(3)) + (per,)

print(f"{'dtype':9s} {'plat_r1':>8s} {'plat_r2':>8s} {'plat_mean':>9s} {'Δ vs float':>10s} | {'t8K_us':>7s} {'t1M_us':>7s} {'floor_us':>8s} {'Δfloor':>7s}")
verdicts = []
for dt in DTYPES:
    p1, p2, pm = plat[dt]
    dplat = (pm - plat["float"][2]) / plat["float"][2]
    t8, t1m, fl, per = small[dt]
    dfl = (fl - small["float"][2]) / small["float"][2]
    d8 = (t8 - small["float"][0]) / small["float"][0]
    d1m = (t1m - small["float"][1]) / small["float"][1]
    ok = abs(dplat) < THR and abs(dfl) < THR and abs(d8) < THR and abs(d1m) < THR
    verdicts.append((dt, ok))
    rows.append(dict(dtype=dt, plateau_r1_GBps=round(p1, 3), plateau_r2_GBps=round(p2, 3), plateau_mean_GBps=round(pm, 3),
                     delta_plateau_vs_float_pct=round(dplat * 100, 2), round_spread_pct=round(abs(p1 - p2) / pm * 100, 2),
                     t8K_mean_us=round(t8, 2), delta_t8K_vs_float_pct=round(d8 * 100, 2),
                     t1M_mean_us=round(t1m, 2), delta_t1M_vs_float_pct=round(d1m * 100, 2),
                     floor_16B_8K_mean_us=round(fl, 2), delta_floor_vs_float_pct=round(dfl * 100, 2),
                     verdict_lt10pct=("PASS" if ok else "FAIL") if dt != "float" else "ref"))
    print(f"{dt:9s} {p1:8.2f} {p2:8.2f} {pm:9.2f} {dplat*100:+9.1f}% | {t8:7.2f} {t1m:7.1f} {fl:8.2f} {dfl*100:+6.1f}%")

# 逐 size 并排表（两轮均值 busbw / time）
os.makedirs(f"{HW}/derived", exist_ok=True)
outp = f"{HW}/derived/{STAMP}_nccl_dtype_table.csv"
if os.path.exists(outp):
    raise SystemExit(f"FATAL: {outp} 已存在")
sizes = sorted(set().union(*[set(data[(dt, rg, 1)]) for dt in DTYPES for rg in ("small", "large")]))
with open(outp, "w", newline="") as fh:
    fh.write(f'# provenance: env=sys sha=717b683182 cmd="python3 scripts/nccl_dtype_scan_analyze.py {STAMP}" '
             f'date={datetime.now(timezone.utc).isoformat(timespec="seconds")} gpu="RTX 4090 x2" driver=610.57.04 exp=EXP-021 '
             f'source={STAMP}_allreduce_size_scan_*_r*.txt threshold="|Δ|<{THR*100:.0f}% vs float on plateau(16M-256M)/t8K/t1M/floor(16B-8K)"\n')
    fh.write("# section=summary\n")
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    fh.write("# section=per_size (mean of r1,r2; small range uses n=100 file, large range n=20 file; 1M appears in both -> large)\n")
    fh.write("size_B," + ",".join(f"{dt}_time_us,{dt}_busbw_GBps" for dt in DTYPES) + "\n")
    for s in sizes:
        rg = "large" if s >= (1 << 20) else "small"
        cells = []
        for dt in DTYPES:
            vals = [data[(dt, rg, rd)][s] for rd in (1, 2) if s in data[(dt, rg, rd)]]
            if len(vals) != 2:
                raise SystemExit(f"FATAL: {dt}/{rg} size={s} 缺轮次")
            cells.append(f"{mean(v[0] for v in vals):.2f},{mean(v[1] for v in vals):.3f}")
        fh.write(f"{s}," + ",".join(cells) + "\n")
print("derived ->", outp)
print("判定:", ", ".join(f"{dt}={'成立' if ok else '不成立'}" for dt, ok in verdicts if dt != "float"))
