#!/usr/bin/env python3
"""EXP-022：解析大 M 3 轮交叉 A/B 日志 → mean±std、Δ、显著性判定（§1 锁定：|Δ|>2×合并std），落 CSV。
铁律 8：任一 (臂, M) 轮数 != 3 或缺 Kernel time 即中止。用法: d2_bigM_analyze.py <raw_dir>"""
import glob, os, re, statistics as st, sys
from datetime import datetime, timezone

d = sys.argv[1].rstrip("/")
stamp = os.path.basename(d).split("_", 1)[1]
pat = re.compile(r"Batch size: (\d+), config: .*?\nKernel time: ([\d.]+) us", re.S)
data = {}
files = sorted(glob.glob(os.path.join(d, "*_kernel_*.log")))
if len(files) != 12:
    raise SystemExit(f"FATAL: 期望 12 个 kernel 日志，实得 {len(files)}")
for f in files:
    m = re.search(r"_kernel_(ep|noep)_(default|tuned)_r(\d)\.log$", f)
    if not m:
        raise SystemExit(f"FATAL: 文件名不合规 {f}")
    mode, arm, _ = m.groups()
    txt = open(f, errors="ignore").read()
    hits = pat.findall(txt)
    if len(hits) != 4:
        raise SystemExit(f"FATAL: {f} 取到 {len(hits)} 个 Kernel time（期望 4）")
    for M, t in hits:
        data.setdefault((mode, arm, int(M)), []).append(float(t))

out = f"/root/projects/vllm/experiments/moe_perf/derived/{stamp}_exp022_bigM_ab.csv"
if os.path.exists(out):
    raise SystemExit(f"FATAL: {out} 已存在")
rows = []
for mode in ("ep", "noep"):
    label = "EP (E=30,N=1408)" if mode == "ep" else "非 EP (E=60,N=704)"
    print(f"\n### {label} —— 3 轮交叉次序，单位 us，mean±std\n")
    print("| M | default | tuned | Δ | 合并 std | 2×合并 std | 判定 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for M in (512, 1024, 2048, 4096):
        dft, tun = data.get((mode, "default", M)), data.get((mode, "tuned", M))
        if not dft or not tun or len(dft) != 3 or len(tun) != 3:
            raise SystemExit(f"FATAL: {mode}/M={M} 轮数 default={len(dft or [])} tuned={len(tun or [])}≠3")
        md, mt = st.mean(dft), st.mean(tun)
        sd, stu = st.stdev(dft), st.stdev(tun)
        pooled = (sd ** 2 + stu ** 2) ** 0.5
        diff = mt - md
        delta = diff / md * 100
        sig = abs(diff) > 2 * pooled
        if sig and delta <= -2: v = "保持收益"
        elif sig and delta < 0: v = "收益缩水"
        elif sig: v = "回退(tuned更慢)"
        else: v = "打平"
        print(f"| {M} | {md:.1f}±{sd:.1f} | {mt:.1f}±{stu:.1f} | {delta:+.2f}% | {pooled:.2f} | {2*pooled:.2f} | {v} |")
        rows.append((mode, M, md, sd, mt, stu, delta, pooled, sig, v, dft, tun))
keep = sum(1 for r in rows if r[9] == "保持收益")
print(f"\n判定：「保持收益」{keep}/8 → 假设{'成立' if keep >= 6 else '不成立'}（阈值 ≥6/8）")
with open(out, "w") as fh:
    fh.write(f'# provenance: env=venvs/main sha=3805e40e17 cmd="python3 moe_perf/d2_bigM_analyze.py {d}" '
             f'date={datetime.now(timezone.utc).isoformat(timespec="seconds")} gpu="2xRTX 4090" driver=610.57.04 exp=EXP-022 '
             f'threshold="significant iff |tuned-default| > 2*sqrt(sd_d^2+sd_t^2); keep iff significant and delta<=-2%"\n')
    fh.write("mode,M,default_mean_us,default_std_us,tuned_mean_us,tuned_std_us,delta_pct,pooled_std_us,significant,verdict,default_r1_r2_r3,tuned_r1_r2_r3\n")
    for mode, M, md, sd, mt, stu, delta, pooled, sig, v, dft, tun in rows:
        fh.write(f"{mode},{M},{md:.2f},{sd:.2f},{mt:.2f},{stu:.2f},{delta:.2f},{pooled:.2f},{int(sig)},{v},"
                 f"{'/'.join(f'{x:.1f}' for x in dft)},{'/'.join(f'{x:.1f}' for x in tun)}\n")
print("derived ->", out)
