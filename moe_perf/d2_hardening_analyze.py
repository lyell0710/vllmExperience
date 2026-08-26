#!/usr/bin/env python3
"""解析 hardening 3 轮交叉 A/B 日志，输出 mean±std 与相对变化。"""
import re, sys, glob, os, statistics as st

d = sys.argv[1]
pat = re.compile(r"Batch size: (\d+), config: .*?\nKernel time: ([\d.]+) us", re.S)
data = {}                       # (mode, arm, M) -> [times]
for f in sorted(glob.glob(os.path.join(d, "*_kernel_*.log"))):
    m = re.search(r"_kernel_(ep|noep)_(default|tuned)_r(\d)\.log$", f)
    if not m: continue
    mode, arm, _ = m.groups()
    for M, t in pat.findall(open(f, errors="ignore").read()):
        data.setdefault((mode, arm, int(M)), []).append(float(t))

Ms = sorted({k[2] for k in data})
for mode in ("ep", "noep"):
    rows = []
    for M in Ms:
        dft = data.get((mode, "default", M), [])
        tun = data.get((mode, "tuned", M), [])
        if not dft or not tun: continue
        md, mt = st.mean(dft), st.mean(tun)
        sd = st.stdev(dft) if len(dft) > 1 else 0.0
        stu = st.stdev(tun) if len(tun) > 1 else 0.0
        delta = (mt - md) / md * 100
        rows.append((M, md, sd, mt, stu, delta, len(dft), len(tun)))
    if not rows: continue
    print(f"\n### {'EP (E=30,N=1408)' if mode=='ep' else '非 EP (E=60,N=704)'}  —— {rows[0][6]} 轮\n")
    print("| M | default (us) | tuned (us) | Δ |")
    print("|---|---:|---:|---:|")
    for M, md, sd, mt, stu, dl, *_ in rows:
        mark = "**" if abs(dl) >= 2.0 else ""
        print(f"| {M} | {md:.1f}±{sd:.1f} | {mt:.1f}±{stu:.1f} | {mark}{dl:+.1f}%{mark} |")
