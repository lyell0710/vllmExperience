#!/usr/bin/env python3
"""EXP-020 step4：解析 nccl 旋钮矩阵 raw → derived/<STAMP>_nccl_knob_matrix.csv + 按 §1 锁定阈值判定。
铁律 8：任何 size 点 / Avg 行 / PCIe 文件取不到即 raise，不回退默认值。
用法: python3 nccl_knob_matrix_analyze.py <STAMP>
"""
import csv, glob, os, re, sys
from datetime import datetime, timezone
from statistics import mean

HW = "/root/projects/vllm/experiments/pd_disagg/hw"
STAMP = sys.argv[1]
LO, HI, HIGH = 1.5, 2.1, 5.0            # §1 锁定：复现窗 [1.5,2.1]；≥5 = 明显偏离
ROW = re.compile(r"^\s+(\d+)\s+\d+\s+float\s+sum\s+-1\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\d+\s+", re.M)
AVG = re.compile(r"# Avg bus bandwidth\s*:\s*([\d.]+)")
VIA = re.compile(r"Channel 00(?:/0)? : 0\[\S+\] -> 1\[\S+\] (?:\[send\] )?via (\S+)")
SIZES = {1 << 20: "1M", 16 << 20: "16M", 256 << 20: "256M"}

def parse_perf(path):
    txt = open(path, errors="replace").read()
    pts = {}
    for m in ROW.finditer(txt):
        pts[int(m.group(1))] = float(m.group(4))          # out-of-place busbw
    for s in SIZES:
        if s not in pts:
            raise SystemExit(f"FATAL: {path} 缺 size={s} 的 busbw 行")
    plateau = [v for s, v in pts.items() if (16 << 20) <= s <= (256 << 20)]
    if len(plateau) != 5:
        raise SystemExit(f"FATAL: {path} 16M–256M 平台点数={len(plateau)}≠5")
    am = AVG.search(txt)
    if not am:
        raise SystemExit(f"FATAL: {path} 缺 'Avg bus bandwidth' 行")
    via = VIA.search(txt)
    if not via:
        raise SystemExit(f"FATAL: {path} 缺 'Channel 00/0 ... via' 自报传输行（需 NCCL_DEBUG=INFO SUBSYS=INIT）")
    if "# exit_code=0" not in txt:
        raise SystemExit(f"FATAL: {path} exit_code 非 0 或缺失")
    return pts, mean(plateau), float(am.group(1)), via.group(1)

def parse_pcie(path):
    rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#") and r[0].strip().isdigit()]
    if not rows:
        raise SystemExit(f"FATAL: {path} 无 PCIe 采样行")
    loaded = [r for r in rows if float(r[4].split()[0]) > 1000]     # SM clock >1 GHz = 负载中
    if not loaded:
        raise SystemExit(f"FATAL: {path} 无负载态样本（SM>1GHz）——采样未覆盖运行期")
    gens = [r[2].strip() for r in loaded]
    widths = {r[3].strip() for r in loaded}
    frac4 = sum(1 for g in gens if g == "4") / len(gens)
    smmax = max(float(r[4].split()[0]) for r in rows)
    return len(rows), len(loaded), frac4, "/".join(sorted(widths)), smmax

def knobs_of(tag):
    shm = p2p = proto = extra = ""
    m = re.match(r"shm(\d)_p2p(\w+)_proto(\w+)$", tag)
    if m:
        shm, p2p, proto = m.group(1), m.group(2), m.group(3)
    elif tag.startswith("extra_"):
        extra = {"extra_nchan1": "NCCL_MAX_NCHANNELS=1",
                 "extra_shmcudamemcpy1": "NCCL_SHM_USE_CUDA_MEMCPY=1",
                 "extra_algoTree": "NCCL_ALGO=Tree"}[tag]
        shm, p2p, proto = "0", "DEF", "DEF"
    else:
        raise SystemExit(f"FATAL: 未知 arm tag {tag}")
    return shm, p2p, proto, extra

files = sorted(glob.glob(f"{HW}/{STAMP}_nccl_knob_*.txt"))
if not files:
    raise SystemExit(f"FATAL: 找不到 {HW}/{STAMP}_nccl_knob_*.txt")
out_rows = []
for f in files:
    tag = os.path.basename(f)[len(STAMP) + len("_nccl_knob_"):-4]
    pcie = f[:-4] + "_pcie.csv"
    if not os.path.exists(pcie):
        raise SystemExit(f"FATAL: 缺 {pcie}")
    pts, plateau, avg, via = parse_perf(f)
    n_all, n_loaded, frac4, widths, smmax = parse_pcie(pcie)
    if LO <= plateau <= HI:
        verdict = "REPRO_1.78(A)"
    elif plateau >= HIGH:
        verdict = "HIGH>=5(B)"
    else:
        verdict = "OTHER(B)"
    shm, p2p, proto, extra = knobs_of(tag)
    out_rows.append(dict(
        arm=tag, NCCL_SHM_DISABLE=shm, NCCL_P2P_LEVEL=p2p, NCCL_PROTO=proto, extra_knob=extra,
        busbw_1M_GBps=pts[1 << 20], busbw_16M_GBps=pts[16 << 20], busbw_256M_GBps=pts[256 << 20],
        plateau_16M_256M_mean_GBps=round(plateau, 3), avg_busbw_GBps=avg,
        nccl_ch00_via=via, pcie_samples=n_all, pcie_loaded_samples=n_loaded,
        pcie_gen4_frac_loaded=round(frac4, 3), pcie_width_loaded=widths, sm_clock_max_MHz=int(smmax),
        verdict=verdict, raw_file=os.path.relpath(f, "/root/projects/vllm/experiments"),
    ))

os.makedirs(f"{HW}/derived", exist_ok=True)
outp = f"{HW}/derived/{STAMP}_nccl_knob_matrix.csv"
if os.path.exists(outp):
    raise SystemExit(f"FATAL: {outp} 已存在，拒绝覆盖")
with open(outp, "w", newline="") as fh:
    fh.write(f'# provenance: env=sys sha=717b683182 cmd="python3 scripts/nccl_knob_matrix_analyze.py {STAMP}" '
             f'date={datetime.now(timezone.utc).isoformat(timespec="seconds")} gpu="RTX 4090 x2" driver=610.57.04 '
             f'exp=EXP-020 source={STAMP}_nccl_knob_*.txt threshold="plateau(16M-256M mean) in [{LO},{HI}] => A; >={HIGH} => B"\n')
    w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
    w.writeheader(); w.writerows(out_rows)
print("derived ->", outp)
print(f"{'arm':28s} {'1M':>6s} {'16M':>6s} {'256M':>6s} {'plat':>6s} {'avg':>6s}  via                 gen4%  smmax  verdict")
for r in out_rows:
    print(f"{r['arm']:28s} {r['busbw_1M_GBps']:6.2f} {r['busbw_16M_GBps']:6.2f} {r['busbw_256M_GBps']:6.2f} "
          f"{r['plateau_16M_256M_mean_GBps']:6.2f} {r['avg_busbw_GBps']:6.2f}  {r['nccl_ch00_via']:18s} "
          f"{r['pcie_gen4_frac_loaded']*100:5.0f}  {r['sm_clock_max_MHz']:5d}  {r['verdict']}")
hits = [r for r in out_rows if r["verdict"].startswith("REPRO") and not r["extra_knob"]]
print("\n判定:", ("A 复现成功: " + ", ".join(h["arm"] for h in hits)) if hits else "B 主矩阵无档落入 [1.5,2.1]")
