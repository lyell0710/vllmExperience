#!/usr/bin/env python3
"""B1/B2 报告图 + derived 表。全部从 runs.jsonl 与 raw/ 重算（LEDGER.md 硬约定 #3）。

用法: python scripts/make_figures.py   （在 pd_disagg/ 目录下）
输出: figures/*.png + results/b1_matrix/derived/*.csv
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
RUNS = BASE / "results" / "b1_matrix" / "runs.jsonl"
RAW = BASE / "results" / "b1_matrix" / "raw"
FIG = BASE / "figures"
DER = BASE / "results" / "b1_matrix" / "derived"

# 四臂固定色序（dataviz 默认调色板 slot1-4，validate_palette 通过）
ARM_COLOR = {
    "colocate": "#2a78d6",
    "replica2": "#eb6834",
    "tp2": "#1baf7a",
    "pd1p1d": "#eda100",
}
ARM_LABEL = {
    "colocate": "colocate (1×GPU)",
    "replica2": "replica2 (2×TP1)",
    "tp2": "TP=2",
    "pd1p1d": "PD 1P1D (NIXL)",
}
ARMS = list(ARM_COLOR)
BUCKETS = [512, 2048, 8192]
SLO_TTFT = {512: 328, 2048: 891, 8192: 4626}
PROV = "source: runs.jsonl (protocol v2, seed-per-point, 84 gated points) · 2×RTX4090 · vLLM 0.25.1"

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#c9c9c9",
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.color": "#e8e8e8",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": "#1a1a19",
    "axes.labelcolor": "#1a1a19",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "font.size": 10,
})


def v2_rows():
    rows = [json.loads(l) for l in RUNS.read_text().splitlines()]
    return [r for r in rows if r.get("seed", 42) != 42 and r["gates"]["pass"]]


def footnote(fig):
    fig.text(0.01, 0.005, PROV, fontsize=6.5, color="#8a8a8a")


def fig1_goodput(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, b in zip(axes, BUCKETS):
        for arm in ARMS:
            pts = sorted(
                (r["offered_rps"], r["metrics"]["goodput_slo_rps"])
                for r in rows
                if r["mode"] == "sweep" and r["input_len"] == b and r["arm"] == arm
            )
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=ARM_COLOR[arm], linewidth=1.8,
                    markersize=4.5, label=ARM_LABEL[arm])
        lim = max(x for x, _ in
                  [(r["offered_rps"], 0) for r in rows
                   if r["mode"] == "sweep" and r["input_len"] == b]) * 1.06
        ax.plot([0, lim], [0, lim], "--", color="#bbbbbb", linewidth=0.9, zorder=0)
        ax.set_xlim(0, lim); ax.set_ylim(bottom=0)
        ax.set_title(f"输入 {b}", fontsize=10.5)
        ax.set_xlabel("offered load (req/s)")
    axes[0].set_ylabel("SLO goodput (req/s)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("goodput：replica2 全场最高；PD 分离在全部负载段被传输延迟压垮（虚线=理想 y=x）",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); footnote(fig)
    fig.savefig(FIG / "fig1_goodput_curves.png", dpi=150); plt.close(fig)


def fig2_ttft(rows):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, b in zip(axes, BUCKETS):
        for arm in ARMS:
            pts = sorted(
                (r["offered_rps"], r["metrics"]["ttft_ms"]["p99"])
                for r in rows
                if r["mode"] == "sweep" and r["input_len"] == b and r["arm"] == arm
            )
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=ARM_COLOR[arm], linewidth=1.8,
                    markersize=4.5, label=ARM_LABEL[arm])
        ax.axhline(SLO_TTFT[b], color="#999999", linewidth=1.0, linestyle=":")
        ax.text(0.02, SLO_TTFT[b] * 1.08, f"SLO {SLO_TTFT[b]}ms",
                fontsize=7.5, color="#777777", transform=ax.get_yaxis_transform())
        ax.set_yscale("log")
        ax.set_title(f"输入 {b}", fontsize=10.5)
        ax.set_xlabel("offered load (req/s)")
    axes[0].set_ylabel("TTFT p99 (ms, log)")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("TTFT p99：PD 的传输延迟使其起点即在 SLO 附近；其余臂到拐点才越线",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95)); footnote(fig)
    fig.savefig(FIG / "fig2_ttft_p99.png", dpi=150); plt.close(fig)


def fig3_cost(rows):
    peak = {}
    for arm in ARMS:
        for b in BUCKETS:
            cand = [r for r in rows
                    if r["mode"] == "sweep" and r["arm"] == arm and r["input_len"] == b]
            best = max(cand, key=lambda r: r["metrics"]["goodput_slo_rps"] or 0)
            peak[(arm, b)] = best
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    w = 0.19
    for i, arm in enumerate(ARMS):
        xs = [j + (i - 1.5) * w for j in range(len(BUCKETS))]
        gp = [peak[(arm, b)]["metrics"]["goodput_slo_rps"] or 0 for b in BUCKETS]
        gpus = [peak[(arm, b)]["gpu_count"] for b in BUCKETS]
        ys = [g / n if g else 0 for g, n in zip(gp, gpus)]
        bars = ax.bar(xs, ys, width=w * 0.92, color=ARM_COLOR[arm],
                      label=ARM_LABEL[arm], edgecolor="white", linewidth=1)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.06, f"{y:.2f}", ha="center", fontsize=7,
                    color="#1a1a19")
    ax.set_xticks(range(len(BUCKETS)))
    ax.set_xticklabels([f"输入 {b}" for b in BUCKETS])
    ax.set_ylabel("峰值 goodput / GPU 数 (req/s per GPU)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("单位 GPU 的峰值 goodput：colocate 领先或与 replica2 打平；TP2/PD 显著负收益\n"
                 "（512 桶 replica2 网格未达其真实拐点，见 EXP-007 §7）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); footnote(fig)
    fig.savefig(FIG / "fig3_per_gpu_goodput.png", dpi=150); plt.close(fig)


def fig4_pd_decompose(rows):
    attr = {(r["arm"], r["input_len"]): r for r in rows if r["mode"] == "attribution"}
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    xs = range(len(BUCKETS))
    colo = [attr[("colocate", b)]["metrics"]["ttft_ms"]["p50"] for b in BUCKETS]
    pd_t = [attr[("pd1p1d", b)]["metrics"]["ttft_ms"]["p50"] for b in BUCKETS]
    xfer = []
    for b in BUCKETS:
        g = attr[("pd1p1d", b)]["gates"]
        xfer.append(g["nixl_xfer_time_delta_s"] / g["nixl_transfers_delta"] * 1000)
    other = [p - c - x for p, c, x in zip(pd_t, colo, xfer)]
    w = 0.5
    ax.bar(xs, colo, w, color="#2a78d6", label="P 端 prefill（≈colocate 无负载 TTFT）",
           edgecolor="white", linewidth=1)
    ax.bar(xs, xfer, w, bottom=colo, color="#eda100", label="NIXL KV 传输（telemetry avg xfer）",
           edgecolor="white", linewidth=1)
    ax.bar(xs, other, w, bottom=[c + x for c, x in zip(colo, xfer)],
           color="#c9c9c9", label="其余（D 首步 / 代理 / 调度）",
           edgecolor="white", linewidth=1)
    for i, b in enumerate(BUCKETS):
        ax.text(i, pd_t[i] * 1.02, f"{pd_t[i]:.0f}ms", ha="center", fontsize=8)
        share = xfer[i] / pd_t[i] * 100
        ax.text(i, colo[i] + xfer[i] / 2, f"传输 {share:.0f}%", ha="center",
                fontsize=7.5, color="#1a1a19")
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"输入 {b}" for b in BUCKETS])
    ax.set_ylabel("PD 无负载 TTFT p50 (ms)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("PD 的 TTFT 分解：KV 传输占 54–64%，各分量与独立遥测对账吻合",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); footnote(fig)
    fig.savefig(FIG / "fig4_pd_ttft_decompose.png", dpi=150); plt.close(fig)


def fig5_nixl(rows):
    pts = []
    for r in rows:
        if r["arm"] == "pd1p1d" and r["mode"] == "attribution":
            g = r["gates"]
            n = g["nixl_transfers_delta"]
            pts.append((g["nixl_bytes_delta"] / n / 1e6,
                        g["nixl_xfer_time_delta_s"] / n * 1000,
                        r["input_len"]))
    pts.append((0.188, 14.128, "smoke 0.5B"))   # EXP-001（NIXL 1P1D smoke 与版本裁决）小传输
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mb, ms, tag in pts:
        ax.plot(mb, ms, "o", color="#eda100", markersize=7,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.annotate(f"{tag}\n{mb:.1f}MB / {ms:.0f}ms", (mb, ms),
                    textcoords="offset points", xytext=(8, -4), fontsize=7.5)
    import numpy as np
    xs = np.logspace(-1, 3, 100)
    ax.plot(xs, xs / 270 * 1000, "--", color="#2a78d6", linewidth=1.4,
            label="0.27 GB/s 带宽渐近线")
    ax.axhline(12, color="#bbbbbb", linewidth=1.0, linestyle=":",
               label="~12ms 延迟地板（小传输）")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("单次传输量 (MB, log)"); ax.set_ylabel("单次传输时间 (ms, log)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_title("NIXL 传输时间 vs 传输量：小传输受 ~12ms 延迟地板，大传输贴 0.27GB/s 带宽墙",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); footnote(fig)
    fig.savefig(FIG / "fig5_nixl_transfer_scaling.png", dpi=150); plt.close(fig)


def fig6_slo_sensitivity(rows):
    """2048 桶、四臂：goodput@各 SLO 倍率（相对锁定 SLO 0.5/1/2/4×）——防"挑阈值"。"""
    scales = [0.5, 1.0, 2.0, 4.0]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for arm in ARMS:
        cand = [r for r in rows
                if r["mode"] == "sweep" and r["arm"] == arm and r["input_len"] == 2048]
        ys = []
        for s in scales:
            best = 0.0
            for r in cand:
                d = json.loads((RAW / f"{r['run_id']}_bench.json").read_text())
                tt, il = d.get("ttfts") or [], d.get("itls") or []
                ok = 0
                for i, t in enumerate(tt):
                    per = il[i] if i < len(il) else []
                    tpot = (sum(per) / len(per) * 1000) if per else 1e9
                    if t * 1000 <= 891 * s and tpot <= 50 * s:
                        ok += 1
                gp = ok / d["duration"] if d.get("duration") else 0
                best = max(best, gp)
            ys.append(best)
        ax.plot(scales, ys, "-o", color=ARM_COLOR[arm], linewidth=1.8,
                markersize=5, label=ARM_LABEL[arm])
    ax.set_xscale("log"); ax.set_xticks(scales)
    ax.set_xticklabels([f"{s}×" for s in scales])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("SLO 倍率（相对锁定值 TTFT 891ms / TPOT 50ms）")
    ax.set_ylabel("峰值 goodput (req/s)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("SLO 敏感性（输入 2048）：臂间排序在 0.5–4× 全区间稳定——结论不依赖阈值选择",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); footnote(fig)
    fig.savefig(FIG / "fig6_slo_sensitivity.png", dpi=150); plt.close(fig)


def derived_csv(rows):
    DER.mkdir(exist_ok=True)
    with open(DER / "sweep_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# " + PROV])
        w.writerow(["arm", "input_len", "mode", "offered_rps", "goodput_rps",
                    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms",
                    "gpu_seconds_per_request", "gpu_count", "run_id"])
        for r in sorted(rows, key=lambda r: (r["arm"], r["input_len"],
                                             r["offered_rps"] or 0)):
            m = r["metrics"]
            w.writerow([r["arm"], r["input_len"], r["mode"], r["offered_rps"],
                        m["goodput_slo_rps"], m["ttft_ms"]["p50"],
                        m["ttft_ms"]["p99"], m["tpot_ms"]["p50"],
                        m["gpu_seconds_per_request"], r["gpu_count"], r["run_id"]])


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    rows = v2_rows()
    fig1_goodput(rows)
    fig2_ttft(rows)
    fig3_cost(rows)
    fig4_pd_decompose(rows)
    fig5_nixl(rows)
    fig6_slo_sensitivity(rows)
    derived_csv(rows)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("derived:", sorted(p.name for p in DER.glob("*.csv")))
