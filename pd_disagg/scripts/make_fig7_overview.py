#!/usr/bin/env python3
"""fig7 · 四臂饱和吞吐总览（README 门面图）。全部从 runs.jsonl 重算（LEDGER.md 硬约定 #3）。

用法: /root/venvs/kernel-opt/bin/python scripts/make_fig7_overview.py
输出: figures/fig7_saturation_overview.png
数字与 LEDGER.md 证据台账 B1 行逐位一致（completed / wall_time_s，饱和模式）。
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt

font_manager.fontManager.addfont("/usr/share/fonts/truetype/arphic/uming.ttc")
plt.rcParams["font.family"] = font_manager.FontProperties(
    fname="/usr/share/fonts/truetype/arphic/uming.ttc"
).get_name()

BASE = Path(__file__).resolve().parent.parent
RUNS = BASE / "results" / "b1_matrix" / "runs.jsonl"
FIG = BASE / "figures"

# 四臂固定配色（与 make_figures.py fig1-6 一色到底，LEDGER.md 硬约定 #6）
ARM_COLOR = {
    "colocate": "#2a78d6",
    "replica2": "#eb6834",
    "tp2": "#1baf7a",
    "pd1p1d": "#eda100",
}
ARM_LABEL = {
    "replica2": "replica2（2×TP1 数据并行）",
    "tp2": "TP=2",
    "colocate": "colocate（单卡基线）",
    "pd1p1d": "PD 1P1D（NIXL）",
}
ORDER = ["replica2", "tp2", "colocate", "pd1p1d"]  # 上→下 = 全桶一致的名次序
BUCKETS = [512, 2048, 8192]
PROV = ("source: results/b1_matrix/runs.jsonl (saturation, protocol v2, seed-per-point)"
        " · 2×RTX4090 · vLLM 0.25.1")

plt.rcParams.update({
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


def saturation_rps():
    rows = [json.loads(l) for l in RUNS.read_text().splitlines()]
    out = {}
    for r in rows:
        if (r.get("mode") == "saturation" and r.get("seed", 42) != 42
                and r["gates"]["pass"] and r["arm"] in ORDER):
            out[(r["arm"], r["input_len"])] = r["completed"] / r["wall_time_s"]
    return out


def main():
    sat = saturation_rps()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ypos = list(range(len(ORDER)))[::-1]  # ORDER[0] 画在最上
    for i, (ax, b) in enumerate(zip(axes, BUCKETS)):
        vals = [sat[(a, b)] for a in ORDER]
        bars = ax.barh(ypos, vals, height=0.62,
                       color=[ARM_COLOR[a] for a in ORDER],
                       edgecolor="white", linewidth=1)
        ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9.5, color="#1a1a19")
        ax.set_yticks(ypos)
        ax.set_yticklabels([ARM_LABEL[a] for a in ORDER] if i == 0 else [])
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("饱和吞吐 (req/s)")
        ax.set_title(f"输入 {b}", fontsize=10.5)
        ax.grid(axis="y", visible=False)
    fig.suptitle("两张卡怎么用：数据并行（replica2）饱和吞吐三个输入桶全部最高，"
                 "PD 分离（NIXL）垫底", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.text(0.01, 0.005, PROV, fontsize=6.5, color="#8a8a8a")
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "fig7_saturation_overview.png", dpi=220)
    plt.close(fig)
    print("wrote", FIG / "fig7_saturation_overview.png")


if __name__ == "__main__":
    main()
