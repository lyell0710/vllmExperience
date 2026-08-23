#!/usr/bin/env python3
"""D1 kernel 分解:nsys stats cuda_gpu_kern_sum → 分类占比表(EXP-014)。

用法: python d1_kernels.py <bs> [...]  # 对 raw/EXP-014/d1_nsys_moe_bs<bs>.nsys-rep
分类按 kernel 名正则,未命中的全部落 other 并打印 top 未识别项(不静默)。
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw" / "EXP-014"
DER = BASE / "derived"
DER.mkdir(exist_ok=True)

BUCKETS = [
    ("grouped GEMM (fused_moe)", re.compile(r"fused_moe_kernel", re.I)),
    ("moe_align_block_size", re.compile(r"moe_align_block_size|count_and_sort_expert", re.I)),
    ("routing (topk/softmax)", re.compile(r"topk|top_k|grouped_topk|moe.*softmax|softmax.*moe|sgl_moe", re.I)),
    ("permute/unpermute/moe_sum", re.compile(r"moe_permute|moe_unpermute|moe_sum|permute_cols|shuffle_rows|expandInput|scatter", re.I)),
    ("AllReduce (NCCL/custom)", re.compile(r"ncclDevKernel|AllReduce|cross_device_reduce|all_reduce|allreduce", re.I)),
    ("attention", re.compile(r"flash|fmha|attn|paged|reshape_and_cache|cache_kernel|kv_cache", re.I)),
    ("dense GEMM/GEMV (proj/shared-exp/lm_head)", re.compile(r"nvjet|cutlass|gemm|gemv|s16816|ampere_|Kernel2mma|splitKreduce|matmul", re.I)),
    ("norm/rope/act/elementwise", re.compile(r"rms_norm|rotary|act_and_mul|silu|elementwise|vectorized_elementwise|fused_add|CatArrayBatched|copy_|index_", re.I)),
]


def stats_csv(rep: Path) -> list[dict]:
    out = subprocess.run(
        ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
         "--force-export=true", str(rep)],
        capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Time"))
    return list(csv.DictReader(lines[start:]))


def classify(name: str) -> str:
    for label, rx in BUCKETS:
        if rx.search(name):
            return label
    return "other"


def main():
    for bs in sys.argv[1:]:
        rep = RAW / f"d1_nsys_moe_bs{bs}.nsys-rep"
        rows = stats_csv(rep)
        total = sum(float(r["Total Time (ns)"]) for r in rows)
        agg: dict[str, float] = {}
        unknown: list[tuple[float, str]] = []
        for r in rows:
            t = float(r["Total Time (ns)"])
            b = classify(r["Name"])
            agg[b] = agg.get(b, 0.0) + t
            if b == "other":
                unknown.append((t, r["Name"]))
        print(f"\n## bs={bs}  (GPU kernel wall-time 总计 {total/1e6:.1f} ms 采集窗)")
        print("| 分类 | 时间 ms | 占比 |")
        print("|---|---|---|")
        out_rows = []
        for b, t in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"| {b} | {t/1e6:.1f} | {t/total*100:.1f}% |")
            out_rows.append({"bucket": b, "time_ms": round(t / 1e6, 2),
                             "share_pct": round(t / total * 100, 2)})
        with open(DER / f"d1_kernel_share_bs{bs}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["bucket", "time_ms", "share_pct"])
            w.writeheader()
            w.writerows(out_rows)
        unknown.sort(reverse=True)
        if unknown:
            print("top other(前5,防静默):")
            for t, n in unknown[:5]:
                print(f"  {t/1e6:8.1f} ms  {n[:100]}")


if __name__ == "__main__":
    main()
