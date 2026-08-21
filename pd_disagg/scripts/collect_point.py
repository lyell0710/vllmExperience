#!/usr/bin/env python3
"""汇集单测量点 → 追加 results/b1_matrix/runs.jsonl 一行（schema 见 results/README.md）。

- 指标来自 bench 结果 JSON（--save-result --save-detailed 产物）
- gate 增量来自 before/after /metrics 快照（直抓引擎端口）
- goodput 仅在给定 --slo-* 时计算（sweep 阶段；SLO 锁定见 results/README.md）
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results" / "b1_matrix"
KV_PAT = re.compile(r"nixl|kv_transfer|kv_load|expired", re.I)
LINE_PAT = re.compile(r"^([^#\s].*?)\s+([0-9.eE+-]+)$")


def read_prom(path):
    vals = {}
    for line in Path(path).read_text().splitlines():
        m = LINE_PAT.match(line)
        if m and KV_PAT.search(m.group(1)):
            vals[m.group(1)] = float(m.group(2))
    return vals


def pick_delta(deltas, *substrings):
    total, found = 0.0, False
    for k, v in deltas.items():
        kl = k.lower()
        if all(s in kl for s in substrings):
            total += v
            found = True
    return total if found else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--output-len", type=int, required=True)
    ap.add_argument("--rps", default="-")
    ap.add_argument("--gpu-count", type=int, required=True)
    ap.add_argument("--engine-ports", nargs="+", required=True)
    ap.add_argument("--slo-ttft-ms", type=float, default=None)
    ap.add_argument("--slo-tpot-ms", type=float, default=None)
    ap.add_argument("--env-label", default="ENV-B")
    ap.add_argument("--sha", default="752a3a5044")
    args = ap.parse_args()

    bench_path = BASE / "raw" / f"{args.prefix}_bench.json"
    b = json.loads(bench_path.read_text())
    g = b.get

    deltas = {}
    for port in args.engine_ports:
        before = read_prom(BASE / "snapshots" / f"{args.prefix}_{port}_before.prom")
        after = read_prom(BASE / "snapshots" / f"{args.prefix}_{port}_after.prom")
        for k in after:
            deltas[f"{port}:{k}"] = after[k] - before.get(k, 0.0)

    is_pd = args.arm == "pd1p1d"
    completed = g("completed", 0)
    duration = g("duration", 0.0)
    failed_requests = len([e for e in (g("errors") or []) if e])

    nixl_bytes = pick_delta(deltas, "bytes") if is_pd else None
    nixl_xfers = pick_delta(deltas, "transfer", "count") if is_pd else None
    failed_xfers = pick_delta(deltas, "fail") if is_pd else None
    expired = pick_delta(deltas, "expired") if is_pd else None

    if is_pd:
        gate_pass = None  # PD 臂首跑需人工核对指标名后在本脚本固化判定
        if None not in (nixl_bytes, nixl_xfers, failed_xfers):
            gate_pass = (
                failed_requests == 0
                and nixl_bytes > 0
                and nixl_xfers == completed
                and failed_xfers == 0
                and (expired or 0) == 0
            )
    else:
        gate_pass = failed_requests == 0

    goodput = None
    if args.slo_ttft_ms is not None and args.slo_tpot_ms is not None:
        ttfts = g("ttfts") or []   # 单位: 秒(detailed 数组)
        itls = g("itls") or []
        ok = 0
        for i, t in enumerate(ttfts):
            per_itl = itls[i] if i < len(itls) else []
            tpot_ms = (sum(per_itl) / len(per_itl) * 1000) if per_itl else float("inf")
            if t * 1000 <= args.slo_ttft_ms and tpot_ms <= args.slo_tpot_ms:
                ok += 1
        goodput = round(ok / duration, 4) if duration else None

    def pct(metric):
        return {p: g(f"{p}_{metric}_ms") for p in ("p50", "p90", "p99")}

    row = {
        "run_id": args.prefix,
        "arm": args.arm,
        "mode": args.mode,
        "model": g("model_id", "Qwen/Qwen2-7B-Instruct"),
        "input_len": args.input_len,
        "output_len": args.output_len,
        "offered_rps": None if args.rps in ("-", "inf") else float(args.rps),
        "num_prompts": g("num_prompts", completed),
        "wall_time_s": round(duration, 3),
        "completed": completed,
        "failed_requests": failed_requests,
        "metrics": {
            "ttft_ms": pct("ttft"),
            "tpot_ms": pct("tpot"),
            "itl_ms": pct("itl"),
            "e2el_ms": pct("e2el"),
            "throughput_tok_s": g("total_token_throughput"),
            "output_tok_s": g("output_throughput"),
            "goodput_slo_rps": goodput,
            "gpu_seconds_per_request": round(args.gpu_count * duration / completed, 3)
            if completed else None,
        },
        "gpu_count": args.gpu_count,
        "gates": {
            "kv_load_failure_policy": "fail",
            "log_stats_enabled": True,
            "metrics_scraped_direct": True,
            "nixl_bytes_delta": nixl_bytes,
            "nixl_transfers_delta": nixl_xfers,
            "transfers_expected": completed if is_pd else None,
            "failed_transfers": failed_xfers,
            "expired_reqs_P": expired,
            "kv_deltas_raw": {k: v for k, v in deltas.items() if v != 0} or None,
            "pass": gate_pass,
        },
        "snapshot_before": [
            f"snapshots/{args.prefix}_{p}_before.prom" for p in args.engine_ports
        ],
        "snapshot_after": [
            f"snapshots/{args.prefix}_{p}_after.prom" for p in args.engine_ports
        ],
        "raw": [f"raw/{args.prefix}_bench.json", f"raw/{args.prefix}_bench.log"],
        "provenance": {
            "env": args.env_label,
            "sha": args.sha,
            "version": __import__("importlib.metadata", fromlist=["version"]).version("vllm"),
            "cmd": f"run_point.sh {args.arm} {args.mode} {args.input_len} "
                   f"{args.output_len} {args.rps} (bench 完整参数见 raw log)",
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": f"{args.gpu_count}x NVIDIA GeForce RTX 4090",
            "driver": "610.57.04",
        },
    }

    out = BASE / "runs.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[collect_point] appended {args.prefix} -> {out} (gate_pass={gate_pass})")


if __name__ == "__main__":
    main()
