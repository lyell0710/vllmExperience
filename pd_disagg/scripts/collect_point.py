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


def exact_delta(deltas, metric_name, label_sub=None):
    """按精确指标名(跨引擎端口求和)取增量。名字来源: 2026-08-21 PD 探针实测,
    v0.25.1 传输计数在 D(consumer)端, P 端仅 failed/expired; _created 是时间戳需排除。"""
    total, found = 0.0, False
    for k, v in deltas.items():
        prom = k.split(":", 1)[1]          # 去掉 "port:" 前缀
        if prom.split("{")[0] != metric_name:
            continue
        if label_sub and label_sub not in prom:
            continue
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
    ap.add_argument("--gpu-csv", default=None)
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

    nixl_bytes = nixl_xfers = xfer_time_s = post_time_s = None
    descriptors = failed_xfers = failed_notifs = expired = ext_kv_tokens = None
    if is_pd:
        nixl_bytes = exact_delta(deltas, "vllm:nixl_bytes_transferred_sum")
        nixl_xfers = exact_delta(deltas, "vllm:nixl_bytes_transferred_count")
        xfer_time_s = exact_delta(deltas, "vllm:nixl_xfer_time_seconds_sum")
        post_time_s = exact_delta(deltas, "vllm:nixl_post_time_seconds_sum")
        descriptors = exact_delta(deltas, "vllm:nixl_num_descriptors_sum")
        failed_xfers = exact_delta(deltas, "vllm:nixl_num_failed_transfers_total")
        failed_notifs = exact_delta(deltas, "vllm:nixl_num_failed_notifications_total")
        expired = exact_delta(deltas, "vllm:nixl_num_kv_expired_reqs_total")
        ext_kv_tokens = exact_delta(
            deltas, "vllm:prompt_tokens_by_source_total",
            label_sub='source="external_kv_transfer"',
        )
        gate_pass = None
        if None not in (nixl_bytes, nixl_xfers, failed_xfers, failed_notifs, expired):
            gate_pass = (
                failed_requests == 0
                and nixl_bytes > 0
                and nixl_xfers == completed
                and failed_xfers == 0
                and failed_notifs == 0
                and expired == 0
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

    gpu_telemetry = None
    if args.gpu_csv and Path(args.gpu_csv).exists():
        per = {}
        for line in Path(args.gpu_csv).read_text().splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 5:
                continue
            idx = parts[0]
            try:
                temp = float(parts[1])
                sm = float(parts[2].split()[0])
                pw = float(parts[3].split()[0])
            except ValueError:
                continue
            d = per.setdefault(idx, {"temps": [], "sms": [], "pws": [], "reasons": set()})
            d["temps"].append(temp); d["sms"].append(sm); d["pws"].append(pw)
            d["reasons"].add(parts[4].split()[0])
        gpu_telemetry = {}
        for idx, d in per.items():
            loaded = [s for s, p_ in zip(d["sms"], d["pws"]) if p_ > 100]
            gpu_telemetry[idx] = {
                "samples": len(d["sms"]),
                "temp_max_c": max(d["temps"]),
                "power_max_w": max(d["pws"]),
                "sm_clock_min_loaded_mhz": min(loaded) if loaded else None,
                "sm_clock_mean_loaded_mhz": round(sum(loaded) / len(loaded), 0)
                if loaded else None,
                "throttle_reasons_seen": sorted(d["reasons"] - {"0x0000000000000000"}),
            }

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
        "gpu_telemetry": gpu_telemetry,
        "gates": {
            "kv_load_failure_policy": "fail",
            "log_stats_enabled": True,
            "metrics_scraped_direct": True,
            "nixl_bytes_delta": nixl_bytes,
            "nixl_transfers_delta": nixl_xfers,
            "transfers_expected": completed if is_pd else None,
            "nixl_xfer_time_delta_s": xfer_time_s,
            "nixl_post_time_delta_s": post_time_s,
            "nixl_descriptors_delta": descriptors,
            "external_kv_tokens_delta": ext_kv_tokens,
            "failed_transfers": failed_xfers,
            "failed_notifications": failed_notifs,
            "expired_reqs": expired,
            "kv_deltas_raw": {
                k: v for k, v in deltas.items()
                if v != 0 and "_bucket{" not in k and "_created" not in k
            } or None,
            "pass": gate_pass,
        },
        "snapshot_before": [
            f"snapshots/{args.prefix}_{p}_before.prom" for p in args.engine_ports
        ],
        "snapshot_after": [
            f"snapshots/{args.prefix}_{p}_after.prom" for p in args.engine_ports
        ],
        "raw": [f"raw/{args.prefix}_bench.json", f"raw/{args.prefix}_bench.log",
                f"raw/{args.prefix}_gpu.csv"],
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
