# SPDX-License-Identifier: Apache-2.0
"""EXT1 analysis: join client / proxy / D-connector records per request.

Identity: client X-Request-Id (ext1-<bucket>-<idx>-<hex6>) == proxy request_id;
engine request ids embed it as substring (both P and D derive ids from the
header). All timestamps are host wall-clock epochs from one machine.

Per-request decomposition of client TTFT:
  pre_proxy   = t_recv(proxy)      - t_send(client)
  p_segment   = t_p_done           - t_p_send        (P prefill incl. HTTP)
  gap_p_to_d  = t_d_send           - t_p_done        (proxy kv param plumbing)
  d_pre_kv    = t0_epoch(D conn)   - t_d_send        (D http+queue+schedule)
  kv_wait     = done_epoch - t0_epoch  == kv_wait_ms (D waits for remote KV)
  post_kv     = t_first_token(cli) - done_epoch      (D first step + stream)

Outputs: derived/ext1_per_request.csv + markdown summary to stdout.
"""

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

RAW = Path(__file__).parent / "raw" / "EXP-013"
DERIVED = Path(__file__).parent / "derived"
DERIVED.mkdir(exist_ok=True)


def load_client():
    rows = {}
    for line in (RAW / "client.jsonl").read_text().splitlines():
        if line.startswith("#"):
            continue
        r = json.loads(line)
        if r.get("ok"):
            rows[r["request_id"]] = r
    return rows


def load_proxy():
    rows = {}
    for line in (RAW / "ext1_proxy_lines.txt").read_text().splitlines():
        if line.startswith("#") or "EXT1_PROXY " not in line:
            continue
        r = json.loads(line.split("EXT1_PROXY ", 1)[1])
        rows[r["request_id"]] = r
    return rows


KV_RE = re.compile(
    r"EXT1_KV req_id=(\S+) remote_request_id=(\S+) kv_wait_ms=([\d.]+) "
    r"t0_epoch=([\d.]+) done_epoch=([\d.]+) bytes=(\d+) xfer_us=(\d+) "
    r"post_us=(\d+) descs=(\d+) handles=(\d+)"
)


def load_kv():
    rows = []
    for line in (RAW / "ext1_kv_lines.txt").read_text().splitlines():
        m = KV_RE.search(line)
        if m:
            rows.append(
                dict(
                    req_id=m.group(1),
                    remote_request_id=m.group(2),
                    kv_wait_ms=float(m.group(3)),
                    t0_epoch=float(m.group(4)),
                    done_epoch=float(m.group(5)),
                    bytes=int(m.group(6)),
                    xfer_us=int(m.group(7)),
                    post_us=int(m.group(8)),
                    descs=int(m.group(9)),
                    handles=int(m.group(10)),
                )
            )
    return rows


def main():
    client = load_client()
    proxy = load_proxy()
    kv = load_kv()

    joined = []
    for rid, c in client.items():
        p = proxy.get(rid)
        matches = [k for k in kv if rid in k["req_id"]]
        if p is None or len(matches) != 1:
            print(f"WARN unmatched {rid}: proxy={p is not None} kv={len(matches)}",
                  file=sys.stderr)
            continue
        k = matches[0]
        row = dict(
            request_id=rid,
            bucket=c["bucket"],
            idx=c["idx"],
            ttft_ms=c["ttft_ms"],
            pre_proxy_ms=(p["t_recv"] - c["t_send"]) * 1e3,
            p_segment_ms=(p["t_p_done"] - p["t_p_send"]) * 1e3,
            gap_p_to_d_ms=(p["t_d_send"] - p["t_p_done"]) * 1e3,
            d_pre_kv_ms=(k["t0_epoch"] - p["t_d_send"]) * 1e3,
            kv_wait_ms=k["kv_wait_ms"],
            post_kv_ms=(c["t_first_token"] - k["done_epoch"]) * 1e3,
            kv_share_of_ttft=k["kv_wait_ms"] / c["ttft_ms"],
            bytes=k["bytes"],
            xfer_ms=k["xfer_us"] / 1e3,
            post_ms=k["post_us"] / 1e3,
            descs=k["descs"],
            handles=k["handles"],
            remote_request_id=k["remote_request_id"],
            identity_match=rid in k["req_id"] and rid in k["remote_request_id"],
        )
        row["sum_segments_ms"] = (
            row["pre_proxy_ms"] + row["p_segment_ms"] + row["gap_p_to_d_ms"]
            + row["d_pre_kv_ms"] + row["kv_wait_ms"] + row["post_kv_ms"]
        )
        joined.append(row)

    if not joined:
        print("no joined rows"); sys.exit(1)

    out = DERIVED / "ext1_per_request.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(joined[0].keys()))
        w.writeheader()
        w.writerows(joined)

    print(f"joined {len(joined)} requests -> {out}\n")
    print("| bucket | n | TTFT p50 ms | kv_wait p50 ms | KV share p50 | "
          "KV share p10-p90 | P seg p50 ms | post_kv p50 ms | closure err p50 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for bucket in sorted({r["bucket"] for r in joined}):
        rs = [r for r in joined if r["bucket"] == bucket and r["idx"] > 0]
        share = np.array([r["kv_share_of_ttft"] for r in rs])
        ttft = np.array([r["ttft_ms"] for r in rs])
        kvw = np.array([r["kv_wait_ms"] for r in rs])
        pseg = np.array([r["p_segment_ms"] for r in rs])
        postkv = np.array([r["post_kv_ms"] for r in rs])
        closure = np.array(
            [abs(r["sum_segments_ms"] - r["ttft_ms"]) / r["ttft_ms"] for r in rs]
        )
        print(
            f"| {bucket} | {len(rs)} | {np.percentile(ttft, 50):.1f} "
            f"| {np.percentile(kvw, 50):.1f} "
            f"| {np.percentile(share, 50) * 100:.1f}% "
            f"| {np.percentile(share, 10) * 100:.1f}–"
            f"{np.percentile(share, 90) * 100:.1f}% "
            f"| {np.percentile(pseg, 50):.1f} | {np.percentile(postkv, 50):.1f} "
            f"| {np.percentile(closure, 50) * 100:.2f}% |"
        )
    n_id = sum(1 for r in joined if r["identity_match"])
    print(f"\nidentity_match (rid in BOTH D req_id and remote_request_id): "
          f"{n_id}/{len(joined)}")
    warm = [r for r in joined if r["idx"] == 0]
    if warm:
        print(f"idx=0 rows (incl. handshake, excluded from stats): "
              f"{[(r['bucket'], round(r['kv_wait_ms'], 1)) for r in warm]}")


if __name__ == "__main__":
    main()
