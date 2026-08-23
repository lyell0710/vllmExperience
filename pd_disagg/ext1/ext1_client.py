# SPDX-License-Identifier: Apache-2.0
"""EXT1 sequential streaming client (concurrency 1, attribution mode).

Per request: unique X-Request-Id, unique random prompt (unique seed per
request => no prefix-cache pollution, protocol v2), streaming completions,
records client-side epochs (t_send / t_first_token / t_done) and TTFT.
Writes one JSONL row per request; first line is provenance.

Usage:
  python ext1_client.py --out raw/EXP-013/client.jsonl \
      --buckets 512 2048 8192 --num-per-bucket 12 --seed-base 13000
"""

import argparse
import json
import time
import uuid

import httpx
import numpy as np
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2-7B-Instruct"


def build_prompt(tokenizer, target_len: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    vocab = tokenizer.vocab_size
    ids = rng.integers(1000, vocab - 1000, size=target_len * 2).tolist()
    text = tokenizer.decode(ids, skip_special_tokens=True)
    ids2 = tokenizer.encode(text, add_special_tokens=False)[:target_len]
    return tokenizer.decode(ids2, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8192/v1/completions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--num-per-bucket", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed-base", type=int, default=13000)
    ap.add_argument("--provenance", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    out = open(args.out, "w")
    if args.provenance:
        out.write("# " + args.provenance + "\n")

    client = httpx.Client(timeout=None)
    n_total = 0
    for bucket in args.buckets:
        for i in range(args.num_per_bucket):
            seed = args.seed_base + bucket + i
            prompt = build_prompt(tokenizer, bucket, seed)
            rid = f"ext1-{bucket}-{i:03d}-{uuid.uuid4().hex[:6]}"
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "stream": True,
            }
            row = {
                "request_id": rid,
                "bucket": bucket,
                "idx": i,
                "seed": seed,
                "prompt_tokens_intended": bucket,
            }
            t_send = time.time()
            t_first = None
            n_chunks = 0
            try:
                with client.stream(
                    "POST", args.url, json=payload,
                    headers={"X-Request-Id": rid},
                ) as r:
                    r.raise_for_status()
                    for chunk in r.iter_bytes():
                        if chunk and t_first is None:
                            t_first = time.time()
                        n_chunks += 1
                t_done = time.time()
                row.update(
                    t_send=t_send,
                    t_first_token=t_first,
                    t_done=t_done,
                    ttft_ms=(t_first - t_send) * 1e3 if t_first else None,
                    total_ms=(t_done - t_send) * 1e3,
                    n_chunks=n_chunks,
                    ok=t_first is not None,
                )
            except Exception as e:
                row.update(ok=False, error=repr(e))
            out.write(json.dumps(row) + "\n")
            out.flush()
            n_total += 1
            print(
                f"[{n_total}] {rid} ttft={row.get('ttft_ms', 'ERR')}",
                flush=True,
            )
    out.close()


if __name__ == "__main__":
    main()
