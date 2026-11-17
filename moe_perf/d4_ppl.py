# SPDX-License-Identifier: Apache-2.0
"""D4 轻量精度:wikitext-2 PPL(vLLM offline, prompt_logprobs)。

用法: python d4_ppl.py <model> <out.json> [--tokens 40000]
方法: wikitext-2-raw-v1 test 拼接 → 按 3584-token 窗切块(相邻块 512 token
重叠仅用于条件,不计分)→ prompt_logprobs=1 取每 token NLL → PPL。
两 checkpoint 用完全相同的窗与计分 token 集合(由各自 tokenizer 切,报告标注)。
"""

import argparse
import json
import math
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

WINDOW = 2048
STRIDE = 1536  # 每窗新计分 token 数;前 512 作条件不计分


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--tokens", type=int, default=40000)
    args = ap.parse_args()

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(r["text"] for r in ds if r["text"].strip())

    llm = LLM(model=args.model, tensor_parallel_size=2,
              enable_expert_parallel=True, max_model_len=WINDOW + 8, max_num_batched_tokens=2048,
              gpu_memory_utilization=0.80, enforce_eager=True)
    tok = llm.get_tokenizer()
    ids = tok.encode(text)
    print(f"total corpus tokens: {len(ids)}")

    sp = SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0)
    nll, count = 0.0, 0
    pos = 0
    t0 = time.time()
    while count < args.tokens and pos + WINDOW <= len(ids):
        window = ids[pos:pos + WINDOW]
        (out,) = llm.generate(
            {"prompt_token_ids": window},
            sampling_params=sp,
            use_tqdm=False,
        )
        score_from = 0 if pos == 0 else WINDOW - STRIDE
        for i, lp in enumerate(out.prompt_logprobs):
            if lp is None or i <= score_from:
                continue
            obj = lp.get(window[i])
            if obj is None:
                continue
            nll -= obj.logprob
            count += 1
        pos += STRIDE
        print(f"scored {count} tokens, ppl so far {math.exp(nll / count):.4f}",
              flush=True)
    ppl = math.exp(nll / count)
    result = dict(model=args.model, ppl=round(ppl, 4), tokens_scored=count,
                  window=WINDOW, stride=STRIDE, seconds=round(time.time() - t0, 1))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT", json.dumps(result))


if __name__ == "__main__":
    main()
