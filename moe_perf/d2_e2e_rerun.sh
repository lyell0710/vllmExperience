#!/bin/bash
# EXP-015（D2 MoE config 调优）补测:tuned e2e c32(+c128 复核)带 warmup,消除 Triton 首次 JIT 伪影
# (首跑 c32 的 TTFT p50 1021ms 被新 tile 配置的现场编译污染,时长多 ~2s)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-015
VENV=/root/venvs/main
MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat

CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
  --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel \
  --gpu-memory-utilization 0.88 > "$RAW/e2e_tuned_rerun_server.log" 2>&1 &
SRV=$!
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }
  sleep 2
done
echo "server up; warmup pass (完整并发梯度预热全部 tile 配置)"
"$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 64 \
  --num-prompts 96 --ignore-eos --seed 19990 --max-concurrency 32 \
  --request-rate inf > "$RAW/e2e_tuned_rerun_warmup.log" 2>&1
echo "warmup rc=$?"
for C in 32 128; do
  "$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
    --dataset-name random --random-input-len 128 --random-output-len 256 \
    --num-prompts $((C * 4)) --ignore-eos --seed $((16000 + C)) \
    --max-concurrency "$C" --request-rate inf \
    --save-result --save-detailed --result-dir "$RAW" \
    --result-filename "e2e_tuned_warm_c${C}.json" \
    > "$RAW/e2e_tuned_warm_c${C}.log" 2>&1
  echo "warm c=$C rc=$?"
done
kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
echo "E2E_RERUN_DONE"
