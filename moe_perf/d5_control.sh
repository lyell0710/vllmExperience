#!/bin/bash
# D5 对照组:同 FP8 服务器、同负载,但 EPLB 关闭。
# 判据:若 probe 前后仍分歧 → token 级比对不是 EPLB 正确性的有效判据(负载/
# 批处理本身即引入非确定);若一致 → EPLB 运行的分歧归因于重排。
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-017/control_noeplb
mkdir -p "$RAW"
source "$DIR/../pd_disagg/scripts/provenance.sh"
prov_env B
VENV=/root/venvs/v0.25.1
MODEL=Qwen/Qwen3-30B-A3B-FP8
prov_line "d5_control.sh ($MODEL TP2+EP, EPLB OFF, same load as fp8 arm)" > "$RAW/manifest.txt"

CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
  --max-model-len 4096 --tensor-parallel-size 2 --enable-expert-parallel \
  --gpu-memory-utilization 0.88 > "$RAW/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 540); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; exit 1; }
  sleep 2
done
echo "control server up (EPLB off)"

probe() {
  local OUT=$RAW/probe_$1.txt
  : > "$OUT"
  for i in 0 1 2 3 4 5 6 7; do
    curl -s -X POST http://127.0.0.1:8100/v1/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"Q${i}: The capital of France is\",\"max_tokens\":48,\"temperature\":0,\"seed\":7}" \
      | "$VENV/bin/python" -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" >> "$OUT" 2>&1
  done
  echo "probe $1 saved"
}

probe before
"$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 256 --ignore-eos --seed 18000 --max-concurrency 16 \
  --request-rate inf > "$RAW/load.log" 2>&1
echo "load rc=$?"
sleep 5
probe after

if diff -q "$RAW/probe_before.txt" "$RAW/probe_after.txt" >/dev/null; then
  echo "CONTROL_IDENTICAL: no-EPLB outputs identical before/after load"
else
  diff "$RAW/probe_before.txt" "$RAW/probe_after.txt" > "$RAW/probe_diff.txt"
  echo "CONTROL_DIFFERS: no-EPLB outputs differ (token-level probe is NOT a valid EPLB gate)"
fi
kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
echo "D5_CONTROL_DONE"
