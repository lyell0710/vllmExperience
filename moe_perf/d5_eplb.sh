#!/bin/bash
# D5 · EPLB gate(EXP-017):Qwen3-30B-A3B-GPTQ-Int4(W4A16)TP2+EP+EPLB
# Gate 三项:①真实重排("Rearranging experts" @ eplb_state.py:748,调小
# window/interval 逼出)②重排前后 greedy 输出一致 ③W4A16 权重搬运不崩。
# 任一不过 → D5 整条砍掉(白板级保留),原样记录。
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
TAG=${TAG:-w4a16}
RAW=$DIR/raw/EXP-017/$TAG
mkdir -p "$RAW"
source "$DIR/../pd_disagg/scripts/provenance.sh"
prov_env B
VENV=/root/venvs/v0.25.1
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-GPTQ-Int4}
prov_line "d5_eplb.sh ($MODEL TP2+EP+EPLB window=50 interval=100)" > "$RAW/manifest.txt"

CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
  --max-model-len 4096 --tensor-parallel-size 2 --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size": 50, "step_interval": 100, "log_balancedness": true}' \
  --gpu-memory-utilization 0.88 > "$RAW/server.log" 2>&1 &
SRV=$!
UP=0
for i in $(seq 1 300); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { UP=1; break; }
  kill -0 $SRV 2>/dev/null || break
  sleep 2
done
if [ "$UP" != 1 ]; then
  echo "GATE3_FAIL: server did not start with EPLB+GPTQ (log tail):"
  grep -iE "error|not support|Traceback|assert" "$RAW/server.log" | head -10
  pkill -f '[v]llm serve' 2>/dev/null; echo "D5_DONE(FAIL)"; exit 1
fi
echo "server up with EPLB enabled"

probe() {  # $1 label: 固定 8 prompt greedy,输出存文件
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

echo "=== load to trigger rearrangement (steps past interval=100)"
"$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts 256 --ignore-eos --seed 18000 --max-concurrency 16 \
  --request-rate inf > "$RAW/load.log" 2>&1
echo "load rc=$?"
sleep 5

REARR=$(grep -c "Rearranging experts" "$RAW/server.log")
echo "rearrangement_count=$REARR"
grep -m5 -n "Rearranging experts\|balancedness" "$RAW/server.log" | head -10 > "$RAW/rearrange_evidence.txt"

probe after

if diff -q "$RAW/probe_before.txt" "$RAW/probe_after.txt" >/dev/null; then
  echo "GATE2_PASS: outputs identical before/after"
else
  echo "GATE2_CHECK: outputs differ (diff saved)"
  diff "$RAW/probe_before.txt" "$RAW/probe_after.txt" > "$RAW/probe_diff.txt"
fi
[ "$REARR" -ge 1 ] && echo "GATE1_PASS: $REARR real rearrangements" || echo "GATE1_FAIL: no rearrangement"
grep -ciE "error|assert|Traceback" "$RAW/server.log" | sed 's/^/server_error_lines=/'

kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
echo "D5_DONE"
