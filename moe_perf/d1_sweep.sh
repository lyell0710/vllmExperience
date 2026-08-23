#!/bin/bash
# D1 · decode 吞吐-batch 曲线:MoE(TP2+EP) vs dense(TP2),同负载轴(EXP-014)
# 输入 128 / 输出 256(decode 主导),并发 1..128,每点唯一 seed(协议 v2)
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-014
mkdir -p "$RAW"
source "$DIR/../pd_disagg/scripts/provenance.sh"
prov_env B
VENV=/root/venvs/v0.25.1
CONCS="1 2 4 8 16 32 64 128"

run_model() {
  local LABEL=$1 MODEL=$2; shift 2
  local EXTRA_ARGS=("$@")
  echo "=== $LABEL: starting server"
  CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
    --max-model-len 8192 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.88 "${EXTRA_ARGS[@]}" \
    > "$RAW/d1_${LABEL}_server.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 240); do
    curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "$LABEL server died"; return 1; }
    sleep 2
  done
  curl -sf http://127.0.0.1:8100/health >/dev/null || { echo "$LABEL start timeout"; return 1; }
  echo "=== $LABEL: up"

  ( while true; do
      nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
        --format=csv,noheader >> "$RAW/d1_${LABEL}_gpu.csv"; sleep 2
    done ) & local SAMPLER=$!

  for C in $CONCS; do
    local NUM=$((C * 4)); [ "$NUM" -lt 16 ] && NUM=16
    echo "=== $LABEL conc=$C num=$NUM"
    "$VENV/bin/vllm" bench serve \
      --host localhost --port 8100 --model "$MODEL" \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$NUM" --ignore-eos --seed $((14000 + C)) \
      --max-concurrency "$C" --request-rate inf \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
      --save-result --save-detailed --result-dir "$RAW" \
      --result-filename "d1_${LABEL}_c${C}_bench.json" \
      > "$RAW/d1_${LABEL}_c${C}_bench.log" 2>&1
    echo "    rc=$?"
  done

  kill $SAMPLER 2>/dev/null
  kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
  echo "=== $LABEL: done"
}

prov_line "d1_sweep.sh (moe_tp2ep + dense_tp2, in128/out256, conc 1..128)" > "$RAW/d1_sweep_manifest.txt"

run_model moe_tp2ep Qwen/Qwen1.5-MoE-A2.7B-Chat --enable-expert-parallel
run_model dense_tp2 Qwen/Qwen2-7B-Instruct
echo "SWEEP_ALL_DONE"
