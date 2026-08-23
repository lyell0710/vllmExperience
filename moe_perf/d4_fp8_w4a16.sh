#!/bin/bash
# D4 · FP8 vs W4A16 同卡对比(EXP-016):Qwen3-30B-A3B-FP8 vs -GPTQ-Int4
# 吞吐曲线(conc 1/32/128,in128/out256)+ attribution(512/128, conc1)
# FP8 on SM89(Ada):加载成败与所选 kernel 路径本身就是数据,失败原样记录
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-016
mkdir -p "$RAW"
source "$DIR/../pd_disagg/scripts/provenance.sh"
prov_env B
VENV=/root/venvs/v0.25.1
prov_line "d4_fp8_w4a16.sh (30B-A3B FP8 vs GPTQ-Int4, TP2+EP)" > "$RAW/manifest.txt"

run_arm() {  # $1 label, $2 model
  local LABEL=$1 MODEL=$2
  echo "=== $LABEL: starting"
  CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
    --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel \
    --gpu-memory-utilization 0.88 > "$RAW/${LABEL}_server.log" 2>&1 &
  local SRV=$!
  local UP=0
  for i in $(seq 1 540); do
    curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && { UP=1; break; }
    kill -0 $SRV 2>/dev/null || break
    sleep 2
  done
  if [ "$UP" != 1 ]; then
    echo "=== $LABEL FAILED to start (log tail below)"
    tail -30 "$RAW/${LABEL}_server.log" | grep -E "Error|error|not support|Traceback|ValueError" | head -10
    pkill -f '[v]llm serve' 2>/dev/null; sleep 5
    return 1
  fi
  grep -iE "marlin|fp8|gptq|quant.*method|Using.*backend" "$RAW/${LABEL}_server.log" | head -8 > "$RAW/${LABEL}_kernel_path.txt"
  ( while true; do
      nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
        --format=csv,noheader >> "$RAW/${LABEL}_gpu.csv"; sleep 2
    done ) & local SAMPLER=$!
  "$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts 16 --ignore-eos --seed 17001 --max-concurrency 1 --request-rate inf \
    --save-result --save-detailed --result-dir "$RAW" \
    --result-filename "${LABEL}_attr512.json" > "$RAW/${LABEL}_attr512.log" 2>&1
  echo "  attr512 rc=$?"
  for C in 1 32 128; do
    local NUM=$((C * 4)); [ "$NUM" -lt 16 ] && NUM=16
    "$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$NUM" --ignore-eos --seed $((17000 + C)) \
      --max-concurrency "$C" --request-rate inf \
      --save-result --save-detailed --result-dir "$RAW" \
      --result-filename "${LABEL}_c${C}.json" > "$RAW/${LABEL}_c${C}.log" 2>&1
    echo "  c=$C rc=$?"
  done
  kill $SAMPLER 2>/dev/null
  kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
  echo "=== $LABEL: done"
}

ARMS=${1:-"fp8 w4a16"}
for a in $ARMS; do
  case $a in
    fp8) run_arm fp8 Qwen/Qwen3-30B-A3B-FP8 ;;
    w4a16) run_arm w4a16 Qwen/Qwen3-30B-A3B-GPTQ-Int4 ;;
  esac
done
echo "D4_BENCH_DONE"
