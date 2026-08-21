#!/usr/bin/env bash
# B1 单测量点执行器：快照(直抓引擎) → vllm bench serve → 快照 → collect_point.py 追加 runs.jsonl
#
# 用法:
#   run_point.sh <arm> <mode> <input_len> <output_len> <rps|-> <bench_port> <engine_port> [engine_port2]
#   arm:  colocate | replica2 | tp2 | pd1p1d
#   mode: attribution(并发1) | sweep(offered load, 需给 rps)
#   bench_port: bench 客户端请求的端口(有代理的臂 = 代理端口)
#   engine_port(s): /metrics 直抓的引擎端口(Gate 要求, 永不抓代理)
# 环境变量: MODEL VENV NUM_PROMPTS GPU_COUNT SLO_TTFT_MS SLO_TPOT_MS
set -euo pipefail
cd "$(dirname "$0")/.."

ARM=$1; MODE=$2; IN=$3; OUT=$4; RPS=$5; BENCH_PORT=$6; shift 6; ENGINE_PORTS=("$@")
VENV=${VENV:-/root/venvs/v0.25.1}
MODEL=${MODEL:-Qwen/Qwen2-7B-Instruct}
NUM=${NUM_PROMPTS:-32}
R=results/b1_matrix

PREFIX=$(date -u +%Y%m%dT%H%M)_${ARM}_${IN}x${OUT}_${MODE}
[ "$RPS" != "-" ] && PREFIX=${PREFIX}_rps${RPS}

for p in "${ENGINE_PORTS[@]}"; do
  scripts/metrics_snapshot.sh snap "$p" "$R/snapshots/${PREFIX}_${p}_before.prom"
done

# GPU 遥测采样(2s): 功率帽节流会使持续 prefill 降频~12%、TTFT 抬升(8/21 实测),
# 每个测量点必须留下工况证据
GPUCSV=$R/raw/${PREFIX}_gpu.csv
( while true; do
    nvidia-smi --query-gpu=index,temperature.gpu,clocks.sm,power.draw,clocks_event_reasons.active \
      --format=csv,noheader >> "$GPUCSV"; sleep 2
  done ) & SAMPLER=$!

if [ "$MODE" = attribution ]; then
  RATE_ARGS=(--max-concurrency 1 --request-rate inf)
else
  RATE_ARGS=(--request-rate "$RPS")
fi

"$VENV/bin/vllm" bench serve \
  --host localhost --port "$BENCH_PORT" --model "$MODEL" \
  --dataset-name random --random-input-len "$IN" --random-output-len "$OUT" \
  --num-prompts "$NUM" --ignore-eos --seed 42 \
  "${RATE_ARGS[@]}" \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
  --save-result --save-detailed --result-dir "$R/raw" \
  --result-filename "${PREFIX}_bench.json" \
  2>&1 | tee "$R/raw/${PREFIX}_bench.log"

kill "$SAMPLER" 2>/dev/null || true

for p in "${ENGINE_PORTS[@]}"; do
  scripts/metrics_snapshot.sh snap "$p" "$R/snapshots/${PREFIX}_${p}_after.prom"
done

GPU_COUNT=${GPU_COUNT:-$([ "$ARM" = colocate ] && echo 1 || echo 2)}
"$VENV/bin/python" scripts/collect_point.py \
  --prefix "$PREFIX" --arm "$ARM" --mode "$MODE" \
  --input-len "$IN" --output-len "$OUT" --rps "$RPS" \
  --gpu-count "$GPU_COUNT" --engine-ports "${ENGINE_PORTS[@]}" \
  --gpu-csv "$GPUCSV" \
  ${SLO_TTFT_MS:+--slo-ttft-ms "$SLO_TTFT_MS"} ${SLO_TPOT_MS:+--slo-tpot-ms "$SLO_TPOT_MS"}
echo "[run_point] done: $PREFIX"
