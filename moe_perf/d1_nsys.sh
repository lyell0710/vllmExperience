#!/bin/bash
# D1 · nsys kernel 分解:MoE TP2+EP decode 稳态,bs=1 与 bs=32 两窗(EXP-014)
# 机制:vllm --profiler-config.profiler=cuda + nsys --capture-range=cudaProfilerApi
#      /start_profile → cudaProfilerStart(全 rank)→ nsys 开采;/stop_profile 收窗。
# 用法: d1_nsys.sh <bs>   (每次一个采集窗,采完服务关闭)
set -u
BS=$1
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-014
mkdir -p "$RAW"
VENV=/root/venvs/v0.25.1
MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat
OUT=$RAW/d1_nsys_moe_bs${BS}

echo "=== nsys capture bs=$BS: starting server under nsys"
CUDA_VISIBLE_DEVICES=0,1 nsys profile \
  --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi --capture-range-end=stop-shutdown \
  --kill=sigkill -o "$OUT" --force-overwrite=true \
  "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
  --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel \
  --gpu-memory-utilization 0.88 \
  --profiler-config.profiler=cuda \
  > "$RAW/d1_nsys_moe_bs${BS}_server.log" 2>&1 &
NSYS_PID=$!

for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break
  kill -0 $NSYS_PID 2>/dev/null || { echo "server died"; exit 1; }
  sleep 2
done
curl -sf http://127.0.0.1:8100/health >/dev/null || { echo "start timeout"; exit 1; }
echo "=== server up, starting load conc=$BS"

NUM=$((BS * 12)); [ "$NUM" -lt 24 ] && NUM=24
"$VENV/bin/vllm" bench serve \
  --host localhost --port 8100 --model "$MODEL" \
  --dataset-name random --random-input-len 128 --random-output-len 512 \
  --num-prompts "$NUM" --ignore-eos --seed $((15000 + BS)) \
  --max-concurrency "$BS" --request-rate inf \
  > "$RAW/d1_nsys_moe_bs${BS}_bench.log" 2>&1 &
BENCH_PID=$!

sleep 15   # 稳态(过 prefill 潮、进 decode 主导段)
echo "=== start_profile (cudaProfilerStart -> nsys capture begins)"
curl -s -X POST http://127.0.0.1:8100/start_profile
sleep 20   # 采集窗
echo "=== stop_profile (capture ends, app shutdown per capture-range-end)"
curl -s -X POST http://127.0.0.1:8100/stop_profile
sleep 5

kill $BENCH_PID 2>/dev/null
for i in $(seq 1 60); do kill -0 $NSYS_PID 2>/dev/null || break; sleep 2; done
kill $NSYS_PID 2>/dev/null; sleep 3; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
ls -la "$OUT"* 2>/dev/null
echo "NSYS_BS${BS}_DONE"
