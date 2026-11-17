#!/bin/bash
# D2 · A/B 验证(EXP-015（D2 MoE config 调优）后半):kernel A/B + e2e serving A/B + correctness
# 前置:d2_tune.sh 完成,configs_{ep,noep} 下有 JSON
# A/B 次序:先测 default(未装 JSON)→ 装 JSON → 测 tuned;e2e 同理
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-015
VENV=/root/venvs/main
MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat
CFGDIR=/root/projects/vllm/vllm/model_executor/layers/fused_moe/configs
BENCH=/root/projects/vllm/benchmarks/kernels/benchmark_moe.py
BS_LIST="1 8 32 64 128 256"
EP_JSON=$(ls "$RAW/configs_ep"/E=30,N=1408,*.json 2>/dev/null | head -1)
NOEP_JSON=$(ls "$RAW/configs_noep"/E=60,N=704,*.json 2>/dev/null | head -1)
[ -z "$EP_JSON" ] && { echo "no EP json"; exit 1; }

kernel_bench() {  # $1 label, $2 extra args
  local LABEL=$1; shift
  CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" "$BENCH" \
    --model "$MODEL" -tp 2 "$@" --seed 0 --batch-size $BS_LIST \
    > "$RAW/kernel_${LABEL}.log" 2>&1
  echo "kernel_bench $LABEL rc=$?"
}

e2e_bench() {  # $1 label
  local LABEL=$1
  CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/vllm" serve "$MODEL" --port 8100 \
    --max-model-len 8192 --tensor-parallel-size 2 --enable-expert-parallel \
    --gpu-memory-utilization 0.88 > "$RAW/e2e_${LABEL}_server.log" 2>&1 &
  local SRV=$!
  for i in $(seq 1 300); do
    curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || { echo "e2e $LABEL server died"; return 1; }
    sleep 2
  done
  for C in 1 32 128; do
    local NUM=$((C * 4)); [ "$NUM" -lt 16 ] && NUM=16
    "$VENV/bin/vllm" bench serve --host localhost --port 8100 --model "$MODEL" \
      --dataset-name random --random-input-len 128 --random-output-len 256 \
      --num-prompts "$NUM" --ignore-eos --seed $((16000 + C)) \
      --max-concurrency "$C" --request-rate inf \
      --save-result --save-detailed --result-dir "$RAW" \
      --result-filename "e2e_${LABEL}_c${C}.json" \
      > "$RAW/e2e_${LABEL}_c${C}.log" 2>&1
    echo "e2e $LABEL c=$C rc=$?"
  done
  kill $SRV 2>/dev/null; sleep 8; pkill -f '[v]llm serve' 2>/dev/null; sleep 5
}

echo "=== phase 1: kernel A (default, JSON 未装)"
kernel_bench ep_default --enable-expert-parallel
kernel_bench noep_default

echo "=== phase 2: e2e A (default)"
e2e_bench default

echo "=== phase 3: install tuned JSONs into repo configs/"
cp -v "$EP_JSON" "$CFGDIR/"
[ -n "$NOEP_JSON" ] && cp -v "$NOEP_JSON" "$CFGDIR/"

echo "=== phase 4: kernel B (tuned)"
kernel_bench ep_tuned --enable-expert-parallel
kernel_bench noep_tuned

echo "=== phase 5: e2e B (tuned)"
e2e_bench tuned

echo "=== phase 6: correctness (pytest moe 子集)"
cd /root/projects/vllm && CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -m pytest \
  tests/kernels/moe/test_moe.py -q -k "not deepseek and not fp8 and not int8 and not wna16" \
  -x --no-header > "$RAW/correctness_pytest.log" 2>&1
echo "pytest rc=$?"
tail -3 "$RAW/correctness_pytest.log"
echo "D2_AB_DONE"
