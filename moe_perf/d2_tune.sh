#!/bin/bash
# D2 · benchmark_moe.py 调优:4090 BF16 两个空缺 tuple(EXP-015（D2 MoE config 调优）)
#   EP:  E=30,N=1408 (TP2+EP)     非EP: E=60,N=704 (TP2)
# ENV-C(main),ray 双卡分摊 batch 档;JSON 落 save-dir 后由 A/B 阶段部署
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-015
mkdir -p "$RAW/configs_ep" "$RAW/configs_noep"
source "$DIR/../pd_disagg/scripts/provenance.sh"
prov_env C
VENV=/root/venvs/main
MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat
cd /root/projects/vllm/benchmarks/kernels

prov_line "d2_tune.sh (benchmark_moe --tune, EP then non-EP)" > "$RAW/manifest.txt"

echo "=== tune EP (E=30, N=1408)"
CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" benchmark_moe.py \
  --model "$MODEL" -tp 2 --enable-expert-parallel --tune \
  --seed 0 --save-dir "$RAW/configs_ep" \
  > "$RAW/tune_ep.log" 2>&1
echo "rc_ep=$?"
ls -la "$RAW/configs_ep/"

echo "=== tune non-EP (E=60, N=704)"
CUDA_VISIBLE_DEVICES=0,1 "$VENV/bin/python" benchmark_moe.py \
  --model "$MODEL" -tp 2 --tune \
  --seed 0 --save-dir "$RAW/configs_noep" \
  > "$RAW/tune_noep.log" 2>&1
echo "rc_noep=$?"
ls -la "$RAW/configs_noep/"
echo "D2_TUNE_DONE"
