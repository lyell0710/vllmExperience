#!/bin/bash
# 剩余 GPU 管线(第二版,无 pkill 隐患):PPL×2 → D5 → D2 tune → D2 AB
DIR=/root/projects/vllm/experiments/moe_perf
RAW16=$DIR/raw/EXP-016
V=/root/venvs/v0.25.1/bin/python

cd "$DIR"
timeout 2400 $V d4_ppl.py Qwen/Qwen3-30B-A3B-FP8 "$RAW16/ppl_fp8.json" --tokens 30000 > "$RAW16/ppl_fp8.log" 2>&1
echo "STAGE_PPL_FP8_DONE rc=$?"
sleep 10

timeout 3000 $V d4_ppl.py Qwen/Qwen3-30B-A3B-GPTQ-Int4 "$RAW16/ppl_w4a16.json" --tokens 30000 > "$RAW16/ppl_w4a16.log" 2>&1
echo "STAGE_PPL_W4A16_DONE rc=$?"
sleep 10

bash "$DIR/d5_eplb.sh" 2>&1 | tail -20
echo "STAGE_D5_DONE"

bash "$DIR/d2_tune.sh"
echo "STAGE_D2_TUNE_DONE"

bash "$DIR/d2_ab.sh"
echo "STAGE_D2_AB_DONE"
echo "GPU_CHAIN_ALL_DONE"
