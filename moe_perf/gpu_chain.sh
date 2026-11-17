#!/bin/bash
# 剩余 GPU 管线自驱动链。每阶段打 STAGE_xxx_DONE 标记。
DIR=/root/projects/vllm/experiments/moe_perf
RAW16=$DIR/raw/EXP-016
V=/root/venvs/v0.25.1/bin/python

bash "$DIR/d4_fp8_w4a16.sh" w4a16
echo "STAGE_W4A16_BENCH_DONE"

cd "$DIR" && timeout 2400 $V d4_ppl.py Qwen/Qwen3-30B-A3B-FP8 "$RAW16/ppl_fp8.json" --tokens 30000 > "$RAW16/ppl_fp8.log" 2>&1
echo "STAGE_PPL_FP8_DONE rc=$?"
pkill -f '[d]4_ppl' 2>/dev/null; sleep 5

timeout 3000 $V d4_ppl.py Qwen/Qwen3-30B-A3B-GPTQ-Int4 "$RAW16/ppl_w4a16.json" --tokens 30000 > "$RAW16/ppl_w4a16.log" 2>&1
echo "STAGE_PPL_W4A16_DONE rc=$?"
pkill -f '[d]4_ppl' 2>/dev/null; sleep 5

bash "$DIR/d5_eplb.sh" 2>&1 | tail -20
echo "STAGE_D5_DONE"

bash "$DIR/d2_tune.sh"
echo "STAGE_D2_TUNE_DONE"

bash "$DIR/d2_ab.sh"
echo "STAGE_D2_AB_DONE"
echo "GPU_CHAIN_ALL_DONE"
