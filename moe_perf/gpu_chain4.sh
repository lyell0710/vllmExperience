#!/bin/bash
# 链4(最终):D5-FP8 变体 → D2 tune 全量(EP+非EP) → D2 AB
DIR=/root/projects/vllm/experiments/moe_perf

TAG=fp8 MODEL=Qwen/Qwen3-30B-A3B-FP8 bash "$DIR/d5_eplb.sh" 2>&1 | tail -25
echo "STAGE_D5_FP8_DONE"

bash "$DIR/d2_tune.sh"
echo "STAGE_D2_TUNE_DONE"

bash "$DIR/d2_ab.sh"
echo "STAGE_D2_AB_DONE"
echo "GPU_CHAIN_ALL_DONE"
