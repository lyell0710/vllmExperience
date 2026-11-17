#!/bin/bash
# 链5(收官):D5 对照组 → D2 tune 全量 → D2 AB。此后不再抢占。
DIR=/root/projects/vllm/experiments/moe_perf

bash "$DIR/d5_control.sh" 2>&1 | tail -8
echo "STAGE_D5_CONTROL_DONE"

bash "$DIR/d2_tune.sh"
echo "STAGE_D2_TUNE_DONE"

bash "$DIR/d2_ab.sh"
echo "STAGE_D2_AB_DONE"
echo "GPU_CHAIN_ALL_DONE"
