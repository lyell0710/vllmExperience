#!/bin/bash
# 链3:D5 重跑(前次被进程清理竞态误杀) → D2 tune 全量 → D2 AB
DIR=/root/projects/vllm/experiments/moe_perf

bash "$DIR/d5_eplb.sh" 2>&1 | tail -25
echo "STAGE_D5_DONE"

bash "$DIR/d2_tune.sh"
echo "STAGE_D2_TUNE_DONE"

bash "$DIR/d2_ab.sh"
echo "STAGE_D2_AB_DONE"
echo "GPU_CHAIN_ALL_DONE"
