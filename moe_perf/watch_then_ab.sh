#!/bin/bash
# 等 d2 tune 链(b6wxc7eak)打出 CHAIN_DONE 再接 d2_ab
TASK_OUT=/tmp/claude-0/-root/52ec8f94-9adc-4524-9efe-67b312329851/tasks/b6wxc7eak.output
until grep -q "CHAIN_DONE" "$TASK_OUT" 2>/dev/null; do sleep 60; done
echo "tune chain done, starting A/B"
bash /root/projects/vllm/experiments/moe_perf/d2_ab.sh
echo "WATCH_AB_DONE"
