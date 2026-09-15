#!/bin/bash
# EXP-022：EXP-015 §7 缺口——大 M(512–4096) kernel A/B，3 轮交叉次序（奇数轮 default 先），复用 d2_hardening.sh 的臂切换+双重断言
# 只做 kernel；不跑 correctness（EXP-015 已 1041 passed）。用法: bash d2_bigM_ab.sh [all|1|2|3]
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
ROUND=${1:-all}
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
OUT=$DIR/raw/EXP-022/bigM_$STAMP
mkdir -p "$OUT"
VENV=/root/venvs/main
MODEL=Qwen/Qwen1.5-MoE-A2.7B-Chat
CFGDIR=/root/projects/vllm/vllm/model_executor/layers/fused_moe/configs
BENCH=/root/projects/vllm/benchmarks/kernels/benchmark_moe.py
HOLD=$DIR/raw/EXP-015/.held      # 放输出目录外，避免清理时连带删除
BS_LIST="512 1024 2048 4096"
SHA=$(cd /root/projects/vllm && git rev-parse --short HEAD)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
mkdir -p "$HOLD"

J1="E=30,N=1408,device_name=NVIDIA_GeForce_RTX_4090.json"
J2="E=60,N=704,device_name=NVIDIA_GeForce_RTX_4090.json"

RAWCFG=$DIR/raw/EXP-015
restore() {   # 双保险：先从 HOLD 拿，拿不到就从 raw 原件补
  for j in "$J1" "$J2"; do
    [ -f "$HOLD/$j" ] && mv -f "$HOLD/$j" "$CFGDIR/$j"
    [ -f "$CFGDIR/$j" ] || { src=$(ls "$RAWCFG"/configs_*/"$j" 2>/dev/null | head -1); [ -n "$src" ] && cp -f "$src" "$CFGDIR/$j"; }
  done
}
trap restore EXIT INT TERM

arm_default() { for j in "$J1" "$J2"; do [ -f "$CFGDIR/$j" ] && mv -f "$CFGDIR/$j" "$HOLD/$j"; done; return 0; }
arm_tuned()   { for j in "$J1" "$J2"; do [ -f "$HOLD/$j" ]   && mv -f "$HOLD/$j" "$CFGDIR/$j"; done; return 0; }

assert_arm() {  # $1 = default|tuned ；装错臂就是废数据，必须当场停
  local ARM=$1 j
  for j in "$J1" "$J2"; do
    if [ "$ARM" = default ]; then
      [ -f "$CFGDIR/$j" ] && { echo "FATAL: default 臂但 $j 仍在 configs/ 内"; exit 1; }
    else
      [ -f "$CFGDIR/$j" ] || { echo "FATAL: tuned 臂但 $j 不在 configs/ 内"; exit 1; }
    fi
  done
  return 0
}

assert_log() {  # $1 = 日志文件, $2 = default|tuned ；核对 kernel 实际读到的 config 来源
  if [ "$2" = default ]; then
    grep -q "Using default MoE config" "$1" || { echo "FATAL: $1 未出现「Using default MoE config」，该臂数据作废"; exit 1; }
  else
    grep -q "Using configuration from" "$1" || { echo "FATAL: $1 未出现「Using configuration from」，该臂数据作废"; exit 1; }
  fi
}

run() {  # $1=ep|noep  $2=default|tuned  $3=round
  local MODE=$1 ARM=$2 R=$3
  local F="$OUT/${STAMP}_kernel_${MODE}_${ARM}_r${R}.log"
  local EXTRA=""; [ "$MODE" = ep ] && EXTRA="--enable-expert-parallel"
  local CMD="$VENV/bin/python $BENCH --model $MODEL -tp 2 $EXTRA --seed 0 --batch-size $BS_LIST"
  echo "# provenance: env=venvs/main sha=$SHA cmd=\"$CMD\" date=$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu=\"2xRTX 4090\" driver=$DRV arm=$ARM round=$R exp=EXP-022 bs_list=\"$BS_LIST\"" > "$F"
  CUDA_VISIBLE_DEVICES=0,1 $CMD >> "$F" 2>&1
  local RC=$?
  assert_log "$F" "$ARM"
  echo "  [$(date -u +%H:%M:%S)] $MODE/$ARM r$R rc=$RC ✓臂已核对 -> $(basename $F)"
}

echo "=== EXP-022 bigM A/B 开始 $(date -u +%FT%TZ)  输出: $OUT  轮次: $ROUND"
ROUNDS="1 2 3"
case "$ROUND" in all) ;; *) ROUNDS="$ROUND" ;; esac
for R in ${ROUNDS:-}; do
  # 交叉次序：奇数轮 default 先，偶数轮 tuned 先，排除热漂移与次序效应
  if [ $((R % 2)) -eq 1 ]; then ORDER="default tuned"; else ORDER="tuned default"; fi
  echo "--- round $R (次序: $ORDER)"
  for ARM in $ORDER; do
    if [ "$ARM" = default ]; then arm_default; else arm_tuned; fi
    assert_arm "$ARM"
    run ep   "$ARM" "$R"
    run noep "$ARM" "$R"
  done
done
arm_tuned   # 恢复 JSON 就位

rmdir "$HOLD" 2>/dev/null
echo "=== EXP022_BIGM_DONE $(date -u +%FT%TZ)"
