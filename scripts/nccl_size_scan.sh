#!/bin/bash
# NCCL allreduce size 扫描（EXP-018）——补 EXP-002 只测大消息(1M–512M)的缺口。
# 目的：①补 decode 级小消息(8 KiB)的纯 NCCL allreduce 延迟地板实测；
#       ②复测大消息区间，核对 EXP-002 的 1.78 GB/s 带宽墙。
# 环境：2×RTX 4090，无 P2P（NCCL 走 SHM）；NCCL 2.28.9（v0.25.1 venv wheel，
#       显式 LD_LIBRARY_PATH，避免落到系统全局 2.25.1 造成版本漂移）。
set -euo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
OUTDIR=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=$(date -u +%Y%m%dT%H%M%S)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
SHA=717b683182

run() {
  local tag=$1; shift
  local out="${OUTDIR}/${STAMP}_allreduce_size_scan_${tag}.txt"
  {
    echo "# provenance: env=sys sha=$SHA cmd=\"all_reduce_perf $*\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"2 NVIDIA GeForce RTX 4090\" driver=$DRIVER tool_sha=$SHA nccl=libnccl.so.2.28.9 note=allreduce-size-scan-补EXP-002小消息缺口-仅代表TP-collective路径"
    env LD_LIBRARY_PATH="$NCCL_LIB" "$BIN" "$@"
  } > "$out"
  echo "落盘: $out"
}

run small -b 8 -e 1M -f 2 -g 2 -n 100
run large -b 1M -e 512M -f 2 -g 2 -n 20
