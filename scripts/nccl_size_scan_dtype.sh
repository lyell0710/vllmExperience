#!/bin/bash
# EXP-021：EXP-018 缺的 dtype 维度——同 size 扫描命令加 -d half / -d bfloat16，与 float 结果并排。
# 假设：allreduce 对 dtype 不敏感（同字节数），阈值：同字节数下 busbw 差 <10% 则成立（§1 锁定）。
# 环境同 EXP-018/020：2×RTX 4090 无 P2P（SHM 路径），NCCL 2.28.9（v0.25.1 venv wheel，显式 LD_LIBRARY_PATH）。
set -uo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
OUTDIR=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
SHA=717b683182

run() {  # $1=round $2=dtype $3=small|large $4...=args
  local rd=$1 dt=$2 tag=$3; shift 3
  local out="${OUTDIR}/${STAMP}_allreduce_size_scan_${dt}_${tag}_r${rd}.txt"
  [ -e "$out" ] && { echo "FATAL: $out 已存在，拒绝覆盖"; exit 1; }
  local cmd="env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT $BIN -d $dt $*"
  {
    echo "# provenance: env=sys sha=$SHA cmd=\"$cmd\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"RTX 4090 x2\" driver=$DRIVER tool_sha=$SHA nccl=libnccl.so.2.28.9 exp=EXP-021 dtype=$dt range=$tag round=$rd note=同EXP-018参数-仅代表TP-collective路径-SHM"
    env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT "$BIN" -d "$dt" "$@" 2>&1
    echo "# exit_code=$?"
  } > "$out"
  echo "[$(date -u +%H:%M:%S)] r$rd $dt/$tag -> $(basename "$out") avg=$(grep 'Avg bus bandwidth' "$out" | awk '{print $NF}') via=$(grep -m1 -oE 'via [A-Za-z/0-9]+' "$out")"
}

echo "=== EXP-021 dtype scan STAMP=$STAMP $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
# 三种 dtype 同一会话内交替跑两轮（r1 顺序 float→half→bfloat16，r2 反序），会话内抖动用两轮差估计
for dt in float half bfloat16; do
  run 1 $dt small -b 8 -e 1M -f 2 -g 2 -n 100
  run 1 $dt large -b 1M -e 512M -f 2 -g 2 -n 20
done
for dt in bfloat16 half float; do
  run 2 $dt small -b 8 -e 1M -f 2 -g 2 -n 100
  run 2 $dt large -b 1M -e 512M -f 2 -g 2 -n 20
done
echo "=== EXP-021 结束 $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
