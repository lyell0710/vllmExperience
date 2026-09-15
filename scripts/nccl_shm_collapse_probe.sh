#!/bin/bash
# EXP-020 附录 A / EXP-021 §7 跟进（post-hoc 探针，非预注册）：
# EXP-021 中 float 大消息第二轮 SHM 路径塌到 2.1–2.3 GB/s，但当时无 PCIe 采样。
# 本探针：float / half 交替各 5 轮大消息扫描，每轮同步 200ms 采 PCIe gen/width + SM/mem clock + pstate + power，
# 目的：抓到塌陷态时看 PCIe 链路/时钟是否同步异常；顺带看塌陷是否 dtype 相关。
set -uo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
OUTDIR=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
SHA=717b683182
ARGS="-b 1M -e 512M -f 2 -g 2 -n 20"
Q="index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm,clocks.mem,pstate,power.draw"

run() {  # $1=round $2=dtype
  local rd=$1 dt=$2
  local out="${OUTDIR}/${STAMP}_nccl_shm_probe_${dt}_r${rd}.txt"
  local pcie="${OUTDIR}/${STAMP}_nccl_shm_probe_${dt}_r${rd}_pcie.csv"
  [ -e "$out" ] && { echo "FATAL: $out 已存在"; exit 1; }
  local cmd="env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT $BIN -d $dt $ARGS"
  echo "# provenance: env=sys sha=$SHA cmd=\"nvidia-smi --query-gpu=$Q --format=csv -lms 200\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"RTX 4090 x2\" driver=$DRIVER exp=EXP-020-appendixA dtype=$dt round=$rd" > "$pcie"
  nvidia-smi --query-gpu=$Q --format=csv -lms 200 >> "$pcie" 2>&1 &
  local S=$!; sleep 0.6
  { echo "# provenance: env=sys sha=$SHA cmd=\"$cmd\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"RTX 4090 x2\" driver=$DRIVER tool_sha=$SHA nccl=libnccl.so.2.28.9 exp=EXP-020-appendixA dtype=$dt round=$rd note=post-hoc-SHM塌陷态探针"
    env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT "$BIN" -d "$dt" $ARGS 2>&1; echo "# exit_code=$?"; } > "$out"
  sleep 0.6; kill "$S" 2>/dev/null; wait "$S" 2>/dev/null
  local gens=$(tail -n +3 "$pcie" | awk -F', ' '$5+0>1000{g[$3]++} END{for(k in g) printf "gen%s=%d ",k,g[k]}')
  echo "[$(date -u +%H:%M:%S)] r$rd $dt avg=$(grep 'Avg bus bandwidth' "$out" | awk '{print $NF}') 1M=$(grep -E '^\s+1048576 ' "$out" | awk '{print $8}') 256M=$(grep -E '^\s+268435456 ' "$out" | awk '{print $8}') loaded:$gens"
}
echo "=== SHM collapse probe STAMP=$STAMP $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
for rd in 1 2 3 4 5; do run $rd float; run $rd half; done
echo "=== 结束 $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
