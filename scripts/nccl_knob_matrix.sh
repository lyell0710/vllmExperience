#!/bin/bash
# EXP-020：EXP-019 §8 复现四步——扫 NCCL 旋钮矩阵找「1.78 GB/s」复现档。
#  step1: NCCL_DEBUG=INFO 完整日志落盘（INIT,NET,GRAPH,ENV）
#  step2: NCCL_SHM_DISABLE{0,1} × NCCL_P2P_LEVEL{默认,LOC,PIX,PHB,SYS} 全矩阵 + NCCL_PROTO{Simple,LL,LL128}(仅默认 SHM/P2P)
#         + 3 个额外候选档（MAX_NCHANNELS=1 / SHM_USE_CUDA_MEMCPY=1 / ALGO=Tree），每档 1M–512M n=20
#  step3: 每档同时 nvidia-smi 500ms 采 PCIe gen/width + SM clock → 同前缀 _pcie.csv
#  step4: 判定由 scripts/nccl_knob_matrix_analyze.py 按 §1 锁定阈值执行，产出 derived/<UTC>_nccl_knob_matrix.csv
# 环境：2×RTX 4090 无 P2P；NCCL 2.28.9（v0.25.1 venv wheel，显式 LD_LIBRARY_PATH，同 EXP-018）
set -uo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
OUTDIR=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
SHA=717b683182
ARGS="-b 1M -e 512M -f 2 -g 2 -n 20"

prov() {  # $1=cmd 描述  $2=额外 note
  echo "# provenance: env=sys sha=$SHA cmd=\"$1\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"RTX 4090 x2\" driver=$DRIVER tool_sha=$SHA nccl=libnccl.so.2.28.9 exp=EXP-020 $2"
}

run_arm() {  # $1=tag  $2...=环境变量 KEY=VAL 列表（可为空）
  local tag=$1; shift
  local envs="$*"
  local out="${OUTDIR}/${STAMP}_nccl_knob_${tag}.txt"
  local pcie="${OUTDIR}/${STAMP}_nccl_knob_${tag}_pcie.csv"
  [ -e "$out" ] && { echo "FATAL: $out 已存在，拒绝覆盖（铁律 3）"; exit 1; }
  local cmd="env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV $envs $BIN $ARGS"
  prov "nvidia-smi --query-gpu=index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm --format=csv -lms 500" "arm=$tag knobs=\"$envs\"" > "$pcie"
  nvidia-smi --query-gpu=index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm --format=csv -lms 500 >> "$pcie" 2>&1 &
  local SAMPLER=$!
  sleep 1
  { prov "$cmd" "arm=$tag knobs=\"$envs\""; env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,ENV $envs "$BIN" $ARGS 2>&1; echo "# exit_code=$?"; } > "$out"
  sleep 1
  kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
  echo "[$(date -u +%H:%M:%S)] $tag -> $(basename "$out")  avg=$(grep 'Avg bus bandwidth' "$out" | awk '{print $NF}')  pcie_rows=$(wc -l < "$pcie")"
}

echo "=== EXP-020 nccl knob matrix  STAMP=$STAMP  $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv

# ---------- step1: 完整 NCCL_DEBUG 日志 ----------
DBG="${OUTDIR}/${STAMP}_nccl_debug_full.log"
DBGP="${OUTDIR}/${STAMP}_nccl_debug_full_pcie.csv"
[ -e "$DBG" ] && { echo "FATAL: $DBG 已存在"; exit 1; }
DBGCMD="env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV $BIN $ARGS"
prov "nvidia-smi ... -lms 500" "arm=debug_full" > "$DBGP"
nvidia-smi --query-gpu=index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm --format=csv -lms 500 >> "$DBGP" 2>&1 &
SAMPLER=$!; sleep 1
{ prov "$DBGCMD" "arm=debug_full note=step1-完整NCCL_DEBUG日志-stdout+stderr"; env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET,GRAPH,ENV "$BIN" $ARGS 2>&1; echo "# exit_code=$?"; } > "$DBG"
sleep 1; kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
echo "[$(date -u +%H:%M:%S)] debug_full -> $(basename "$DBG") lines=$(wc -l < "$DBG") avg=$(grep 'Avg bus bandwidth' "$DBG" | awk '{print $NF}')"

# ---------- step2+3: 矩阵 ----------
for SHM in 0 1; do
  for P2P in DEF LOC PIX PHB SYS; do
    E="NCCL_SHM_DISABLE=$SHM"
    [ "$P2P" != DEF ] && E="$E NCCL_P2P_LEVEL=$P2P"
    run_arm "shm${SHM}_p2p${P2P}_protoDEF" $E
  done
done
for PROTO in Simple LL LL128; do
  run_arm "shm0_p2pDEF_proto${PROTO}" NCCL_SHM_DISABLE=0 NCCL_PROTO=$PROTO
done
# 额外候选档（§1 预注册，与主矩阵分开判读）
run_arm "extra_nchan1"      NCCL_SHM_DISABLE=0 NCCL_MAX_NCHANNELS=1
run_arm "extra_shmcudamemcpy1" NCCL_SHM_DISABLE=0 NCCL_SHM_USE_CUDA_MEMCPY=1
run_arm "extra_algoTree"    NCCL_SHM_DISABLE=0 NCCL_ALGO=Tree

echo "=== EXP-020 矩阵结束 $(date -u +%FT%TZ)"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
