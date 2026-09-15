#!/usr/bin/env bash
# EXP-026 第二段：UCX 传输层（TLS）两步骤筛选。
#
# 动机：EXP-026 第一段发现 NIXL 的 `skip_desc_merge` 默认为 False（已合并描述符），
# 粒度从 16 KiB 到 16 MiB 带宽几乎不变（0.383→0.376 GB/s），而 UCX 自报
#   ucp_context_0 intra-node cfg#1 rma_am(tcp/eth0) ... am(tcp/eth0 cma/memory cuda_ipc/cuda)
# ——RMA（KV 的 RDMA READ 路径）走的是 **eth0 上的 TCP**，GPU 缓冲是 software emulation。
#
# 步骤 1（screen）：对每个 UCX_TLS 只做 agent 初始化 + 内存注册 + 远端握手（不传输），
#   快速判定该组合能不能起来（`UCX_TLS=shm` 实测在 createBackend 就报 NIXL_ERR_BACKEND）。
# 步骤 2（bench）：只对 screen 通过的组合跑完整传输扫描。
#
# 每档都落盘 D/P 日志、CSV 与 UCX 自报 transport；失败档同样登记（负结果也是结果）。
set -uo pipefail
cd "$(dirname "$0")/.."

BENCH=scripts/nixl_desc_granularity_bench.py
PY=/root/venvs/v0.25.1/bin/python
STAMP=${STAMP_OVERRIDE:-$(date -u +%Y%m%dT%H%M)}
VOL=$((64*1024*1024))
REG=$((128*1024*1024))
ITERS=${ITERS:-3}
OUT=pd_disagg/hw/exp026_tls_$STAMP
mkdir -p "$OUT"

SETTINGS=(
  "baseline|"
  "shm|shm"
  "cuda_copy_shm|cuda_copy,shm"
  "cuda_ipc_shm|cuda_ipc,shm"
  "cuda_ipc_cuda_copy_shm|cuda_ipc,cuda_copy,shm"
  "all|all"
  "tcp_cuda_copy|cuda_copy,tcp"
  "tcp_cuda_ipc_cuda_copy|cuda_ipc,cuda_copy,tcp"
  "tcp_sm|sm,tcp"
  "tcp_cuda_ipc|cuda_ipc,tcp"
)
# 保留 TCP（控制面需要）但额外放开更快的候选传输，看 NIXL/UCX 会不会改选：
# TLS_LIST 用 ';' 分隔（值里含逗号），例：TLS_LIST='tcp_sm|sm,tcp;tcp_cuda_ipc|cuda_ipc,tcp'
if [ -n "${TLS_LIST:-}" ]; then
  IFS=';' read -ra SETTINGS <<< "$TLS_LIST"
fi

cleanup_p() {
  for p in /proc/[0-9]*; do
    pid=$(basename "$p")
    [ "$pid" = "$$" ] && continue
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in *nixl_desc_granularity_bench.py*--role\ P*) kill -9 "$pid" 2>/dev/null;; esac
  done
}

run_arm() {  # $1=name $2=tls $3=mode(screen|bench)
  local NAME="$1" TLS="$2" MODE="$3"
  local W="$OUT/$MODE/$NAME"; mkdir -p "$W"
  if [ -n "$TLS" ]; then export UCX_TLS="$TLS"; else unset UCX_TLS; fi
  local EXTRA=""; [ "$MODE" = "screen" ] && EXTRA="--init_only"
  ( CUDA_VISIBLE_DEVICES=0 UCX_LOG_LEVEL=info "$PY" "$BENCH" --role P --workdir "$W" \
      --region "$REG" --volume "$VOL" --iters 1 --wait_timeout 60 $EXTRA > "$W/P.log" 2>&1 & )
  sleep 3
  CUDA_VISIBLE_DEVICES=1 UCX_LOG_LEVEL=info "$PY" "$BENCH" --role D --workdir "$W" \
      --region "$REG" --volume "$VOL" --iters "$ITERS" --xfer_timeout 20 --wait_timeout 60 \
      $EXTRA --out "$W/D.csv" > "$W/D.log" 2>&1
  local RC=$?
  cleanup_p
  local TR
  TR=$(grep -oE 'rma_am\([^)]*\)' "$W/D.log" 2>/dev/null | head -1)
  [ -z "$TR" ] && TR=$(grep -oE 'rma_am\([^)]*\)' "$W/P.log" 2>/dev/null | head -1)
  local BW16
  BW16=$(grep -E 'A contiguous  G=   16KiB' "$W/D.log" 2>/dev/null | grep -oE '[0-9.]+ GB/s' | head -1)
  local ERR
  ERR=$(grep -hoE 'NIXL_ERR_[A-Z_]+|nixlBackendError|TimeoutError' "$W/P.log" "$W/D.log" 2>/dev/null | head -1)
  local TLS_SAFE="${TLS//,/\+}"   # TLS 值含逗号会把 CSV 撑列（实测踩到）
  printf '%s,%s,%s,%s,%s,%s,%s\n' "$MODE" "$NAME" "${TLS_SAFE:-未设}" "$RC" "${TR:-}" "${BW16:-}" "${ERR:-}" \
      >> "$OUT/tls_summary.csv"
  echo "  [$MODE] $NAME exit=$RC transport=[${TR:-未捕获}] 16KiB=${BW16:-n/a} err=[${ERR:-}]"
  unset UCX_TLS
}

printf 'mode,name,ucx_tls,exit_code,transport,bw_16KiB_contig,error\n' > "$OUT/tls_summary.csv"
echo "=== EXP-026 UCX_TLS 两步骤筛选 STAMP=$STAMP vol=$((VOL/1024/1024))MiB reg=$((REG/1024/1024))MiB"
nvidia-smi --query-compute-apps=pid,process_name --format=csv

echo
echo "########## 步骤 1：screen（只验证能否初始化）"
for S in "${SETTINGS[@]}"; do run_arm "${S%%|*}" "${S#*|}" screen; done

echo
echo "########## 步骤 2：bench（只跑 screen 通过的档）"
for S in "${SETTINGS[@]}"; do
  N="${S%%|*}"
  RC=$(awk -F, -v n="$N" '$1=="screen" && $2==n {print $4}' "$OUT/tls_summary.csv" | tail -1)
  [ "$RC" = "0" ] || { echo "  跳过 $N（screen exit=$RC）"; continue; }
  run_arm "$N" "${S#*|}" bench
done

echo
echo "=== 汇总 $OUT/tls_summary.csv"
column -t -s, "$OUT/tls_summary.csv" 2>/dev/null || cat "$OUT/tls_summary.csv"
nvidia-smi --query-compute-apps=pid,process_name --format=csv
