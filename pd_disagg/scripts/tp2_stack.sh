#!/usr/bin/env bash
# tp2 栈起/拆（EXP-024 新增；配置同 EXP-005/007：TP=2，--gpu-memory-utilization 0.88）
#   up <UTC前缀> : 双卡起一个 TP=2 实例 :8100，等就绪；日志 results/b1_matrix/raw/<前缀>_tp2_conc128_server_8100.log
#   down         : 按 /proc 定位并终止，不用 pkill 字面量（防自匹配；见 EXP-023 §7 教训）
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/root/venvs/v0.25.1}
MODEL=${MODEL:-Qwen/Qwen2-7B-Instruct}
R=results/b1_matrix/raw
PORT=8100

down() {
  local killed=0
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in
      *"vllm serve $MODEL"*) echo "TERM $(basename "$p"): $(echo "$c" | cut -c1-70)"; kill -TERM "$(basename "$p")" 2>/dev/null; killed=$((killed+1));;
    esac
  done
  sleep 6
  for p in /proc/[0-9]*; do
    exe=$(readlink "$p/exe" 2>/dev/null); case "$exe" in *venvs/v0.25.1*) echo "残留 $(basename "$p") $exe";; esac
  done
  echo "killed=$killed"; nvidia-smi --query-compute-apps=pid,process_name --format=csv
}

case "${1:-}" in
  up)
    P=${2:?需要 UTC 前缀}
    ss -ltn 2>/dev/null | grep -q ":$PORT " && { echo "FATAL: 端口 $PORT 被占用"; exit 2; }
    LOG="$R/${P}_tp2_conc128_server_${PORT}.log"
    [ -e "$LOG" ] && { echo "FATAL: $LOG 已存在，拒绝覆盖"; exit 2; }
    echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"CUDA_VISIBLE_DEVICES=0,1 $VENV/bin/vllm serve $MODEL --port $PORT --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.88\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=610.57.04 exp=EXP-024" > "$LOG"
    ( CUDA_VISIBLE_DEVICES=0,1 nohup "$VENV/bin/vllm" serve "$MODEL" --port "$PORT" \
        --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization 0.88 \
        >> "$LOG" 2>&1 & )
    echo "started tp2 :$PORT -> $LOG"
    t0=$(date +%s)
    until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do
      sleep 5
      grep -qE 'OutOfMemoryError|CUDA out of memory|Address already in use' "$LOG" && { echo "FATAL: OOM/端口冲突"; grep -m2 -E 'OutOfMemoryError|Address already' "$LOG"; exit 4; }
      [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "FATAL: 600s 未就绪"; tail -5 "$LOG"; exit 3; }
    done
    echo "ready :$PORT at +$(( $(date +%s) - t0 ))s"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ;;
  down) down ;;
  *) echo "用法: $0 up <UTC前缀> | down"; exit 1;;
esac
