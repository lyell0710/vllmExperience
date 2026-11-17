#!/usr/bin/env bash
# replica2 栈起/拆（EXP-023 新增；配置同 EXP-004/005/007）：
#   up <UTC前缀>  : GPU0:8100 + GPU1:8200 两个 Qwen2-7B-Instruct 实例 + rr_proxy:8300，等就绪；日志落 results/b1_matrix/raw/<前缀>_replica2_conc128_*.log
#   down          : pkill -f '[v]llm serve'（方括号技巧，防误杀自身 shell）+ 代理；/proc/*/exe 复核无残留
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/root/venvs/v0.25.1}
MODEL=${MODEL:-Qwen/Qwen2-7B-Instruct}
R=results/b1_matrix/raw
case "${1:-}" in
  up)
    P=${2:?需要 UTC 前缀}
    for port in 8100 8200 8300; do
      if ss -ltn 2>/dev/null | grep -q ":$port "; then echo "FATAL: 端口 $port 已被占用"; exit 2; fi
    done
    for f in "$R/${P}_replica2_conc128_server_8100.log" "$R/${P}_replica2_conc128_server_8200.log" "$R/${P}_replica2_conc128_proxy.log"; do
      [ -e "$f" ] && { echo "FATAL: $f 已存在，拒绝覆盖"; exit 2; }
    done
    for i in 0 1; do
      port=$(( 8100 + i*100 ))
      log="$R/${P}_replica2_conc128_server_${port}.log"
      echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"CUDA_VISIBLE_DEVICES=$i $VENV/bin/vllm serve $MODEL --port $port --max-model-len 16384\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"RTX 4090 x2 (this=GPU$i)\" driver=610.57.04 exp=EXP-023" > "$log"
      CUDA_VISIBLE_DEVICES=$i nohup "$VENV/bin/vllm" serve "$MODEL" --port "$port" --max-model-len 16384 >> "$log" 2>&1 &
      echo "started vllm serve :$port (pid $!) -> $log"
    done
    plog="$R/${P}_replica2_conc128_proxy.log"
    echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"$VENV/bin/python matrix/rr_proxy.py --port 8300 --backends 127.0.0.1:8100 127.0.0.1:8200\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"n/a\" driver=610.57.04 exp=EXP-023" > "$plog"
    nohup "$VENV/bin/python" matrix/rr_proxy.py --port 8300 --backends 127.0.0.1:8100 127.0.0.1:8200 >> "$plog" 2>&1 &
    echo "started rr_proxy :8300 (pid $!)"
    # 等两个引擎 /v1/models 就绪（最长 300s）
    t0=$(date +%s)
    for port in 8100 8200; do
      until curl -sf "http://localhost:$port/v1/models" >/dev/null; do
        sleep 3
        if [ $(( $(date +%s) - t0 )) -gt 300 ]; then echo "FATAL: :$port 300s 未就绪"; tail -5 "$R/${P}_replica2_conc128_server_${port}.log"; exit 3; fi
        if grep -q 'OutOfMemoryError\|CUDA out of memory\|Address already in use' "$R/${P}_replica2_conc128_server_${port}.log"; then echo "FATAL: :$port OOM/端口冲突"; exit 4; fi
      done
      echo "ready :$port at +$(( $(date +%s) - t0 ))s"
    done
    until curl -sf http://localhost:8300/health >/dev/null; do sleep 1; done
    echo "ready :8300 $(curl -s http://localhost:8300/health)"
    nvidia-smi --query-gpu=index,memory.used --format=csv
    ;;
  down)
    pkill -f '[v]llm serve' || true
    pkill -f '[r]r_proxy.py' || true
    sleep 5
    left=0
    for p in /proc/[0-9]*; do exe=$(readlink "$p/exe" 2>/dev/null); case "$exe" in *venvs/v0.25.1*) left=1; echo "残留: $p $exe";; esac; done
    [ $left -eq 0 ] && echo "clean: 无 v0.25.1 进程残留"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv
    ;;
  *) echo "用法: $0 up <UTC前缀> | down"; exit 1;;
esac
