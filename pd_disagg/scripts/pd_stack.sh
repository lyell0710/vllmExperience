#!/usr/bin/env bash
# pd1p1d 栈起/拆（EXP-024 新增；配置同 EXP-006/007：NIXL pull，1P1D）
#   P: GPU0:8100 side-channel 5600 kv_producer ｜ D: GPU1:8200 side 5601 kv_consumer
#   proxy: smoke/toy_proxy_v0251.py --port 8192 （bench 打 8192，/metrics 直抓 8100/8200）
#   up <UTC前缀> / down
set -uo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-/root/venvs/v0.25.1}
MODEL=${MODEL:-Qwen/Qwen2-7B-Instruct}
R=results/b1_matrix/raw
PDCFG_P='{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}'
PDCFG_D='{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}'

down() {
  local killed=0
  for p in /proc/[0-9]*; do
    c=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$c" in
      *"vllm serve $MODEL"*|*toy_proxy_v0251*) echo "TERM $(basename "$p"): $(echo "$c" | cut -c1-70)"; kill -TERM "$(basename "$p")" 2>/dev/null; killed=$((killed+1));;
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
    for port in 8100 8200 8192; do
      ss -ltn 2>/dev/null | grep -q ":$port " && { echo "FATAL: 端口 $port 被占用"; exit 2; }
    done
    LOGP="$R/${P}_pd1p1d_conc128_P_8100.log"; LOGD="$R/${P}_pd1p1d_conc128_D_8200.log"; LOGX="$R/${P}_pd1p1d_conc128_proxy.log"
    for f in "$LOGP" "$LOGD" "$LOGX"; do [ -e "$f" ] && { echo "FATAL: $f 已存在"; exit 2; }; done
    echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 $VENV/bin/vllm serve $MODEL --port 8100 --max-model-len 16384 --kv-transfer-config '$PDCFG_P'\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2 (this=GPU0 P)\" driver=610.57.04 exp=EXP-024" > "$LOGP"
    echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 $VENV/bin/vllm serve $MODEL --port 8200 --max-model-len 16384 --kv-transfer-config '$PDCFG_D'\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2 (this=GPU1 D)\" driver=610.57.04 exp=EXP-024" > "$LOGD"
    ( CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 nohup "$VENV/bin/vllm" serve "$MODEL" --port 8100 --max-model-len 16384 --kv-transfer-config "$PDCFG_P" >> "$LOGP" 2>&1 & )
    ( CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 nohup "$VENV/bin/vllm" serve "$MODEL" --port 8200 --max-model-len 16384 --kv-transfer-config "$PDCFG_D" >> "$LOGD" 2>&1 & )
    echo "started P:8100 D:8200"
    t0=$(date +%s)
    for port in 8100 8200; do
      until curl -sf "http://localhost:$port/v1/models" >/dev/null; do
        sleep 5
        grep -qE 'OutOfMemoryError|CUDA out of memory|Address already in use' "$LOGP" "$LOGD" && { echo "FATAL: OOM/端口冲突"; exit 4; }
        [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "FATAL: :$port 600s 未就绪"; exit 3; }
      done
      echo "ready :$port at +$(( $(date +%s) - t0 ))s"
    done
    echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"$VENV/bin/python smoke/toy_proxy_v0251.py --port 8192 --prefiller-ports 8100 --decoder-ports 8200\" date=$(date -u +%FT%T+00:00) gpu=n/a driver=610.57.04 exp=EXP-024" > "$LOGX"
    ( nohup "$VENV/bin/python" smoke/toy_proxy_v0251.py --port 8192 --prefiller-ports 8100 --decoder-ports 8200 >> "$LOGX" 2>&1 & )
    until curl -sf http://localhost:8192/health >/dev/null 2>&1 || curl -sf http://localhost:8192/v1/models >/dev/null 2>&1; do
      sleep 2; [ $(( $(date +%s) - t0 )) -gt 120 ] && { echo "WARN: proxy :8192 未探到 /health 或 /v1/models（照常继续，bench 会暴露）"; break; }
    done
    echo "proxy :8192 up"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ;;
  down) down ;;
  *) echo "用法: $0 up <UTC前缀> | down"; exit 1;;
esac
