#!/bin/bash
# NIXL 1P1D smoke test — 版本裁决用
# 用法: run_1p1d_smoke.sh <venv路径> <标签> <toy_proxy路径>
# 验收三项: ①1P1D 请求跑通 ②日志有 "KV Transfer metrics:" ③failure_policy=fail 被接受
VENV=$1; LABEL=$2; PROXY_PY=$3
DIR=$(dirname "$0")
MODEL=Qwen/Qwen2.5-0.5B-Instruct
LOG_P=$DIR/smoke_${LABEL}_P.log
LOG_D=$DIR/smoke_${LABEL}_D.log
LOG_X=$DIR/smoke_${LABEL}_proxy.log
RESULT=$DIR/smoke_${LABEL}_result.txt
PIDS=()

cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 3; for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT

VLLM_VER=$("$VENV/bin/python" -c "import vllm; print(vllm.__version__)")
SHA=$(git -C /root/vllm rev-parse --short HEAD 2>/dev/null)
{
echo "# provenance: sha=$SHA version=$VLLM_VER cmd=\"run_1p1d_smoke.sh $VENV $LABEL\" kv_load_failure_policy=fail date=$(date -Is)"
echo "label=$LABEL venv=$VENV"
} > "$RESULT"

KTC='{"kv_connector":"NixlConnector","kv_role":"KVROLE","kv_load_failure_policy":"fail"}'

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  "$VENV/bin/vllm" serve $MODEL --port 8100 --enforce-eager --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --kv-transfer-config "${KTC/KVROLE/kv_producer}" > "$LOG_P" 2>&1 &
PIDS+=($!)

CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  "$VENV/bin/vllm" serve $MODEL --port 8200 --enforce-eager --max-model-len 2048 \
  --gpu-memory-utilization 0.7 \
  --kv-transfer-config "${KTC/KVROLE/kv_consumer}" > "$LOG_D" 2>&1 &
PIDS+=($!)

wait_up() {
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    kill -0 "$2" 2>/dev/null || { echo "process for port $1 died" >> "$RESULT"; return 1; }
    sleep 2
  done
  echo "port $1 timeout" >> "$RESULT"; return 1
}
wait_up 8100 "${PIDS[0]}" || { echo "VERDICT=FAIL(P 启动失败)" >> "$RESULT"; exit 1; }
wait_up 8200 "${PIDS[1]}" || { echo "VERDICT=FAIL(D 启动失败)" >> "$RESULT"; exit 1; }
echo "check3_failure_policy=PASS (both instances started with fail policy)" >> "$RESULT"

"$VENV/bin/python" "$PROXY_PY" --port 8192 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$LOG_X" 2>&1 &
PIDS+=($!)
sleep 5

ok=0
for i in 1 2; do
  resp=$(curl -sf -X POST http://127.0.0.1:8192/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"'$MODEL'","prompt":"San Francisco is a","max_tokens":16,"temperature":0}')
  text=$(echo "$resp" | "$VENV/bin/python" -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null)
  [ -n "$text" ] && ok=$((ok+1)) && echo "req$i text: $text" >> "$RESULT"
done
if [ "$ok" -eq 2 ]; then echo "check1_e2e=PASS (2/2 requests)" >> "$RESULT"; else echo "check1_e2e=FAIL ($ok/2)" >> "$RESULT"; fi

sleep 8   # 等 metrics 周期性日志刷出
if grep -q "KV Transfer metrics" "$LOG_P" "$LOG_D"; then
  echo "check2_metrics=PASS" >> "$RESULT"
  grep -h "KV Transfer metrics" "$LOG_P" "$LOG_D" | tail -2 >> "$RESULT"
else
  echo "check2_metrics=FAIL(日志未出现)" >> "$RESULT"
fi

grep -c "ERROR" "$LOG_P" "$LOG_D" | sed 's/^/errors_in_/' >> "$RESULT"
cat "$RESULT"
