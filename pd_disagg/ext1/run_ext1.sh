#!/bin/bash
# EXT1 · request 级 KV-wait 关联测量（EXP-013）
# 1P1D pull 臂（EXP-006 同配置）+ EXT1 patch（nixl_req_telemetry_v0251.patch 已打入 ENV-B）
# 产物: raw/EXP-013/{client.jsonl, proxy.log, P.log, D.log, ext1_kv_lines.txt,
#        metrics_{8100,8200}_{before,after}.prom}
set -u
DIR=$(cd "$(dirname "$0")" && pwd)
RAW=$DIR/raw/EXP-013
mkdir -p "$RAW"
source "$DIR/../scripts/provenance.sh"
prov_env B
VENV=/root/venvs/v0.25.1
MODEL=Qwen/Qwen2-7B-Instruct
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 5; pkill -f '[v]llm serve' 2>/dev/null; }
trap cleanup EXIT

KTC='{"kv_connector":"NixlConnector","kv_role":"KVROLE","kv_load_failure_policy":"fail"}'

CUDA_VISIBLE_DEVICES=0 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5600 \
  "$VENV/bin/vllm" serve $MODEL --port 8100 --max-model-len 16384 \
  --kv-transfer-config "${KTC/KVROLE/kv_producer}" > "$RAW/P.log" 2>&1 &
PIDS+=($!)
CUDA_VISIBLE_DEVICES=1 UCX_NET_DEVICES=all VLLM_NIXL_SIDE_CHANNEL_PORT=5601 \
  "$VENV/bin/vllm" serve $MODEL --port 8200 --max-model-len 16384 \
  --kv-transfer-config "${KTC/KVROLE/kv_consumer}" > "$RAW/D.log" 2>&1 &
PIDS+=($!)

wait_up() {
  for i in $(seq 1 200); do
    curl -sf "http://127.0.0.1:$1/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  return 1
}
wait_up 8100 || { echo "P start FAIL"; exit 1; }
wait_up 8200 || { echo "D start FAIL"; exit 1; }
echo "P/D up"

"$VENV/bin/python" "$DIR/ext1_proxy.py" --port 8192 \
  --prefiller-hosts 127.0.0.1 --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 --decoder-ports 8200 > "$RAW/proxy.log" 2>&1 &
PIDS+=($!)
sleep 5

curl -s http://127.0.0.1:8100/metrics > "$RAW/metrics_8100_before.prom"
curl -s http://127.0.0.1:8200/metrics > "$RAW/metrics_8200_before.prom"

PROV=$(prov_line "run_ext1.sh (1P1D pull, EXT1 patch, attribution conc=1)" patch=nixl_req_telemetry_v0251.patch)
"$VENV/bin/python" "$DIR/ext1_client.py" --out "$RAW/client.jsonl" \
  --buckets 512 2048 8192 --num-per-bucket 12 --max-tokens 32 \
  --seed-base 13000 --provenance "$PROV"
CLIENT_RC=$?

curl -s http://127.0.0.1:8100/metrics > "$RAW/metrics_8100_after.prom"
curl -s http://127.0.0.1:8200/metrics > "$RAW/metrics_8200_after.prom"

{ prov_line "grep EXT1_KV from D.log"; grep "EXT1_KV" "$RAW/D.log"; } > "$RAW/ext1_kv_lines.txt"
{ prov_line "grep EXT1_PROXY from proxy.log"; grep "EXT1_PROXY" "$RAW/proxy.log"; } > "$RAW/ext1_proxy_lines.txt"

echo "client_rc=$CLIENT_RC"
echo "EXT1_KV lines: $(grep -c EXT1_KV "$RAW/D.log")"
echo "EXT1_PROXY lines: $(grep -c EXT1_PROXY "$RAW/proxy.log")"
echo "done"
