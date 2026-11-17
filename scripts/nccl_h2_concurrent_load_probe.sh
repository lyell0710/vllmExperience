#!/bin/bash
# EXP-020 附录 B（post-hoc）：H2「并发 vLLM 负载压低 SHM 路径 allreduce」旁证。
# 前提：replica2 栈（8100/8200/8300）已起；两卡各剩 ~900 MiB，故 allreduce 缩到 -e 64M。
# 序列：control（栈空闲）×2 → 起负载 bench（8300, conc128, 非记录点）→ 负载中 allreduce ×2 → 等负载结束。
set -uo pipefail
NCCL_LIB=/root/venvs/v0.25.1/lib/python3.12/site-packages/nvidia/nccl/lib
BIN=/root/tools/nccl-tests/build/all_reduce_perf
HW=/root/projects/vllm/experiments/pd_disagg/hw
STAMP=$(date -u +%Y%m%dT%H%M)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
ARGS="-b 1M -e 64M -f 2 -g 2 -n 20"
Q="index,timestamp,pcie.link.gen.current,pcie.link.width.current,clocks.sm,utilization.gpu,memory.used"
run() { # $1=tag
  local out="$HW/${STAMP}_nccl_h2_$1.txt" pcie="$HW/${STAMP}_nccl_h2_$1_pcie.csv"
  [ -e "$out" ] && { echo "FATAL: $out exists"; exit 1; }
  echo "# provenance: env=sys sha=717b683182 cmd=\"nvidia-smi --query-gpu=$Q --format=csv -lms 200\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=$DRV exp=EXP-020-appendixB tag=$1" > "$pcie"
  nvidia-smi --query-gpu=$Q --format=csv -lms 200 >> "$pcie" 2>&1 & local S=$!; sleep 0.5
  { echo "# provenance: env=sys sha=717b683182 cmd=\"env LD_LIBRARY_PATH=$NCCL_LIB NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT $BIN -d float $ARGS\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=$DRV tool_sha=717b683182 nccl=libnccl.so.2.28.9 exp=EXP-020-appendixB tag=$1 note=post-hoc-H2旁证-vllm双实例常驻-显存受限故-e64M"
    env LD_LIBRARY_PATH="$NCCL_LIB" NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT "$BIN" -d float $ARGS 2>&1; echo "# exit_code=$?"; } > "$out"
  sleep 0.5; kill $S 2>/dev/null; wait $S 2>/dev/null
  echo "[$(date -u +%T)] $1 avg=$(grep 'Avg bus' "$out" | awk '{print $NF}') 1M=$(grep -E '^\s+1048576 ' "$out" | awk '{print $8}') 16M=$(grep -E '^\s+16777216 ' "$out" | awk '{print $8}') 64M=$(grep -E '^\s+67108864 ' "$out" | awk '{print $8}') exit=$(grep -o 'exit_code=.*' "$out") util=$(tail -n +3 "$pcie" | awk -F', ' '{u+=$6; n++} END{if(n) printf "%.0f%%", u/n}')"
}
echo "=== H2 probe STAMP=$STAMP $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
run control_idle_r1; run control_idle_r2
BL="$HW/${STAMP}_nccl_h2_loadgen_bench.log"
echo "# provenance: env=ENV-B sha=752a3a5044 cmd=\"vllm bench serve --host localhost --port 8300 --model Qwen/Qwen2-7B-Instruct --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 1200 --ignore-eos --seed 7777 --max-concurrency 128 --request-rate inf\" date=$(date -u +%FT%T+00:00) gpu=\"RTX 4090 x2\" driver=$DRV exp=EXP-020-appendixB note=负载发生器-非测量点-不进runs.jsonl" > "$BL"
/root/venvs/v0.25.1/bin/vllm bench serve --host localhost --port 8300 --model Qwen/Qwen2-7B-Instruct --dataset-name random --random-input-len 512 --random-output-len 128 --num-prompts 1200 --ignore-eos --seed 7777 --max-concurrency 128 --request-rate inf >> "$BL" 2>&1 &
BP=$!; sleep 12
run under_vllm_load_r1; run under_vllm_load_r2
wait $BP; echo "loadgen done: $(grep -E 'Request throughput|Successful' "$BL" | tr '\n' ' ')"
echo "=== H2 probe end $(date -u +%FT%TZ)"
