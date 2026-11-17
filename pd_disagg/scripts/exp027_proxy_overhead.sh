#!/usr/bin/env bash
# EXP-027：rr_proxy 开销拆分。
#   A 臂（直连对照）：两个 benchmark 进程【同时】各打自己那个实例（各 600 请求、conc64）
#   B 臂（经代理）  ：一个 benchmark 打 rr_proxy（1200 请求、conc128）
# 两臂并发形态一致（两实例并发、每实例约 64 路在飞、总量 1200、512×128），唯一变量是代理。
#
# 用法: bash exp027_proxy_overhead.sh <STAMP>       （栈需已起）
set -uo pipefail
cd "$(dirname "$0")/../.."          # 到 experiments/
VENV=${VENV:-/root/venvs/v0.25.1}
MODEL=${MODEL:-Qwen/Qwen2-7B-Instruct}
R=pd_disagg/results/b1_matrix/raw
STAMP=${1:?需要 UTC 前缀}

bench() {  # $1=tag $2=port $3=num_prompts $4=conc $5=seed
  local tag=$1 port=$2 n=$3 c=$4 seed=$5
  local out="$R/${STAMP}_exp027_${tag}_bench"
  [ -e "${out}.json" ] && { echo "FATAL: ${out}.json 已存在"; return 1; }
  "$VENV/bin/vllm" bench serve --host localhost --port "$port" --model "$MODEL" \
    --dataset-name random --random-input-len 512 --random-output-len 128 \
    --num-prompts "$n" --max-concurrency "$c" --request-rate inf --ignore-eos --seed "$seed" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
    --save-result --save-detailed --result-dir "$R" --result-filename "${tag}_bench.json" \
    > "${out}.log" 2>&1
  local rc=$?
  local res="$R/${tag}_bench.json"
  [ -f "$res" ] && mv "$res" "${out}.json"
  return $rc
}

echo "=== EXP-027 rr_proxy 开销拆分 STAMP=$STAMP"
nvidia-smi --query-compute-apps=pid,process_name --format=csv

echo
echo "########## A 臂：两个直连客户端并发（600+600，各 conc64）"
A_T0=$(date +%s)
bench direct8100 8100 600 64 1099 & P1=$!
bench direct8200 8200 600 64 2099 & P2=$!
wait $P1; RC1=$?
wait $P2; RC2=$?
A_T1=$(date +%s)
echo "  A 臂 exit: 8100=$RC1 8200=$RC2  墙钟=$((A_T1 - A_T0))s"

echo
echo "########## B 臂：经代理（1200，conc128）"
bench proxy 8300 1200 128 1099
RCB=$?
echo "  B 臂 exit=$RCB"

echo
echo "########## 汇总（python 算）"
"$VENV/bin/python" - "$STAMP" <<'PY'
import json, sys, glob
from pathlib import Path
STAMP = sys.argv[1]
R = Path("pd_disagg/results/b1_matrix/raw")

def load(tag):
    p = R / f"{STAMP}_exp027_{tag}_bench.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d["completed"], d["duration"], d["request_throughput"]

a1, a2, b = load("direct8100"), load("direct8200"), load("proxy")
print(f"  A1(8100): completed={a1[0]} wall={a1[1]:.2f}s rate={a1[2]:.3f} req/s")
print(f"  A2(8200): completed={a2[0]} wall={a2[1]:.2f}s rate={a2[2]:.3f} req/s")
tot = a1[0] + a2[0]
agg = tot / max(a1[1], a2[1])
sumrate = a1[2] + a2[2]
print(f"  A_agg（保守: {tot}/max(wall)）= {agg:.3f} req/s ；速率之和（乐观）= {sumrate:.3f} req/s")
print(f"  B(proxy): completed={b[0]} wall={b[1]:.2f}s rate={b[2]:.3f} req/s")
print()
R_cons = (agg - b[2]) / agg * 100
R_opt  = (sumrate - b[2]) / sumrate * 100
print(f"  代理开销 R（保守口径）= {R_cons:+.2f}%")
print(f"  代理开销 R（乐观口径，速率之和）= {R_opt:+.2f}%")
print(f"  判定：{'A 代理有可测开销（≥5%）' if R_cons >= 5 else 'B 代理开销在噪声内（<5%）'}")
PY
nvidia-smi --query-compute-apps=pid,process_name --format=csv
