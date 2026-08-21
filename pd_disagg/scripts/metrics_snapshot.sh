#!/usr/bin/env bash
# B1 Gate 机械化：直抓引擎 /metrics（不抓代理），保存全量快照；
# 对比两份快照输出 KV-transfer 相关计数器增量（誊入 runs.jsonl 的 gates 字段）。
#
# 用法:
#   metrics_snapshot.sh snap 8100 snapshots/<前缀>_8100_before.prom
#   metrics_snapshot.sh snap 8100 snapshots/<前缀>_8100_after.prom
#   metrics_snapshot.sh diff snapshots/<前缀>_8100_before.prom snapshots/<前缀>_8100_after.prom
set -euo pipefail
PAT='nixl|kv_transfer|kv_load|expired|kv_cache_transfer'

case "${1:-}" in
  snap)
    curl -sf "http://localhost:$2/metrics" > "$3"
    echo "saved $3 ($(grep -cE "$PAT" "$3" || true) 行命中 KV-transfer 模式)"
    ;;
  diff)
    norm() { grep -E "$PAT" "$1" | grep -v '^#' \
             | sed -E 's/[[:space:]]+([0-9.eE+-]+)$/\t\1/' | sort; }
    join -t $'\t' -j 1 <(norm "$2") <(norm "$3") \
    | awk -F'\t' '{ d = $3 - $2
        # failed/expired 是"必须为 0"的 gate 证据，Δ=0 也要打印
        if (d != 0 || $1 ~ /failed|expired/)
            printf "%-90s Δ=%g (%g -> %g)\n", $1, d, $2, $3 }'
    echo "--- (其余 Δ=0 计数器已省略；gate 关注 bytes/transfers 的 sum/count 增量与 failed/expired=0) ---"
    ;;
  *)
    echo "用法: $0 snap <port> <file> | diff <before> <after>" >&2; exit 1
    ;;
esac
