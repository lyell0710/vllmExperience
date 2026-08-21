#!/usr/bin/env bash
# R0-5: torch profiler 直控引擎端口。
# 代理不转发 start/stop_profile（EXPERIMENT_PLAN.md 核验 #12），
# 所以 P/D 各自的端口分别打，不打代理。
# 引擎侧前提(v0.25.1): 启动参数带
#   --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=<绝对路径>
# 注意: VLLM_TORCH_PROFILER_DIR 环境变量在 v0.25.1 已弃用(日志报 Unknown env var)；
#       0.17.1 仍用环境变量——版本差异记入 B3。
#
# 用法: profile_ctl.sh start 8100 8200
#       profile_ctl.sh stop  8100 8200
set -e
action=$1; shift
[ "$action" = start ] || [ "$action" = stop ] || { echo "用法: $0 start|stop <port...>" >&2; exit 1; }
for port in "$@"; do
  curl -sf -X POST "http://localhost:${port}/${action}_profile" >/dev/null \
    && echo "[${port}] ${action}_profile OK" \
    || { echo "[${port}] ${action}_profile FAILED" >&2; exit 1; }
done
