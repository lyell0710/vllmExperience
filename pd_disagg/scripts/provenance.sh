# provenance.sh — 所有实验脚本 source 此文件，结果文件第一行必须来自 prov_line
# （清单 R0-2 硬性要求；模板见 DECISION.md）
#
# 用法:
#   source .../scripts/provenance.sh
#   prov_env B                     # 预设 ENV-A / ENV-B / ENV-C
#   prov_line "<完整命令>" [额外k=v ...] >> result.txt
#
# 语义约定（修正 smoke 早期记录的 sha 混用问题）:
#   sha = 该 env 里 vllm 代码的 SHA（wheel 安装 = 对应 tag 的 SHA，不是仓库 HEAD）
#   version = 该 env 的 importlib.metadata vllm 版本

prov_env() {
  case "$1" in
    A|a) PROV_SHA=n/a-pypi-0.17.1; PROV_PY=/root/venvs/v0.17.1/bin/python; PROV_ENV=ENV-A ;;
    B|b) PROV_SHA=752a3a5044;      PROV_PY=/root/venvs/v0.25.1/bin/python; PROV_ENV=ENV-B ;;
    C|c) PROV_SHA=$(git -C /root/projects/vllm rev-parse --short=10 HEAD); PROV_PY=/root/venvs/main/bin/python; PROV_ENV=ENV-C ;;
    *) echo "prov_env: unknown env $1" >&2; return 1 ;;
  esac
  export PROV_SHA PROV_PY PROV_ENV
}

prov_line() {
  local cmd="$1"; shift || true
  local ver=n/a
  [ -n "${PROV_PY:-}" ] && ver=$("$PROV_PY" -c \
    "from importlib.metadata import version; print(version('vllm'))" 2>/dev/null || echo n/a)
  local gpu drv
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | paste -sd'+' -)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  echo "# provenance: env=${PROV_ENV:-n/a} sha=${PROV_SHA:-n/a} version=$ver cmd=\"$cmd\" date=$(date -u +%Y-%m-%dT%H:%M:%S+00:00) gpu=\"$gpu\" driver=$drv $*"
}
