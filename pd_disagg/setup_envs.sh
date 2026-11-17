#!/bin/bash
# 建三个 vLLM 环境: v0.17.1 / v0.25.1 / main(editable, 预编译)
# 日志: setup_envs.log。幂等:已装成功的环境跳过。
set -x
export PATH="$HOME/.local/bin:$PATH"

ok() { "$1/bin/python" -c "import vllm" >/dev/null 2>&1; }

if ! ok ~/venvs/v0.17.1; then
  uv venv ~/venvs/v0.17.1 --python 3.12 &&
  uv pip install -p ~/venvs/v0.17.1/bin/python "vllm==0.17.1" --torch-backend=auto
fi

if ! ok ~/venvs/v0.25.1; then
  uv venv ~/venvs/v0.25.1 --python 3.12 &&
  uv pip install -p ~/venvs/v0.25.1/bin/python "vllm==0.25.1" --torch-backend=auto
fi

if ! ok ~/venvs/main; then
  uv venv ~/venvs/main --python 3.12 &&
  cd /root/vllm &&
  VLLM_USE_PRECOMPILED=1 uv pip install -p ~/venvs/main/bin/python -e /root/vllm --torch-backend=auto
fi

echo "=== FINAL CHECK ==="
for v in v0.17.1 v0.25.1 main; do
  ~/venvs/$v/bin/python -c "import vllm; print('$v ->', vllm.__version__)" 2>&1 | tail -1
done
