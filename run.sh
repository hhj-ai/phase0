#!/usr/bin/env bash

# 允许用 sh 运行：自动切回 bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/venv/p0_env"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[P0][run] ERROR: venv 不存在：${VENV_DIR}。先跑 gpu.sh。" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

NPROC="${NPROC:-8}"

set -x
python main.py probe

# 分布式 worker：优先 torchrun
if command -v torchrun >/dev/null 2>&1; then
  torchrun --nproc_per_node="${NPROC}" main.py worker
else
  python -m torch.distributed.run --nproc_per_node="${NPROC}" main.py worker
fi

python main.py analyze
python main.py summary
set +x
