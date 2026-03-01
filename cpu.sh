#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/common.sh"

p0_log "[cpu] ROOT_DIR   : $ROOT_DIR"
p0_log "[cpu] 目标：自举 python3.10 + 构建 wheelhouse + 创建 CPU venv"

# 1) bootstrap python
bash "$ROOT_DIR/tools/bootstrap_python310.sh"

# 2) build wheelhouse (idempotent)
bash "$ROOT_DIR/tools/build_wheelhouse.sh"

# 3) create venv and install cpu deps offline
bash "$ROOT_DIR/tools/create_venv.sh" "$ROOT_DIR/requirements.cpu.txt" "$ROOT_DIR/constraints.cpu.txt"

p0_log "[cpu] DONE"
p0_log "[cpu] 之后直接用：./venv/p0_env/bin/python main.py ..."
