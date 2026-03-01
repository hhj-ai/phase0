#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/tools/common.sh"
p0_pip_sanitize_env

PY_BIN="$ROOT_DIR/.runtime/python310/bin/python3.10"
VENV_DIR="$ROOT_DIR/venv/p0_env"
WHEELHOUSE="$ROOT_DIR/offline_wheels/py310"

REQ="${1:-$ROOT_DIR/requirements.cpu.txt}"
CONSTRAINT="${2:-$ROOT_DIR/constraints.cpu.txt}"

if [[ ! -x "$PY_BIN" ]]; then
  p0_die "找不到 python3.10：$PY_BIN"
fi
if [[ ! -d "$WHEELHOUSE" ]]; then
  p0_die "找不到 wheelhouse：$WHEELHOUSE（先跑 tools/build_wheelhouse.sh）"
fi

mkdir -p "$(dirname "$VENV_DIR")"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  p0_log "[venv] Creating venv: $VENV_DIR"
  "$PY_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

p0_log "[venv] Upgrading pip/setuptools/wheel..."
"$VENV_PY" -m pip install --upgrade "pip>=24.3" "setuptools>=70" "wheel>=0.41" -i https://pypi.org/simple --isolated

p0_log "[venv] Installing from wheelhouse (offline) ..."
"$VENV_PY" -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ" -c "$CONSTRAINT"

p0_log "[venv] OK: $("$VENV_PY" -V)"
