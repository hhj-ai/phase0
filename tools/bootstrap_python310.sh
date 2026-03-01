#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/tools/common.sh"

p0_pip_sanitize_env

ARCH="$(uname -m || true)"
OS="$(uname -s || true)"

# A stable, widely mirrored CPython 3.10 standalone build (tar.gz; no zstd needed).
# Source: indygreg/python-build-standalone release assets (install_only build).
PY_URL_DEFAULT="https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz"

PY_URL="${P0_PYTHON_TARBALL_URL:-$PY_URL_DEFAULT}"

RUNTIME_DIR="$ROOT_DIR/.runtime"
PY_DIR="$RUNTIME_DIR/python310"
PY_BIN="$PY_DIR/bin/python3.10"

p0_log "[py] ROOT_DIR   : $ROOT_DIR"
p0_log "[py] RUNTIME   : $RUNTIME_DIR"
p0_log "[py] PY_DIR    : $PY_DIR"
p0_log "[py] PY_URL    : $PY_URL"

if [[ "$OS" != "Linux" ]]; then
  p0_die "这套脚本默认只支持 Linux（当前 OS=$OS）。"
fi
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
  p0_die "这套脚本默认只支持 x86_64（当前 ARCH=$ARCH）。"
fi

if [[ -x "$PY_BIN" ]]; then
  p0_log "[py] Python 3.10 已存在：$PY_BIN"
  "$PY_BIN" -V
  exit 0
fi

mkdir -p "$PY_DIR"
TMP="$RUNTIME_DIR/_python310.tgz"

p0_log "[py] 下载 Python 3.10 standalone..."
p0_download "$PY_URL" "$TMP"

p0_log "[py] 解压..."
tar -xzf "$TMP" -C "$PY_DIR"
rm -f "$TMP"

if [[ ! -x "$PY_BIN" ]]; then
  p0_die "解压后仍找不到 $PY_BIN（可能 tarball 结构变了）。"
fi

p0_log "[py] 初始化 pip (ensurepip)..."
"$PY_BIN" -m ensurepip --upgrade || true

p0_log "[py] 升级 pip/setuptools/wheel..."
"$PY_BIN" -m pip install --upgrade "pip>=24.3" "setuptools>=70" "wheel>=0.41" -i https://pypi.org/simple --isolated

p0_log "[py] OK: $("$PY_BIN" -V)"
