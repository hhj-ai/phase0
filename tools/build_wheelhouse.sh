#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/tools/common.sh"
p0_pip_sanitize_env

WHEELHOUSE="$ROOT_DIR/offline_wheels/py310"
REQ="$ROOT_DIR/requirements.cpu.txt"
CONSTRAINT="$ROOT_DIR/constraints.cpu.txt"
MARK="$WHEELHOUSE/.wheelhouse.ok"

PY_BIN="$ROOT_DIR/.runtime/python310/bin/python3.10"

if [[ ! -x "$PY_BIN" ]]; then
  p0_die "找不到自举 python：$PY_BIN（先跑 tools/bootstrap_python310.sh 或 cpu.sh）"
fi

mkdir -p "$WHEELHOUSE"

# Fingerprint = requirements + constraints + this script hash
fp_req="$(p0_sha256 "$REQ")"
fp_con="$(p0_sha256 "$CONSTRAINT")"
fp_self="$(p0_sha256 "$ROOT_DIR/tools/build_wheelhouse.sh")"
FP="req=$fp_req con=$fp_con self=$fp_self"

if [[ -f "$MARK" ]]; then
  old="$(cat "$MARK" 2>/dev/null || true)"
  if [[ "$old" == "$FP" ]]; then
    p0_log "[wheelhouse] 已存在且指纹一致，跳过下载/构建。"
    exit 0
  fi
  p0_log "[wheelhouse] 指纹变化，将增量更新 wheelhouse（不会删除你手工放进去的轮子）。"
fi

# Create downloader venv (only used for download/build wheels)
DL_VENV="$ROOT_DIR/.runtime/downloader_venv"
DL_PY="$DL_VENV/bin/python"
if [[ ! -x "$DL_PY" ]]; then
  p0_log "[wheelhouse] 创建 downloader venv: $DL_VENV"
  "$PY_BIN" -m venv "$DL_VENV"
fi

p0_log "[wheelhouse] 升级 downloader venv 的构建工具..."
"$DL_PY" -m pip install --upgrade "pip>=24.3" "setuptools>=70" "wheel>=0.41" "build>=1.2" -i https://pypi.org/simple --isolated

# Build deps for sdist-only packages
"$DL_PY" -m pip install --upgrade "cython>=0.29" "numpy" -i https://pypi.org/simple --isolated

p0_log "[wheelhouse] 下载 requirements (wheel/sdist) ..."
"$DL_PY" -m pip download -r "$REQ" -c "$CONSTRAINT" -d "$WHEELHOUSE" -i https://pypi.org/simple --isolated

# Build wheels for sdist-only packages and drop wheels into wheelhouse
build_one_sdist() {
  local name="$1"   # package name in file pattern
  local pattern="$WHEELHOUSE/${name//-/_}-"*"tar.gz"
  local sdist=""
  # pick newest by sort
  sdist="$(ls -1 $pattern 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "$sdist" ]]; then
    # some sdists use '-' not '_' or vice versa
    sdist="$(ls -1 "$WHEELHOUSE/$name-"*.tar.gz 2>/dev/null | sort | tail -n 1 || true)"
  fi
  if [[ -z "$sdist" ]]; then
    p0_die "没找到 $name 的 sdist（$WHEELHOUSE 下没有 *.tar.gz）。"
  fi

  p0_log "[wheelhouse] build wheel from sdist: $(basename "$sdist")"
  "$DL_PY" -m pip wheel --no-deps --no-build-isolation "$sdist" -w "$WHEELHOUSE" -i https://pypi.org/simple --isolated
}

# These two commonly don't ship manylinux wheels on PyPI:
build_one_sdist "pycocotools"
build_one_sdist "qwen_vl_utils"

# Validate we have wheels now
p0_log "[wheelhouse] 核查关键 wheel..."
ls -1 "$WHEELHOUSE" | head -n 5 >/dev/null

echo -n "$FP" > "$MARK"
p0_log "[wheelhouse] OK -> $WHEELHOUSE"
