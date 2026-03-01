#!/usr/bin/env bash
set -euo pipefail

# This file defines:
#   - p0_conda      : wrapper for conda command
#   - p0_conda_run  : run command inside a conda env prefix (no "activate" needed)
#
# Behavior:
#   1) If an existing conda is available, use it.
#   2) Otherwise, bootstrap a local Miniforge under $ROOT_DIR/.runtime/miniforge3
#
# It avoids "system python" problems, and also avoids global pip index configs
# by letting scripts set PIP_CONFIG_FILE=/dev/null.

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNTIME_DIR="${P0_RUNTIME_DIR:-$ROOT_DIR/.runtime}"
MINIFORGE_DIR="${P0_MINIFORGE_DIR:-$RUNTIME_DIR/miniforge3}"
MINIFORGE_INSTALLER_URL_DEFAULT="https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
MINIFORGE_INSTALLER_URL="${P0_MINIFORGE_INSTALLER_URL:-$MINIFORGE_INSTALLER_URL_DEFAULT}"

p0_have_cmd() { command -v "$1" >/dev/null 2>&1; }

p0_conda() {
  if [[ -n "${P0_CONDA_EXE:-}" && -x "${P0_CONDA_EXE:-}" ]]; then
    "${P0_CONDA_EXE}" "$@"
    return
  fi
  if p0_have_cmd conda; then
    conda "$@"
    return
  fi
  if [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
    "$MINIFORGE_DIR/bin/conda" "$@"
    return
  fi
  p0_die "找不到 conda，也没有本地 Miniforge：$MINIFORGE_DIR/bin/conda"
}

p0_conda_run() {
  local prefix="$1"; shift
  p0_conda run -p "$prefix" "$@"
}

p0_bootstrap_miniforge() {
  # Only bootstrap on Linux x86_64, which matches your cluster logs.
  local os arch
  os="$(uname -s || true)"
  arch="$(uname -m || true)"
  [[ "$os" == "Linux" ]] || p0_die "当前脚本默认 Linux；你现在 OS=$os"
  [[ "$arch" == "x86_64" || "$arch" == "amd64" ]] || p0_die "当前脚本默认 x86_64；你现在 ARCH=$arch"

  mkdir -p "$RUNTIME_DIR"
  if [[ -x "$MINIFORGE_DIR/bin/conda" ]]; then
    p0_log "[conda] 使用本地 Miniforge：$MINIFORGE_DIR"
    return
  fi

  p0_log "[conda] 系统无 conda：开始自举 Miniforge -> $MINIFORGE_DIR"
  local installer="$RUNTIME_DIR/Miniforge3-Linux-x86_64.sh"
  # Prefer curl; fallback to wget
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 5 --retry-delay 2 -o "$installer" "$MINIFORGE_INSTALLER_URL"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$installer" "$MINIFORGE_INSTALLER_URL"
  else
    p0_die "缺少 curl/wget，无法下载 Miniforge 安装包。"
  fi

  bash "$installer" -b -p "$MINIFORGE_DIR"
  rm -f "$installer"

  # Speed/robustness tweaks
  "$MINIFORGE_DIR/bin/conda" config --set channel_priority strict >/dev/null 2>&1 || true
  "$MINIFORGE_DIR/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true

  p0_log "[conda] Miniforge OK：$("$MINIFORGE_DIR/bin/conda" --version)"
}

# If "conda" exists, keep using it. Otherwise bootstrap Miniforge.
if ! command -v conda >/dev/null 2>&1; then
  p0_bootstrap_miniforge
fi
