#!/usr/bin/env bash
# Common helpers for phase0 scripts.
set -euo pipefail

# If user runs "sh cpu.sh", re-exec with bash (dash doesn't support some things).
if [[ -z "${BASH_VERSION:-}" ]]; then
  exec bash "$0" "$@"
fi

p0_root_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

p0_log() {
  echo "[P0] $*"
}

p0_die() {
  echo "[P0] ERROR: $*" 1>&2
  exit 1
}

# Make pip ignore any global pip.conf / env mirrors.
p0_pip_sanitize_env() {
  export PIP_CONFIG_FILE=/dev/null
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  export PIP_NO_PYTHON_VERSION_WARNING=1
  export PYTHONDONTWRITEBYTECODE=1
}

p0_find_downloader() {
  if command -v curl >/dev/null 2>&1; then
    echo "curl"
  elif command -v wget >/dev/null 2>&1; then
    echo "wget"
  else
    return 1
  fi
}

p0_download() {
  # p0_download URL OUTFILE
  local url="$1"
  local out="$2"
  local dl
  dl="$(p0_find_downloader)" || p0_die "需要 curl 或 wget 才能下载：$url"
  if [[ "$dl" == "curl" ]]; then
    curl -L --retry 6 --retry-delay 2 --fail -o "$out" "$url"
  else
    wget -O "$out" "$url"
  fi
}

p0_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    p0_die "没有 sha256sum/shasum，无法做稳定指纹。"
  fi
}
