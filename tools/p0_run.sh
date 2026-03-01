#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT_DIR/tools/common.sh"
source "$ROOT_DIR/tools/bootstrap_miniforge.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {cpu|gpu} <command...>"
  exit 2
fi

which_env="$1"; shift
case "$which_env" in
  cpu) ENV_DIR="${P0_CPU_ENV_DIR:-$ROOT_DIR/venv/p0_cpu}" ;;
  gpu) ENV_DIR="${P0_GPU_ENV_DIR:-$ROOT_DIR/venv/p0_gpu}" ;;
  *) p0_die "unknown env: $which_env (use cpu|gpu)" ;;
esac

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  p0_die "env 不存在或未初始化：$ENV_DIR（先跑 cpu.sh / gpu.sh）"
fi

# Run inside env without needing 'conda activate'
exec p0_conda_run "$ENV_DIR" "$@"
