#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/common.sh"
source "$ROOT_DIR/tools/bootstrap_miniforge.sh"

# -----------------------------
# CPU env config
# -----------------------------
ENV_DIR="${P0_CPU_ENV_DIR:-$ROOT_DIR/venv/p0_cpu}"
REQ_FILE="${P0_CPU_REQ_FILE:-$ROOT_DIR/requirements.cpu.txt}"

p0_log "[cpu] ROOT_DIR : $ROOT_DIR"
p0_log "[cpu] ENV_DIR  : $ENV_DIR"
p0_log "[cpu] REQ_FILE : $REQ_FILE"

p0_require_file "$REQ_FILE"

# A simple "don't redownload/reinstall" stamp:
# - If requirements file doesn't change, we skip pip install.
REQ_HASH="$(p0_sha256_file "$REQ_FILE")"
STAMP_FILE="$ENV_DIR/.p0_cpu_${REQ_HASH}.stamp"

if [[ -f "$STAMP_FILE" && -z "${P0_FORCE:-}" ]]; then
  p0_log "[cpu] 已安装且 requirements 未变化：跳过安装（P0_FORCE=1 可强制重装）"
  p0_log "[cpu] 用法："
  p0_log "      $ROOT_DIR/tools/p0_run.sh cpu python -c \"import pycocotools; print('pycocotools OK')\""
  exit 0
fi

# (Re)create env if missing
if [[ ! -d "$ENV_DIR" || ! -x "$ENV_DIR/bin/python" ]]; then
  p0_log "[cpu] 创建 conda env (python=3.10) ..."
  p0_conda create -y -p "$ENV_DIR" python=3.10 pip
else
  p0_log "[cpu] conda env 已存在：$ENV_DIR"
fi

# Make pip ignore user/global configs that may force a bad index.
export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

p0_log "[cpu] 升级 pip/setuptools/wheel ..."
p0_conda_run "$ENV_DIR" python -m pip install -U pip setuptools wheel --index-url https://pypi.org/simple

# Prefer conda-forge for pycocotools to avoid "No matching distribution" from strange indexes.
p0_log "[cpu] 安装 pycocotools (conda-forge) ..."
p0_conda install -y -p "$ENV_DIR" -c conda-forge pycocotools

p0_log "[cpu] pip 安装 requirements（强制走 PyPI HTTPS）..."
p0_conda_run "$ENV_DIR" python -m pip install -r "$REQ_FILE" --index-url https://pypi.org/simple

p0_log "[cpu] 写入安装 stamp ..."
mkdir -p "$ENV_DIR"
rm -f "$ENV_DIR"/.p0_cpu_*.stamp || true
date > "$STAMP_FILE"

p0_log "[cpu] 记录 freeze（方便以后复现）..."
p0_conda_run "$ENV_DIR" python -m pip freeze > "$ENV_DIR/requirements.freeze.txt" || true

p0_log "[cpu] OK"
p0_log "[cpu] 运行示例："
p0_log "      $ROOT_DIR/tools/p0_run.sh cpu python -c \"import pycocotools; print('pycocotools OK')\""
