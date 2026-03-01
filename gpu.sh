#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${BASH_VERSION:-}" ]]; then exec bash "$0" "$@"; fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/tools/common.sh"
source "$ROOT_DIR/tools/bootstrap_miniforge.sh"

# -----------------------------
# GPU env config
# -----------------------------
ENV_DIR="${P0_GPU_ENV_DIR:-$ROOT_DIR/venv/p0_gpu}"
REQ_FILE="${P0_GPU_REQ_FILE:-$ROOT_DIR/requirements.gpu.txt}"

p0_log "[gpu] ROOT_DIR : $ROOT_DIR"
p0_log "[gpu] ENV_DIR  : $ENV_DIR"
p0_log "[gpu] REQ_FILE : $REQ_FILE"

p0_require_file "$REQ_FILE"

REQ_HASH="$(p0_sha256_file "$REQ_FILE")"
STAMP_FILE="$ENV_DIR/.p0_gpu_${REQ_HASH}.stamp"

if [[ -f "$STAMP_FILE" && -z "${P0_FORCE:-}" ]]; then
  p0_log "[gpu] 已安装且 requirements 未变化：跳过安装（P0_FORCE=1 可强制重装）"
  p0_log "[gpu] 用法："
  p0_log "      $ROOT_DIR/tools/p0_run.sh gpu python -c \"import torch; print(torch.__version__, torch.version.cuda)\""
  exit 0
fi

# (Re)create env if missing
if [[ ! -d "$ENV_DIR" || ! -x "$ENV_DIR/bin/python" ]]; then
  p0_log "[gpu] 创建 conda env (python=3.10) ..."
  p0_conda create -y -p "$ENV_DIR" python=3.10 pip
else
  p0_log "[gpu] conda env 已存在：$ENV_DIR"
fi

export PIP_CONFIG_FILE="${PIP_CONFIG_FILE:-/dev/null}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1

p0_log "[gpu] 安装 PyTorch + CUDA (conda) ..."
# This tends to be the most stable way to avoid libnvJitLink / libcusparse mismatch issues.
# Channels order matters: pytorch, nvidia, conda-forge.
p0_conda install -y -p "$ENV_DIR" -c pytorch -c nvidia -c conda-forge \
  pytorch torchvision torchaudio pytorch-cuda=12.4

p0_log "[gpu] 安装 pycocotools (conda-forge) ..."
p0_conda install -y -p "$ENV_DIR" -c conda-forge pycocotools

p0_log "[gpu] 升级 pip/setuptools/wheel ..."
p0_conda_run "$ENV_DIR" python -m pip install -U pip setuptools wheel --index-url https://pypi.org/simple

# requirements.gpu.txt currently references requirements.cpu.txt via "-r ...".
# Keep that behavior; just use pip for the non-torch stack.
p0_log "[gpu] pip 安装 requirements（强制走 PyPI HTTPS）..."
p0_conda_run "$ENV_DIR" python -m pip install -r "$REQ_FILE" --index-url https://pypi.org/simple

p0_log "[gpu] 写入安装 stamp ..."
mkdir -p "$ENV_DIR"
rm -f "$ENV_DIR"/.p0_gpu_*.stamp || true
date > "$STAMP_FILE"

p0_log "[gpu] 记录 freeze（方便以后复现）..."
p0_conda_run "$ENV_DIR" python -m pip freeze > "$ENV_DIR/requirements.freeze.txt" || true

p0_log "[gpu] OK"
p0_log "[gpu] 运行示例："
p0_log "      $ROOT_DIR/tools/p0_run.sh gpu python -c \"import torch; print(torch.__version__, torch.version.cuda)\""
