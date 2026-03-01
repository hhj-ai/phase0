#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Phase0 multi-GPU launcher (safe-by-default)
# - Runs IN-PLACE inside your git repo (no copying / no rsync / no deleting).
# - Uses a local venv with --system-site-packages to reuse your existing CUDA
#   PyTorch/Transformers install (so it doesn't "touch your env").
# -----------------------------------------------------------------------------

BASE_DIR="${P0_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_DIR"

# Optional: clean outputs (off by default)
if [[ "${P0_CLEAN:-0}" == "1" ]]; then
  echo "[P0] Cleaning results/logs (P0_CLEAN=1)"
  rm -rf "$BASE_DIR/results" "$BASE_DIR/logs"
fi
mkdir -p "$BASE_DIR/results" "$BASE_DIR/logs" "$BASE_DIR/data"

VENV_DIR="${P0_VENV_DIR:-$BASE_DIR/venv/p0_env}"
PYTHON_BIN="${P0_PYTHON:-python3}"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[P0] Creating venv (with --system-site-packages) at: $VENV_DIR"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Quick sanity checks (no installs unless you explicitly request it)
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
  print('cuda device:', torch.cuda.get_device_name(0))
PY

# Install lightweight deps only if missing
python - <<'PY'
import importlib, sys
need = ['numpy','pandas','PIL','tqdm','pycocotools','matplotlib']
missing=[]
for m in need:
  try:
    importlib.import_module(m)
  except Exception:
    missing.append(m)
if missing:
  print('MISSING:' + ' '.join(missing))
  sys.exit(2)
print('OK')
PY

if [[ $? -eq 2 ]]; then
  if [[ "${P0_INSTALL_DEPS:-1}" == "1" ]]; then
    echo "[P0] Installing missing lightweight deps into venv..."
    WHEELHOUSE="${P0_WHEELHOUSE:-$BASE_DIR/data/wheels}"
    if [[ -d "$WHEELHOUSE" ]] && compgen -G "$WHEELHOUSE/*.whl" > /dev/null; then
      python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.notorch.txt
    else
      python -m pip install -r requirements.notorch.txt
    fi
  else
    echo "[P0] Missing deps detected, but P0_INSTALL_DEPS=0, aborting."
    exit 2
  fi
fi

MODEL_PATH="${P0_MODEL_PATH:-$BASE_DIR/models/Qwen3-VL-8B-Instruct}"
COCO_ROOT="${P0_COCO_ROOT:-$BASE_DIR/data/coco}"
GPU_NUM="${P0_GPU_NUM:-8}"
N_SAMPLES="${P0_N_SAMPLES:-2000}"
LAYERS=( ${P0_LAYERS:-16 20 24 28 32} )

# 1) Probe
python main.py probe \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_PATH" \
  --coco_root "$COCO_ROOT" \
  --n_samples 8

# 2) Worker (sharded automatically by torchrun env RANK/WORLD_SIZE)
LOG_PREFIX="$BASE_DIR/logs/worker_$(date +%Y%m%d_%H%M%S)"

torchrun --standalone --nproc_per_node "$GPU_NUM" \
  main.py worker \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_PATH" \
  --coco_root "$COCO_ROOT" \
  --device cuda \
  --n_samples "$N_SAMPLES" \
  --layers "${LAYERS[@]}" \
  2>&1 | tee "${LOG_PREFIX}.log"

# 3) Analyze + Summary
python main.py analyze --base_dir "$BASE_DIR"
python main.py summary --base_dir "$BASE_DIR"

echo "[P0] Done. Results: $BASE_DIR/results  Logs: $BASE_DIR/logs"
