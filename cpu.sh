#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Phase0 CPU launcher (safe-by-default)
# - Runs IN-PLACE inside your git repo (no copying / no rsync / no deleting).
# - Creates a local venv that *reuses* your current site-packages to avoid
#   reinstalling heavy deps (torch/transformers) unless missing.
# -----------------------------------------------------------------------------

BASE_DIR="${P0_BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_DIR"

# Optional: restore tracked files that were deleted by an earlier rsync --delete
# (won't recover untracked directories).
if [[ "${P0_RESTORE_GIT:-0}" == "1" ]]; then
  if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "[P0] Restoring tracked files from git (checkout -- .)"
    git checkout -- .
    git submodule update --init --recursive || true
  else
    echo "[P0] P0_RESTORE_GIT=1 set but current dir is not a git repo. Skipping."
  fi
fi

VENV_DIR="${P0_VENV_DIR:-$BASE_DIR/venv/p0_env}"
PYTHON_BIN="${P0_PYTHON:-python3}"

mkdir -p "$(dirname "$VENV_DIR")" "$BASE_DIR/data" "$BASE_DIR/results" "$BASE_DIR/logs"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "[P0] Creating venv (with --system-site-packages) at: $VENV_DIR"
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Lightweight dependency check: install only if missing.
python - <<'PY'
import importlib, sys
need = [
  'numpy','pandas','PIL','tqdm','pycocotools','matplotlib'
]
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

status=$?
if [[ $status -eq 2 ]]; then
  echo "[P0] Installing missing lightweight deps into venv..."
  # If you have an offline wheelhouse, set P0_WHEELHOUSE to it.
  WHEELHOUSE="${P0_WHEELHOUSE:-$BASE_DIR/data/wheels}"
  if [[ -d "$WHEELHOUSE" ]] && compgen -G "$WHEELHOUSE/*.whl" > /dev/null; then
    python -m pip install --no-index --find-links "$WHEELHOUSE" -r requirements.notorch.txt
  else
    python -m pip install -r requirements.notorch.txt
  fi
fi

# Run a single-process worker on CPU (small N by default)
MODEL_PATH="${P0_MODEL_PATH:-$BASE_DIR/models/Qwen3-VL-8B-Instruct}"
COCO_ROOT="${P0_COCO_ROOT:-$BASE_DIR/data/coco}"
N_SAMPLES="${P0_N_SAMPLES:-100}"

python main.py probe \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_PATH" \
  --coco_root "$COCO_ROOT" \
  --n_samples 8

python main.py worker \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_PATH" \
  --coco_root "$COCO_ROOT" \
  --device cpu \
  --n_samples "$N_SAMPLES" \
  --num_shards 1 \
  --shard_idx 0 \
  --layers ${P0_LAYERS:-16 20 24 28 32}

python main.py analyze --base_dir "$BASE_DIR"
python main.py summary --base_dir "$BASE_DIR"

echo "[P0] Done. Results: $BASE_DIR/results  Logs: $BASE_DIR/logs"
