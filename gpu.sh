#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# P0 GPU runner (offline)
# - Creates venv (py3.10)
# - Installs from wheelhouse ONLY
# - Fixes CUDA runtime lib selection (nvJitLink) for torch+cu12
# - Runs: probe -> worker (8 GPUs) -> analyze -> summary
# ============================================================

BASE_DIR="${P0_BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="$DATA_DIR/wheels"
MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"
VENV_DIR="$BASE_DIR/venv/p0_env"

CLEAN_RESULTS="${CLEAN_RESULTS:-1}"  # default: clean old outputs

echo "================================================================"
echo "[gpu] BASE_DIR    : $BASE_DIR"
echo "[gpu] CODE_DIR    : $CODE_DIR"
echo "[gpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[gpu] MODEL_DIR   : $MODEL_DIR"
echo "[gpu] COCO_IMG_DIR: $COCO_IMG_DIR"
echo "[gpu] COCO_ANN    : $COCO_ANN_PATH"
echo "[gpu] RESULT_DIR  : $RESULT_DIR"
echo "[gpu] LOG_DIR     : $LOG_DIR"
echo "[gpu] VENV        : $VENV_DIR"
echo "================================================================"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$BASE_DIR/venv"

# -------------------------------
# 0) Preconditions
# -------------------------------
if [[ ! -f "$CODE_DIR/main.py" ]]; then
  echo "[gpu] ERROR: missing $CODE_DIR/main.py" >&2
  exit 1
fi
if [[ ! -d "$WHEELHOUSE" ]]; then
  echo "[gpu] ERROR: missing wheelhouse: $WHEELHOUSE" >&2
  exit 1
fi
if [[ ! -d "$MODEL_DIR" ]]; then
  echo "[gpu] ERROR: missing model dir: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -d "$COCO_IMG_DIR" || ! -f "$COCO_ANN_PATH" ]]; then
  echo "[gpu] ERROR: missing COCO val2017 or annotations." >&2
  exit 1
fi

# -------------------------------
# 1) Create venv
# -------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[gpu] creating venv..."
  python3.10 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

# -------------------------------
# 2) Install (offline)
# -------------------------------
echo "[gpu] installing from wheelhouse ONLY..."
python -m pip install --no-index --find-links "$WHEELHOUSE" -U pip setuptools wheel || true

# Install torch first (will also install nvidia-* cuda libs from wheelhouse)
python -m pip install --no-index --find-links "$WHEELHOUSE" "torch==2.4.1"

# Install the rest
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.lock.txt"

# -------------------------------
# 3) Fix CUDA runtime library picking (nvJitLink)
#    Symptom: ImportError in torch:
#      libcusparse.so.12: symbol __nvJitLinkComplete_12_4 ... not defined in libnvJitLink.so.12
#    Root cause: dynamic linker picks an older system libnvJitLink.so.12.
#    Fix: prioritize the pip-packaged CUDA libs shipped with torch deps (nvidia/*).
# -------------------------------
SITEPKG="$(python -c "import site; print(site.getsitepackages()[0])")"

NVJIT_DIR="$SITEPKG/nvidia/nvjitlink/lib"
CUSPARSE_DIR="$SITEPKG/nvidia/cusparse/lib"
TORCH_LIB_DIR="$SITEPKG/torch/lib"

if [[ -d "$NVJIT_DIR" ]]; then
  export LD_LIBRARY_PATH="$NVJIT_DIR:$CUSPARSE_DIR:$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"
  if [[ -f "$NVJIT_DIR/libnvJitLink.so.12" ]]; then
    export LD_PRELOAD="$NVJIT_DIR/libnvJitLink.so.12:${LD_PRELOAD:-}"
  fi
fi

# Helpful allocator setting (optional)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "== sanity check =="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"

# -------------------------------
# 4) Clean previous outputs (optional)
# -------------------------------
if [[ "$CLEAN_RESULTS" == "1" ]]; then
  echo "[gpu] cleaning old results/logs..."
  rm -f "$RESULT_DIR"/p0a_* "$RESULT_DIR"/p0b_* "$RESULT_DIR"/*.csv "$RESULT_DIR"/*.png "$RESULT_DIR"/*.json 2>/dev/null || true
  rm -f "$LOG_DIR"/*.log 2>/dev/null || true
fi

# -------------------------------
# 5) Run pipeline
# -------------------------------
echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

# probe (single process)
python "$CODE_DIR/main.py" \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  probe

# worker (8 GPUs)
torchrun --nproc_per_node 8 "$CODE_DIR/main.py" \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  worker \
  --num_shards 8 \
  --num_samples 400 \
  --replace_mode moment_noise \
  --noise_scale 0.10 \
  --t_key_size 4 \
  --lambda_e_values 0.0 0.05 0.1 0.2 0.5 \
  --layers 16 20 24 28 32

# analyze + summary
python "$CODE_DIR/main.py" \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  analyze

python "$CODE_DIR/main.py" \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  summary

echo "================================================================"
echo "✓ DONE"
echo "  logs   : $LOG_DIR"
echo "  results: $RESULT_DIR"
echo "================================================================"
