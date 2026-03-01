#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# P0 GPU runner (offline)
# - Create venv
# - Install from wheelhouse (no internet)
# - Run probe + 8GPU worker + analyze + summary
# ============================================================

BASE_DIR="${P0_BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"

MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"
VENV_DIR="$BASE_DIR/venv/p0_env"

NUM_SAMPLES="${P0_NUM_SAMPLES:-400}"
SEED="${P0_SEED:-0}"
NPROC="${P0_NPROC:-8}"

# Clean old results by default (set P0_CLEAN=0 to keep)
P0_CLEAN="${P0_CLEAN:-1}"

echo "================================================================"
echo "[gpu] BASE_DIR    : $BASE_DIR"
echo "[gpu] CODE_DIR    : $CODE_DIR"
echo "[gpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[gpu] MODEL_DIR   : $MODEL_DIR"
echo "[gpu] COCO_IMG_DIR: $COCO_IMG_DIR"
echo "[gpu] COCO_ANN    : $COCO_ANN_PATH"
echo "[gpu] RESULT_DIR  : $RESULT_DIR"
echo "[gpu] LOG_DIR     : $LOG_DIR"
echo "[gpu] VENV_DIR    : $VENV_DIR"
echo "================================================================"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# Helpful allocator setting to reduce fragmentation in long runs
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# ------------------------------------------------------------
# 0) Create venv
# ------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python -m pip install --no-index --find-links "$WHEELHOUSE" -U pip setuptools wheel || true

# ------------------------------------------------------------
# 1) Install deps offline
# ------------------------------------------------------------
echo "[gpu] installing requirements offline..."
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.lock.txt"

python -c "import torch; print('[gpu] torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"

# ------------------------------------------------------------
# 2) Optional cleanup
# ------------------------------------------------------------
if [[ "$P0_CLEAN" == "1" ]]; then
  echo "[gpu] cleaning old results in $RESULT_DIR ..."
  rm -f "$RESULT_DIR"/p0a_*.json "$RESULT_DIR"/p0b_shard_*_of_*.csv "$RESULT_DIR"/p0b_merged.csv "$RESULT_DIR"/p0b_summary.json "$RESULT_DIR"/*.png || true
fi

# ------------------------------------------------------------
# 3) Run: probe -> worker (torchrun) -> analyze -> summary
# NOTE: global args MUST come before the subcommand.
# ------------------------------------------------------------
COMMON_ARGS=( \
  --base_dir "$BASE_DIR" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
)

echo "================================================================"
echo "[run] probe"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" probe --seed "$SEED"

echo "================================================================"
echo "[run] worker (torchrun)"
echo "================================================================"
torchrun --standalone --nproc_per_node="$NPROC" "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" \
  worker --seed "$SEED" --num_samples "$NUM_SAMPLES"

echo "================================================================"
echo "[run] analyze"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" analyze

echo "================================================================"
echo "[run] summary"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" summary

echo "================================================================"
echo "✓ DONE"
echo "  logs   : $LOG_DIR"
echo "  results: $RESULT_DIR"
echo "================================================================"
