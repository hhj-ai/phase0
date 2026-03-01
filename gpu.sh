#!/usr/bin/env bash
set -euo pipefail

# =========================
# P0 GPU (offline) runner
# =========================
# Assumes cpu.sh has prepared:
#   - $BASE_DIR/code/main.py
#   - $BASE_DIR/code/requirements.*.txt
#   - $BASE_DIR/data/wheels (wheelhouse)
#   - $BASE_DIR/data/models/Qwen3-VL-8B-Instruct
#   - $BASE_DIR/data/datasets/coco_val2017
#
# You can override BASE_DIR and NGPU:
#   BASE_DIR=/path/to/p0_qwen3vl NGPU=4 bash gpu.sh

BASE_DIR="${BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
NGPU="${NGPU:-8}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="$DATA_DIR/wheels"
MODEL_PATH="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"
VENV_DIR="$BASE_DIR/venv/p0_env"

echo "================================================================"
echo "[gpu] BASE_DIR    : $BASE_DIR"
echo "[gpu] CODE_DIR    : $CODE_DIR"
echo "[gpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[gpu] MODEL_PATH  : $MODEL_PATH"
echo "[gpu] COCO_IMG    : $COCO_IMG_DIR"
echo "[gpu] COCO_ANN    : $COCO_ANN_PATH"
echo "[gpu] RESULT_DIR  : $RESULT_DIR"
echo "[gpu] LOG_DIR     : $LOG_DIR"
echo "[gpu] VENV_DIR    : $VENV_DIR"
echo "[gpu] NGPU        : $NGPU"
echo "================================================================"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$BASE_DIR/venv"

if [[ ! -f "$CODE_DIR/main.py" ]]; then
  echo "[gpu][FATAL] missing $CODE_DIR/main.py (run cpu.sh first)" >&2
  exit 1
fi
if [[ ! -d "$WHEELHOUSE" ]] || [[ -z "$(ls -A "$WHEELHOUSE" 2>/dev/null || true)" ]]; then
  echo "[gpu][FATAL] wheelhouse empty: $WHEELHOUSE (run cpu.sh first)" >&2
  exit 1
fi

# -------------------------
# 0) Clean old result artifacts (avoid mixing old+new)
# -------------------------
echo "[gpu] cleaning old result artifacts..."
rm -f "$RESULT_DIR"/p0a_probe_info.json \
      "$RESULT_DIR"/p0b_shard_*_of_*.csv \
      "$RESULT_DIR"/p0b_merged.csv \
      "$RESULT_DIR"/p0b_summary.json \
      "$RESULT_DIR"/p0b_hist_cp_vs_hal.png \
      "$RESULT_DIR"/p0b_analysis.png || true

# -------------------------
# 1) Create / activate venv
# -------------------------
if [[ ! -d "$VENV_DIR" ]]; then
  python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# -------------------------
# 2) Install from wheelhouse (offline)
# -------------------------
echo "[gpu] upgrading pip/setuptools/wheel from wheelhouse (offline)..."
python -m pip install --no-index --find-links "$WHEELHOUSE" -U pip setuptools wheel

echo "[gpu] installing torch (offline)..."
python -m pip install --no-index --find-links "$WHEELHOUSE" "torch==2.4.1+cu124"

echo "[gpu] installing other deps (offline)..."
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.notorch.txt"

# Make sure CUDA libs from torch deps are visible
echo "[gpu] exporting LD_LIBRARY_PATH for nvidia/*/lib inside venv..."
for d in "$VENV_DIR"/lib/python3.10/site-packages/nvidia/*/lib; do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=1

# Default to use all GPUs if not set
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # 0..(NGPU-1)
  CSV=""
  for ((i=0; i<NGPU; i++)); do
    if [[ -z "$CSV" ]]; then CSV="$i"; else CSV="$CSV,$i"; fi
  done
  export CUDA_VISIBLE_DEVICES="$CSV"
fi
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print('[gpu] torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"

COMMON_ARGS=(
  --base_dir "$BASE_DIR"
  --model_path "$MODEL_PATH"
  --coco_img_dir "$COCO_IMG_DIR"
  --coco_ann_path "$COCO_ANN_PATH"
  --result_dir "$RESULT_DIR"
  --log_dir "$LOG_DIR"
)

# -------------------------
# 3) Run pipeline
# -------------------------
echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" probe --seed 0

torchrun --standalone --nproc_per_node "$NGPU" \
  "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" worker \
  --max_samples 400

python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" analyze
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" summary

echo "================================================================"
echo "[gpu] DONE. outputs in: $RESULT_DIR"
echo "  - p0a_probe_info.json"
echo "  - p0b_shard_XXX_of_YYY.csv"
echo "  - p0b_merged.csv"
echo "  - p0b_summary.json"
echo "  - p0b_hist_cp_vs_hal.png (if matplotlib available)"
echo "================================================================"
