#!/usr/bin/env bash
set -euo pipefail

# =========================
# P0 GPU runner (offline)
# - creates/uses venv
# - installs from wheelhouse
# - cleans old results
# - probe -> torchrun worker -> analyze -> summary
# =========================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

CODE_DIR="$SHARED/code"
DATA_DIR="$SHARED/data"
WHEELHOUSE="$DATA_DIR/wheels"
VENV="$SHARED/venv/p0_env"

MODEL_PATH="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

RESULT_DIR="$SHARED/results"
LOG_DIR="$SHARED/logs"

NGPU="${NGPU:-8}"
SEED="${SEED:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-400}"
SCORE_MODE="${SCORE_MODE:-ratio}"   # ced / delta / ratio

echo "================================================================"
echo "[gpu] SHARED      : $SHARED"
echo "[gpu] CODE_DIR    : $CODE_DIR"
echo "[gpu] VENV        : $VENV"
echo "[gpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[gpu] MODEL_PATH  : $MODEL_PATH"
echo "[gpu] COCO_IMG    : $COCO_IMG_DIR"
echo "[gpu] COCO_ANN    : $COCO_ANN_PATH"
echo "[gpu] RESULT_DIR  : $RESULT_DIR"
echo "[gpu] LOG_DIR     : $LOG_DIR"
echo "[gpu] NGPU        : $NGPU"
echo "[gpu] SCORE_MODE  : $SCORE_MODE"
echo "================================================================"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

# If scheduler doesn't set CUDA_VISIBLE_DEVICES, set it explicitly (prevents all ranks piling onto cuda:0 in some weird envs)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(python - <<PY
import torch
n = torch.cuda.device_count()
print(",".join(str(i) for i in range(n)))
PY
)"
  export CUDA_VISIBLE_DEVICES
fi
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Reduce fragmentation risk
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Offline mode: never hit the network on GPU nodes
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ------- Create venv if missing -------
if [ ! -x "$VENV/bin/python3.10" ]; then
  echo "[gpu] creating venv: $VENV"
  python3.10 -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"

# ------- Prefer venv NVIDIA libs (fixes libnvJitLink/cusparse symbol mismatch) -------
NVIDIA_LIBS="$("$VENV/bin/python3.10" - <<'PY'
import site, glob, os
sp = site.getsitepackages()[0]
paths = sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib")))
print(":".join(paths))
PY
)"
if [ -n "$NVIDIA_LIBS" ]; then
  export LD_LIBRARY_PATH="$NVIDIA_LIBS:${LD_LIBRARY_PATH:-}"
fi

# ------- Install from wheelhouse (no index) -------
echo "[gpu] installing pip tooling (offline)..."
pip install --no-index --find-links "$WHEELHOUSE" "pip==26.0.1" "setuptools==82.0.0" "wheel==0.45.1" >/dev/null

echo "[gpu] installing requirements (offline)..."
pip install --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.lock.txt" >/dev/null

echo "[gpu] sanity check..."
python - <<'PY'
import torch, tokenizers, transformers, huggingface_hub
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

# ------- Clean old results (avoid mixing runs) -------
echo "[gpu] cleaning old results under: $RESULT_DIR"
rm -f "$RESULT_DIR"/p0a_* "$RESULT_DIR"/p0b_* || true

COMMON_ARGS=(
  --base_dir "$SHARED"
  --model_path "$MODEL_PATH"
  --coco_img_dir "$COCO_IMG_DIR"
  --coco_ann_path "$COCO_ANN_PATH"
  --result_dir "$RESULT_DIR"
  --log_dir "$LOG_DIR"
)

echo "================================================================"
echo "[run] probe"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" probe --seed "$SEED"

echo "================================================================"
echo "[run] worker (torchrun)"
echo "================================================================"
torchrun --standalone --nproc_per_node="$NGPU" \
  "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" worker \
  --seed "$SEED" --num_shards "$NGPU" --num_samples "$NUM_SAMPLES"

echo "================================================================"
echo "[run] analyze"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" analyze --score_mode "$SCORE_MODE"

echo "================================================================"
echo "[run] summary"
echo "================================================================"
python "$CODE_DIR/main.py" "${COMMON_ARGS[@]}" summary --score_mode "$SCORE_MODE"

echo "================================================================"
echo "✓ DONE"
echo "  results: $RESULT_DIR"
echo "================================================================"
