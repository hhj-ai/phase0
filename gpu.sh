\
#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
CODE_DIR="$BASE_DIR/code"
WHEELHOUSE="$BASE_DIR/data/wheels"
VENV_DIR="$BASE_DIR/venv/p0_env"
RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"

MODEL_DIR="$BASE_DIR/data/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$BASE_DIR/data/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

echo "[gpu] BASE_DIR    : $BASE_DIR"
echo "[gpu] CODE_DIR    : $CODE_DIR"
echo "[gpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[gpu] VENV_DIR    : $VENV_DIR"
echo "[gpu] MODEL_DIR   : $MODEL_DIR"
echo "[gpu] COCO_IMG_DIR: $COCO_IMG_DIR"
echo "================================================================"

# ----- env -----
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# ----- venv -----
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[gpu] creating venv..."
  python3.10 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# bootstrap pip from wheelhouse (avoid network)
python -m pip install --no-index --find-links "$WHEELHOUSE" -U "pip==25.0.1" "setuptools==70.3.0" "wheel==0.43.0" >/dev/null

# install torch + deps from wheelhouse
TORCH_WHL="$WHEELHOUSE/torch-2.4.1+cu124-cp310-cp310-linux_x86_64.whl"
if [[ ! -f "$TORCH_WHL" ]]; then
  echo "[gpu] ERROR: missing torch wheel: $TORCH_WHL"
  exit 2
fi

python -m pip install --no-index --find-links "$WHEELHOUSE" "$TORCH_WHL" >/dev/null

# Ensure torch-bundled CUDA libs win over system libs to avoid nvJitLink/cusparse mismatch.
TORCH_SITE="$VENV_DIR/lib/python3.10/site-packages/torch"
export LD_LIBRARY_PATH="$TORCH_SITE/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/cusparse/lib:$VENV_DIR/lib/python3.10/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

# install rest deps (offline)
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.lock.txt" >/dev/null

python -c "import torch; print('[gpu] torch', torch.__version__, 'cuda', torch.version.cuda, 'ngpu', torch.cuda.device_count())"

# ----- clean outputs (fresh run) -----
echo "[gpu] cleaning old results/logs..."
rm -f "$RESULT_DIR"/p0a_probe_info.json "$RESULT_DIR"/p0b_shard_* "$RESULT_DIR"/p0b_merged.csv "$RESULT_DIR"/p0b_summary.json "$RESULT_DIR"/*.png 2>/dev/null || true
mkdir -p "$RESULT_DIR" "$LOG_DIR"

echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

# 1) probe (single GPU)
python "$CODE_DIR/main.py" probe \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  --layers 16 20 24 28 32

# 2) worker (multi-GPU via torchrun)
torchrun --standalone --nproc_per_node 8 "$CODE_DIR/main.py" worker \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  --num_shards 8 \
  --num_samples 400 \
  --layers 16 20 24 28 32

# 3) analyze
python "$CODE_DIR/main.py" analyze --result_dir "$RESULT_DIR" --log_dir "$LOG_DIR"

# 4) summary
python "$CODE_DIR/main.py" summary --result_dir "$RESULT_DIR" --log_dir "$LOG_DIR"

echo "================================================================"
echo "✓ DONE. outputs in: $RESULT_DIR"
echo "================================================================"
