#!/usr/bin/env bash
set -euo pipefail

# =========================
# GPU node (NO internet)
# - creates venv
# - installs from wheelhouse
# - runs: probe + workers + analyze + summary
# =========================

SHARED="${P0_SHARED_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
CODE_DIR="$SHARED/code"
DATA_DIR="$SHARED/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"
VENV_DIR="$SHARED/venv/p0_env"
LOG_DIR="$SHARED/logs"
RESULT_DIR="$SHARED/results"

NGPU="${1:-8}"
PYTHON_BIN="${P0_PYTHON_BIN:-python3}"

MODEL_DIR="${P0_MODEL_DIR:-$DATA_DIR/models/Qwen3-VL-8B-Instruct}"
COCO_IMG_DIR="${P0_COCO_IMG_DIR:-$DATA_DIR/datasets/coco_val2017/val2017}"
COCO_ANN_PATH="${P0_COCO_ANN_PATH:-$DATA_DIR/datasets/coco_val2017/annotations/instances_val2017.json}"

echo "================================================================"
echo "[gpu] SHARED     : $SHARED"
echo "[gpu] CODE_DIR   : $CODE_DIR"
echo "[gpu] WHEELHOUSE : $WHEELHOUSE"
echo "[gpu] VENV       : $VENV_DIR"
echo "[gpu] NGPU       : $NGPU"
echo "================================================================"

mkdir -p "$CODE_DIR" "$LOG_DIR" "$RESULT_DIR"

# 0) Sync main.py into shared/code (expects main.py next to gpu.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/main.py" ]]; then
  cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"
else
  echo "[gpu] ERROR: main.py not found next to gpu.sh"
  exit 1
fi

# 1) Create venv
if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

REQ_FILE="$CODE_DIR/requirements.lock.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[gpu] ERROR: $REQ_FILE not found. Run cpu.sh first."
  exit 1
fi

# 2) Offline install
python -m pip install --no-index --find-links "$WHEELHOUSE" "pip==26.0.1" "setuptools==82.0.0" "wheel==0.45.1"
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ_FILE"
# ensure transformers is installed from wheelhouse
python -m pip install --no-index --find-links "$WHEELHOUSE" transformers

python -c "import transformers; print('transformers', transformers.__version__)"

# 3) Force offline behavior for HF
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="$SHARED/.hf"
export TRANSFORMERS_CACHE="$SHARED/.hf/transformers"
export HF_DATASETS_CACHE="$SHARED/.hf/datasets"

# 4) Fix CUDA lib selection (avoid nvJitLink symbol mismatch):
export LD_LIBRARY_PATH="$(python - <<'PY'
import site, os, glob
sp = site.getsitepackages()[0]
libs=[]
for p in glob.glob(os.path.join(sp, "nvidia", "*", "lib")):
    libs.append(p)
print(":".join(libs))
PY
):${LD_LIBRARY_PATH:-}"

echo "== sanity check =="
python - <<'PY'
import torch, transformers, huggingface_hub, tokenizers
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

# 5) Probe
echo "== run: probe =="
python "$CODE_DIR/main.py" \
  --base_dir "$SHARED" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  probe \
  --device 0 \
  --replace_mode moment_noise \
  --noise_scale 0.15 \
  --t_key_size 4 \
  > "$LOG_DIR/p0a_probe.log" 2>&1
echo "[gpu] probe log: $LOG_DIR/p0a_probe.log"

# 6) Workers (one per GPU, explicit sharding)
echo "== run: workers ($NGPU shards) =="
pids=()
for ((i=0;i<NGPU;i++)); do
  echo "[gpu] start worker $i"
  CUDA_VISIBLE_DEVICES=$i \
  python "$CODE_DIR/main.py" \
    --base_dir "$SHARED" \
    --model_path "$MODEL_DIR" \
    --coco_img_dir "$COCO_IMG_DIR" \
    --coco_ann_path "$COCO_ANN_PATH" \
    --result_dir "$RESULT_DIR" \
    --log_dir "$LOG_DIR" \
    worker \
    --device 0 \
    --shard_idx $i \
    --num_shards $NGPU \
    --num_samples 400 \
    --seed 42 \
    --replace_mode moment_noise \
    --noise_scale 0.15 \
    --t_key_size 4 \
    > "$LOG_DIR/p0b_w${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [[ "$fail" -ne 0 ]]; then
  echo "[gpu] ERROR: some workers failed. Check $LOG_DIR/p0b_w*.log"
  exit 1
fi

# 7) Analyze + summary
echo "== run: analyze =="
python "$CODE_DIR/main.py" --base_dir "$SHARED" --result_dir "$RESULT_DIR" --log_dir "$LOG_DIR" analyze \
  > "$LOG_DIR/p0b_analyze.log" 2>&1

echo "== run: summary =="
python "$CODE_DIR/main.py" --base_dir "$SHARED" --result_dir "$RESULT_DIR" --log_dir "$LOG_DIR" summary \
  | tee "$LOG_DIR/p0b_summary.log"

echo "================================================================"
echo "[gpu] DONE"
echo "  logs   : $LOG_DIR"
echo "  results: $RESULT_DIR"
echo "================================================================"
