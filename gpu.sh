#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# gpu.sh  (GPU服务器 / 无网)
# - 离线创建 venv
# - 从 wheelhouse 离线安装 requirements.lock.txt
# - 修复 CUDA 动态库优先级：把 venv 的 nvidia/*/lib 放到 LD_LIBRARY_PATH 最前
#
# 关键修复（你当前报错的点）：
# argparse 的“全局参数”必须放在子命令(probe/worker/...)之前：
#   ✅ python main.py --model_path ... --coco_img_dir ... probe
#   ❌ python main.py probe --model_path ...
# ============================================================

# ---- Hardcoded shared dir (按你给的) ----
SHARED="${P0_SHARED_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
CODE_DIR="$SHARED/code"
DATA_DIR="$SHARED/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"

VENV_DIR="$SHARED/venv/p0_env"
LOG_DIR="$SHARED/logs"
RESULT_DIR="$SHARED/results"

NGPU="${1:-8}"
PYTHON_BIN="${P0_PYTHON_BIN:-python3.10}"

MODEL_DIR="${P0_MODEL_DIR:-$DATA_DIR/models/Qwen3-VL-8B-Instruct}"
COCO_IMG_DIR="${P0_COCO_IMG_DIR:-$DATA_DIR/datasets/coco_val2017/val2017}"
COCO_ANN_PATH="${P0_COCO_ANN_PATH:-$DATA_DIR/datasets/coco_val2017/annotations/instances_val2017.json}"

# 是否强制重建 venv（0/1）
FORCE_RECREATE_VENV="${FORCE_RECREATE_VENV:-0}"
# 是否清理旧结果（0/1）
CLEAN_OLD_RESULTS="${CLEAN_OLD_RESULTS:-0}"

echo "================================================================"
echo "[gpu] SHARED     : $SHARED"
echo "[gpu] CODE_DIR   : $CODE_DIR"
echo "[gpu] WHEELHOUSE : $WHEELHOUSE"
echo "[gpu] VENV       : $VENV_DIR"
echo "[gpu] NGPU       : $NGPU"
echo "================================================================"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

if [ "$CLEAN_OLD_RESULTS" = "1" ]; then
  echo "[gpu] cleaning old results/logs (p0a/p0b only)..."
  rm -f "$RESULT_DIR"/p0a_probe_info.json || true
  rm -f "$RESULT_DIR"/p0b_* || true
  rm -f "$LOG_DIR"/p0* || true
fi

REQ_LOCK="$CODE_DIR/requirements.lock.txt"
MAIN_PY="$CODE_DIR/main.py"

if [ ! -f "$REQ_LOCK" ]; then
  echo "[gpu] ERROR: requirements.lock.txt not found at: $REQ_LOCK"
  exit 1
fi
if [ ! -f "$MAIN_PY" ]; then
  echo "[gpu] ERROR: main.py not found at: $MAIN_PY"
  exit 1
fi

if [ "$FORCE_RECREATE_VENV" = "1" ] && [ -d "$VENV_DIR" ]; then
  echo "[gpu] FORCE_RECREATE_VENV=1 -> removing old venv..."
  rm -rf "$VENV_DIR"
fi

echo "[1/4] create/activate venv..."
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

export PYTHONNOUSERSITE=1

echo "[2/4] offline install from wheelhouse..."
if [ ! -d "$WHEELHOUSE" ]; then
  echo "[gpu] ERROR: wheelhouse not found: $WHEELHOUSE"
  exit 1
fi

python -m pip install --no-index --find-links "$WHEELHOUSE" --upgrade pip setuptools wheel >/dev/null
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ_LOCK"

echo "[3/4] fix CUDA libs search path..."
PURELIB="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
SITE="$PURELIB"

# prepend venv nvidia libs
if [ -d "$SITE/nvidia" ]; then
  if [ -d "$SITE/nvidia/nvjitlink/lib" ]; then
    export LD_LIBRARY_PATH="$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
  fi
  for d in "$SITE"/nvidia/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
fi
[ -d "$SITE/torch/lib" ] && export LD_LIBRARY_PATH="$SITE/torch/lib:${LD_LIBRARY_PATH:-}"

# optional preload
NVJIT="$SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12"
if [ -f "$NVJIT" ]; then
  export LD_PRELOAD="${NVJIT}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "  PURELIB=$PURELIB"
echo "  NVJIT  =${NVJIT:-<none>}"
echo "  ✓ LD_LIBRARY_PATH/LD_PRELOAD set"

echo "[4/4] sanity check..."
python - <<'PY'
import os, sys
print("== sanity check ==")
print("python", sys.version.split()[0])
print("LD_LIBRARY_PATH head:", ":".join(os.environ.get("LD_LIBRARY_PATH","").split(":")[:3]))
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
import transformers, huggingface_hub, tokenizers
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

COMMON_ARGS=( \
  --base_dir "$SHARED" \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
)

# probe（单卡）
python "$MAIN_PY" "${COMMON_ARGS[@]}" probe \
  --device 0 | tee "$LOG_DIR/p0a_probe.log"

# worker（多卡）
torchrun --standalone --nproc_per_node="$NGPU" \
  "$MAIN_PY" "${COMMON_ARGS[@]}" worker \
  --num_shards "$NGPU" | tee "$LOG_DIR/p0b_worker_all.log"

# analyze + summary
python "$MAIN_PY" "${COMMON_ARGS[@]}" analyze | tee "$LOG_DIR/p0b_analyze.log"
python "$MAIN_PY" "${COMMON_ARGS[@]}" summary | tee "$LOG_DIR/p0b_summary.log"

echo "================================================================"
echo "✓ DONE"
echo "  logs   : $LOG_DIR"
echo "  results: $RESULT_DIR"
echo "================================================================"
