#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# gpu.sh  (GPU服务器 / 无网)
# - 离线创建 venv
# - 从 wheelhouse 离线安装 requirements.lock.txt
# - 关键修复：安装完成后，把 venv 里的 nvidia/*/lib 放到 LD_LIBRARY_PATH 最前面
#            解决 libnvJitLink / cusparse 版本不匹配导致的 ImportError
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
if [ ! -f "$REQ_LOCK" ]; then
  echo "[gpu] ERROR: requirements.lock.txt not found at: $REQ_LOCK"
  echo "      先在 CPU 机跑 cpu.sh，把 code/requirements.lock.txt 写出来。"
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

# 避免用户 site-packages 污染
export PYTHONNOUSERSITE=1

echo "[2/4] offline install from wheelhouse..."
if [ ! -d "$WHEELHOUSE" ]; then
  echo "[gpu] ERROR: wheelhouse not found: $WHEELHOUSE"
  exit 1
fi

# 先用 wheelhouse 里的 pip/setuptools/wheel 把安装器本身抬到新版本（离线）
python -m pip install --no-index --find-links "$WHEELHOUSE" --upgrade pip setuptools wheel >/dev/null

# 离线安装：严格 no-index
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQ_LOCK"

echo "[3/4] fix CUDA libs search path..."
PURELIB="$(python - <<'PY'
import sysconfig
print(sysconfig.get_paths()["purelib"])
PY
)"
SITE="$PURELIB"

# 把 venv 里的 nvidia/*/lib 放到 LD_LIBRARY_PATH 最前面（覆盖系统旧 CUDA 库）
if [ -d "$SITE/nvidia" ]; then
  # nvjitlink 最优先（你这个错误就是它）
  if [ -d "$SITE/nvidia/nvjitlink/lib" ]; then
    export LD_LIBRARY_PATH="$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
  fi
  for d in "$SITE"/nvidia/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  done
fi
[ -d "$SITE/torch/lib" ] && export LD_LIBRARY_PATH="$SITE/torch/lib:${LD_LIBRARY_PATH:-}"

# LD_PRELOAD 兜底：强制加载 venv 的 nvJitLink（可选，但很稳）
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
import transformers
import huggingface_hub
import tokenizers
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

echo "================================================================"
echo "[run] main.py (probe -> worker -> analyze -> summary)"
echo "================================================================"

# 你这套 main.py 是我前面给你封装的入口
# 约定：main.py 支持子命令：probe / worker / analyze / summary / all
# 如果你的 main.py 不是这个接口，把这里的命令改成你自己的即可

# probe（单卡）
python "$CODE_DIR/main.py" probe \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" | tee "$LOG_DIR/p0a_probe.log"

# worker（多卡）
torchrun --standalone --nproc_per_node="$NGPU" \
  "$CODE_DIR/main.py" worker \
  --model_path "$MODEL_DIR" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH" \
  --result_dir "$RESULT_DIR" \
  --log_dir "$LOG_DIR" \
  --num_shards "$NGPU" | tee "$LOG_DIR/p0b_worker_all.log"

# analyze + summary
python "$CODE_DIR/main.py" analyze --result_dir "$RESULT_DIR" --log_dir "$LOG_DIR" | tee "$LOG_DIR/p0b_analyze.log"
python "$CODE_DIR/main.py" summary --result_dir "$RESULT_DIR" | tee "$LOG_DIR/p0b_summary.log"

echo "================================================================"
echo "✓ DONE"
echo "  logs   : $LOG_DIR"
echo "  results: $RESULT_DIR"
echo "================================================================"
