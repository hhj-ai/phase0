#!/usr/bin/env bash
set -euo pipefail

# =========================
# P0 CPU (online) bootstrap
# =========================
# - Sync code (main.py) to shared directory
# - Download/prepare:
#   * wheelhouse (for GPU offline install, Python 3.10)
#   * Qwen3-VL model snapshot
#   * COCO val2017 (images + instances_val2017.json)
#
# You can override BASE_DIR by:
#   BASE_DIR=/path/to/p0_qwen3vl bash cpu.sh

BASE_DIR="${BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="$DATA_DIR/wheels"
MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_DIR="$COCO_ROOT/annotations"
COCO_ANN_PATH="$COCO_ANN_DIR/instances_val2017.json"

PIP_PYPI="https://pypi.org/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

# Target GPU env: Python 3.10 on Linux x86_64
DL_FLAGS=(
  --only-binary=:all:
  --platform manylinux2014_x86_64
  --python-version 3.10
  --implementation cp
  --abi cp310
  --abi abi3
)

echo "================================================================"
echo "[cpu] BASE_DIR    : $BASE_DIR"
echo "[cpu] CODE_DIR    : $CODE_DIR"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$WHEELHOUSE" "$DATA_DIR/models" "$DATA_DIR/datasets"

# -------------------------
# 1) Sync code
# -------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "$SCRIPT_DIR/main.py" ]]; then
  echo "[cpu][FATAL] main.py not found next to cpu.sh (expected: $SCRIPT_DIR/main.py)" >&2
  exit 1
fi

echo "[cpu] syncing code -> $CODE_DIR"
cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"

# -------------------------
# 2) Write requirements
# -------------------------
REQ_NO_TORCH="$CODE_DIR/requirements.notorch.txt"
REQ_LOCK="$CODE_DIR/requirements.lock.txt"

cat > "$REQ_NO_TORCH" <<'EOF'
# ==== Core runtime ====
transformers==5.2.0
tokenizers==0.22.2
huggingface-hub==1.5.0
safetensors>=0.5.2
accelerate>=0.34.0
sentencepiece>=0.2.0

# ==== Vision / data / eval ====
pillow>=10.4.0
numpy>=1.26.4
scipy>=1.10.1
pandas>=2.0.3
matplotlib>=3.7.5
scikit-learn>=1.3.2
pycocotools>=2.0.7
tqdm>=4.67.1
einops>=0.8.0

# Qwen utils (if needed by processor)
qwen-vl-utils>=0.0.10
EOF

# Lock file includes torch
cat > "$REQ_LOCK" <<'EOF'
torch==2.4.1+cu124
# (optional) torchvision/torchaudio are not required for this P0,
# but can be useful. Uncomment if you want them in the wheelhouse.
# torchvision==0.19.1+cu124
# torchaudio==2.4.1+cu124

# Mirror the rest:
transformers==5.2.0
tokenizers==0.22.2
huggingface-hub==1.5.0
safetensors>=0.5.2
accelerate>=0.34.0
sentencepiece>=0.2.0
pillow>=10.4.0
numpy>=1.26.4
scipy>=1.10.1
pandas>=2.0.3
matplotlib>=3.7.5
scikit-learn>=1.3.2
pycocotools>=2.0.7
tqdm>=4.67.1
einops>=0.8.0
qwen-vl-utils>=0.0.10
EOF

echo "[cpu] requirements written:"
echo "  - $REQ_NO_TORCH"
echo "  - $REQ_LOCK"

# -------------------------
# 3) Download wheels (official indices)
# -------------------------
PYTHON_BIN="$(command -v python3.10 || true)"
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "[cpu][FATAL] python3 not found" >&2
  exit 1
fi
echo "[cpu] python: $PYTHON_BIN"

# Download pip/setuptools/wheel for GPU venv upgrade (avoid CPU python version constraints)
echo "[cpu] downloading bootstrap wheels (pip/setuptools/wheel) from PyPI for py3.10..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  "${DL_FLAGS[@]}" \
  --no-deps \
  --ignore-requires-python \
  --index-url "$PIP_PYPI" \
  "pip==26.0.1" "setuptools==82.0.0" "wheel==0.43.0"

# Torch (and deps) from official PyTorch index (cu124)
echo "[cpu] downloading torch wheels (cu124) into wheelhouse..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  "${DL_FLAGS[@]}" \
  --index-url "$TORCH_INDEX" \
  --extra-index-url "$PIP_PYPI" \
  "torch==2.4.1+cu124"

# Everything else from PyPI (py310 wheels)
echo "[cpu] downloading remaining wheels into wheelhouse..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  "${DL_FLAGS[@]}" \
  --index-url "$PIP_PYPI" \
  -r "$REQ_NO_TORCH"

echo "[cpu] wheelhouse ready: $WHEELHOUSE"
echo "[cpu] (note) if you previously downloaded cp38/cp39 wheels, you may keep them; GPU install uses --only-binary and will pick compatible ones."

# -------------------------
# 4) Download COCO val2017 (official)
# -------------------------
mkdir -p "$COCO_ROOT" "$COCO_ANN_DIR"
if [[ ! -d "$COCO_IMG_DIR" ]] || [[ -z "$(ls -A "$COCO_IMG_DIR" 2>/dev/null || true)" ]]; then
  echo "[cpu] downloading COCO val2017 images..."
  mkdir -p "$COCO_ROOT/tmp"
  VAL_ZIP="$COCO_ROOT/tmp/val2017.zip"
  wget -c -O "$VAL_ZIP" "https://images.cocodataset.org/zips/val2017.zip"
  unzip -q -o "$VAL_ZIP" -d "$COCO_ROOT"
  rm -f "$VAL_ZIP"
else
  echo "[cpu] COCO images already present: $COCO_IMG_DIR"
fi

if [[ ! -f "$COCO_ANN_PATH" ]]; then
  echo "[cpu] downloading COCO annotations..."
  mkdir -p "$COCO_ROOT/tmp"
  ANN_ZIP="$COCO_ROOT/tmp/annotations_trainval2017.zip"
  wget -c -O "$ANN_ZIP" "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  unzip -q -o "$ANN_ZIP" -d "$COCO_ROOT"
  rm -f "$ANN_ZIP"
  # instances_val2017.json should now exist under $COCO_ROOT/annotations
else
  echo "[cpu] COCO annotations already present: $COCO_ANN_PATH"
fi

# -------------------------
# 5) Download model snapshot (HF official)
# -------------------------
if [[ -d "$MODEL_DIR" ]] && [[ -n "$(ls -A "$MODEL_DIR" 2>/dev/null || true)" ]]; then
  echo "[cpu] model already present: $MODEL_DIR"
else
  echo "[cpu] downloading model snapshot: Qwen/Qwen3-VL-8B-Instruct -> $MODEL_DIR"
  CPU_DL_VENV="$BASE_DIR/venv/cpu_dl_env"
  if [[ ! -d "$CPU_DL_VENV" ]]; then
    "$PYTHON_BIN" -m venv "$CPU_DL_VENV"
  fi
  source "$CPU_DL_VENV/bin/activate"
  python -m pip install -U --index-url "$PIP_PYPI" "pip>=24.1" >/dev/null
  python -m pip install -U --index-url "$PIP_PYPI" "huggingface_hub==1.5.0" "tqdm" >/dev/null
  python - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
  repo_id="Qwen/Qwen3-VL-8B-Instruct",
  local_dir="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/data/models/Qwen3-VL-8B-Instruct",
  local_dir_use_symlinks=False,
  resume_download=True,
)
print("done")
PY
  deactivate
fi

echo "================================================================"
echo "[cpu] DONE"
echo "  - code:      $CODE_DIR/main.py"
echo "  - wheelhouse:$WHEELHOUSE"
echo "  - model:     $MODEL_DIR"
echo "  - coco img:  $COCO_IMG_DIR"
echo "  - coco ann:  $COCO_ANN_PATH"
echo "Next: on GPU node run: bash gpu.sh"
echo "================================================================"
