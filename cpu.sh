#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# P0 CPU bootstrap (with internet)
# - Sync code to shared dir
# - Download model + COCO val2017
# - Build offline wheelhouse for Python 3.10 / manylinux2014
# ============================================================

BASE_DIR="${P0_BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"

MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_DIR="$COCO_ROOT/annotations"
COCO_ANN_PATH="$COCO_ANN_DIR/instances_val2017.json"

echo "================================================================"
echo "[cpu] BASE_DIR    : $BASE_DIR"
echo "[cpu] CODE_DIR    : $CODE_DIR"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$DATA_DIR" "$WHEELHOUSE" "$COCO_ROOT" "$COCO_ANN_DIR"

# ------------------------------------------------------------
# 0) Sync code to shared dir
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[cpu] syncing code -> $CODE_DIR"
cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"

# ------------------------------------------------------------
# 1) Write requirements (single source of truth for GPU offline install)
# ------------------------------------------------------------
REQ_NOTORCH="$CODE_DIR/requirements.notorch.txt"
REQ_LOCK="$CODE_DIR/requirements.lock.txt"

cat > "$REQ_NOTORCH" << 'EOF'
# Core runtime
transformers==5.2.0
tokenizers==0.22.2
huggingface-hub==1.5.0
safetensors==0.5.2
sentencepiece==0.2.0
einops==0.8.0
qwen-vl-utils==0.0.10

# Data / eval
numpy==1.26.4
pandas==2.2.3
scipy==1.15.0
scikit-learn==1.5.2
matplotlib==3.10.0
pillow==11.0.0
pycocotools==2.0.7

# Utilities
requests==2.32.3
pyyaml==6.0.2
protobuf==5.29.3
tqdm==4.67.1
filelock==3.16.1
jinja2==3.1.4
packaging==24.2
EOF

cat > "$REQ_LOCK" << 'EOF'
# Torch family (CUDA 12.4 build comes from PyTorch wheel index; version string is torch==2.4.1)
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1

-r requirements.notorch.txt
EOF

echo "[cpu] requirements written:"
echo "  - $REQ_NOTORCH"
echo "  - $REQ_LOCK"

# ------------------------------------------------------------
# 2) Force official PyPI (avoid any site-wide custom index)
# ------------------------------------------------------------
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

PIP_PY="$(command -v python3 || true)"
if [[ -z "$PIP_PY" ]]; then
  echo "[cpu] ERROR: python3 not found" >&2
  exit 1
fi

# ------------------------------------------------------------
# 3) Build wheelhouse for target (py310, manylinux2014_x86_64)
# ------------------------------------------------------------
TARGET_PYVER="310"
TARGET_PLATFORM="manylinux2014_x86_64"

echo "[cpu] cleaning mismatched wheels (keep cp310 only)..."
find "$WHEELHOUSE" -maxdepth 1 -type f \( -name '*cp38*' -o -name '*cp39*' -o -name '*cp311*' -o -name '*cp312*' \) -print -delete || true

echo "[cpu] downloading torch wheels (cu124) into wheelhouse..."
"$PIP_PY" -m pip download \
  -d "$WHEELHOUSE" \
  --only-binary=:all: \
  --platform "$TARGET_PLATFORM" \
  --python-version "$TARGET_PYVER" \
  --implementation cp \
  --abi cp310 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --extra-index-url https://pypi.org/simple \
  "torch==2.4.1" "torchvision==0.19.1" "torchaudio==2.4.1"

echo "[cpu] downloading remaining wheels from PyPI..."
"$PIP_PY" -m pip download \
  -d "$WHEELHOUSE" \
  --only-binary=:all: \
  --platform "$TARGET_PLATFORM" \
  --python-version "$TARGET_PYVER" \
  --implementation cp \
  --abi cp310 \
  -i https://pypi.org/simple \
  -r "$REQ_NOTORCH"

echo "[cpu] wheelhouse ready: $(ls -1 "$WHEELHOUSE" | wc -l) files"

# ------------------------------------------------------------
# 4) Download model (Hugging Face)
# ------------------------------------------------------------
if [[ -f "$MODEL_DIR/config.json" ]]; then
  echo "[cpu] model already present: $MODEL_DIR"
else
  echo "[cpu] installing huggingface-cli..."
  "$PIP_PY" -m pip install -i https://pypi.org/simple -U "huggingface_hub[cli]==1.5.0"
  echo "[cpu] downloading model -> $MODEL_DIR"
  huggingface-cli download "Qwen/Qwen3-VL-8B-Instruct" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False
fi

# ------------------------------------------------------------
# 5) Download COCO val2017 (images + annotations)
# ------------------------------------------------------------
mkdir -p "$COCO_IMG_DIR" "$COCO_ANN_DIR"

if [[ -f "$COCO_ANN_PATH" ]]; then
  echo "[cpu] COCO annotations already present: $COCO_ANN_PATH"
else
  echo "[cpu] downloading COCO annotations..."
  TMP_ZIP="$COCO_ROOT/annotations_trainval2017.zip"
  wget -O "$TMP_ZIP" -c "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  unzip -o "$TMP_ZIP" -d "$COCO_ROOT"
  rm -f "$TMP_ZIP"
fi

if [[ -d "$COCO_IMG_DIR" ]] && [[ "$(ls -1 "$COCO_IMG_DIR" 2>/dev/null | wc -l)" -gt 10 ]]; then
  echo "[cpu] COCO val images already present: $COCO_IMG_DIR"
else
  echo "[cpu] downloading COCO val2017 images..."
  TMP_ZIP="$COCO_ROOT/val2017.zip"
  wget -O "$TMP_ZIP" -c "http://images.cocodataset.org/zips/val2017.zip"
  unzip -o "$TMP_ZIP" -d "$COCO_ROOT"
  rm -f "$TMP_ZIP"
fi

echo "================================================================"
echo "✓ CPU bootstrap done."
echo "  code      : $CODE_DIR/main.py"
echo "  wheelhouse : $WHEELHOUSE"
echo "  model      : $MODEL_DIR"
echo "  coco imgs  : $COCO_IMG_DIR"
echo "  coco ann   : $COCO_ANN_PATH"
echo "================================================================"
