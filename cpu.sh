#!/usr/bin/env bash
set -euo pipefail

# =========================
# P0 CPU bootstrap (official sources)
# - sync code to SHARED/code
# - download COCO val2017
# - download model (optional, if missing)
# - download wheels into SHARED/data/wheels for offline GPU install
# =========================

# -------- Hardcoded project root --------
SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

CODE_DIR="$SHARED/code"
DATA_DIR="$SHARED/data"
WHEELHOUSE="$DATA_DIR/wheels"
MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"

COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_DIR="$COCO_ROOT/annotations"
COCO_ANN_PATH="$COCO_ANN_DIR/instances_val2017.json"

echo "================================================================"
echo "[cpu] SHARED      : $SHARED"
echo "[cpu] CODE_DIR    : $CODE_DIR"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$WHEELHOUSE" "$MODEL_DIR" "$COCO_ROOT" "$COCO_ANN_DIR"

# -------- Sync code (expects main.py/cpu.sh/gpu.sh in the same folder as this script) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[cpu] syncing code -> $CODE_DIR"
cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"
cp -f "$SCRIPT_DIR/cpu.sh"  "$CODE_DIR/cpu.sh"  || true
cp -f "$SCRIPT_DIR/gpu.sh"  "$CODE_DIR/gpu.sh"  || true
chmod +x "$CODE_DIR/"*.sh || true

# -------- Write requirements (pin versions that exist on PyPI) --------
REQ_NO_TORCH="$CODE_DIR/requirements.notorch.txt"
REQ_LOCK="$CODE_DIR/requirements.lock.txt"

cat > "$REQ_NO_TORCH" <<'REQ'
# Core runtime (pin to known-good versions)
transformers==5.2.0
tokenizers==0.22.2
huggingface-hub==1.5.0
safetensors>=0.4.3
accelerate>=0.30.0
# vision/io
pillow>=10.3.0
opencv-python-headless>=4.9.0.80
# data
numpy>=1.26.4
pandas>=2.2.2
tqdm>=4.66.4
pyyaml>=6.0.1
requests>=2.32.3
# plotting (optional, but nice for p0b_hist)
matplotlib>=3.8.4
# Qwen VL helper (if your code uses it)
qwen-vl-utils>=0.0.8
REQ

cat > "$REQ_LOCK" <<'REQ'
# Torch stack (use PyTorch CUDA12.4 index to get the right wheels)
torch==2.4.1
# optional: torchvision/torchaudio if needed
# torchvision==0.19.1
# torchaudio==2.4.1
REQ

# append the no-torch stack
cat "$REQ_NO_TORCH" >> "$REQ_LOCK"

echo "[cpu] requirements written:"
echo "  - $REQ_NO_TORCH"
echo "  - $REQ_LOCK"

# -------- Clean mismatched wheels (keep cp310) --------
echo "[cpu] cleaning mismatched torch wheels (keep cp310)..."
find "$WHEELHOUSE" -maxdepth 1 -type f \( -name "*cp38*.whl" -o -name "*cp39*.whl" -o -name "*cp311*.whl" -o -name "*cp312*.whl" \) -print -delete || true

# -------- Ensure pip tooling is present --------
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

# Always force official sources (ignore any global pip.conf mirror)
PIP_PYPI="https://pypi.org/simple"
PIP_TORCH_CU124="https://download.pytorch.org/whl/cu124"

echo "[cpu] downloading bootstrap wheels (pip/setuptools/wheel) from PyPI..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  --only-binary=:all: --no-deps \
  --index-url "$PIP_PYPI" \
  "pip==26.0.1" "setuptools==82.0.0" "wheel==0.45.1"

echo "[cpu] downloading torch wheels (CUDA 12.4) into wheelhouse..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 \
  --index-url "$PIP_TORCH_CU124" --extra-index-url "$PIP_PYPI" \
  -r "$REQ_LOCK"

echo "[cpu] downloading remaining wheels (no-torch) into wheelhouse..."
"$PYTHON_BIN" -m pip download -d "$WHEELHOUSE" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310 \
  --index-url "$PIP_PYPI" \
  -r "$REQ_NO_TORCH"

echo "[cpu] wheelhouse size: $(ls -1 "$WHEELHOUSE" | wc -l) files"

# -------- Download COCO val2017 (official COCO host) --------
download_and_unzip () {
  local url="$1"
  local zip_path="$2"
  local dst_dir="$3"

  if [ -d "$dst_dir" ] && [ "$(find "$dst_dir" -maxdepth 1 -type f | wc -l)" -gt 10 ]; then
    echo "[cpu] already present: $dst_dir"
    return
  fi

  mkdir -p "$(dirname "$zip_path")"
  echo "[cpu] downloading: $url"
  curl -L --retry 5 --retry-delay 2 -o "$zip_path" "$url"

  echo "[cpu] testing zip: $zip_path"
  unzip -t "$zip_path" >/dev/null

  echo "[cpu] extracting -> $dst_dir"
  unzip -q "$zip_path" -d "$COCO_ROOT"
}

download_and_unzip "https://images.cocodataset.org/zips/val2017.zip" "$COCO_ROOT/val2017.zip" "$COCO_IMG_DIR"

if [ ! -f "$COCO_ANN_PATH" ]; then
  echo "[cpu] downloading COCO annotations..."
  curl -L --retry 5 --retry-delay 2 -o "$COCO_ROOT/annotations_trainval2017.zip" \
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
  unzip -q "$COCO_ROOT/annotations_trainval2017.zip" -d "$COCO_ROOT"
fi

# -------- Optional: download model from HF if missing --------
if [ -f "$MODEL_DIR/config.json" ]; then
  echo "[cpu] model already present: $MODEL_DIR"
else
  echo "[cpu] model missing; attempting HF download (requires internet + permissions)..."
  "$PYTHON_BIN" -m pip install --user --index-url "$PIP_PYPI" "huggingface-hub==1.5.0" >/dev/null
  # NOTE: replace repo id if your local folder maps to a different HF repo
  huggingface-cli download "Qwen/Qwen3-VL-8B-Instruct" \
    --local-dir "$MODEL_DIR" --local-dir-use-symlinks False || {
      echo "[cpu] HF download failed. If you already have the model elsewhere, copy it into:"
      echo "      $MODEL_DIR"
    }
fi

echo "================================================================"
echo "✓ DONE"
echo "  code    : $CODE_DIR"
echo "  wheels  : $WHEELHOUSE"
echo "  coco    : $COCO_ROOT"
echo "  model   : $MODEL_DIR"
echo "================================================================"
