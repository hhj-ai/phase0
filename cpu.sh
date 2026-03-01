\
#!/usr/bin/env bash
set -euo pipefail

# ========== hardcoded base ==========
BASE_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
CODE_DIR="$BASE_DIR/code"
WHEELHOUSE="$BASE_DIR/data/wheels"
VENV_DIR="$BASE_DIR/venv/p0_env"
MODEL_DIR="$BASE_DIR/data/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$BASE_DIR/data/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_PATH="$COCO_ROOT/annotations/instances_val2017.json"

echo "[cpu] BASE_DIR    : $BASE_DIR"
echo "[cpu] CODE_DIR    : $CODE_DIR"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$WHEELHOUSE" "$BASE_DIR/results" "$BASE_DIR/logs"

echo "[cpu] syncing code -> $CODE_DIR"
# assume you run this script from the folder containing main.py cpu.sh gpu.sh
rsync -av --delete \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  ./ "$CODE_DIR/"

REQ_NOTORCH="$CODE_DIR/requirements.notorch.txt"
REQ_LOCK="$CODE_DIR/requirements.lock.txt"

cat > "$REQ_NOTORCH" <<'EOF'
# core scientific stack
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.5.2
matplotlib==3.9.0
pillow==10.4.0

# hf/qwen
transformers==5.2.0
tokenizers==0.22.2
huggingface-hub==1.5.0
safetensors==0.7.0
accelerate==0.34.2
qwen-vl-utils==0.0.10

# coco
pycocotools==2.0.7

# avoid datasets/fsspec conflict if datasets exists in environment
fsspec[http]==2024.9.0
EOF

# lock file: torch + nvidia deps are installed from wheelhouse on GPU node
cat > "$REQ_LOCK" <<'EOF'
-r requirements.notorch.txt
EOF

echo "[cpu] requirements written:"
echo "  - $REQ_NOTORCH"
echo "  - $REQ_LOCK"

# ----- clean mismatched torch wheels (keep cp310) -----
echo "[cpu] cleaning mismatched torch wheels (keep cp310)..."
find "$WHEELHOUSE" -maxdepth 1 -type f -name "*.whl" | grep -E "cp3(8|9)" || true
find "$WHEELHOUSE" -maxdepth 1 -type f -name "*cp38*.whl" -delete || true
find "$WHEELHOUSE" -maxdepth 1 -type f -name "*cp39*.whl" -delete || true

# ----- ensure bootstrap wheels -----
echo "[cpu] downloading bootstrap wheels (pip/setuptools/wheel) from PyPI..."
python -m pip download -q -d "$WHEELHOUSE" --no-deps --only-binary=:all: \
  "pip==25.0.1" "setuptools==70.3.0" "wheel==0.43.0"

# ----- download non-torch deps for py310 -----
echo "[cpu] downloading non-torch wheels for py310..."
DL_FLAGS=(--only-binary=:all: --platform manylinux2014_x86_64 --python-version 310 --implementation cp --abi cp310)
python -m pip download -q -d "$WHEELHOUSE" "${DL_FLAGS[@]}" -r "$REQ_NOTORCH" --no-deps

# ----- torch wheels (CU124) -----
TORCH_WHL="$WHEELHOUSE/torch-2.4.1+cu124-cp310-cp310-linux_x86_64.whl"
if [[ -f "$TORCH_WHL" ]]; then
  echo "[cpu] torch wheel already present: $TORCH_WHL"
else
  echo "[cpu] fetching torch wheel from official PyTorch index..."
  # official filename uses URL-encoded '+' as %2B
  TORCH_URL="https://download.pytorch.org/whl/cu124/torch-2.4.1%2Bcu124-cp310-cp310-linux_x86_64.whl"
  wget -q -O "$TORCH_WHL" "$TORCH_URL"
  echo "[cpu] saved: $TORCH_WHL"
fi

echo "[cpu] downloading torch runtime deps (nvidia-* / triton) for py310..."
python -m pip download -q -d "$WHEELHOUSE" "${DL_FLAGS[@]}" \
  --index-url "https://download.pytorch.org/whl/cu124" --extra-index-url "https://pypi.org/simple" --only-binary=:all: \
  "triton==3.0.0" \
  "nvidia-cublas-cu12" "nvidia-cuda-nvrtc-cu12" "nvidia-cuda-runtime-cu12" \
  "nvidia-cudnn-cu12" "nvidia-cufft-cu12" "nvidia-curand-cu12" \
  "nvidia-cusolver-cu12" "nvidia-cusparse-cu12" "nvidia-nccl-cu12" \
  "nvidia-nvjitlink-cu12" "nvidia-nvtx-cu12" || true

echo "================================================================"
echo "[cpu] DONE. wheelhouse ready at: $WHEELHOUSE"
echo "      code synced to: $CODE_DIR"
echo "================================================================"
