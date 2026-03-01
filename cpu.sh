#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# P0 CPU downloader (official sources only)
# - Downloads: wheelhouse (py310), COCO val2017 + annotations
# - Syncs code (main.py) into $BASE_DIR/code
# ============================================================

BASE_DIR="${P0_BASE_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="$DATA_DIR/wheels"
MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_DIR="$COCO_ROOT/annotations"
COCO_ANN_PATH="$COCO_ANN_DIR/instances_val2017.json"

# Official package indexes (NO internal mirrors)
PYPI_INDEX="https://pypi.org/simple"
TORCH_INDEX_CU124="https://download.pytorch.org/whl/cu124"

echo "================================================================"
echo "[cpu] BASE_DIR    : $BASE_DIR"
echo "[cpu] CODE_DIR    : $CODE_DIR"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$WHEELHOUSE" "$COCO_ROOT" "$DATA_DIR/models" "$DATA_DIR/datasets"

# -------------------------------
# 0) Sync code
# -------------------------------
echo "[cpu] syncing code -> $CODE_DIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/main.py" ]]; then
  cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"
else
  echo "[cpu] WARNING: main.py not found next to cpu.sh. You must copy main.py into $CODE_DIR manually."
fi

# -------------------------------
# 1) Write pinned requirements
# -------------------------------
REQ_NOTORCH="$CODE_DIR/requirements.notorch.txt"
REQ_LOCK="$CODE_DIR/requirements.lock.txt"

cat > "$REQ_NOTORCH" <<'REQEOF'
# NOTE: torch is installed separately from the official PyTorch index.
transformers==5.2.0
tokenizers==0.22.2
huggingface_hub==1.5.0
qwen-vl-utils==0.0.10

# common runtime deps
numpy==1.26.4
pandas==2.2.3
pillow==11.0.0
matplotlib==3.10.0
tqdm==4.67.1
scikit-learn==1.5.2
scipy==1.15.0
pycocotools==2.0.7
protobuf==5.29.3
pyyaml==6.0.2
requests==2.32.3
einops==0.8.0
filelock==3.16.1
packaging==24.2
jinja2==3.1.4
fsspec==2024.10.0
typing_extensions==4.12.2
regex==2024.11.6
safetensors==0.5.2
sentencepiece==0.2.0
REQEOF

# lock == notorch (we keep one lock file to install on GPU)
cp -f "$REQ_NOTORCH" "$REQ_LOCK"

echo "[cpu] requirements written:"
echo "  - $REQ_NOTORCH"
echo "  - $REQ_LOCK"

# -------------------------------
# 2) Clean obviously wrong wheels
# -------------------------------
echo "[cpu] cleaning mismatched wheels (drop cp38 only; keep abi3) ..."
find "$WHEELHOUSE" -maxdepth 1 -type f -name '*cp38*' -print -delete || true

# -------------------------------
# 3) Download wheelhouse (py310)
# -------------------------------
echo "[cpu] downloading wheels for target: py310 manylinux2014_x86_64 ..."

# Use pip download with target python/platform so the CPU host python version doesn't matter.
PYDL_FLAGS=(
  "--dest" "$WHEELHOUSE"
  "--only-binary=:all:"
  "--platform" "manylinux2014_x86_64"
  "--implementation" "cp"
  "--python-version" "310"
  "--abi" "cp310"
  "--index-url" "$PYPI_INDEX"
  "--trusted-host" "pypi.org"
  "--trusted-host" "files.pythonhosted.org"
)

python -m pip download "${PYDL_FLAGS[@]}" -r "$REQ_LOCK"

# Download torch (+ CUDA deps) from the official torch index.
# IMPORTANT: Use torch==2.4.1 (no "+cu124" in spec); the cu124 index provides the CUDA build.
echo "[cpu] downloading torch (+ CUDA deps) from official PyTorch index (cu124)..."
python -m pip download \
  --dest "$WHEELHOUSE" \
  --only-binary=:all: \
  --platform manylinux2014_x86_64 \
  --implementation cp \
  --python-version 310 \
  --abi cp310 \
  --index-url "$TORCH_INDEX_CU124" \
  --extra-index-url "$PYPI_INDEX" \
  --trusted-host "download.pytorch.org" \
  --trusted-host "pypi.org" \
  --trusted-host "files.pythonhosted.org" \
  "torch==2.4.1"

# -------------------------------
# 4) COCO val2017 + annotations
# -------------------------------
echo "[cpu] preparing COCO val2017 ..."
mkdir -p "$COCO_ROOT" "$COCO_ANN_DIR"

download_if_missing () {
  local url="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "[cpu] exists: $out"
    return 0
  fi
  echo "[cpu] downloading: $url"
  curl -L --retry 5 --retry-delay 3 -o "$out" "$url"
}

if [[ ! -d "$COCO_IMG_DIR" || ! -f "$COCO_ANN_PATH" ]]; then
  TMPDIR="$COCO_ROOT/.tmp"
  mkdir -p "$TMPDIR"
  VAL_ZIP="$TMPDIR/val2017.zip"
  ANN_ZIP="$TMPDIR/annotations_trainval2017.zip"

  download_if_missing "http://images.cocodataset.org/zips/val2017.zip" "$VAL_ZIP"
  download_if_missing "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$ANN_ZIP"

  echo "[cpu] unzipping val2017..."
  mkdir -p "$COCO_ROOT"
  unzip -q -n "$VAL_ZIP" -d "$COCO_ROOT"
  echo "[cpu] unzipping annotations..."
  unzip -q -n "$ANN_ZIP" -d "$COCO_ROOT"

  if [[ ! -d "$COCO_IMG_DIR" ]]; then
    echo "[cpu] ERROR: val2017 not found after unzip: $COCO_IMG_DIR" >&2
    exit 1
  fi
  if [[ ! -f "$COCO_ANN_PATH" ]]; then
    echo "[cpu] ERROR: instances_val2017.json not found after unzip: $COCO_ANN_PATH" >&2
    exit 1
  fi
else
  echo "[cpu] COCO already present: $COCO_IMG_DIR"
fi

# -------------------------------
# 5) Model check (download is optional)
# -------------------------------
if [[ -d "$MODEL_DIR" ]]; then
  echo "[cpu] model already present: $MODEL_DIR"
else
  echo "[cpu] WARNING: model dir not found: $MODEL_DIR"
  echo "[cpu] This script does NOT auto-download the model by default."
  echo "[cpu] Place Qwen3-VL-8B-Instruct under: $MODEL_DIR"
fi

echo "================================================================"
echo "[cpu] DONE"
echo "  - code synced to: $CODE_DIR"
echo "  - wheelhouse     : $WHEELHOUSE"
echo "  - COCO img dir   : $COCO_IMG_DIR"
echo "  - COCO ann path  : $COCO_ANN_PATH"
echo "================================================================"
