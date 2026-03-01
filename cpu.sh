#!/usr/bin/env bash
set -euo pipefail

# =========================
# CPU node (has internet)
# - downloads: model + COCO + wheels
# - writes: $SHARED/code/main.py and $SHARED/code/requirements.lock.txt
# =========================

SHARED="${P0_SHARED_DIR:-/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl}"
CODE_DIR="$SHARED/code"
DATA_DIR="$SHARED/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"

MODEL_ID="${P0_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
MODEL_DIR="${P0_MODEL_DIR:-$DATA_DIR/models/Qwen3-VL-8B-Instruct}"

COCO_ROOT="${P0_COCO_ROOT:-$DATA_DIR/datasets/coco_val2017}"
COCO_IMG_DIR="$COCO_ROOT/val2017"
COCO_ANN_DIR="$COCO_ROOT/annotations"
COCO_VAL_ZIP="$COCO_ROOT/val2017.zip"
COCO_ANN_ZIP="$COCO_ROOT/annotations_trainval2017.zip"

PYTHON_BIN="${P0_PYTHON_BIN:-python3}"

echo "================================================================"
echo "[cpu] SHARED     : $SHARED"
echo "[cpu] WHEELHOUSE : $WHEELHOUSE"
echo "[cpu] MODEL_DIR  : $MODEL_DIR"
echo "[cpu] COCO_ROOT  : $COCO_ROOT"
echo "================================================================"

mkdir -p "$CODE_DIR" "$DATA_DIR" "$WHEELHOUSE" "$COCO_ROOT" "$COCO_ANN_DIR"

# 0) Copy main.py into shared/code
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/main.py" ]]; then
  cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"
else
  echo "[cpu] ERROR: main.py not found next to cpu.sh"
  exit 1
fi

# 1) CPU helper venv
CPU_VENV="$SHARED/venv/cpu_env"
if [[ ! -d "$CPU_VENV" ]]; then
  "$PYTHON_BIN" -m venv "$CPU_VENV"
fi
source "$CPU_VENV/bin/activate"
python -m pip install -U pip setuptools wheel huggingface_hub

# 2) COCO downloads (official COCO URLs)
download_zip () {
  local url="$1"
  local out="$2"
  mkdir -p "$(dirname "$out")"
  if [[ -f "$out" ]]; then
    echo "[cpu] zip exists: $out (skip download)"
    return 0
  fi
  echo "[cpu] downloading: $url"
  wget -c -O "$out" "$url"
}

extract_zip () {
  local zip="$1"
  local outdir="$2"
  mkdir -p "$outdir"
  echo "[cpu] verifying zip: $zip"
  unzip -t "$zip" >/dev/null
  echo "[cpu] extracting: $zip -> $outdir"
  unzip -q -o "$zip" -d "$outdir"
}

if [[ ! -d "$COCO_IMG_DIR" || -z "$(ls -A "$COCO_IMG_DIR" 2>/dev/null || true)" ]]; then
  download_zip "http://images.cocodataset.org/zips/val2017.zip" "$COCO_VAL_ZIP"
  extract_zip "$COCO_VAL_ZIP" "$COCO_ROOT"
else
  echo "[cpu] COCO images already present: $COCO_IMG_DIR"
fi

if [[ ! -f "$COCO_ANN_DIR/instances_val2017.json" ]]; then
  download_zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$COCO_ANN_ZIP"
  extract_zip "$COCO_ANN_ZIP" "$COCO_ROOT"
fi

# 3) Model snapshot (Hugging Face) if missing
if [[ ! -d "$MODEL_DIR" || -z "$(ls -A "$MODEL_DIR" 2>/dev/null || true)" ]]; then
  echo "[cpu] snapshot_download model: $MODEL_ID -> $MODEL_DIR"
  export P0_MODEL_ID="$MODEL_ID"
  export P0_MODEL_DIR="$MODEL_DIR"
  python - <<PY
import os
from huggingface_hub import snapshot_download
model_id = os.environ.get("P0_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")
local_dir = os.environ.get("P0_MODEL_DIR", os.environ.get("MODEL_DIR"))
os.makedirs(local_dir, exist_ok=True)
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
print("done:", local_dir)
PY
else
  echo "[cpu] model already present: $MODEL_DIR"
fi

# 4) Wheelhouse for GPU offline install
REQ_FILE="$CODE_DIR/requirements.lock.txt"
cat > "$REQ_FILE" <<'EOF'
pip==26.0.1
setuptools==82.0.0
wheel==0.45.1

torch==2.4.1+cu124

huggingface_hub==1.5.0
tokenizers==0.22.2
qwen-vl-utils==0.0.10
safetensors==0.5.2
sentencepiece==0.2.0

numpy==1.26.4
scipy==1.15.0
pandas==2.2.3
pillow==11.0.0
tqdm==4.67.1
pyyaml==6.0.2
requests==2.32.3
filelock==3.16.1
packaging==24.2
jinja2==3.1.4

matplotlib==3.10.0
scikit-learn==1.5.2
opencv-python-headless==4.10.0.84
EOF

echo "[cpu] requirements written: $REQ_FILE"

echo "[cpu] downloading torch wheels (cu124) into wheelhouse..."
python -m pip download -d "$WHEELHOUSE" --extra-index-url https://download.pytorch.org/whl/cu124 "torch==2.4.1+cu124"

echo "[cpu] downloading PyPI wheels into wheelhouse (excluding torch)..."
WHEELHOUSE="$WHEELHOUSE" REQ_FILE="$REQ_FILE" python - <<'PY'
import os, subprocess, sys, pathlib
wheelhouse=os.environ["WHEELHOUSE"]
req_file=os.environ["REQ_FILE"]
lines=pathlib.Path(req_file).read_text().splitlines()
keep=[]
for ln in lines:
    ln=ln.strip()
    if not ln or ln.startswith("#"):
        continue
    if ln.startswith("torch=="):
        continue
    keep.append(ln)
tmp=pathlib.Path(req_file).with_suffix(".pypi.tmp.txt")
tmp.write_text("\n".join(keep)+"\n")
subprocess.check_call([sys.executable, "-m", "pip", "download", "-d", wheelhouse, "-r", str(tmp)])
print("done")
PY

# transformers wheel: use existing if present; else build from git
if ! ls "$WHEELHOUSE"/transformers-*.whl >/dev/null 2>&1; then
  echo "[cpu] transformers wheel not found; building from git..."
  python -m pip wheel -w "$WHEELHOUSE" "git+https://github.com/huggingface/transformers.git"
else
  echo "[cpu] transformers wheel already present; skip build"
fi

echo "================================================================"
echo "[cpu] DONE"
echo "  code      : $CODE_DIR/main.py"
echo "  req       : $REQ_FILE"
echo "  wheelhouse: $WHEELHOUSE"
echo "  model     : $MODEL_DIR"
echo "  coco      : $COCO_ROOT"
echo "================================================================"
