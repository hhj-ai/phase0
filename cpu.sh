#!/bin/bash
set -euo pipefail

# ============================================================
# cpu.sh  (CPU服务器 / 有网)
# - 只负责：准备离线依赖(wheelhouse)、模型、COCO val2017
# - 强制使用官方源：PyPI + PyTorch 官方 cu124
# - 目标运行环境：GPU 侧 Python 3.10 (cp310), Linux x86_64 manylinux2014
# ============================================================

# ----------------------------
# 硬编码共享目录（按你给的）
# ----------------------------
SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELHOUSE="$SHARED/data/wheels"
DATA_DIR="$SHARED/data"
MODEL_DIR="$DATA_DIR/models/Qwen3-VL-8B-Instruct"
COCO_ROOT="$DATA_DIR/datasets/coco_val2017"
CODE_DIR="$SHARED/code"

# 是否清理旧结果(只删 results/logs 里 p0a/p0b 的产物，不动数据/模型)
CLEAN_OLD_RESULTS="${CLEAN_OLD_RESULTS:-0}"

# ----------------------------
# 强制官方 pip 源（忽略系统 pip.conf / 环境变量）
# ----------------------------
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX || true

PYPI_URL="https://pypi.org/simple"
TORCH_CU124_URL="https://download.pytorch.org/whl/cu124"

# 目标平台（给 GPU 用的 wheels）
TARGET_PLATFORM="manylinux2014_x86_64"
TARGET_PY="310"
TARGET_ABI="cp310"
TARGET_IMPL="cp"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WHEELHOUSE" "$MODEL_DIR" "$COCO_ROOT" "$CODE_DIR" "$SHARED/results" "$SHARED/logs"

echo "================================================================"
echo "[cpu] SHARED      : $SHARED"
echo "[cpu] WHEELHOUSE  : $WHEELHOUSE"
echo "[cpu] MODEL_DIR   : $MODEL_DIR"
echo "[cpu] COCO_ROOT   : $COCO_ROOT"
echo "================================================================"

if [ "$CLEAN_OLD_RESULTS" = "1" ]; then
  echo "[cpu] cleaning old results/logs (p0a/p0b only)..."
  rm -f "$SHARED/results"/p0a_probe_info.json || true
  rm -f "$SHARED/results"/p0b_* || true
  rm -f "$SHARED/logs"/p0* || true
fi

# ----------------------------
# 0) 同步代码到共享目录（方便 GPU 直接跑）
# ----------------------------
echo "[cpu] syncing code -> $CODE_DIR"
cp -f "$SCRIPT_DIR/main.py" "$CODE_DIR/main.py"
cp -f "$SCRIPT_DIR/gpu.sh"  "$CODE_DIR/gpu.sh"
cp -f "$SCRIPT_DIR/cpu.sh"  "$CODE_DIR/cpu.sh" 2>/dev/null || true

# ----------------------------
# 1) 写 requirements（锁住关键项）
#    说明：transformers 5.x Requires-Python>=3.10。
#    CPU 机器如果不是 3.10 也没关系，我们只用 pip download 的 cross-platform 模式。
# ----------------------------
REQ="$CODE_DIR/requirements.lock.txt"
cat > "$REQ" <<'EOF'
# --- core ---
torch==2.4.1+cu124
torchvision==0.19.1+cu124
torchaudio==2.4.1+cu124

transformers==5.2.0
tokenizers==0.23.0
huggingface-hub==1.5.0

qwen-vl-utils==0.0.14

# --- common runtime ---
numpy
pillow
pandas
matplotlib
scipy
scikit-learn
pycocotools
tqdm
protobuf
sentencepiece
safetensors
einops
accelerate
EOF
echo "[cpu] requirements written: $REQ"

# ----------------------------
# 2) 清理 wheelhouse 里明显不匹配的 torch（比如 cp38）
# ----------------------------
echo "[cpu] cleaning mismatched torch wheels (keep cp310)..."
rm -f "$WHEELHOUSE"/torch-*-cp3[789]*.whl "$WHEELHOUSE"/torchvision-*-cp3[789]*.whl "$WHEELHOUSE"/torchaudio-*-cp3[789]*.whl 2>/dev/null || true

# ----------------------------
# 3) 下载 pip/setuptools/wheel 自身（GPU 离线 venv 更稳）
# ----------------------------
echo "[cpu] downloading bootstrap wheels (pip/setuptools/wheel)..."
python3 -m pip download -d "$WHEELHOUSE"   -i "$PYPI_URL"   --no-cache-dir --disable-pip-version-check   "pip" "setuptools" "wheel" >/dev/null

# ----------------------------
# 4) 下载全部依赖 wheels（面向 GPU 的 cp310 manylinux）
#    强制 PyPI + PyTorch 官方 cu124
# ----------------------------
echo "[cpu] downloading all wheels for target: py${TARGET_PY} ${TARGET_PLATFORM} ..."
python3 -m pip download -r "$REQ" -d "$WHEELHOUSE"   --only-binary=:all:   --platform "$TARGET_PLATFORM"   --python-version "$TARGET_PY"   --abi "$TARGET_ABI"   --implementation "$TARGET_IMPL"   -i "$PYPI_URL"   --extra-index-url "$TORCH_CU124_URL"   --no-cache-dir --disable-pip-version-check

echo "[cpu] wheelhouse count: $(ls -1 "$WHEELHOUSE" | wc -l)"

# ----------------------------
# 5) 下载 COCO val2017 + annotations（官方 COCO 源）
# ----------------------------
download_if_needed () {
  local url="$1"
  local out="$2"
  local name
  name="$(basename "$out")"
  if [ -f "$out" ]; then
    # zip 就测一下
    if [[ "$out" == *.zip ]]; then
      unzip -t "$out" >/dev/null 2>&1 && { echo "[cpu] coco ok: $name"; return 0; }
      echo "[cpu] coco zip corrupt -> re-download: $name"
      rm -f "$out"
    else
      echo "[cpu] coco ok: $name"
      return 0
    fi
  fi
  echo "[cpu] downloading: $name"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 10 --retry-delay 2 -o "$out" "$url"
  else
    wget -c -t 10 -O "$out" "$url"
  fi
  if [[ "$out" == *.zip ]]; then
    unzip -t "$out" >/dev/null 2>&1 || { echo "[cpu] ERROR: zip invalid: $out"; exit 1; }
  fi
}

VAL_ZIP="$COCO_ROOT/val2017.zip"
ANN_ZIP="$COCO_ROOT/annotations_trainval2017.zip"
VAL_DIR="$COCO_ROOT/val2017"
ANN_DIR="$COCO_ROOT/annotations"

COCO_URL_BASE="https://images.cocodataset.org/zips"
COCO_ANN_BASE="https://images.cocodataset.org/annotations"

download_if_needed "${COCO_URL_BASE}/val2017.zip" "$VAL_ZIP"
download_if_needed "${COCO_ANN_BASE}/annotations_trainval2017.zip" "$ANN_ZIP"

if [ ! -d "$VAL_DIR" ]; then
  echo "[cpu] extracting val2017..."
  unzip -q "$VAL_ZIP" -d "$COCO_ROOT"
fi
if [ ! -d "$ANN_DIR" ]; then
  echo "[cpu] extracting annotations..."
  unzip -q "$ANN_ZIP" -d "$COCO_ROOT"
fi

# ----------------------------
# 6) 模型：如果不存在再下载（HF 官方）
# ----------------------------
if [ -f "$MODEL_DIR/config.json" ] || [ -f "$MODEL_DIR/model.safetensors.index.json" ] || ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
  echo "[cpu] model already present: $MODEL_DIR"
else
  echo "[cpu] model not found -> downloading from Hugging Face (official)..."

  DL_VENV="$SHARED/venv/cpu_dl_env"
  if [ ! -d "$DL_VENV" ]; then
    python3 -m venv "$DL_VENV"
  fi
  source "$DL_VENV/bin/activate"
  python -m pip install -U pip -i "$PYPI_URL" --no-cache-dir --disable-pip-version-check >/dev/null
  python -m pip install -U "huggingface_hub>=0.24" "hf_transfer" -i "$PYPI_URL" --no-cache-dir --disable-pip-version-check >/dev/null || true

  export HF_HUB_ENABLE_HF_TRANSFER=1

  # 如果需要 token：export HF_TOKEN=...
  MODEL_DIR="$MODEL_DIR" python - <<'PY'
import os
from huggingface_hub import snapshot_download
repo_id = "Qwen/Qwen3-VL-8B-Instruct"
local_dir = os.environ["MODEL_DIR"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    token=token,
)
print("[cpu] downloaded:", repo_id, "->", local_dir)
PY

  deactivate || true
fi

echo "================================================================"
echo "[cpu] DONE"
echo "  wheelhouse : $WHEELHOUSE"
echo "  coco imgs  : $VAL_DIR"
echo "  coco ann   : $ANN_DIR/instances_val2017.json"
echo "  model      : $MODEL_DIR"
echo "  code       : $CODE_DIR/main.py  $CODE_DIR/gpu.sh"
echo "================================================================"
