#!/bin/bash
set -euo pipefail

# ============================================================
# cpu_fixed.sh - CPU服务器专用（有网环境）
# 功能：下载所有依赖、模型和数据集到共享存储（供GPU离线使用）
# 重点修复：
#   1) transformers(main) 依赖升级：huggingface_hub>=1.3.0 / tokenizers>=0.22
#   2) wheelhouse完整化：torch + 依赖（含nvjitlink等）+ pip/setuptools/wheel
#   3) COCO zip 下载校验，避免半包导致 unzip 报错
#   4) pip 配置污染（老 find-links 路径）→ 全部用 --isolated + PIP_CONFIG_FILE=/dev/null
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"
CODE_DIR="$SHARED/code"

# 可配置参数
PYTHON_VERSION="${PYTHON_VERSION:-3.10.13}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"
# PyTorch wheel URL uses short form: cu118, cu121, cu124
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-cu124}"

# 关键：这些版本要满足 transformers(main) 的依赖约束
HF_HUB_VERSION="${HF_HUB_VERSION:-1.5.0}"       # 2026-02 最新稳定之一
TOKENIZERS_VERSION="${TOKENIZERS_VERSION:-0.22.2}"  # 2026-01 最新稳定之一

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_CONFIG_FILE=/dev/null

echo "================================================================"
echo "  P0实验准备（CPU服务器 - 有网环境）"
echo "================================================================"
echo "共享目录: $SHARED"
echo "wheelhouse: $WHEELS"
echo ""

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
zip_ok () {
  local f="$1"
  python - <<PY
import sys, zipfile
f=sys.argv[1]
try:
    with zipfile.ZipFile(f,'r') as z:
        bad=z.testzip()
        if bad is not None:
            print(f"[zipcheck] BAD member: {bad}")
            sys.exit(2)
except Exception as e:
    print(f"[zipcheck] FAIL: {e}")
    sys.exit(2)
print("[zipcheck] OK")
PY "$f" >/dev/null
}

download_zip () {
  local url="$1"
  local out="$2"
  local tries="${3:-3}"

  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ]; then
    if zip_ok "$out"; then
      echo "    ✓ 已存在且通过校验: $(basename "$out")"
      return 0
    else
      echo "    ⚠️  发现损坏zip，删除重下: $(basename "$out")"
      rm -f "$out"
    fi
  fi

  for t in $(seq 1 "$tries"); do
    echo "    ↓ 下载($t/$tries): $(basename "$out")"
    wget -q --show-progress --progress=dot:giga -O "$out" "$url" || true
    if [ -f "$out" ] && zip_ok "$out"; then
      echo "    ✓ 下载完成且通过校验: $(basename "$out")"
      return 0
    fi
    echo "    ⚠️  下载文件仍不合法，重试..."
    rm -f "$out"
    sleep 2
  done

  echo "    ✗ 下载失败（zip校验始终不过）：$out"
  return 1
}

# ------------------------------------------------------------
# Directory layout
# ------------------------------------------------------------
mkdir -p "$SHARED"/{code,data/models/Qwen3-VL-8B-Instruct,data/datasets/hallusion_bench,data/datasets/coco_val2017,data/wheels,results,logs,tools,venv}
mkdir -p "$WHEELS" "$CODE_DIR"

# ------------------------------------------------------------
# Write requirements (GPU离线安装会用到)
# ------------------------------------------------------------
cat > "$CODE_DIR/requirements.txt" <<EOF
# --- core ---
accelerate==1.2.1
qwen-vl-utils==0.0.10

# --- transformers依赖：必须满足 transformers(main) ---
huggingface_hub==${HF_HUB_VERSION}
tokenizers==${TOKENIZERS_VERSION}

# --- numerics / io ---
pillow==11.0.0
numpy==1.26.4
scipy==1.15.0
pandas==2.2.3
tqdm==4.67.1

# --- eval / plotting ---
scikit-learn==1.6.0
datasets==3.2.0
pycocotools==2.0.8
matplotlib==3.10.0

# --- misc ---
typing-extensions==4.12.2
filelock==3.16.1
pyyaml==6.0.2
requests==2.32.3
jinja2==3.1.4
networkx==3.4.2
sympy==1.13.3
regex==2024.11.6
safetensors==0.5.2
packaging==24.2
einops==0.8.0
protobuf==5.29.3
sentencepiece==0.2.0

# --- optional downloader ---
gdown==5.2.0
EOF

echo "[1/6] 下载 Python ${PYTHON_VERSION} standalone..."
if [ ! -d "$SHARED/tools/python3.10" ]; then
  cd "$SHARED/tools"
  PYTHON_TARBALL="cpython-${PYTHON_VERSION}+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"
  if [ ! -f "$PYTHON_TARBALL" ]; then
    wget -q "https://github.com/indygreg/python-build-standalone/releases/download/20240107/${PYTHON_TARBALL}"
  fi
  tar -xzf "$PYTHON_TARBALL"
  mv python python3.10
  rm -f "$PYTHON_TARBALL"
  echo "  ✓ Python ${PYTHON_VERSION} 已下载"
else
  echo "  ✓ Python 已存在，跳过"
fi

echo ""
echo "[2/6] 创建 build_env 并构建 wheelhouse..."
"$SHARED/tools/python3.10/bin/python3.10" -m venv "$SHARED/venv/build_env"
source "$SHARED/venv/build_env/bin/activate"

python -m pip install --isolated --upgrade pip setuptools wheel build packaging

# 2.1 download pip / setuptools / wheel（GPU离线创建venv后可升级pip）
echo "  下载 pip / setuptools / wheel..."
python -m pip download --isolated --no-cache-dir --dest "$WHEELS" \
  --index-url https://pypi.org/simple \
  pip setuptools wheel

# 2.2 download python deps wheels
echo "  下载 requirements wheels..."
python -m pip download --isolated --no-cache-dir --dest "$WHEELS" \
  --index-url https://pypi.org/simple \
  --only-binary=:all: -r "$CODE_DIR/requirements.txt"

# 2.3 download torch family (allow deps from PyPI)
echo "  下载 torch / torchvision / torchaudio (${TORCH_CUDA_VERSION})..."
python -m pip download --isolated --no-cache-dir --dest "$WHEELS" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}" \
  --extra-index-url "https://pypi.org/simple" \
  --only-binary=:all: \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

echo ""
echo "[3/6] 构建 transformers wheel（Qwen3-VL 通常需要 main/dev）..."
cd "$SHARED/code"
if ! ls "$WHEELS"/transformers*.whl >/dev/null 2>&1; then
  rm -rf "$SHARED/code/transformers"
  git clone --depth 1 https://github.com/huggingface/transformers.git "$SHARED/code/transformers"
  cd "$SHARED/code/transformers"
  python -m pip wheel . -w "$WHEELS" --no-deps
  echo "  ✓ transformers wheel 已构建"
else
  echo "  ✓ transformers wheel 已存在，跳过"
fi

echo ""
echo "[4/6] 下载 Qwen3-VL-8B-Instruct 模型（HuggingFace）..."
python - <<PY
from huggingface_hub import snapshot_download
import os
dst = os.path.join(r"$SHARED", "data/models/Qwen3-VL-8B-Instruct")
if os.path.exists(os.path.join(dst, "config.json")):
    print("  ✓ 模型已存在，跳过")
else:
    snapshot_download(
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        local_dir=dst,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.bin"]
    )
    print("  ✓ 模型下载完成")
PY

echo ""
echo "[5/6] 下载数据集..."
mkdir -p "$SHARED/data/datasets"

# HallusionBench（尽量下载；失败不致命）
echo "  下载 HallusionBench..."
cd "$SHARED/data/datasets"
if [ ! -d "hallusion_bench/images" ]; then
  mkdir -p hallusion_bench/images
  rm -rf HallusionBench_temp || true
  git clone --depth 1 https://github.com/tianyi-lab/HallusionBench.git HallusionBench_temp
  cp HallusionBench_temp/HallusionBench.json hallusion_bench/ || true
  # gdown 可选：没有也不影响主实验
  python -m pip install --isolated -q "gdown==5.2.0" || true
  python -m gdown --id 1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0 -O hallusion_bench.zip --quiet || true
  if [ -f "hallusion_bench.zip" ]; then
    unzip -q hallusion_bench.zip -d hallusion_bench/images || true
  fi
  rm -rf HallusionBench_temp
  echo "    ✓ HallusionBench 处理完成（若图片缺失可忽略）"
else
  echo "    ✓ HallusionBench 已存在"
fi

# COCO val2017（P0默认用它）
echo "  下载 COCO val2017..."
COCO_DIR="$SHARED/data/datasets/coco_val2017"
mkdir -p "$COCO_DIR"
download_zip "http://images.cocodataset.org/zips/val2017.zip" \
            "$COCO_DIR/val2017.zip"
download_zip "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
            "$COCO_DIR/annotations_trainval2017.zip"

if [ ! -d "$COCO_DIR/val2017" ]; then
  unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR"
fi
if [ ! -d "$COCO_DIR/annotations" ]; then
  unzip -q "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR"
fi
echo "    ✓ COCO val2017 完成"

# 可选：COCO train2014（你如果要大规模跑再开）
if [ "${COCO_USE_TRAIN2014:-0}" = "1" ]; then
  echo "  [可选] 下载 COCO train2014..."
  COCO14="$SHARED/data/datasets/coco_train2014"
  mkdir -p "$COCO14"
  download_zip "http://images.cocodataset.org/zips/train2014.zip" \
              "$COCO14/train2014.zip"
  download_zip "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" \
              "$COCO14/annotations_trainval2014.zip"
  [ -d "$COCO14/train2014" ] || unzip -q "$COCO14/train2014.zip" -d "$COCO14"
  [ -d "$COCO14/annotations" ] || unzip -q "$COCO14/annotations_trainval2014.zip" -d "$COCO14"
  echo "    ✓ COCO train2014 完成"
fi

echo ""
echo "[6/6] 把本目录下的代码文件同步到共享目录（可选但推荐）..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for f in p0_experiment.py p0_experiment_fixed.py gpu.sh gpu_fixed.sh; do
  if [ -f "$SCRIPT_DIR/$f" ]; then
    cp -f "$SCRIPT_DIR/$f" "$CODE_DIR/"
    echo "  ✓ 已复制 $f → $CODE_DIR/"
  fi
done

echo ""
echo "================================================================"
echo "  ✓ CPU侧准备完成"
echo "  下一步到 GPU 服务器运行: bash gpu_fixed.sh 8"
echo "================================================================"
