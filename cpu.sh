#!/bin/bash
set -euo pipefail

# ============================================================
# cpu.sh - CPU服务器（有网）下载 wheelhouse（含依赖）
# 修复点：
#  1) 用引号包住 'huggingface-hub>=1.3.0,<2.0'，避免 bash 把 <2.0 当重定向
#  2) --isolated + PIP_CONFIG_FILE=/dev/null 防止旧 pip.conf / PIP_FIND_LINKS 污染
#  3) 清理旧的 huggingface_hub 0.x / tokenizers 0.21.x 轮子，避免GPU离线装到旧版本
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"
BUILD_ENV="$SHARED/venv/build_env"

TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"
TORCH_CUDA_VERSION="${TORCH_CUDA_VERSION:-cu124}"

mkdir -p "$WHEELS" "$SHARED/venv" "$SHARED/code"

echo "================================================================"
echo "CPU wheelhouse prepare"
echo "  SHARED   : $SHARED"
echo "  WHEELS   : $WHEELS"
echo "================================================================"

# 屏蔽全局 pip 配置污染
unset PIP_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

# 建一个干净的 venv 来下载 wheels（避免系统 pip 各种奇葩）
python3 -m venv "$BUILD_ENV"
source "$BUILD_ENV/bin/activate"
python -m pip install -U pip setuptools wheel

echo ""
echo "[1/4] 清理旧冲突 wheels（huggingface_hub 0.x / tokenizers 0.21.x）..."
rm -f "$WHEELS"/huggingface_hub-0*.whl || true
rm -f "$WHEELS"/tokenizers-0.21*.whl || true

echo ""
echo "[2/4] 下载 torch 家族（带依赖）..."
python -m pip download --isolated -d "$WHEELS" \
  torch=="$TORCH_VERSION" torchvision=="$TORCHVISION_VERSION" torchaudio=="$TORCHAUDIO_VERSION" \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VERSION}" \
  --extra-index-url "https://pypi.org/simple"

echo ""
echo "[3/4] 下载 transformers 依赖（注意：含 <2.0 必须加引号）..."
# 这里不强制下载 transformers 本体（你如果用源码 wheel，就在 wheels 里放你构建的 transformers-*.whl）
# 但依赖必须满足 transformers(main/dev) 的要求
python -m pip download --isolated -d "$WHEELS" \
  'huggingface-hub>=1.3.0,<2.0' \
  'tokenizers>=0.22.0,<=0.23.0' \
  accelerate datasets pandas numpy scipy tqdm pillow matplotlib scikit-learn pycocotools \
  safetensors sentencepiece regex pyyaml requests packaging jinja2 filelock

echo ""
echo "[4/4] 写一份GPU侧离线安装用的 requirements_offline.txt（同样注意引号）..."
cat > "$SHARED/code/requirements_offline.txt" <<'EOF'
# --- core ---
accelerate
datasets
pandas
numpy
scipy
tqdm
pillow
matplotlib
scikit-learn
pycocotools
safetensors
sentencepiece
regex
pyyaml
requests
packaging
jinja2
filelock
# --- transformers main/dev compatible ---
huggingface-hub>=1.3.0,<2.0
tokenizers>=0.22.0,<=0.23.0
EOF

echo ""
echo "================================================================"
echo "✓ CPU done."
echo "  wheelhouse: $WHEELS"
echo "  req file  : $SHARED/code/requirements_offline.txt"
echo "================================================================"
