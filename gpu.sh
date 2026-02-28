#!/bin/bash
set -e

# ============================================================
# gpu.sh - GPU服务器专用（无网环境）
# 功能：创建venv、离线安装（wheels）、运行P0实验
# ============================================================

NUM_GPUS="${1:-8}"

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
CODE_DIR="$SHARED/code"
WHEELS="$SHARED/data/wheels"
VENV="$SHARED/venv/p0_env"
PY="$SHARED/tools/python3.10/bin/python3.10"

export HF_HOME="$SHARED/data/hf_cache"
export TRANSFORMERS_CACHE="$SHARED/data/hf_cache"

mkdir -p "$SHARED/results" "$SHARED/logs" "$SHARED/data/hf_cache"

echo "================================================================"
echo "  P0实验 - GPU服务器（无网环境）"
echo "================================================================"
echo "NUM_GPUS=$NUM_GPUS"
echo "SHARED=$SHARED"
echo "WHEELS=$WHEELS"
echo ""

# 1) venv
echo "[1/4] 创建/激活 venv..."
if [ ! -d "$VENV" ]; then
  $PY -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 2) 离线安装（全部从 wheels）
echo ""
echo "[2/4] 离线安装依赖（全部从 wheels）..."

# 先用 wheels 里的 pip/setuptools/wheel 升级（不联网）
pip install --no-index --find-links="$WHEELS" --upgrade pip setuptools wheel || true

# 安装 torch + CUDA 依赖（CPU端已下载到 wheelhouse）
pip install --no-index --find-links="$WHEELS" torch torchvision torchaudio

# 安装 transformers（CPU端从源码打包的 wheel）
TRANS_WHL=$(ls -t "$WHEELS"/transformers*.whl 2>/dev/null | head -1 || true)
if [ -n "$TRANS_WHL" ] && [ -f "$TRANS_WHL" ]; then
  pip install --no-index --find-links="$WHEELS" --no-deps "$TRANS_WHL"
else
  echo "✗ 未找到 transformers wheel: $WHEELS/transformers*.whl"
  exit 1
fi

# 安装其余依赖
if [ -f "$CODE_DIR/requirements.txt" ]; then
  pip install --no-index --find-links="$WHEELS" -r "$CODE_DIR/requirements.txt"
else
  echo "✗ 未找到 requirements.txt: $CODE_DIR/requirements.txt"
  exit 1
fi

# 3) 环境检查
echo ""
echo "[3/4] 环境检查..."
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"
python -c "import transformers; print('transformers', transformers.__version__)"
python -c "import qwen_vl_utils; print('qwen_vl_utils OK')"

# 4) 运行实验
echo ""
echo "[4/4] 运行 P0..."
cd "$CODE_DIR"

export CUDA_VISIBLE_DEVICES=$(python - <<'PY'
n=int(__import__('os').environ.get('NUM_GPUS','8'))
print(",".join(str(i) for i in range(n)))
PY
)

# P0-a probe (单卡) ——建议先跑一次，写入 probe_info.json
python p0_experiment.py --mode probe --replace_mode moment_noise --noise_scale 1.0

# P0-b worker (多卡并行)
torchrun --nproc_per_node="$NUM_GPUS" p0_experiment.py \
  --mode worker \
  --num_shards "$NUM_GPUS" \
  --num_samples 400 \
  --replace_mode moment_noise \
  --noise_scale 1.0

# Merge + Analyze
python p0_experiment.py --mode analyze --result_dir "$SHARED/results"

echo ""
echo "================================================================"
echo "  ✓ GPU端完成"
echo "结果目录: $SHARED/results"
echo "================================================================"
