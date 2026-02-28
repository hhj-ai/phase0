#!/bin/bash
set -euo pipefail

# ============================================================
# gpu.sh - GPU服务器（无网）离线安装 + 运行
# 修复点：
#  1) --isolated + PIP_CONFIG_FILE=/dev/null，彻底无视系统 pip.conf / PIP_FIND_LINKS
#  2) nvJitLink / cusparse 符号错误：LD_LIBRARY_PATH 优先 nvjitlink
#  3) 依赖版本按 requirements_offline.txt 约束安装，避免装到旧 huggingface_hub / tokenizers
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"
VENV="$SHARED/venv/p0_env"
REQ="$SHARED/code/requirements_offline.txt"

GPUS="${1:-8}"
MODEL_PATH="$SHARED/data/models/Qwen3-VL-8B-Instruct"

echo "================================================================"
echo "GPU offline install + run"
echo "  SHARED: $SHARED"
echo "  WHEELS: $WHEELS"
echo "  GPUS  : $GPUS"
echo "================================================================"

# 屏蔽全局 pip 配置污染
unset PIP_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

python3 -m venv "$VENV"
source "$VENV/bin/activate"

# 离线升级 pip 工具链（这些 wheel 要在 wheelhouse 里；没有也不致命）
python -m pip install --isolated --no-index --find-links "$WHEELS" -U pip setuptools wheel || true

echo ""
echo "[1/3] 离线安装 torch 家族（从 wheelhouse）..."
python -m pip install --isolated --no-index --find-links "$WHEELS" \
  torch torchvision torchaudio

echo ""
echo "[2/3] 修复 nvJitLink / cusparse 动态库优先级..."
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
NVJ="$SITE/nvidia/nvjitlink/lib"
CUS="$SITE/nvidia/cusparse/lib"
export LD_LIBRARY_PATH="$NVJ:$CUS:${LD_LIBRARY_PATH:-}"

# 可选：更强硬的方式，确保先加载 venv 里那份 libnvJitLink
if [ -f "$NVJ/libnvJitLink.so.12" ]; then
  export LD_PRELOAD="$NVJ/libnvJitLink.so.12:${LD_PRELOAD:-}"
fi

echo ""
echo "[3/3] 离线安装其余依赖（带版本约束，避免冲突）..."
if [ ! -f "$REQ" ]; then
  echo "✗ missing $REQ (请先在CPU跑 cpu.sh 生成它)"
  exit 1
fi

python -m pip install --isolated --no-index --find-links "$WHEELS" -r "$REQ"

# 安装 transformers：优先用你 wheelhouse 里的 transformers-*.whl（你如果从源码 build wheel 就放这里）
TRANS_WHL="$(ls -t "$WHEELS"/transformers-*.whl 2>/dev/null | head -1 || true)"
if [ -n "$TRANS_WHL" ]; then
  python -m pip install --isolated --no-index --find-links "$WHEELS" "$TRANS_WHL"
else
  # 如果你没自建 transformers wheel，那就从 wheelhouse 装已下载的 transformers（前提是你CPU也下载了）
  python -m pip install --isolated --no-index --find-links "$WHEELS" transformers
fi

echo ""
echo "== sanity check =="
python - <<PY
import torch
import huggingface_hub, tokenizers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

echo ""
echo "== run (torchrun) =="
cd "$SHARED/code"
torchrun --nproc_per_node="$GPUS" p0_experiment.py \
  --mode worker \
  --model_path "$MODEL_PATH"
