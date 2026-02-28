#!/bin/bash
set -euo pipefail

# ============================================================
# gpu_fixed.sh - GPU服务器专用（无网环境）
# 功能：离线安装 + 修复 PyTorch CUDA 12.x 的 nvJitLink / cusparse 链接问题 + 跑P0
#
# 重点修复：
#   1) pip 配置污染（老 find-links）→ 全部用 --isolated + PIP_CONFIG_FILE=/dev/null
#   2) transformers(main) 依赖升级：huggingface_hub==1.5.0 / tokenizers==0.22.2
#   3) torch 导入报错：libcusparse.so.12 找不到 __nvJitLinkComplete_12_4
#      → 安装后再设置 LD_LIBRARY_PATH，让 site-packages/nvidia/nvjitlink/lib 优先
# ============================================================

NUM_GPUS="${1:-8}"

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH="$SHARED/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG="$SHARED/data/datasets/coco_val2017/val2017"
COCO_ANN="$SHARED/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR="$SHARED/results"
LOG_DIR="$SHARED/logs"
CODE_DIR="$SHARED/code"
WHEELS="$SHARED/data/wheels"
VENV="$SHARED/venv/p0_env"

# 可配置版本（要和CPU侧 wheelhouse 一致）
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.4.1}"

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export PIP_CONFIG_FILE=/dev/null

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$CODE_DIR"

echo "================================================================"
echo "  P0实验 - GPU服务器（${NUM_GPUS}卡并行 / 无网）"
echo "================================================================"
echo "共享目录: $SHARED"
echo ""

# ============================================================
# STEP 1: 创建 venv + 离线安装
# ============================================================
echo "[1/4] 搭建环境（全离线）..."

if [ ! -d "$VENV" ]; then
  "$SHARED/tools/python3.10/bin/python3.10" -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 强制干净：避免系统 pip.conf / 环境变量污染
unset PIP_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL || true

# 用 wheelhouse 升级 pip / setuptools / wheel（如果 wheelhouse 不全也不致命）
python -m pip install --isolated --no-index --find-links "$WHEELS" -U pip setuptools wheel || true

echo "  安装 torch 家族..."
python -m pip install --isolated --no-index --find-links "$WHEELS" \
  "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}"

echo "  安装其余依赖（requirements.txt）..."
if [ ! -f "$CODE_DIR/requirements.txt" ]; then
  echo "  ✗ 找不到 $CODE_DIR/requirements.txt（先在CPU侧运行 cpu_fixed.sh）"
  exit 1
fi
python -m pip install --isolated --no-index --find-links "$WHEELS" -r "$CODE_DIR/requirements.txt"

echo "  安装 transformers wheel..."
TRANS_WHL="$(ls -t "$WHEELS"/transformers*.whl 2>/dev/null | head -1 || true)"
if [ -z "$TRANS_WHL" ]; then
  echo "  ✗ 找不到 transformers wheel（CPU侧需要 build transformers wheel）"
  exit 1
fi
python -m pip install --isolated --no-index --find-links "$WHEELS" --no-deps "$TRANS_WHL"

# ============================================================
# STEP 2: CUDA 动态库路径修复（安装完成后再做）
# ============================================================
echo ""
echo "[2/4] 修复 CUDA 库路径 + 环境自检..."

SITE="$(python - <<PY
import site
print(site.getsitepackages()[0])
PY
)"

# 关键：把 nvjitlink 放最前（避免系统旧版 libnvJitLink.so.12 抢先被加载）
prepend_paths=(
  "$SITE/nvidia/nvjitlink/lib"
  "$SITE/nvidia/cusparse/lib"
  "$SITE/nvidia/cublas/lib"
  "$SITE/nvidia/cudnn/lib"
  "$SITE/nvidia/cusolver/lib"
  "$SITE/nvidia/cuda_runtime/lib"
  "$SITE/nvidia/nccl/lib"
  "$SITE/torch/lib"
)

for p in "${prepend_paths[@]}"; do
  if [ -d "$p" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done

# 可选兜底：强制 preload nvJitLink（有些机器 LD_LIBRARY_PATH 仍会被系统注入覆盖）
NVJIT="$SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12"
if [ -f "$NVJIT" ]; then
  export LD_PRELOAD="${NVJIT}${LD_PRELOAD:+:$LD_PRELOAD}"
fi

echo "  ✓ LD_LIBRARY_PATH 已更新（nvjitlink 优先）"

echo ""
echo "  检查 PyTorch CUDA..."
python - <<PY
import torch
assert torch.cuda.is_available(), "CUDA不可用"
print(f"✓ torch {torch.__version__}, torch CUDA={torch.version.cuda}, visible GPUs={torch.cuda.device_count()}")
for i in range(min(torch.cuda.device_count(), int("${NUM_GPUS}"))):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU {i}: {p.name}, {p.total_memory//1024**3}GB")
PY

echo ""
echo "  检查 transformers / Qwen3VL / hub / tokenizers..."
python - <<PY
import huggingface_hub, tokenizers, transformers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print("✓ imports OK")
print("  huggingface_hub:", huggingface_hub.__version__)
print("  tokenizers:", tokenizers.__version__)
print("  transformers:", transformers.__version__)
PY

echo ""
echo "  测试模型路径..."
if [ ! -d "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "  ✗ 模型路径不存在或不完整: $MODEL_PATH"
  exit 1
fi
if [ ! -d "$COCO_IMG" ] || [ ! -f "$COCO_ANN" ]; then
  echo "  ✗ COCO 路径不存在或不完整: $COCO_IMG / $COCO_ANN"
  exit 1
fi

# ============================================================
# STEP 3: 复制 Python 脚本到共享 code（保证用最新）
# ============================================================
echo ""
echo "[3/4] 同步代码文件到 $CODE_DIR ..."
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/p0_experiment_fixed.py" ]; then
  cp -f "$SCRIPT_DIR/p0_experiment_fixed.py" "$CODE_DIR/p0_experiment.py"
  echo "  ✓ 使用 p0_experiment_fixed.py → $CODE_DIR/p0_experiment.py"
elif [ -f "$SCRIPT_DIR/p0_experiment.py" ]; then
  cp -f "$SCRIPT_DIR/p0_experiment.py" "$CODE_DIR/p0_experiment.py"
  echo "  ✓ 复制 p0_experiment.py → $CODE_DIR/"
else
  echo "  ✗ 找不到 p0_experiment_fixed.py 或 p0_experiment.py（请把文件和脚本放同一目录）"
  exit 1
fi

# ============================================================
# STEP 4: 运行实验
# ============================================================
echo ""
echo "[4/4] 运行 P0 实验..."

export P0_MODEL="$MODEL_PATH"
export P0_COCO_IMG="$COCO_IMG"
export P0_COCO_ANN="$COCO_ANN"
export P0_RESULTS="$RESULT_DIR"

cd "$CODE_DIR"

# 4a: P0-a probe
echo ""
echo "[4a] P0-a 架构探测（GPU 0）..."
CUDA_VISIBLE_DEVICES=0 python p0_experiment.py --mode probe --seed 42 2>&1 | tee "$LOG_DIR/p0a.log"

if [ ! -f "$RESULT_DIR/p0a_probe_info.json" ]; then
  echo "  ✗ P0-a 探测结果不存在：$RESULT_DIR/p0a_probe_info.json"
  exit 1
fi

# 4b: P0-b worker multi-GPU
echo ""
echo "[4b] P0-b CED验证（${NUM_GPUS}卡并行）..."
rm -f "$RESULT_DIR"/p0b_shard_*.csv || true

PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
  CUDA_VISIBLE_DEVICES="$i" python p0_experiment.py \
    --mode worker --shard_idx "$i" --num_shards "$NUM_GPUS" \
    --num_samples 400 --seed 42 \
    --cf_mode moment_noise --noise_scale 1.0 \
    > "$LOG_DIR/p0b_w${i}.log" 2>&1 &
  PIDS+=($!)
  echo "  Worker $i → GPU $i (PID ${PIDS[-1]})"
done

echo "  等待 ${NUM_GPUS} 个 worker 完成..."
FAIL=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}" || { echo "  ✗ Worker $i 失败! → $LOG_DIR/p0b_w${i}.log"; FAIL=1; }
done
if [ "$FAIL" -ne 0 ]; then
  echo "  ✗ 部分 worker 失败"
  exit 1
fi
echo "  ✓ 全部 worker 完成"

# 4c: analyze
echo ""
echo "[4c] 合并结果 + 分析..."
python p0_experiment.py --mode analyze --seed 42 2>&1 | tee "$LOG_DIR/p0b_analyze.log"

echo ""
echo "================================================================"
echo "  ✓ P0实验完成"
echo "输出文件:"
echo "  探测信息: $RESULT_DIR/p0a_probe_info.json"
echo "  合并结果: $RESULT_DIR/p0b_results.csv"
echo "  分析图表: $RESULT_DIR/p0b_analysis.png"
echo "  日志目录: $LOG_DIR/"
echo "================================================================"
