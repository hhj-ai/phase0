#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"

GPUS=${1:-8}

echo "== 创建 venv =="
python3 -m venv $SHARED/venv/p0_env
source $SHARED/venv/p0_env/bin/activate

unset PIP_FIND_LINKS
export PIP_CONFIG_FILE=/dev/null

echo "== 离线安装 =="
pip install --isolated --no-index --find-links $WHEELS torch torchvision torchaudio
pip install --isolated --no-index --find-links $WHEELS transformers huggingface-hub tokenizers

echo "== CUDA 动态库修复 =="
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/nvjitlink/lib:$SITE/nvidia/cusparse/lib:$LD_LIBRARY_PATH"

echo "== 启动多卡 =="
torchrun --nproc_per_node=$GPUS p0_experiment.py \
    --mode worker \
    --model_path $SHARED/data/models/Qwen3-VL-8B-Instruct
