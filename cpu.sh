#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"

mkdir -p $WHEELS
mkdir -p $SHARED/venv

echo "== 清理 pip 全局污染 =="
unset PIP_FIND_LINKS
unset PIP_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

echo "== 下载 wheelhouse =="
pip download --isolated \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124 \
    -d $WHEELS

pip download --isolated \
    transformers \
    huggingface-hub>=1.3.0,<2.0 \
    tokenizers>=0.22.0,<=0.23.0 \
    -d $WHEELS

echo "CPU准备完成"
