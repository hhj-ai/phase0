#!/bin/bash
# ==============================================================================
# 实验名称：P0实验 —— JS散度视觉锚定验证实验 (Phase 0)
# 环境要求：GPU服务器（H200 * 8），无网环境。
# 逻辑：创建虚拟环境 -> 顺序离线安装 -> 启动分布式推理与分析。
# ==============================================================================

set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
VENV_PATH="$SHARED/venv_p0"
WHEELS_DIR="$SHARED/data/wheels"

echo "[P0] 初始化 GPU 离线运行环境..."

# 1. 创建并激活 Venv
python3.11 -m venv $VENV_PATH
source $VENV_PATH/bin/activate

# 2. 严格顺序离线安装 (Torch -> Deps -> Transformers)
echo "[P0] 执行离线 Wheel 安装..."
pip install --no-index --find-links=$WHEELS_DIR torch torchvision torchaudio
pip install --no-index --find-links=$WHEELS_DIR accelerate flash-attn qwen-vl-utils
pip install --no-index --find-links=$WHEELS_DIR transformers*.whl
pip install --no-index --find-links=$WHEELS_DIR pillow numpy scipy pandas tqdm scikit-learn datasets pycocotools matplotlib

# 3. 运行 P0 核心验证程序 (8卡分布式)
echo "[P0] 启动分布式核心实验: JS散度作为物理测谎仪验证..."
export OMP_NUM_THREADS=8
torchrun --nproc_per_node=8 $SHARED/code/p0_main.py \
    --model_path "$SHARED/data/models/Qwen3-VL-8B-Instruct" \
    --data_root "$SHARED/data/datasets" \
    --output_dir "$SHARED/results" \
    --num_samples 800 \
    --batch_size 1

echo "================================================================"
echo "[P0] 实验运行结束。结果已存入 $SHARED/results"
cat $SHARED/results/summary.log
echo "================================================================"
