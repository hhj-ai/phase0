#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

echo "=== P0实验启动（GPU服务器 8×H200） ==="
echo "实验目的：验证JS散度作为视觉锚定度物理测谎仪的可行性，用于未来RL奖励信号。"

python -m venv $SHARED/venv/p0_env --system-site-packages
source $SHARED/venv/p0_env/bin/activate

cd $SHARED/data/wheels
pip install --no-index --no-cache-dir --find-links=. torch torchvision torchaudio
pip install --no-index --no-cache-dir --find-links=. accelerate huggingface_hub qwen-vl-utils pillow numpy scipy pandas tqdm scikit-learn datasets pycocotools gdown matplotlib
pip install --no-index --no-cache-dir --find-links=. --no-deps $SHARED/code/transformers/*.whl

cd $SHARED/code

echo "启动8卡并行运行p0_main.py..."
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 p0_main.py \
  --num_samples 800 \
  --batch_size 4 \
  2>&1 | tee $SHARED/logs/run.log

echo "=== P0实验运行结束！结果在 $SHARED/results ==="
echo "请查看 logs/run.log 开头实验目的说明和最终AUC结论。"
