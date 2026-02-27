#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

echo "=== P0实验：CED公式验证（GPU服务器 8×H200） ==="
echo "CED_i = Σ D_JS(P(·|V)||P(·|V^cf)) + λ_e·[H(P^cf)-H(P)]_+"
echo "通过标准: AUC > 0.85"
echo ""

echo "创建虚拟环境..."
$SHARED/tools/python3.10/bin/python3.10 -m venv $SHARED/venv/p0_env
source $SHARED/venv/p0_env/bin/activate

cd $SHARED/data/wheels
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    torch torchvision torchaudio
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    accelerate huggingface_hub qwen-vl-utils pillow numpy scipy pandas \
    tqdm scikit-learn datasets pycocotools gdown matplotlib
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    --no-deps transformers*.whl 2>/dev/null || \
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    --no-deps $SHARED/code/transformers*.whl

cd $SHARED/code

echo ""
echo "启动8卡并行..."
echo "每卡加载完整模型(8B bf16 ~ 16GB)，数据并行分片"
echo ""

torchrun \
    --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    p0_main.py \
    --num_samples 600 \
    --layers 16 20 24 28 32 \
    --lambda_e_values 0.0 0.05 0.1 0.2 0.5 \
    --seed 42 \
    2>&1 | tee $SHARED/logs/p0_run.log

echo ""
echo "=== P0实验完成 ==="
echo "结果: $SHARED/results/js_results_all.csv"
echo "图表: $SHARED/results/p0_analysis.png"
echo "日志: $SHARED/logs/p0_run.log"
