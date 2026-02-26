#!/bin/bash
# ==============================================================================
# 实验名称：P0实验 —— JS散度视觉锚定验证实验 (Phase 0)
# 实验目的：验证 [视觉Token均值替换] 后的 JS散度 是否能作为 VLM 幻觉缓解的“物理测谎仪”
# 核心假设：JS散度在正例（物体存在）时显著大，在反例（幻觉）时显著小。
# 环境要求：CPU服务器联网执行，所有资源下载至硬编码共享盘路径。
# ==============================================================================

set -e

# 1. 硬编码根路径
SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS_DIR="$SHARED/data/wheels"
MODEL_DIR="$SHARED/data/models/Qwen3-VL-8B-Instruct"
DATA_DIR="$SHARED/data/datasets"
CODE_DIR="$SHARED/code"

echo "[P0] 创建生产级目录结构..."
mkdir -p $SHARED/{code,results,logs}
mkdir -p $WHEELS_DIR
mkdir -p $MODEL_DIR
mkdir -p $DATA_DIR/{hallusion_bench,coco_val2017}

# 2. 生成精确版本 requirements.txt
cat <<EOF > $CODE_DIR/requirements.txt
torch==2.5.1
torchvision==0.20.1
torchaudio==2.5.1
accelerate==1.3.0
qwen-vl-utils==0.0.14
flash-attn==2.7.4.post1
huggingface_hub==0.28.1
pillow==11.1.0
numpy==2.2.1
scipy==1.15.1
pandas==2.2.3
tqdm==4.67.1
scikit-learn==1.6.0
datasets==3.2.0
pycocotools==2.0.8
gdown==5.2.0
matplotlib==3.10.0
EOF

# 3. 下载 Wheels (离线安装准备)
echo "[P0] 正在下载依赖库 Wheels..."
cd $WHEELS_DIR
pip download -r $CODE_DIR/requirements.txt \
    --dest . --no-deps \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple

# 4. Git Clone 并构建 Transformers/Accelerate Main 分支 (针对 Qwen3-VL 适配)
echo "[P0] 构建 Transformers/Accelerate 最新分支..."
git clone https://github.com/huggingface/transformers.git $CODE_DIR/transformers_git
cd $CODE_DIR/transformers_git && python3 setup.py bdist_wheel && mv dist/*.whl $WHEELS_DIR/
git clone https://github.com/huggingface/accelerate.git $CODE_DIR/accelerate_git
cd $CODE_DIR/accelerate_git && python3 setup.py bdist_wheel && mv dist/*.whl $WHEELS_DIR/

# 5. 模型 Snapshot 下载 (强制 BF16，严禁量化)
echo "[P0] 下载 Qwen3-VL-8B-Instruct 模型 (BF16 全精度)..."
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-VL-7B-Instruct', # 注：此处按最新QwenVL命名逻辑下载
    local_dir='$MODEL_DIR',
    ignore_patterns=['*.bin', '*.pth', '*.msgpack'],
    local_dir_use_symlinks=False
)
"

# 6. 下载数据集 (HallusionBench & COCO Val2017)
echo "[P0] 下载验证数据集..."
cd $DATA_DIR/hallusion_bench
wget https://github.com/tianyi-ji/HallusionBench/raw/main/HallusionBench.json
# COCO val2017 仅下载标注和必要部分
cd $DATA_DIR/coco_val2017
wget http://images.cocodataset.org/zips/val2017.zip && unzip -q val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip -q annotations_trainval2017.zip

echo "================================================================"
echo "[P0] CPU准备完成，所有文件已硬编码存入 $SHARED"
echo "================================================================"
