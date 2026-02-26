#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

echo "=== P0实验准备开始（CPU服务器） ==="
echo "实验目的：验证JS散度作为视觉锚定度物理测谎仪的可行性，用于未来RL奖励信号。"

mkdir -p $SHARED/{code,data/models/Qwen3-VL-8B-Instruct,data/datasets/hallusion_bench,data/datasets/coco_val2017,data/wheels,results,logs}

cd $SHARED/code

cat > requirements.txt << EOF
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
--index-url https://download.pytorch.org/whl/cu124
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

echo "下载wheels（--no-cache-dir）..."
python -m pip download -r requirements.txt -d $SHARED/data/wheels --no-cache-dir --no-deps

echo "构建transformers主分支wheel（Qwen3-VL必需）..."
git clone --depth 1 https://github.com/huggingface/transformers.git $SHARED/code/transformers
cd $SHARED/code/transformers && python -m pip wheel . -w $SHARED/data/wheels --no-deps
cd $SHARED/code

echo "下载Qwen3-VL-8B-Instruct模型（bf16全精度）..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-VL-8B-Instruct',
                  local_dir='$SHARED/data/models/Qwen3-VL-8B-Instruct',
                  local_dir_use_symlinks=False,
                  ignore_patterns=['*.bin'])
"

echo "下载HallusionBench..."
cd $SHARED/data/datasets
git clone --depth 1 https://github.com/tianyi-lab/HallusionBench.git hallusion_bench_temp
cp hallusion_bench_temp/HallusionBench.json hallusion_bench/
gdown --id 1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0 -O hallusion_bench.zip --quiet
unzip -q hallusion_bench.zip -d hallusion_bench/images
rm -rf hallusion_bench_temp

echo "下载COCO val2017..."
cd $SHARED/data/datasets/coco_val2017
wget -q http://images.cocodataset.org/zips/val2017.zip
wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

echo "=== CPU准备完成！所有文件已硬编码存入 $SHARED ==="
echo "请到GPU服务器执行 setup_and_run.sh"
