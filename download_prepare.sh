#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"

echo "=== P0实验准备开始（CPU服务器） ==="
echo "实验目的：验证JS散度作为视觉锚定度物理测谎仪的可行性，用于未来RL奖励信号。"

mkdir -p $SHARED/{code,data/models/Qwen3-VL-8B-Instruct,data/datasets/hallusion_bench,data/datasets/coco_val2017,data/wheels,results,logs}
cd $SHARED/code

cat > requirements_official.txt << EOF
accelerate==0.33.0
qwen-vl-utils==0.0.8
huggingface_hub==0.25.2
pillow==10.4.0
numpy==1.24.4
scipy==1.10.1
pandas==2.0.3
tqdm==4.66.5
scikit-learn==1.3.2
datasets==2.19.2
pycocotools==2.0.7
gdown==5.2.0
matplotlib==3.7.5
EOF

echo "强制使用官方PyPI + 跳过已下载包（全部Python 3.8兼容版）..."
cd $WHEELS
for pkg in $(cat $SHARED/code/requirements_official.txt); do
    pkg_name=$(echo $pkg | cut -d= -f1 | sed 's/==.*//')
    if ls $WHEELS | grep -qi "$pkg_name"; then
        echo "✅ 已存在，跳过: $pkg"
    else
        echo "下载: $pkg"
        python -m pip download "$pkg" --no-deps \
          --index-url https://pypi.org/simple \
          --trusted-host pypi.org \
          --trusted-host files.pythonhosted.org \
          --no-cache-dir -d $WHEELS
    fi
done

echo "下载torch家族（cu124，兼容Python 3.8）..."
python -m pip download torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124 \
  --no-deps --no-cache-dir -d $WHEELS

echo "构建transformers主分支wheel（Qwen3-VL必需）..."
if [ ! -f $WHEELS/transformers*.whl ]; then
    git clone --depth 1 https://github.com/huggingface/transformers.git $SHARED/code/transformers
    cd $SHARED/code/transformers && python -m pip wheel . -w $WHEELS --no-deps
    cd $SHARED/code
else
    echo "✅ transformers wheel 已存在，跳过"
fi

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
if [ ! -d hallusion_bench ]; then
    git clone --depth 1 https://github.com/tianyi-lab/HallusionBench.git hallusion_bench_temp
    mkdir -p hallusion_bench/images
    cp hallusion_bench_temp/HallusionBench.json hallusion_bench/
    gdown --id 1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0 -O hallusion_bench.zip --quiet
    unzip -q hallusion_bench.zip -d hallusion_bench/images
    rm -rf hallusion_bench_temp
else
    echo "✅ HallusionBench 已存在，跳过"
fi

echo "下载COCO val2017..."
cd $SHARED/data/datasets/coco_val2017
if [ ! -f annotations_trainval2017.zip ]; then
    wget -q http://images.cocodataset.org/zips/val2017.zip
    wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip -q val2017.zip
    unzip -q annotations_trainval2017.zip
else
    echo "✅ COCO 已存在，跳过"
fi

echo "=== CPU准备完成！所有文件已硬编码存入 $SHARED ==="
echo "请到GPU服务器执行 setup_and_run.sh"
