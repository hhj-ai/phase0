#!/bin/bash
set -e

# ============================================================
# cpu.sh - CPU服务器专用（有网环境）
# 功能：下载所有依赖、模型和数据集到共享存储
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"

# 可配置参数
PYTHON_VERSION="${PYTHON_VERSION:-3.10.13}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
CUDA_VERSION="${CUDA_VERSION:-cu124}"

echo "================================================================"
echo "  P0实验准备（CPU服务器 - 有网环境）"
echo "================================================================"
echo "目标目录: $SHARED"
echo ""

# 创建目录结构
mkdir -p $SHARED/{code,data/models/Qwen3-VL-8B-Instruct,data/datasets/hallusion_bench,data/datasets/coco_val2017,data/wheels,results,logs,tools,venv}
cd $SHARED/code

# 写入requirements
cat > requirements.txt << 'EOF'
accelerate==1.2.1
qwen-vl-utils==0.0.10
huggingface_hub==0.28.1
pillow==11.0.0
numpy==1.26.4
scipy==1.15.0
pandas==2.2.3
tqdm==4.67.1
scikit-learn==1.6.0
datasets==3.2.0
pycocotools==2.0.8
gdown==5.2.0
matplotlib==3.10.0
typing-extensions==4.12.2
filelock==3.16.1
pyyaml==6.0.2
requests==2.32.3
jinja2==3.1.4
networkx==3.4.2
sympy==1.13.3
regex==2024.11.6
tokenizers==0.21.0
safetensors==0.5.2
packaging==24.2
EOF

echo "[1/6] 下载 Python ${PYTHON_VERSION} standalone..."
if [ ! -d "$SHARED/tools/python3.10" ]; then
    cd $SHARED/tools
    PYTHON_TARBALL="cpython-${PYTHON_VERSION}+20240107-x86_64-unknown-linux-gnu-install_only.tar.gz"
    if [ ! -f "$PYTHON_TARBALL" ]; then
        wget -q "https://github.com/indygreg/python-build-standalone/releases/download/20240107/${PYTHON_TARBALL}"
    fi
    tar -xzf "$PYTHON_TARBALL"
    mv python python3.10
    rm -f "$PYTHON_TARBALL"
    echo "  ✓ Python ${PYTHON_VERSION} 已下载"
else
    echo "  ✓ Python 已存在，跳过"
fi

echo ""
echo "[2/6] 创建虚拟环境并下载依赖..."
$SHARED/tools/python3.10/bin/python3.10 -m venv $SHARED/venv/build_env
source $SHARED/venv/build_env/bin/activate

cd $WHEELS

echo "  下载requirements依赖..."
pip download -r $SHARED/code/requirements.txt \
    --index-url https://pypi.org/simple \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --no-cache-dir -d $WHEELS

echo "  下载torch家族 (${CUDA_VERSION})..."
pip download torch==${TORCH_VERSION} torchvision==0.19.1 torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION} \
    --no-deps --no-cache-dir -d $WHEELS

echo ""
echo "[3/6] 构建transformers wheel（Qwen3-VL必需）..."
if [ ! -f $WHEELS/transformers*.whl ]; then
    if [ -d "$SHARED/code/transformers" ]; then
        echo "  删除旧transformers目录..."
        rm -rf $SHARED/code/transformers
    fi
    git clone --depth 1 https://github.com/huggingface/transformers.git $SHARED/code/transformers
    cd $SHARED/code/transformers && python -m pip wheel . -w $WHEELS --no-deps
    cd $SHARED/code
    echo "  ✓ transformers wheel 已构建"
else
    echo "  ✓ transformers wheel 已存在，跳过"
fi

echo ""
echo "[4/6] 安装依赖到构建环境..."
pip install --no-cache-dir --index-url https://pypi.org/simple \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    -r $SHARED/code/requirements.txt

echo ""
echo "[5/6] 下载Qwen3-VL-8B-Instruct模型..."
if [ ! -d "$SHARED/data/models/Qwen3-VL-8B-Instruct/.git" ] && [ ! -f "$SHARED/data/models/Qwen3-VL-8B-Instruct/config.json" ]; then
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-VL-8B-Instruct',
    local_dir='$SHARED/data/models/Qwen3-VL-8B-Instruct',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.bin']
)
print('  ✓ 模型下载完成')
"
else
    echo "  ✓ 模型已存在，跳过"
fi

echo ""
echo "[6/6] 下载数据集..."

# HallusionBench
echo "  下载HallusionBench..."
cd $SHARED/data/datasets
if [ ! -d "hallusion_bench/images" ]; then
    mkdir -p hallusion_bench/images
    if [ ! -d "HallusionBench_temp" ]; then
        git clone --depth 1 https://github.com/tianyi-lab/HallusionBench.git HallusionBench_temp
    fi
    cp HallusionBench_temp/HallusionBench.json hallusion_bench/
    if [ ! -f "hallusion_bench.zip" ]; then
        gdown --id 1eeO1i0G9BSZTE1yd5XeFwmrbe1hwyf_0 -O hallusion_bench.zip --quiet || true
    fi
    if [ -f "hallusion_bench.zip" ]; then
        unzip -q hallusion_bench.zip -d hallusion_bench/images
    fi
    rm -rf HallusionBench_temp
    echo "    ✓ HallusionBench 完成"
else
    echo "    ✓ HallusionBench 已存在"
fi

# COCO val2017
echo "  下载COCO val2017..."
cd $SHARED/data/datasets/coco_val2017
if [ ! -d "val2017" ]; then
    if [ ! -f "val2017.zip" ]; then
        wget -q http://images.cocodataset.org/zips/val2017.zip
    fi
    if [ ! -f "annotations_trainval2017.zip" ]; then
        wget -q http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    fi
    unzip -q val2017.zip
    unzip -q annotations_trainval2017.zip
    echo "    ✓ COCO val2017 完成"
else
    echo "    ✓ COCO val2017 已存在"
fi

echo ""
echo "================================================================"
echo "  ✓ CPU准备完成！"
echo ""
echo "已准备内容:"
echo "  - Python 3.10: $SHARED/tools/python3.10"
echo "  - Wheels: $WHEELS"
echo "  - 模型: $SHARED/data/models/Qwen3-VL-8B-Instruct"
echo "  - 数据集: $SHARED/data/datasets/"
echo ""
echo "下一步: 到GPU服务器执行 bash gpu.sh"
echo "================================================================"
