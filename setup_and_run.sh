#!/bin/bash
set -e

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

echo "=== P0实验：CED公式验证 ==="

# ============================================================
# 1. 环境搭建
# ============================================================
echo "[1/5] 创建虚拟环境..."
if [ ! -d "$SHARED/venv/p0_env" ]; then
    $SHARED/tools/python3.10/bin/python3.10 -m venv $SHARED/venv/p0_env
fi
source $SHARED/venv/p0_env/bin/activate

cd $SHARED/data/wheels

echo "[2/5] 安装依赖..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    torch torchvision torchaudio 2>/dev/null || true
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    accelerate huggingface_hub qwen-vl-utils pillow numpy scipy pandas \
    tqdm scikit-learn datasets pycocotools gdown matplotlib regex 2>/dev/null || true

for whl in transformers*.whl; do
    [ -f "$whl" ] && pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location --no-deps "$whl" 2>/dev/null && break
done

# ============================================================
# 2. 修复 CUDA nvJitLink 版本冲突
#    报错: libcusparse.so.12: symbol __nvJitLinkComplete_12_4 not defined
#    根因: torch 2.4.1+cu124 自带的 cusparse 需要 nvJitLink 12.4，
#          但系统 LD_LIBRARY_PATH 中的旧版 libnvJitLink.so.12 先被加载
#    修复: 让 torch 自带的 nvidia 库路径排在最前面
# ============================================================
echo "[3/5] 修复CUDA库路径（nvJitLink冲突）..."

# 关键：用底层方式获取 site-packages 路径，不依赖 python -c（此时python可能还import不了torch）
SITE_PACKAGES="$SHARED/venv/p0_env/lib/python3.10/site-packages"

# 收集所有torch自带nvidia库路径
NVIDIA_LIB_PATHS=""
if [ -d "$SITE_PACKAGES/nvidia" ]; then
    for lib_dir in "$SITE_PACKAGES"/nvidia/*/lib; do
        [ -d "$lib_dir" ] && NVIDIA_LIB_PATHS="$lib_dir:$NVIDIA_LIB_PATHS"
    done
fi

# 也检查 torch/lib 下的
if [ -d "$SITE_PACKAGES/torch/lib" ]; then
    NVIDIA_LIB_PATHS="$SITE_PACKAGES/torch/lib:$NVIDIA_LIB_PATHS"
fi

if [ -n "$NVIDIA_LIB_PATHS" ]; then
    # 放到最前面，覆盖系统的旧版本
    export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "  ✓ 已将torch nvidia库加入LD_LIBRARY_PATH (优先于系统库)"
    # 列出关键库
    echo "  nvJitLink: $(ls "$SITE_PACKAGES"/nvidia/nvjitlink/lib/libnvJitLink* 2>/dev/null | head -1 || echo '未找到')"
    echo "  cusparse:  $(ls "$SITE_PACKAGES"/nvidia/cusparse/lib/libcusparse* 2>/dev/null | head -1 || echo '未找到')"
else
    echo "  WARNING: 未找到torch自带nvidia库"
fi

# 如果LD_LIBRARY_PATH还不够，用LD_PRELOAD兜底
NVJIT_LIB="$SITE_PACKAGES/nvidia/nvjitlink/lib/libnvJitLink.so.12"
if [ -f "$NVJIT_LIB" ]; then
    export LD_PRELOAD="${NVJIT_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "  ✓ 已设置LD_PRELOAD兜底: $NVJIT_LIB"
fi

# ============================================================
# 3. 验证torch能正常导入
# ============================================================
echo "[4/5] 验证PyTorch..."

python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA版本: {torch.version.cuda}')
    n = torch.cuda.device_count()
    print(f'  GPU数量: {n}')
    for i in range(min(n, 4)):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}, {p.total_mem/1024**3:.0f}GB')
    # 触发cusparse确保无报错
    x = torch.randn(4, 4, device='cuda')
    print(f'  CUDA运算测试: OK')
" || {
    echo ""
    echo "=== PyTorch导入仍然失败 ==="
    echo "可能原因及解决方案："
    echo ""
    echo "1. 系统CUDA驱动太老，不支持CUDA 12.4"
    echo "   检查: nvidia-smi | grep 'CUDA Version'"
    nvidia-smi 2>/dev/null | grep "CUDA Version" || echo "   nvidia-smi不可用"
    echo "   如果版本 < 12.4，需要换torch版本:"
    echo "   pip download torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "2. 系统的libnvJitLink太老且无法被覆盖"
    echo "   查找所有版本:"
    find / -name "libnvJitLink.so*" -type f 2>/dev/null | head -5
    echo ""
    echo "   如果存在 /usr/local/cuda-12.4 之类的路径，手动设置:"
    echo "   export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:\$LD_LIBRARY_PATH"
    echo ""
    echo "3. 下载匹配的nvjitlink包:"
    echo "   pip download nvidia-nvjitlink-cu12==12.4.127 -d $SHARED/data/wheels"
    echo "   pip install $SHARED/data/wheels/nvidia_nvjitlink_cu12*.whl"
    exit 1
}

# ============================================================
# 4. 运行实验
# ============================================================
echo "[5/5] 运行P0实验..."
mkdir -p $SHARED/logs $SHARED/results
cd $SHARED/code

echo ""
echo "=== P0-a: 架构探测（单卡，~5分钟）==="
CUDA_VISIBLE_DEVICES=0 python p0a_probe.py 2>&1 | tee $SHARED/logs/p0a_probe.log

if grep -q "P0-a PASSED" $SHARED/logs/p0a_probe.log; then
    echo ""
    echo "=== P0-b: CED信号验证（单卡，逐样本推理）==="
    CUDA_VISIBLE_DEVICES=0 python p0b_ced_validation.py \
        --num_samples 400 \
        --seed 42 \
        2>&1 | tee $SHARED/logs/p0b_ced.log
else
    echo ""
    echo "P0-a未通过，查看: $SHARED/logs/p0a_probe.log"
    exit 1
fi

echo ""
echo "=== 完成 ==="
echo "日志: $SHARED/logs/"
echo "结果: $SHARED/results/"
