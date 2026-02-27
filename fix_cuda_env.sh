#!/bin/bash
# ============================================================
# fix_cuda_env.sh — 修复 nvJitLink 版本冲突
#
# 用法: source fix_cuda_env.sh
# 必须用 source，不能直接 bash（因为要修改当前shell的环境变量）
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

echo "=== CUDA环境诊断 ==="

# 激活venv
source $SHARED/venv/p0_env/bin/activate

# 1. 找到torch自带的nvidia库路径
SITE=$(python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
echo "site-packages: $SITE"

# 2. 把torch自带的所有nvidia子库路径加到LD_LIBRARY_PATH最前面
# 这样torch的12.4版本会覆盖系统的旧版本
NVIDIA_PATHS=""
if [ -d "$SITE/nvidia" ]; then
    for d in "$SITE"/nvidia/*/lib; do
        [ -d "$d" ] && NVIDIA_PATHS="$d:$NVIDIA_PATHS"
    done
fi

if [ -n "$NVIDIA_PATHS" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "已添加torch nvidia库路径到LD_LIBRARY_PATH"
else
    echo "WARNING: 未找到torch自带的nvidia库目录"
fi

# 3. 检查nvJitLink版本
echo ""
echo "--- nvJitLink库搜索 ---"
# 列出所有能找到的libnvJitLink
for f in $(ldconfig -p 2>/dev/null | grep nvJitLink | awk '{print $NF}'); do
    echo "  系统: $f"
done
if [ -f "$SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12" ]; then
    echo "  torch: $SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12"
fi
# 用python看实际加载的路径
python -c "
import ctypes, ctypes.util
try:
    lib = ctypes.CDLL('libnvJitLink.so.12')
    print(f'  实际加载: OK')
except Exception as e:
    print(f'  实际加载失败: {e}')
" 2>/dev/null

# 4. 测试torch是否能正常导入
echo ""
echo "--- 测试torch导入 ---"
python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    # 快速测试cusparse（触发报错的库）
    x = torch.randn(10, 10, device='cuda').to_sparse()
    print(f'  cusparse测试: OK')
" 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 环境修复成功！"
    echo "   现在可以运行: python p0a_probe.py"
else
    echo ""
    echo "❌ LD_LIBRARY_PATH方案不够，尝试备用方案..."
    echo ""
    
    # 方案B: 检查是否有nvidia-nvjitlink-cu12 wheel可装
    echo "--- 方案B: 安装匹配的nvjitlink包 ---"
    NVJIT_WHL=$(find $SHARED/data/wheels -name "nvidia_nvjitlink_cu12*.whl" 2>/dev/null | head -1)
    if [ -n "$NVJIT_WHL" ]; then
        echo "  找到wheel: $NVJIT_WHL"
        pip install --no-cache-dir --force-reinstall "$NVJIT_WHL" 2>/dev/null
    else
        echo "  未找到离线wheel"
        echo "  如果能联网: pip install nvidia-nvjitlink-cu12==12.4.127"
        echo "  如果不能联网: 需要从别的机器下载wheel后放到 $SHARED/data/wheels/"
        echo "    命令: pip download nvidia-nvjitlink-cu12==12.4.127 -d ./wheels"
    fi
    
    # 方案C: 用LD_PRELOAD强制加载
    echo ""
    echo "--- 方案C: LD_PRELOAD强制加载 ---"
    TORCH_NVJIT="$SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12"
    if [ -f "$TORCH_NVJIT" ]; then
        export LD_PRELOAD="$TORCH_NVJIT${LD_PRELOAD:+:$LD_PRELOAD}"
        echo "  已设置 LD_PRELOAD=$TORCH_NVJIT"
        python -c "import torch; print('  LD_PRELOAD方案: OK')" 2>&1 && echo "✅ 方案C成功！" || echo "❌ 方案C也失败"
    else
        echo "  未找到torch自带的libnvJitLink.so.12"
        
        # 方案D: 找系统上所有版本
        echo ""
        echo "--- 方案D: 搜索系统所有nvJitLink ---"
        find / -name "libnvJitLink.so*" 2>/dev/null | head -10
        echo ""
        echo "  如果上面列出了12.4版本的路径，手动设置:"
        echo "  export LD_LIBRARY_PATH=/path/to/12.4/lib:\$LD_LIBRARY_PATH"
    fi
fi

echo ""
echo "=== 诊断结束 ==="
echo "如果环境OK，运行P0实验："
echo "  cd $SHARED/code"
echo "  python p0a_probe.py"
echo "  python p0b_ced_validation.py --num_samples 400"
