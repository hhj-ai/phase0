#!/bin/bash
set -e

# ============================================================
# gpu.sh - GPU服务器专用（无网环境）
# 功能：修复CUDA、离线安装、运行P0实验
# ============================================================

NUM_GPUS="${1:-8}"

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH="$SHARED/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG="$SHARED/data/datasets/coco_val2017/val2017"
COCO_ANN="$SHARED/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR="$SHARED/results"
LOG_DIR="$SHARED/logs"
CODE_DIR="$SHARED/code"
WHEELS="$SHARED/data/wheels"
VENV="$SHARED/venv/p0_env"
SITE="$VENV/lib/python3.10/site-packages"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$CODE_DIR"

echo "================================================================"
echo "  P0实验 - GPU服务器（${NUM_GPUS}卡并行）"
echo "================================================================"
echo ""

# ============================================================
# Screen Session Management
# ============================================================
SCREEN_NAME="p0_qwen3vl"
echo "[*] Screen session management..."
if screen -ls | grep -q "$SCREEN_NAME"; then
    echo "  ⚠️  Screen session '$SCREEN_NAME' already exists"
    echo "  Options: (r) reattach, (k) kill and create new, (q) quit"
    # Default to kill and create new for automated runs
    echo "  Auto-select: kill and create new"
    screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true
fi
screen -dmS "$SCREEN_NAME"
echo "  ✓ Screen session '$SCREEN_NAME' created"
echo "  To monitor: screen -r $SCREEN_NAME"
echo ""

# ============================================================
# STEP 0: CUDA环境修复（必须在任何Python之前！）
# ============================================================
echo "[0/4] 修复CUDA库路径..."

# 将torch自带的nvidia库路径加到最前面，覆盖系统的旧版本
if [ -d "$SITE/nvidia" ]; then
    for d in "$SITE"/nvidia/*/lib; do
        [ -d "$d" ] && export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
    done
fi
[ -d "$SITE/torch/lib" ] && export LD_LIBRARY_PATH="$SITE/torch/lib:${LD_LIBRARY_PATH:-}"

# LD_PRELOAD兜底：nvJitLink
NVJIT="$SITE/nvidia/nvjitlink/lib/libnvJitLink.so.12"
[ -f "$NVJIT" ] && export LD_PRELOAD="${NVJIT}${LD_PRELOAD:+:$LD_PRELOAD}"

echo "  ✓ LD_LIBRARY_PATH和LD_PRELOAD已设置"

# ============================================================
# STEP 1: 虚拟环境 + 离线安装
# ============================================================
echo ""
echo "[1/4] 搭建环境（全离线）..."

if [ ! -d "$VENV" ]; then
    $SHARED/tools/python3.10/bin/python3.10 -m venv "$VENV"
fi
source "$VENV/bin/activate"

cd "$WHEELS"

# 关键：全部用 --no-index，绝不联网
# Installation order matters: torch -> huggingface_hub -> tokenizers -> transformers -> others

echo "  [1/6] 安装torch..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    torch torchvision torchaudio 2>/dev/null || {
    echo "  ✗ torch安装失败，检查wheel文件是否存在"
    ls -la torch*.whl 2>/dev/null || echo "    未找到torch wheel文件"
    exit 1
}

echo "  [2/6] 安装huggingface_hub..."
pip install --no-index --no-cache-dir --no-deps --force-reinstall \
    --no-warn-script-location \
    huggingface_hub*.whl 2>/dev/null || {
    echo "  ✗ huggingface_hub安装失败"
    exit 1
}

echo "  [3/6] 安装依赖包..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    numpy pillow filelock packaging pyyaml requests \
    regex typing-extensions filelock networkx sympy 2>/dev/null || true

echo "  [4/6] 安装tokenizers..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    tokenizers 2>/dev/null || true

echo "  [5/6] 安装transformers..."
TRANS_WHL=$(ls -t "$WHEELS"/transformers*.whl 2>/dev/null | head -1)
[ -n "$TRANS_WHL" ] && pip install --no-index --no-cache-dir --no-deps \
    --no-warn-script-location "$TRANS_WHL" 2>/dev/null || {
    echo "  ✗ transformers安装失败"
    exit 1
}

echo "  [6/6] 安装其他依赖..."
pip install --no-index --no-cache-dir --find-links=. --no-warn-script-location \
    accelerate qwen-vl-utils scipy pandas tqdm scikit-learn \
    pycocotools matplotlib safetensors datasets \
    einops protobuf sentencepiece gdown 2>/dev/null || {
    echo "  ⚠️  部分可选依赖安装失败，尝试继续..."
}

# 确认huggingface_hub版本
echo ""
echo "  确认关键包版本..."
python -c "
import sys
print(f'Python: {sys.version}')
try:
    import huggingface_hub
    print(f'huggingface_hub: {huggingface_hub.__version__}')
except Exception as e:
    print(f'huggingface_hub: 导入失败 - {e}')
    sys.exit(1)
try:
    import transformers
    print(f'transformers: {transformers.__version__}')
except Exception as e:
    print(f'transformers: 导入失败 - {e}')
    sys.exit(1)
try:
    import einops
    print(f'einops: {einops.__version__}')
except:
    print('einops: 未安装（可选）')
" || {
    echo "  ✗ 关键包导入失败"
    exit 1
}

# ============================================================
# STEP 2: 验证环境
# ============================================================
echo ""
echo "[2/4] 验证环境..."

echo "  检查PyTorch CUDA..."
python -c "
import torch
if not torch.cuda.is_available():
    print('✗ CUDA不可用！')
    exit(1)
print(f'✓ torch {torch.__version__}, CUDA {torch.version.cuda}, {torch.cuda.device_count()} GPUs')
for i in range(min(torch.cuda.device_count(), $NUM_GPUS)):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_memory//1024**3}GB')
" || {
    echo "  ✗ torch CUDA失败"
    exit 1
}

echo ""
echo "  检查transformers和Qwen3VL..."
python -c "
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
print('✓ transformers + Qwen3VL导入成功')
import huggingface_hub
print(f'✓ huggingface_hub {huggingface_hub.__version__}')
import einops
print(f'✓ einops {einops.__version__}')
" || {
    echo "  ✗ transformers导入失败"
    exit 1
}

# Test model loading
echo ""
echo "  测试模型加载..."
python -c "
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import os

model_path = '$MODEL_PATH'
if not os.path.exists(model_path):
    print(f'✗ 模型路径不存在: {model_path}')
    exit(1)

device = torch.device('cuda:0')
print(f'正在加载模型从 {model_path}...')
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
).to(device).eval()
print(f'✓ 模型加载成功')
print(f'  视觉编码器类型: {type(model.model.visual).__name__}')

processor = AutoProcessor.from_pretrained(model_path)
print(f'✓ Processor加载成功')
" || {
    echo "  ✗ 模型加载失败"
    exit 1
}

echo ""
echo "  ✓ 环境验证通过"

# ============================================================
# STEP 3: 复制Python脚本
# ============================================================
echo ""
echo "[3/4] 复制p0_experiment.py..."

if [ ! -f "$CODE_DIR/p0_experiment.py" ]; then
    # 优先从phase0目录（脚本所在目录）复制
    SCRIPT_DIR="$(dirname $(readlink -f $0))"
    if [ -f "$SCRIPT_DIR/p0_experiment.py" ]; then
        cp "$SCRIPT_DIR/p0_experiment.py" "$CODE_DIR/"
    elif [ -f "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0/p0_experiment.py" ]; then
        cp "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/phase0/p0_experiment.py" "$CODE_DIR/"
    else
        echo "  ✗ 找不到p0_experiment.py！"
        exit 1
    fi
fi
echo "  ✓ p0_experiment.py已就绪"

# ============================================================
# STEP 4: 运行实验
# ============================================================
echo ""
echo "[4/4] 运行P0实验..."

export P0_MODEL="$MODEL_PATH"
export P0_COCO_IMG="$COCO_IMG"
export P0_COCO_ANN="$COCO_ANN"
export P0_RESULTS="$RESULT_DIR"

cd "$CODE_DIR"

# ── 4a: P0-a 架构探测 ──
echo ""
echo "[4a] P0-a 架构探测（GPU 0）..."
CUDA_VISIBLE_DEVICES=0 python p0_experiment.py --mode probe --seed 42 2>&1 | tee "$LOG_DIR/p0a.log"

# 检查P0-a结果
if [ ! -f "$RESULT_DIR/p0a_probe_info.json" ]; then
    echo "  ✗ P0-a探测结果不存在！"
    exit 1
fi

PROBE_PASSED=$(python -c "import json; d=json.load(open('$RESULT_DIR/p0a_probe_info.json')); print('true' if d.get('all_passed') else 'false')")
if [ "$PROBE_PASSED" != "true" ]; then
    echo "  ✗ P0-a探测未通过！请检查架构兼容性。"
    exit 1
fi
echo "  ✓ P0-a通过"

# ── 4b: P0-b 多卡并行 ──
echo ""
echo "[4b] P0-b CED验证（${NUM_GPUS}卡并行）..."
rm -f "$RESULT_DIR"/p0b_shard_*.csv

PIDS=()
for i in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$i python p0_experiment.py \
        --mode worker --shard_id $i --num_shards $NUM_GPUS \
        --num_samples 400 --seed 42 \
        > "$LOG_DIR/p0b_w${i}.log" 2>&1 &
    PIDS+=($!)
    echo "  Worker $i → GPU $i (PID ${PIDS[-1]})"
done

echo "  等待${NUM_GPUS}个worker完成..."
FAIL=0
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]} || { echo "  ✗ Worker $i 失败! → $LOG_DIR/p0b_w${i}.log"; FAIL=1; }
done

if [ $FAIL -ne 0 ]; then
    echo "  ✗ 部分worker失败"
    exit 1
fi
echo "  ✓ 全部worker完成"

# ── 4c: 合并分析 ──
echo ""
echo "[4c] 合并结果 + 分析..."
python p0_experiment.py --mode analyze --seed 42 2>&1 | tee "$LOG_DIR/p0b_analyze.log"

echo ""
echo "================================================================"
echo "  ✓ P0实验完成！"
echo ""
echo "输出文件:"
echo "  探测信息: $RESULT_DIR/p0a_probe_info.json"
echo "  原始结果: $RESULT_DIR/p0b_results.csv"
echo "  分析图表: $RESULT_DIR/p0b_analysis.png"
echo "  日志文件: $LOG_DIR/"
echo "================================================================"
