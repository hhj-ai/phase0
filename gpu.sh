#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# gpu.sh
# ------------------------------------------------------------
# 目标：
# 1) 不碰你现有环境：默认在 $BASE_DIR/venv/p0_env 里装依赖
# 2) 优先离线：使用 $WHEELHOUSE (pip --no-index --find-links)
# 3) 不再调用 cpu.sh，也不会 rsync --delete
# ============================================================

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${P0_BASE_DIR:-$HERE_DIR}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
OUT_DIR="$BASE_DIR/outputs"

WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"
COCO_DIR="${P0_COCO_DIR:-$DATA_DIR/coco2017}"
COCO_IMG_DIR="${P0_COCO_IMG_DIR:-$COCO_DIR/val2017}"
COCO_ANN_PATH="${P0_COCO_ANN_PATH:-$COCO_DIR/annotations/instances_val2017.json}"

VENV_DIR="${P0_VENV_DIR:-$BASE_DIR/venv/p0_env}"
PYBIN="${P0_PYBIN:-python3}"

# Torch 版本：默认 cu124。你也可以覆写：
#   P0_TORCH_SPEC='torch==2.5.1+cu124' bash gpu.sh
TORCH_SPEC="${P0_TORCH_SPEC:-torch==2.4.1+cu124}"

# 默认跑全流程（probe -> worker -> analyze -> summary）。
# 如果你只想装环境：P0_SKIP_RUN=1 bash gpu.sh
SKIP_RUN="${P0_SKIP_RUN:-0}"

mkdir -p "$CODE_DIR" "$DATA_DIR" "$OUT_DIR"

# 确保代码文件在 code/（不删除旧文件）
cp -f "$HERE_DIR/main.py" "$CODE_DIR/main.py"
cp -f "$HERE_DIR/gpu.sh"  "$CODE_DIR/gpu.sh"
cp -f "$HERE_DIR/cpu.sh"  "$CODE_DIR/cpu.sh"

# 如果 requirements 还没生成，cpu.sh 会写；这里兜底写一份最小的
if [[ ! -f "$CODE_DIR/requirements.lock.txt" ]]; then
  echo "[gpu] requirements.lock.txt missing; generating minimal one via cpu.sh (no download)"
  P0_ALLOW_DOWNLOAD=0 bash "$HERE_DIR/cpu.sh"
fi

# --- 1) 创建 venv（不覆盖已有） ---
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[gpu] creating venv: $VENV_DIR"
  "$PYBIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 关键：忽略系统 pip 配置（很多集群把 index-url 指到奇怪的 http 源）
export PIP_CONFIG_FILE=/dev/null

# --- 2) 安装依赖（优先离线 wheelhouse；缺了就明确报错） ---
# 判断 wheelhouse 是否可用
if [[ -d "$WHEELHOUSE" ]] && [[ "$(ls -A "$WHEELHOUSE" 2>/dev/null | wc -l)" -gt 0 ]]; then
  HAVE_WHEELHOUSE=1
else
  HAVE_WHEELHOUSE=0
fi

# 小工具：检查导入是否 OK
check_imports() {
  python - <<'PY'
import importlib
mods = ['torch','transformers','qwen_vl_utils','PIL','cv2','numpy']
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print('MISSING_OR_BROKEN')
    for m,e in missing:
        print(m, e)
else:
    print('OK')
PY
}

STATUS="$(check_imports | head -n 1)"
if [[ "$STATUS" != "OK" ]]; then
  echo "[gpu] python env missing deps; installing..."
  if [[ "$HAVE_WHEELHOUSE" -ne 1 ]]; then
    echo "[gpu] ERROR: wheelhouse not found or empty: $WHEELHOUSE"
    echo "       - 把 wheels 放到该目录，或设置 P0_WHEELHOUSE 指向正确路径"
    echo "       - 有网的话可先: P0_ALLOW_DOWNLOAD=1 bash cpu.sh 生成 wheelhouse"
    exit 1
  fi

  # 安装 torch（GPU）
  echo "[gpu] installing torch: $TORCH_SPEC"
  python -m pip install --isolated --no-index --find-links "$WHEELHOUSE" "$TORCH_SPEC" || {
    echo "[gpu] ERROR: failed to install $TORCH_SPEC from wheelhouse."
    echo "       Check wheels in: $WHEELHOUSE"
    exit 1
  }

  # 安装其余依赖
  echo "[gpu] installing other requirements from: $CODE_DIR/requirements.lock.txt"
  python -m pip install --isolated --no-index --find-links "$WHEELHOUSE" -r "$CODE_DIR/requirements.lock.txt"
fi

# --- 3) 处理 CUDA 动态库优先级（避免 libnvJitLink / cusparse 版本被系统抢走） ---
PY_SITE=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

for d in \
  "$PY_SITE/nvidia/nvjitlink/lib" \
  "$PY_SITE/nvidia/cusparse/lib" \
  "$PY_SITE/nvidia/cublas/lib" \
  "$PY_SITE/nvidia/cudnn/lib" \
  "$PY_SITE/torch/lib"; do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

# sanity
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda device:', torch.cuda.get_device_name(0))
PY

if [[ "$SKIP_RUN" == "1" ]]; then
  echo "[gpu] env ready. (P0_SKIP_RUN=1)"
  exit 0
fi

# --- 4) 跑实验 ---
mkdir -p "$OUT_DIR"

MODEL_ID="${P0_MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
RUN_TAG="${P0_RUN_TAG:-p0_run}"

# 分布式配置
NNODES="${P0_NNODES:-1}"
NPROC="${P0_NPROC_PER_NODE:-4}"
MASTER_ADDR="${P0_MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${P0_MASTER_PORT:-29500}"

MAX_SAMPLES="${P0_MAX_SAMPLES:-200}"
CHUNK_SIZE="${P0_CHUNK_SIZE:-50}"
SEED="${P0_SEED:-123}"

# 你之前传的 layers（当前版本 main.py 里“接受但不强依赖”，用来兼容旧脚本）
LAYERS="${P0_LAYERS:-16 20 24 28 32}"

echo "[gpu] running probe..."
python "$CODE_DIR/main.py" probe \
  --model_id "$MODEL_ID" \
  --out_dir "$OUT_DIR/$RUN_TAG" \
  --coco_img_dir "$COCO_IMG_DIR" \
  --coco_ann_path "$COCO_ANN_PATH"

echo "[gpu] running worker (torchrun)..."
torchrun --standalone --nnodes "$NNODES" --nproc_per_node "$NPROC" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  "$CODE_DIR/main.py" worker \
    --model_id "$MODEL_ID" \
    --out_dir "$OUT_DIR/$RUN_TAG" \
    --coco_img_dir "$COCO_IMG_DIR" \
    --coco_ann_path "$COCO_ANN_PATH" \
    --chunk_size "$CHUNK_SIZE" \
    --seed "$SEED" \
    --max_samples "$MAX_SAMPLES" \
    --layers $LAYERS

echo "[gpu] analyze..."
python "$CODE_DIR/main.py" analyze \
  --out_dir "$OUT_DIR/$RUN_TAG"

echo "[gpu] summary..."
python "$CODE_DIR/main.py" summary \
  --out_dir "$OUT_DIR/$RUN_TAG"

echo "[gpu] done. outputs at: $OUT_DIR/$RUN_TAG"
