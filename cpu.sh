#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# cpu.sh  (SAFE MODE)
# ------------------------------------------------------------
# 这个脚本只做“准备工作”，默认**不改你的现有环境**：
#   - 不 rsync --delete（不会乱删你目录里的东西）
#   - 不强行下载 pip / setuptools / wheel
#   - 只把运行所需的脚本/requirements 写到 $BASE_DIR/code
#
# 如果你确实需要它去下载 COCO / wheelhouse，显式打开：
#   P0_ALLOW_DOWNLOAD=1 bash cpu.sh
# ============================================================

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${P0_BASE_DIR:-$HERE_DIR}"

CODE_DIR="$BASE_DIR/code"
DATA_DIR="$BASE_DIR/data"
WHEELHOUSE="${P0_WHEELHOUSE:-$DATA_DIR/wheels}"
COCO_DIR="${P0_COCO_DIR:-$DATA_DIR/coco2017}"

mkdir -p "$CODE_DIR" "$WHEELHOUSE" "$COCO_DIR"

# --- 1) 同步必要文件（不删除旧文件） ---
cp -f "$HERE_DIR/main.py" "$CODE_DIR/main.py"
cp -f "$HERE_DIR/gpu.sh"  "$CODE_DIR/gpu.sh"
cp -f "$HERE_DIR/cpu.sh"  "$CODE_DIR/cpu.sh"

# --- 2) 写 requirements（不含 torch，GPU 脚本负责） ---
# 重点：
# - fsspec 固定到 2024.9.0（兼容 datasets 3.2.0 的约束，避免 resolver 警告）
# - 不强拉 datasets（你的任务不需要它；有的话也能兼容）
cat > "$CODE_DIR/requirements.notorch.txt" <<'REQ'
# Core
numpy==2.0.1
pillow==10.4.0
opencv-python-headless==4.10.0.84
ujson==5.10.0
tqdm==4.66.5

# HF stack
transformers==5.2.0
huggingface_hub==1.5.0
tokenizers==0.22.2
safetensors==0.5.2
accelerate==1.10.1

# Qwen-VL utils
qwen-vl-utils==0.0.13
einops==0.8.2

# Compatibility guard (datasets 3.2.0 requires fsspec[http]<=2024.9.0)
fsspec[http]==2024.9.0
REQ

# “lock” 这里就先等同 notorch，方便离线装；你也可以自己生成更严格的 lock。
cp -f "$CODE_DIR/requirements.notorch.txt" "$CODE_DIR/requirements.lock.txt"

echo "[cpu] wrote:"
echo "  - $CODE_DIR/requirements.notorch.txt"
echo "  - $CODE_DIR/requirements.lock.txt"
echo "[cpu] synced: main.py / gpu.sh / cpu.sh -> $CODE_DIR (NO DELETE)"

# --- 3) 可选：下载数据与 wheelhouse（默认关闭） ---
if [[ "${P0_ALLOW_DOWNLOAD:-0}" != "1" ]]; then
  echo "[cpu] download step skipped (set P0_ALLOW_DOWNLOAD=1 to enable)"
  exit 0
fi

# 下面下载只在你明确允许时执行。
# 注意：集群/容器通常有自己的 pip 镜像配置（甚至是 http 源），
# 我们这里用 --isolated 避免读系统 pip.conf，尽量减少被“怪镜像”坑。

PYBIN="${P0_PYBIN:-python3}"

echo "[cpu] downloading COCO2017 val + annotations (optional) ..."
mkdir -p "$COCO_DIR"

VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

if [[ ! -d "$COCO_DIR/val2017" ]]; then
  curl -L "$VAL_URL" -o "$COCO_DIR/val2017.zip"
  unzip -q "$COCO_DIR/val2017.zip" -d "$COCO_DIR"
fi

if [[ ! -f "$COCO_DIR/annotations/instances_val2017.json" ]]; then
  curl -L "$ANN_URL" -o "$COCO_DIR/annotations_trainval2017.zip"
  unzip -q "$COCO_DIR/annotations_trainval2017.zip" -d "$COCO_DIR"
fi

echo "[cpu] building wheelhouse into: $WHEELHOUSE"
# 仅下载 notorch 依赖（torch 在 gpu.sh 里处理）
$PYBIN -m pip --isolated download -r "$CODE_DIR/requirements.notorch.txt" -d "$WHEELHOUSE"

echo "[cpu] done. wheelhouse contains $(ls -1 "$WHEELHOUSE" | wc -l) files"
