#!/bin/bash
set -euo pipefail

# ============================================================
# gpu.sh - GPU服务器（无网）离线安装 + 8卡分片运行（不使用torchrun）
# 修复点：
#  1) 去掉 --model_path（p0_experiment.py 不支持这个参数）
#  2) 不用 torchrun（p0_experiment.py 不是DDP脚本，应该用 shard_idx/num_shards）
#  3) 每个 shard 绑定一张 GPU：CUDA_VISIBLE_DEVICES=i
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
WHEELS="$SHARED/data/wheels"
VENV="$SHARED/venv/p0_env"
REQ="$SHARED/code/requirements_offline.txt"

GPUS="${1:-8}"

echo "================================================================"
echo "GPU offline install + run (manual sharding)"
echo "  SHARED: $SHARED"
echo "  WHEELS: $WHEELS"
echo "  GPUS  : $GPUS"
echo "================================================================"

# 彻底屏蔽 pip 全局配置污染
unset PIP_FIND_LINKS PIP_INDEX_URL PIP_EXTRA_INDEX_URL
export PIP_CONFIG_FILE=/dev/null

mkdir -p "$SHARED/logs" "$SHARED/results"

# 1) venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

# 2) 离线升级基础工具链（wheelhouse里有就升；没有也不致命）
python -m pip install --isolated --no-index --find-links "$WHEELS" -U pip setuptools wheel || true

# 3) 离线安装 torch
python -m pip install --isolated --no-index --find-links "$WHEELS" torch torchvision torchaudio

# 4) nvJitLink/cusparse 动态库优先级修复（你之前那个 ImportError 就靠这个）
SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
NVJ="$SITE/nvidia/nvjitlink/lib"
CUS="$SITE/nvidia/cusparse/lib"
export LD_LIBRARY_PATH="$NVJ:$CUS:${LD_LIBRARY_PATH:-}"
if [ -f "$NVJ/libnvJitLink.so.12" ]; then
  export LD_PRELOAD="$NVJ/libnvJitLink.so.12:${LD_PRELOAD:-}"
fi

# 5) 离线安装其余依赖（带版本约束，确保 hub/tokenizers 满足 transformers dev）
if [ ! -f "$REQ" ]; then
  echo "✗ missing $REQ (请先在CPU跑 cpu.sh 生成它)"
  exit 1
fi
python -m pip install --isolated --no-index --find-links "$WHEELS" -r "$REQ"

# 6) transformers（优先装 wheelhouse 里的 transformers-*.whl）
TRANS_WHL="$(ls -t "$WHEELS"/transformers-*.whl 2>/dev/null | head -1 || true)"
if [ -n "$TRANS_WHL" ]; then
  python -m pip install --isolated --no-index --find-links "$WHEELS" "$TRANS_WHL"
else
  python -m pip install --isolated --no-index --find-links "$WHEELS" transformers
fi

echo ""
echo "== sanity check =="
python - <<'PY'
import torch, huggingface_hub, tokenizers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "ngpu", torch.cuda.device_count())
print("huggingface_hub", huggingface_hub.__version__)
print("tokenizers", tokenizers.__version__)
print("transformers", transformers.__version__)
PY

echo ""
echo "== run probe on GPU0 =="
cd "$SHARED/code"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
python p0_experiment.py --mode probe > "$SHARED/logs/p0a_probe.log" 2>&1 || (echo "probe failed, see $SHARED/logs/p0a_probe.log"; exit 1)

echo ""
echo "== run workers (${GPUS} shards) =="
pids=()
for ((i=0; i<GPUS; i++)); do
  echo "  launch shard $i on GPU $i ..."
  CUDA_VISIBLE_DEVICES=$i OMP_NUM_THREADS=1 \
  python p0_experiment.py \
    --mode worker \
    --shard_idx $i \
    --num_shards $GPUS \
    --num_samples 400 \
    --output_dir "$SHARED/results" \
    --result_dir "$SHARED/results" \
    > "$SHARED/logs/p0b_w${i}.log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "✗ some workers failed. Check logs: $SHARED/logs/p0b_w*.log"
  exit 1
fi

echo ""
echo "== analyze =="
python p0_experiment.py --mode analyze --result_dir "$SHARED/results" > "$SHARED/logs/p0b_analyze.log" 2>&1 \
  || (echo "analyze failed, see $SHARED/logs/p0b_analyze.log"; exit 1)

echo ""
echo "================================================================"
echo "✓ DONE"
echo "  logs   : $SHARED/logs"
echo "  results: $SHARED/results"
echo "================================================================"
