#!/bin/bash
set -euo pipefail

# ============================================================
# result_hardcoded.sh
# 一键汇总 P0-b 结果（自动合并 shards、统计、AUC、出图、写 summary.json）
# 完全硬编码你的路径；脚本放哪都能跑；不需要传参
# ============================================================

SHARED="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
RESULT_DIR="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/results"
VENV="/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/venv/p0_env"

echo "================================================================"
echo "P0 Result Summary (hardcoded)"
echo "  SHARED    : $SHARED"
echo "  RESULT_DIR: $RESULT_DIR"
echo "  VENV      : $VENV"
echo "================================================================"

if [ ! -d "$RESULT_DIR" ]; then
  echo "[error] RESULT_DIR not found: $RESULT_DIR" >&2
  exit 2
fi

# 激活 venv（如果存在）
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  echo "[warn] venv not found, will use system python."
fi

export P0_RESULT_DIR="$RESULT_DIR"

python - <<'PY'
import os
from pathlib import Path
import pandas as pd

def find_score_col(df, preferred=None):
    if preferred and preferred in df.columns:
        return preferred
    for c in ["ced","CED","ced_sum","ced_total","avg_ced_per_token","js_sum","JS_sum","kl_sum","KL_sum"]:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise SystemExit("No numeric columns found to use as score_col.")
    return num_cols[0]

def merge_shards(result_dir: Path, pattern="p0b_shard_*_of_*.csv") -> Path:
    shard_paths = sorted(result_dir.glob(pattern))
    if not shard_paths:
        raise SystemExit(f"No shard csv found in {result_dir} matching {pattern}")
    dfs = []
    for p in shard_paths:
        dfs.append(pd.read_csv(p))
    merged = pd.concat(dfs, ignore_index=True)
    out = result_dir / "p0b_merged.csv"
    merged.to_csv(out, index=False)
    return out

def auc_rank(y, s):
    # sklearn-free AUC via rank statistic (Mann–Whitney U)
    import numpy as np
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            avg_rank = (i + 1 + j + 1) / 2.0
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[y == 1].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

result_dir = Path(os.environ.get("P0_RESULT_DIR")).resolve()
out_dir = result_dir

merged_path = result_dir / "p0b_merged.csv"
if not merged_path.exists():
    print(f"[info] {merged_path} not found, merging shards...")
    merged_path = merge_shards(result_dir)

df = pd.read_csv(merged_path)

print("============================================================")
print("P0 Results Summary")
print("============================================================")
print("merged_csv :", merged_path)
print("rows       :", len(df))
print("columns    :", len(df.columns))
print()

if "behavior" not in df.columns:
    raise SystemExit(f"Missing 'behavior' column. Available columns: {list(df.columns)[:50]}")

score_col = find_score_col(df)
print("score_col  :", score_col)
print()

g = df.groupby("behavior")[score_col].agg(["count","mean","std","min","max"]).sort_values("count", ascending=False)
print("--- Behavior Distribution ---")
print(g.to_string())
print()

sub = df[df["behavior"].isin(["correct_positive","hallucination"])].copy()
cp_n = int((sub["behavior"]=="correct_positive").sum())
hal_n = int((sub["behavior"]=="hallucination").sum())
print("--- Core AUC: correct_positive vs hallucination ---")
print(f"  cp={cp_n} hal={hal_n} total={len(sub)}")

if cp_n > 0 and hal_n > 0:
    y = (sub["behavior"]=="correct_positive").astype(int).values
    s = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).values
    auc = auc_rank(y, s)
    print(f"  AUC({score_col}) = {auc:.4f}")
else:
    auc = float("nan")
    print("  Not enough samples for AUC.")
print()

# save summary json
import json
summary = {
    "merged_csv": str(merged_path),
    "rows": int(len(df)),
    "score_col": score_col,
    "behavior_stats": g.reset_index().to_dict(orient="records"),
    "cp_n": cp_n,
    "hal_n": hal_n,
    "auc": auc,
}
summary_path = out_dir / "p0b_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[saved]", summary_path)

# plots
try:
    import matplotlib.pyplot as plt
    if cp_n > 0 and hal_n > 0:
        cp = pd.to_numeric(sub[sub["behavior"]=="correct_positive"][score_col], errors="coerce").dropna().values
        hal = pd.to_numeric(sub[sub["behavior"]=="hallucination"][score_col], errors="coerce").dropna().values

        plt.figure()
        plt.hist(cp, bins=50, alpha=0.5, label="correct_positive")
        plt.hist(hal, bins=50, alpha=0.5, label="hallucination")
        plt.xlabel(score_col)
        plt.ylabel("count")
        plt.legend()
        hist_path = out_dir / "p0b_hist_cp_vs_hal.png"
        plt.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.close()
        print("[saved]", hist_path)
except Exception as e:
    print("[warn] plotting skipped:", e)
PY

echo "================================================================"
echo "✓ 完成。输出在：$RESULT_DIR"
echo "  - p0b_summary.json"
echo "  - p0b_hist_cp_vs_hal.png (若 matplotlib 可用)"
echo "  - p0b_merged.csv (若原来没有会自动生成)"
echo "================================================================"
