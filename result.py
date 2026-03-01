#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from pathlib import Path

import pandas as pd

def _find_score_col(df: pd.DataFrame, preferred: str | None = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    candidates = [
        "ced", "CED", "ced_sum", "ced_total", "avg_ced_per_token",
        "js_sum", "JS_sum", "kl_sum", "KL_sum",
        "axis_delta_yesno", "axis_delta",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        raise SystemExit("No numeric columns found to use as score_col.")
    return num_cols[0]

def merge_shards(result_dir: Path, pattern: str = "p0b_shard_*_of_*.csv") -> Path:
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

def compute_auc_binary(y, s):
    # sklearn-free AUC using rank statistic (Mann–Whitney U)
    import numpy as np
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)

    # tie handling
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
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def main():
    ap = argparse.ArgumentParser(description="Summarize P0 results (merge shards, compute AUC/stats, save plots).")
    ap.add_argument("--result_dir", type=str, default=os.environ.get("P0_RESULT_DIR", "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/results"))
    ap.add_argument("--merged_csv", type=str, default="", help="Optional: explicit path to merged csv")
    ap.add_argument("--score_col", type=str, default="", help="Column to use as score (auto if empty)")
    ap.add_argument("--behavior_col", type=str, default="behavior")
    ap.add_argument("--out_dir", type=str, default="", help="Where to write summary + plots (default: result_dir)")
    ap.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    ap.add_argument("--cp_label", type=str, default="correct_positive")
    ap.add_argument("--hal_label", type=str, default="hallucination")
    args = ap.parse_args()

    result_dir = Path(args.result_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else result_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve merged csv
    if args.merged_csv:
        merged_path = Path(args.merged_csv).expanduser().resolve()
        if not merged_path.exists():
            raise SystemExit(f"--merged_csv not found: {merged_path}")
    else:
        merged_path = result_dir / "p0b_merged.csv"
        if not merged_path.exists():
            print(f"[info] {merged_path} not found, merging shards...")
            merged_path = merge_shards(result_dir)

    df = pd.read_csv(merged_path)
    print("============================================================")
    print("P0 Results Summary")
    print("============================================================")
    print(f"merged_csv : {merged_path}")
    print(f"rows       : {len(df)}")
    print(f"columns    : {len(df.columns)}")
    print("")

    if args.behavior_col not in df.columns:
        raise SystemExit(f"Missing behavior column '{args.behavior_col}'. Available columns: {list(df.columns)[:50]}")

    score_col = _find_score_col(df, args.score_col or None)
    print(f"score_col  : {score_col}")
    print("")

    g = df.groupby(args.behavior_col)[score_col].agg(["count","mean","std","min","max"]).sort_values("count", ascending=False)
    print("--- Behavior Distribution ---")
    print(g.to_string())
    print("")

    sub = df[df[args.behavior_col].isin([args.cp_label, args.hal_label])].copy()
    cp_n = int((sub[args.behavior_col] == args.cp_label).sum())
    hal_n = int((sub[args.behavior_col] == args.hal_label).sum())
    print("--- Core AUC: cp vs halluc ---")
    print(f"  cp={cp_n} hal={hal_n} total={len(sub)}")

    if len(sub) > 0 and cp_n > 0 and hal_n > 0:
        y = (sub[args.behavior_col] == args.cp_label).astype(int).values
        s = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).values
        auc = compute_auc_binary(y, s)
        print(f"  AUC({score_col}) = {auc:.4f}")
    else:
        auc = float("nan")
        print("  Not enough samples for AUC.")
    print("")

    # Save summary json
    import json
    summary = {
        "merged_csv": str(merged_path),
        "rows": int(len(df)),
        "score_col": score_col,
        "behavior_stats": g.reset_index().to_dict(orient="records"),
        "cp_label": args.cp_label,
        "hal_label": args.hal_label,
        "cp_n": cp_n,
        "hal_n": hal_n,
        "auc": auc,
    }
    summary_path = out_dir / "p0b_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[saved] {summary_path}")

    if args.no_plots:
        return

    # Plots (matplotlib only; do not set specific colors)
    import matplotlib.pyplot as plt

    if cp_n > 0 and hal_n > 0:
        cp = pd.to_numeric(sub[sub[args.behavior_col]==args.cp_label][score_col], errors="coerce").dropna().values
        hal = pd.to_numeric(sub[sub[args.behavior_col]==args.hal_label][score_col], errors="coerce").dropna().values

        plt.figure()
        plt.hist(cp, bins=50, alpha=0.5, label=args.cp_label)
        plt.hist(hal, bins=50, alpha=0.5, label=args.hal_label)
        plt.xlabel(score_col)
        plt.ylabel("count")
        plt.legend()
        hist_path = out_dir / "p0b_hist_cp_vs_hal.png"
        plt.savefig(hist_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[saved] {hist_path}")

        # ROC curve (sklearn optional)
        try:
            from sklearn.metrics import roc_curve, auc as sk_auc
            y = (sub[args.behavior_col] == args.cp_label).astype(int).values
            s = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).values
            fpr, tpr, _ = roc_curve(y, s)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC={sk_auc(fpr,tpr):.4f}")
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            roc_path = out_dir / "p0b_roc_cp_vs_hal.png"
            plt.savefig(roc_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[saved] {roc_path}")
        except Exception as e:
            print(f"[warn] ROC plot skipped: {e}")

if __name__ == "__main__":
    main()
