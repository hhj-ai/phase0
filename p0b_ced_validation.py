#!/usr/bin/env python3
"""
P0-b: CED公式验证（按新计划重写）

=== 核心改动（vs旧版） ===
1. 行为分组：先让模型回答，再按回答与GT的一致性分组
   - correct_positive: GT=存在 & 模型答对 → 替换物体区域 → 期望CED高
   - hallucination:    GT=不存在 & 模型答"有" → 替换区域 → 期望CED低
   - correct_negative: GT=不存在 & 模型答对 → 替换区域 → 期望CED低
   - miss:             GT=存在 & 模型答"无" → 替换区域 → 期望CED中
   + 每个正例样本还做控制组（替换无物体区域）→ 期望CED低

2. CED公式消融：同时计算6种变体
   A. 裸JS散度
   B. CED = JS + λ_e·[H(P^cf)-H(P)]_+  （你的公式）
   C. JS + λ_e·H(P^cf)                  （只看反事实熵）
   D. JS + λ_e·|H(P^cf)-H(P)|           （对称版）
   E. KL(P||P^cf)                        （非对称散度）
   F. cosine_distance(h, h^cf)            （特征空间距离）

3. 跨任务：存在性/空间/属性/计数 四类问题
4. 跨层：logits + 中间层 16,20,24,28,32

=== 通过标准 ===
correct_positive vs hallucination 的 AUC > 0.85

单卡运行（8B bf16 ≈ 16GB，逐样本推理）
"""

import os
import json
import random
import argparse
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ============================================================
# 路径
# ============================================================
BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH = f"{BASE}/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG_DIR = f"{BASE}/data/datasets/coco_val2017/val2017"
COCO_ANN_PATH = f"{BASE}/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR = f"{BASE}/results"


# ============================================================
# 1. 度量函数
# ============================================================

def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _to_prob(x):
    p = np.asarray(x, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    return p

def js_divergence(p, q):
    p, q = _to_prob(p), _to_prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))

def kl_divergence(p, q):
    p, q = _to_prob(p), _to_prob(q)
    return float(entropy(p, q, base=2))

def shannon_entropy(p):
    return float(entropy(_to_prob(p), base=2))

def compute_all_metrics(orig_logits, cf_logits, lambda_e_values=[0.0, 0.05, 0.1, 0.2, 0.5]):
    """计算所有6种度量变体"""
    p = _softmax(orig_logits)
    q = _softmax(cf_logits)

    js = js_divergence(p, q)
    kl = kl_divergence(p, q)
    h_orig = shannon_entropy(p)
    h_cf = shannon_entropy(q)
    entropy_penalty = max(0.0, h_cf - h_orig)

    result = {
        "js": js,                                  # A: 裸JS
        "kl": kl,                                  # E: KL(P||P^cf)
        "h_orig": h_orig,
        "h_cf": h_cf,
        "entropy_penalty": entropy_penalty,
    }

    for lam in lambda_e_values:
        result[f"ced_lam{lam:.2f}"] = js + lam * entropy_penalty           # B: CED
        result[f"ced_abs_lam{lam:.2f}"] = js + lam * abs(h_cf - h_orig)    # D: 对称
        result[f"ced_hcf_lam{lam:.2f}"] = js + lam * h_cf                  # C: 只看H^cf

    return result

def compute_hidden_js(h_orig, h_cf, top_k=4096):
    """中间层JS（softmax转伪分布后计算）"""
    h1 = h_orig.cpu().float().numpy()
    h2 = h_cf.cpu().float().numpy()
    top_dims = np.union1d(np.argsort(np.abs(h1))[-top_k:], np.argsort(np.abs(h2))[-top_k:])
    return js_divergence(_softmax(h1[top_dims]), _softmax(h2[top_dims]))

def compute_cosine_dist(h_orig, h_cf):
    """F: cosine distance"""
    h1 = h_orig.cpu().float().numpy()
    h2 = h_cf.cpu().float().numpy()
    return float(cosine_dist(h1, h2))


# ============================================================
# 2. Hook
# ============================================================

class VisualTokenHook:
    def __init__(self):
        self.captured = None
        self._modifications = None
        self._handle = None

    def register(self, model):
        self._handle = model.visual.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._handle:
            self._handle.remove()

    def reset(self):
        self.captured = None
        self._modifications = None

    def set_replace(self, mods):
        self._modifications = mods

    def _fn(self, module, input, output):
        if self._modifications is None:
            self.captured = output.detach().clone()
            return output
        else:
            mod = self.captured.clone().to(device=output.device, dtype=output.dtype)
            for idx, val in self._modifications.items():
                if 0 <= idx < mod.shape[0]:
                    mod[idx] = val.to(device=mod.device, dtype=mod.dtype)
            return mod


# ============================================================
# 3. 数据构造
# ============================================================

SPATIAL_TEMPLATES = [
    ("Is the {obj} on the left side of the image?", lambda b, w, h: b[0] + b[2]/2 < w/2),
    ("Is the {obj} on the right side of the image?", lambda b, w, h: b[0] + b[2]/2 > w/2),
    ("Is the {obj} in the upper half of the image?", lambda b, w, h: b[1] + b[3]/2 < h/2),
    ("Is the {obj} in the lower half of the image?", lambda b, w, h: b[1] + b[3]/2 > h/2),
]

ATTRIBUTE_MAP = {
    "traffic light": ("Is the traffic light red?", None),
    "banana": ("Is the banana yellow?", None),
    "broccoli": ("Is the broccoli green?", None),
    "fire hydrant": ("Is the fire hydrant red?", None),
    "stop sign": ("Is there a red stop sign?", None),
}

def load_samples(ann_path, img_dir, n_samples=400, min_bbox_ratio=0.02):
    """构造四类问题的样本"""
    print(f"Loading COCO from {ann_path}...")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {i["id"]: i for i in coco["images"]}
    all_cat_ids = set(cat_map.keys())

    img_anns = defaultdict(list)
    for a in coco["annotations"]:
        img_anns[a["image_id"]].append(a)

    img_cats = {iid: set(a["category_id"] for a in anns) for iid, anns in img_anns.items()}
    samples = []
    img_ids = list(img_anns.keys())
    random.shuffle(img_ids)

    for img_id in img_ids:
        if len(samples) >= n_samples:
            break

        anns = img_anns[img_id]
        info = img_map[img_id]
        w, h = info["width"], info["height"]
        path = os.path.join(img_dir, info["file_name"])
        if not os.path.exists(path):
            continue

        valid = [a for a in anns if a["bbox"][2]*a["bbox"][3]/(w*h) > min_bbox_ratio and not a.get("iscrowd", 0)]
        if not valid:
            continue

        target = max(valid, key=lambda a: a["bbox"][2]*a["bbox"][3])
        cat = cat_map[target["category_id"]]
        bbox = target["bbox"]
        all_bboxes = [a["bbox"] for a in anns]

        # --- 存在性正例 ---
        samples.append({
            "task": "existence", "img_path": path, "img_w": w, "img_h": h,
            "question": f"Is there a {cat} in this image? Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": 1,  # 物体存在
        })

        # --- 存在性反例（问不存在的物体） ---
        absent = all_cat_ids - img_cats.get(img_id, set())
        if absent:
            abs_name = cat_map[random.choice(list(absent))]
            samples.append({
                "task": "existence", "img_path": path, "img_w": w, "img_h": h,
                "question": f"Is there a {abs_name} in this image? Answer yes or no.",
                "object": abs_name, "bbox": bbox, "all_bboxes": all_bboxes,
                "gt": 0,
            })

        # --- 空间关系 ---
        tmpl, check_fn = random.choice(SPATIAL_TEMPLATES)
        gt_spatial = 1 if check_fn(bbox, w, h) else 0
        samples.append({
            "task": "spatial", "img_path": path, "img_w": w, "img_h": h,
            "question": tmpl.format(obj=cat) + " Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": gt_spatial,
        })

        # --- 计数 ---
        same_cat = [a for a in valid if a["category_id"] == target["category_id"]]
        count = len(same_cat)
        wrong_count = count + random.choice([1, 2]) if random.random() < 0.5 else max(1, count - 1)
        ask_count = wrong_count if random.random() < 0.5 else count
        samples.append({
            "task": "counting", "img_path": path, "img_w": w, "img_h": h,
            "question": f"Are there exactly {ask_count} {cat}(s) in this image? Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": 1 if ask_count == count else 0,
        })

    task_counts = defaultdict(int)
    for s in samples:
        task_counts[s["task"]] += 1
    print(f"Loaded {len(samples)} samples: {dict(task_counts)}")
    return samples


# ============================================================
# 4. 单样本处理
# ============================================================

def get_model_answer(model, processor, image, question, device):
    """让模型实际回答问题，提取yes/no"""
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=20)

    # 只取新生成的token
    input_len = inputs["input_ids"].shape[1]
    answer_text = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip().lower()

    if "yes" in answer_text:
        return 1, answer_text
    elif "no" in answer_text:
        return 0, answer_text
    else:
        return -1, answer_text  # 无法解析


def bbox_to_indices(bbox, img_w, img_h, grid_h, grid_w):
    x, y, w, h = bbox
    cs = max(0, int(x / img_w * grid_w))
    ce = min(grid_w, int(np.ceil((x+w) / img_w * grid_w)))
    rs = max(0, int(y / img_h * grid_h))
    re = min(grid_h, int(np.ceil((y+h) / img_h * grid_h)))
    return [r * grid_w + c for r in range(rs, re) for c in range(cs, ce)]


def get_control_indices(all_bboxes, img_w, img_h, grid_h, grid_w, n):
    occupied = set()
    for b in all_bboxes:
        occupied.update(bbox_to_indices(b, img_w, img_h, grid_h, grid_w))
    free = sorted(set(range(grid_h * grid_w)) - occupied)
    if len(free) >= n:
        start = random.randint(0, len(free) - n)
        return free[start:start+n]
    return free if free else list(range(min(n, grid_h * grid_w)))


def process_sample(sample, model, processor, hook, device, layers, lambda_e_values):
    from qwen_vl_utils import process_vision_info

    try:
        image = Image.open(sample["img_path"]).convert("RGB")
    except:
        return None

    # --- Step 1: 获取模型实际回答 ---
    pred, pred_text = get_model_answer(model, processor, image, sample["question"], device)
    if pred == -1:
        return None  # 无法解析

    # 行为分组
    gt = sample["gt"]
    if gt == 1 and pred == 1:
        behavior = "correct_positive"
    elif gt == 0 and pred == 1:
        behavior = "hallucination"
    elif gt == 0 and pred == 0:
        behavior = "correct_negative"
    elif gt == 1 and pred == 0:
        behavior = "miss"
    else:
        return None

    # --- Step 2: 构造输入（用于CED计算，不generate，只取logits） ---
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": sample["question"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    grid_thw = inputs.get("image_grid_thw")
    if grid_thw is None:
        return None
    grid_h = grid_thw[0, 1].item()
    grid_w = grid_thw[0, 2].item()

    # --- Step 3: 原始forward (capture) ---
    hook.reset()
    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    if hook.captured is None:
        return None

    # --- Step 4: 物体区域替换 ---
    target_idx = bbox_to_indices(sample["bbox"], sample["img_w"], sample["img_h"], grid_h, grid_w)
    if not target_idx:
        return None

    vis_feat = hook.captured
    surr_idx = sorted(set(range(vis_feat.shape[0])) - set(target_idx))
    replacement = vis_feat[surr_idx].mean(0) if surr_idx else vis_feat.mean(0)

    mods = {i: replacement for i in target_idx}
    hook.set_replace(mods)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)
    hook.reset()

    # --- Step 5: 计算所有度量 ---
    orig_logits = orig_out.logits[0, -1].cpu().float().numpy()
    cf_logits = cf_out.logits[0, -1].cpu().float().numpy()

    metrics = compute_all_metrics(orig_logits, cf_logits, lambda_e_values)

    # cosine distance (F)
    h_orig_last = orig_out.hidden_states[-1][0, -1]
    h_cf_last = cf_out.hidden_states[-1][0, -1]
    metrics["cosine_dist"] = compute_cosine_dist(h_orig_last, h_cf_last)

    # 中间层
    for layer in layers:
        if layer < len(orig_out.hidden_states):
            metrics[f"layer_{layer}_js"] = compute_hidden_js(
                orig_out.hidden_states[layer][0, -1],
                cf_out.hidden_states[layer][0, -1]
            )

    result = {
        "task": sample["task"],
        "object": sample["object"],
        "gt": gt,
        "pred": pred,
        "pred_text": pred_text,
        "behavior": behavior,
        "n_target_tokens": len(target_idx),
        "n_total_tokens": grid_h * grid_w,
        **metrics,
    }

    # --- Step 6: 控制组（仅对correct_positive做）---
    if behavior == "correct_positive":
        n_ctrl = max(4, len(target_idx))
        ctrl_idx = get_control_indices(sample["all_bboxes"], sample["img_w"], sample["img_h"],
                                       grid_h, grid_w, n_ctrl)
        surr_ctrl = sorted(set(range(vis_feat.shape[0])) - set(ctrl_idx))
        repl_ctrl = vis_feat[surr_ctrl].mean(0) if surr_ctrl else vis_feat.mean(0)

        hook.reset()
        # 需要重新capture
        with torch.no_grad():
            _ = model(**inputs, output_hidden_states=False, return_dict=True)

        hook.set_replace({i: repl_ctrl for i in ctrl_idx})
        with torch.no_grad():
            ctrl_out = model(**inputs, output_hidden_states=False, return_dict=True)
        hook.reset()

        ctrl_logits = ctrl_out.logits[0, -1].cpu().float().numpy()
        ctrl_metrics = compute_all_metrics(orig_logits, ctrl_logits, lambda_e_values)
        result["ctrl_js"] = ctrl_metrics["js"]
        for lam in lambda_e_values:
            result[f"ctrl_ced_lam{lam:.2f}"] = ctrl_metrics[f"ced_lam{lam:.2f}"]

    return result


# ============================================================
# 5. 主循环
# ============================================================

def main(args):
    print("=" * 60)
    print("P0-b: CED公式验证 + 行为分组实验")
    print("=" * 60)
    print(f"通过标准: correct_positive vs hallucination AUC > 0.85\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    hook = VisualTokenHook().register(model)

    # 加载probe信息
    probe_path = f"{BASE}/results/p0a_probe_info.json"
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            probe = json.load(f)
        layers = [l for l in args.layers if l < probe["n_hidden_layers"]]
        print(f"P0-a探测: grid={probe['grid_h']}x{probe['grid_w']}, layers={layers}")
    else:
        layers = args.layers

    samples = load_samples(COCO_ANN_PATH, COCO_IMG_DIR, n_samples=args.num_samples)

    results = []
    behavior_counts = defaultdict(int)

    for sample in tqdm(samples, desc="P0-b"):
        r = process_sample(sample, model, processor, hook, device, layers, args.lambda_e_values)
        if r:
            results.append(r)
            behavior_counts[r["behavior"]] += 1

        # 实时进度
        if len(results) % 50 == 0 and len(results) > 0:
            df_t = pd.DataFrame(results)
            cp = df_t[df_t["behavior"] == "correct_positive"]
            hal = df_t[df_t["behavior"] == "hallucination"]
            if len(cp) > 5 and len(hal) > 5:
                sub = pd.concat([cp, hal])
                labels = (sub["behavior"] == "correct_positive").astype(int)
                auc = roc_auc_score(labels, sub["js"])
                tqdm.write(f"  n={len(results)}, behaviors={dict(behavior_counts)}, "
                           f"cp_vs_hal AUC(js)={auc:.3f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"{RESULT_DIR}/p0b_results.csv", index=False)
    print(f"\nSaved {len(df)} results to {RESULT_DIR}/p0b_results.csv")

    analyze(df, args.lambda_e_values, layers)
    hook.remove()


# ============================================================
# 6. 分析
# ============================================================

def safe_auc(labels, scores):
    if len(set(labels)) < 2 or len(labels) < 10:
        return float("nan")
    return roc_auc_score(labels, scores)

def analyze(df, lambda_e_values, layers):
    print("\n" + "=" * 60)
    print("P0-b 结果分析")
    print("=" * 60)

    # 6.1 行为分布
    print("\n--- 行为分布 ---")
    for b in ["correct_positive", "hallucination", "correct_negative", "miss"]:
        sub = df[df["behavior"] == b]
        if len(sub) > 0:
            print(f"  {b:20s}: n={len(sub):4d}, JS={sub['js'].mean():.4f}±{sub['js'].std():.4f}")

    # 6.2 核心AUC: correct_positive vs hallucination
    print("\n--- 核心AUC: correct_positive vs hallucination ---")
    cp = df[df["behavior"] == "correct_positive"]
    hal = df[df["behavior"] == "hallucination"]

    if len(cp) < 5 or len(hal) < 5:
        print(f"WARNING: 样本不足 (cp={len(cp)}, hal={len(hal)})")
        print("  幻觉样本太少——模型在该数据上不怎么犯错")
        print("  建议: 增加样本量或换更难的问题模板")
        # 退而求其次：用gt=1 vs gt=0
        print("\n--- 退化方案: gt=1 vs gt=0 ---")
        labels = df["gt"]
        if labels.nunique() > 1:
            for metric in ["js"] + [f"ced_lam{l:.2f}" for l in lambda_e_values]:
                if metric in df.columns:
                    auc = safe_auc(labels, df[metric])
                    print(f"  {metric:25s}: AUC = {auc:.4f}")
        return

    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)])

    print("\n  [Logits层]")

    # A: 裸JS
    auc_js = safe_auc(sub["label"], sub["js"])
    print(f"  A. 裸JS散度:              AUC = {auc_js:.4f}")

    # B: CED各lambda
    best_auc, best_config = auc_js, "js"
    for lam in lambda_e_values:
        col = f"ced_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            tag = " ★" if auc > best_auc else ""
            print(f"  B. CED(λ={lam:.2f}):           AUC = {auc:.4f}{tag}")
            if auc > best_auc:
                best_auc, best_config = auc, col

    # C: ced_hcf
    for lam in [0.1]:
        col = f"ced_hcf_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  C. JS+λH(P^cf) (λ={lam:.2f}): AUC = {auc:.4f}")

    # D: ced_abs
    for lam in [0.1]:
        col = f"ced_abs_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  D. JS+λ|ΔH| (λ={lam:.2f}):    AUC = {auc:.4f}")

    # E: KL
    auc_kl = safe_auc(sub["label"], sub["kl"])
    print(f"  E. KL(P||P^cf):            AUC = {auc_kl:.4f}")
    if auc_kl > best_auc:
        best_auc, best_config = auc_kl, "kl"

    # F: cosine
    if "cosine_dist" in sub.columns:
        auc_cos = safe_auc(sub["label"], sub["cosine_dist"])
        print(f"  F. Cosine distance:        AUC = {auc_cos:.4f}")
        if auc_cos > best_auc:
            best_auc, best_config = auc_cos, "cosine_dist"

    print(f"\n  → 最优: {best_config}, AUC = {best_auc:.4f}")

    # 6.3 中间层
    print("\n  [中间层]")
    for layer in layers:
        col = f"layer_{layer}_js"
        if col in sub.columns and sub[col].notna().sum() > 10:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  Layer {layer}: AUC = {auc:.4f}")

    # 6.4 跨任务
    print("\n--- 跨任务一致性 ---")
    for task in ["existence", "spatial", "counting"]:
        task_sub = df[df["task"] == task]
        cp_t = task_sub[task_sub["behavior"] == "correct_positive"]
        hal_t = task_sub[task_sub["behavior"] == "hallucination"]
        if len(cp_t) >= 3 and len(hal_t) >= 3:
            s = pd.concat([cp_t.assign(label=1), hal_t.assign(label=0)])
            auc = safe_auc(s["label"], s["js"])
            print(f"  {task:12s}: AUC = {auc:.4f} (cp={len(cp_t)}, hal={len(hal_t)})")
        else:
            print(f"  {task:12s}: 样本不足 (cp={len(cp_t)}, hal={len(hal_t)})")

    # 6.5 控制组对比
    print("\n--- 控制组（仅correct_positive） ---")
    ctrl_rows = cp[cp["ctrl_js"].notna()] if "ctrl_js" in cp.columns else pd.DataFrame()
    if len(ctrl_rows) > 0:
        print(f"  物体区域JS:  {ctrl_rows['js'].mean():.4f} ± {ctrl_rows['js'].std():.4f}")
        print(f"  无物体区域JS: {ctrl_rows['ctrl_js'].mean():.4f} ± {ctrl_rows['ctrl_js'].std():.4f}")
        ratio = ctrl_rows['js'].mean() / max(ctrl_rows['ctrl_js'].mean(), 1e-8)
        print(f"  比值: {ratio:.1f}x")

    # 6.6 熵项分析
    print("\n--- 熵项分析 ---")
    for b in ["correct_positive", "hallucination", "correct_negative"]:
        s = df[df["behavior"] == b]
        if len(s) > 0:
            print(f"  {b:20s}: H(P)={s['h_orig'].mean():.3f}, "
                  f"H(P^cf)={s['h_cf'].mean():.3f}, "
                  f"Δ_+={s['entropy_penalty'].mean():.4f}")

    # 6.7 Pass/Fail
    print("\n" + "=" * 60)
    if best_auc > 0.85:
        print(f"✅ P0-b PASSED: AUC = {best_auc:.4f} > 0.85 ({best_config})")
        print(f"   → 进入 Phase 1 GRPO!")
    elif best_auc > 0.75:
        print(f"⚠️  P0-b MARGINAL: AUC = {best_auc:.4f}")
        print(f"   建议: 调整token替换策略或增加样本")
    else:
        print(f"❌ P0-b FAILED: AUC = {best_auc:.4f}")
        print(f"   建议: 切换VCD噪声/MaskCD/PROJECTAWAY")
    print("=" * 60)

    # 可视化
    plot_results(df, sub, lambda_e_values, layers)


def plot_results(df, sub, lambda_e_values, layers):
    os.makedirs(RESULT_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. JS分布（按行为）
    ax = axes[0, 0]
    colors = {"correct_positive": "green", "hallucination": "red",
              "correct_negative": "blue", "miss": "orange"}
    for b, c in colors.items():
        s = df[df["behavior"] == b]
        if len(s) > 2:
            ax.hist(s["js"], bins=25, alpha=0.4, color=c, label=f"{b} (n={len(s)})", density=True)
    ax.set_xlabel("JS Divergence")
    ax.set_title("JS Distribution by Behavior")
    ax.legend(fontsize=8)

    # 2. 公式消融AUC
    ax = axes[0, 1]
    if len(sub) > 0 and sub["label"].nunique() > 1:
        names, aucs = [], []
        names.append("JS"); aucs.append(safe_auc(sub["label"], sub["js"]))
        for lam in lambda_e_values:
            col = f"ced_lam{lam:.2f}"
            if col in sub.columns:
                names.append(f"CED λ={lam}")
                aucs.append(safe_auc(sub["label"], sub[col]))
        names.append("KL"); aucs.append(safe_auc(sub["label"], sub["kl"]))
        if "cosine_dist" in sub.columns:
            names.append("Cosine"); aucs.append(safe_auc(sub["label"], sub["cosine_dist"]))
        ax.barh(names, aucs, color=["green" if a > 0.85 else "orange" if a > 0.75 else "red" for a in aucs])
        ax.axvline(0.85, color="red", ls="--", alpha=0.5)
        ax.set_xlabel("AUC")
        ax.set_title("Formula Ablation (cp vs hal)")
        ax.set_xlim(0.5, 1.0)

    # 3. 各层AUC
    ax = axes[0, 2]
    if len(sub) > 0 and sub["label"].nunique() > 1:
        layer_names, layer_aucs = ["logits"], [safe_auc(sub["label"], sub["js"])]
        for l in layers:
            col = f"layer_{l}_js"
            if col in sub.columns and sub[col].notna().sum() > 10:
                layer_names.append(f"L{l}")
                layer_aucs.append(safe_auc(sub["label"], sub[col]))
        ax.bar(layer_names, layer_aucs,
               color=["green" if a > 0.85 else "orange" if a > 0.75 else "red" for a in layer_aucs])
        ax.axhline(0.85, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("AUC")
        ax.set_title("AUC by Layer")
        ax.set_ylim(0.5, 1.0)

    # 4. ROC
    ax = axes[1, 0]
    if len(sub) > 0 and sub["label"].nunique() > 1:
        fpr, tpr, _ = roc_curve(sub["label"], sub["js"])
        ax.plot(fpr, tpr, label=f"JS (AUC={safe_auc(sub['label'], sub['js']):.3f})", lw=2)
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve"); ax.legend()

    # 5. 控制组对比
    ax = axes[1, 1]
    cp = df[df["behavior"] == "correct_positive"]
    if "ctrl_js" in cp.columns and cp["ctrl_js"].notna().sum() > 5:
        ax.boxplot([cp["js"].dropna(), cp["ctrl_js"].dropna()],
                   labels=["Object Region", "Control Region"])
        ax.set_ylabel("JS Divergence")
        ax.set_title("Object vs Control Region (correct_positive only)")

    # 6. 跨任务
    ax = axes[1, 2]
    task_aucs = {}
    for task in ["existence", "spatial", "counting"]:
        t = df[df["task"] == task]
        cp_t = t[t["behavior"] == "correct_positive"]
        hal_t = t[t["behavior"] == "hallucination"]
        if len(cp_t) >= 3 and len(hal_t) >= 3:
            s = pd.concat([cp_t.assign(label=1), hal_t.assign(label=0)])
            task_aucs[task] = safe_auc(s["label"], s["js"])
    if task_aucs:
        ax.bar(task_aucs.keys(), task_aucs.values(),
               color=["green" if v > 0.85 else "orange" for v in task_aucs.values()])
        ax.axhline(0.85, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("AUC"); ax.set_title("Cross-task Consistency")
        ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig(f"{RESULT_DIR}/p0b_analysis.png", dpi=150)
    plt.close()
    print(f"\nCharts saved to {RESULT_DIR}/p0b_analysis.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=400)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    parser.add_argument("--lambda_e_values", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
