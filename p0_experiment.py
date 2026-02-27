#!/usr/bin/env python3
"""
P0 Experiment: Unified script for Qwen3-VL CED validation

Modes:
  probe   - Architecture probing (single GPU, ~5 minutes)
  worker  - P0-b worker for parallel processing (sharded data)
  analyze - Merge shards and analyze results

Usage:
  python p0_experiment.py --mode probe
  python p0_experiment.py --mode worker --shard_idx 0 --num_shards 8
  python p0_experiment.py --mode analyze --result_dir /path/to/results
"""

import os
import sys
import json
import random
import argparse
import warnings
import glob
from copy import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

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
# Hardcoded Paths
# ============================================================
BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH = f"{BASE}/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG_DIR = f"{BASE}/data/datasets/coco_val2017/val2017"
COCO_ANN_PATH = f"{BASE}/data/datasets/coco_val2017/annotations/instances_val2017.json"
RESULT_DIR = f"{BASE}/results"

# ============================================================
# Utility Functions
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
    """Jensen-Shannon divergence."""
    p, q = _to_prob(p), _to_prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))

def kl_divergence(p, q):
    """KL divergence from p to q."""
    p, q = _to_prob(p), _to_prob(q)
    return float(entropy(p, q, base=2))

def shannon_entropy(p):
    """Shannon entropy in bits."""
    return float(entropy(_to_prob(p), base=2))

def compute_all_metrics(orig_logits, cf_logits, lambda_e_values=[0.0, 0.05, 0.1, 0.2, 0.5]):
    """Compute all 6 metric variants."""
    p = _softmax(orig_logits)
    q = _softmax(cf_logits)

    js = js_divergence(p, q)
    kl = kl_divergence(p, q)
    h_orig = shannon_entropy(p)
    h_cf = shannon_entropy(q)
    entropy_penalty = max(0.0, h_cf - h_orig)

    result = {
        "js": js,
        "kl": kl,
        "h_orig": h_orig,
        "h_cf": h_cf,
        "entropy_penalty": entropy_penalty,
    }

    for lam in lambda_e_values:
        result[f"ced_lam{lam:.2f}"] = js + lam * entropy_penalty
        result[f"ced_abs_lam{lam:.2f}"] = js + lam * abs(h_cf - h_orig)
        result[f"ced_hcf_lam{lam:.2f}"] = js + lam * h_cf

    return result

def compute_hidden_js(h_orig, h_cf, top_k=4096):
    """JS divergence for hidden states."""
    h1 = h_orig.cpu().float().numpy()
    h2 = h_cf.cpu().float().numpy()
    top_dims = np.union1d(np.argsort(np.abs(h1))[-top_k:], np.argsort(np.abs(h2))[-top_k:])
    return js_divergence(_softmax(h1[top_dims]), _softmax(h2[top_dims]))

def compute_cosine_dist(h_orig, h_cf):
    """Cosine distance between hidden states."""
    h1 = h_orig.cpu().float().numpy()
    h2 = h_cf.cpu().float().numpy()
    return float(cosine_dist(h1, h2))

def safe_auc(labels, scores):
    """Safe AUC calculation."""
    if len(set(labels)) < 2 or len(labels) < 10:
        return float("nan")
    return roc_auc_score(labels, scores)

# ============================================================
# Visual Token Hook
# ============================================================

class VisualTokenHook:
    """Hook for capturing and modifying visual encoder output."""

    def __init__(self):
        self.captured = None
        self._modifications = None
        self._handle = None

    def register(self, model):
        self._handle = model.model.visual.register_forward_hook(self._fn)
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
        """Handle dataclass output with pooler_output field."""
        if self._modifications is None:
            # Capture mode: store pooler_output (merger output that goes to LLM)
            self.captured = output.pooler_output.detach().clone()
            return output
        else:
            # Replace mode: modify pooler_output and return new dataclass
            from copy import copy
            mod = self.captured.clone().to(
                device=output.pooler_output.device,
                dtype=output.pooler_output.dtype
            )
            for idx, val in self._modifications.items():
                if 0 <= idx < mod.shape[0]:
                    mod[idx] = val.to(device=mod.device, dtype=mod.dtype)
            # Return modified dataclass
            out_copy = copy(output)
            out_copy.pooler_output = mod
            return out_copy

# ============================================================
# Data Loading
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

def load_samples(ann_path: str, img_dir: str, n_samples: int = 400,
                 min_bbox_ratio: float = 0.02, seed: int = 42) -> List[Dict]:
    """Construct samples for four task types."""
    print(f"Loading COCO from {ann_path}...")
    random.seed(seed)

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

        # Existence positive
        samples.append({
            "task": "existence", "img_path": path, "img_w": w, "img_h": h,
            "question": f"Is there a {cat} in this image? Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": 1,
        })

        # Existence negative
        absent = all_cat_ids - img_cats.get(img_id, set())
        if absent:
            abs_name = cat_map[random.choice(list(absent))]
            samples.append({
                "task": "existence", "img_path": path, "img_w": w, "img_h": h,
                "question": f"Is there a {abs_name} in this image? Answer yes or no.",
                "object": abs_name, "bbox": bbox, "all_bboxes": all_bboxes,
                "gt": 0,
            })

        # Spatial relation
        tmpl, check_fn = random.choice(SPATIAL_TEMPLATES)
        gt_spatial = 1 if check_fn(bbox, w, h) else 0
        samples.append({
            "task": "spatial", "img_path": path, "img_w": w, "img_h": h,
            "question": tmpl.format(obj=cat) + " Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": gt_spatial,
        })

        # Counting
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

def bbox_to_indices(bbox, img_w, img_h, grid_h, grid_w, debug=False):
    """Map bbox coordinates to visual token indices."""
    x, y, w, h = bbox
    col_start = max(0, int(x / img_w * grid_w))
    col_end = min(grid_w, int(np.ceil((x+w) / img_w * grid_w)))
    row_start = max(0, int(y / img_h * grid_h))
    row_end = min(grid_h, int(np.ceil((y+h) / img_h * grid_h)))
    indices = [r * grid_w + c for r in range(row_start, row_end) for c in range(col_start, col_end)]

    if debug:
        print(f"  [bbox映射] 图像({img_w}x{img_h}px) -> Grid({grid_w}x{grid_h} tokens)")
        print(f"  [bbox映射] bbox({x:.0f},{y:.0f},{w:.0f},{h:.0f}px) -> tokens[{col_start}:{col_end}, {row_start}:{row_end}]")
        print(f"  [bbox映射] 共{len(indices)}个tokens ({len(indices)/(grid_h*grid_w)*100:.1f}%)")

    return indices

def get_control_indices(all_bboxes, img_w, img_h, grid_h, grid_w, n):
    """Get control region indices (unoccupied by any bbox)."""
    occupied = set()
    for b in all_bboxes:
        occupied.update(bbox_to_indices(b, img_w, img_h, grid_h, grid_w))
    free = sorted(set(range(grid_h * grid_w)) - occupied)
    if len(free) >= n:
        start = random.randint(0, len(free) - n)
        return free[start:start+n]
    return free if free else list(range(min(n, grid_h * grid_w)))

# ============================================================
# Model Interaction
# ============================================================

def get_model_answer(model, processor, image, question, device):
    """Get actual answer from model (yes/no)."""
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

    input_len = inputs["input_ids"].shape[1]
    answer_text = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip().lower()

    if "yes" in answer_text:
        return 1, answer_text
    elif "no" in answer_text:
        return 0, answer_text
    else:
        return -1, answer_text

def process_sample(sample, model, processor, hook, device, layers, lambda_e_values):
    """Process a single sample through CED pipeline."""
    from qwen_vl_utils import process_vision_info

    try:
        image = Image.open(sample["img_path"]).convert("RGB")
    except Exception:
        return None

    # Step 1: Get model answer
    pred, pred_text = get_model_answer(model, processor, image, sample["question"], device)
    if pred == -1:
        return None

    # Behavior grouping
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

    # Step 2: Construct inputs for CED (no generation, just logits)
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

    # Step 3: Original forward (capture)
    hook.reset()
    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    if hook.captured is None:
        return None

    # Step 4: Object region replacement
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

    # Step 5: Compute all metrics
    orig_logits = orig_out.logits[0, -1].cpu().float().numpy()
    cf_logits = cf_out.logits[0, -1].cpu().float().numpy()

    metrics = compute_all_metrics(orig_logits, cf_logits, lambda_e_values)

    # Cosine distance
    h_orig_last = orig_out.hidden_states[-1][0, -1]
    h_cf_last = cf_out.hidden_states[-1][0, -1]
    metrics["cosine_dist"] = compute_cosine_dist(h_orig_last, h_cf_last)

    # Hidden layers
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

    # Step 6: Control group (only for correct_positive)
    if behavior == "correct_positive":
        n_ctrl = max(4, len(target_idx))
        ctrl_idx = get_control_indices(sample["all_bboxes"], sample["img_w"], sample["img_h"],
                                       grid_h, grid_w, n_ctrl)
        surr_ctrl = sorted(set(range(vis_feat.shape[0])) - set(ctrl_idx))
        repl_ctrl = vis_feat[surr_ctrl].mean(0) if surr_ctrl else vis_feat.mean(0)

        hook.reset()
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
# Mode: Probe
# ============================================================

def run_probe(args):
    """Architecture probing mode (P0-a equivalent)."""
    print("=" * 60)
    print("P0-a: Qwen3-VL Architecture Probing")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Load model
    print("\n--- Step 1: Loading Model ---")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"\nmodel.model.visual type: {type(model.model.visual).__name__}")
    if hasattr(model.model.visual, 'merger'):
        print(f"model.model.visual.merger type: {type(model.model.visual.merger).__name__}")

    # Architecture parameter detection
    print("\n--- Architecture Parameters ---")
    spatial_merge_size = getattr(model.model.visual, 'spatial_merge_size', None)
    if spatial_merge_size is None and hasattr(model.model.visual, 'config'):
        spatial_merge_size = getattr(model.model.visual.config, 'spatial_merge_size', None)
    if spatial_merge_size is None:
        spatial_merge_size = 2
        print(f"  spatial_merge_size: not detected, using default {spatial_merge_size}")
    else:
        print(f"  spatial_merge_size: {spatial_merge_size} (merges {spatial_merge_size}x{spatial_merge_size} patches per token)")

    patch_size = getattr(model.model.visual, 'patch_size', None)
    if patch_size is None and hasattr(model.model.visual, 'config'):
        patch_size = getattr(model.model.visual.config, 'patch_size', None)
    if patch_size is None:
        patch_size = 14
        print(f"  patch_size: not detected, using default {patch_size}")
    else:
        print(f"  patch_size: {patch_size}")

    vis_params = sum(p.numel() for p in model.model.visual.parameters())
    print(f"\nVisual encoder parameters: {vis_params / 1e6:.1f}M")

    # Step 2: Single image forward pass
    print("\n--- Step 2: Single Image Forward Pass ---")

    with open(COCO_ANN_PATH) as f:
        coco = json.load(f)
    test_img_info = coco["images"][0]
    test_img_path = os.path.join(COCO_IMG_DIR, test_img_info["file_name"])
    img = Image.open(test_img_path).convert("RGB")
    img_w, img_h = img.size
    print(f"Test image: {test_img_info['file_name']}, size: {img_w}x{img_h}")

    # Theoretical token calculation
    theoretical_patches_h = img_h / patch_size
    theoretical_patches_w = img_w / patch_size
    theoretical_tokens_h = theoretical_patches_h / spatial_merge_size
    theoretical_tokens_w = theoretical_patches_w / spatial_merge_size
    theoretical_total = theoretical_tokens_h * theoretical_tokens_w

    print(f"\n  Theory:")
    print(f"    Image size: {img_w} x {img_h}")
    print(f"    After patch: {theoretical_patches_h:.1f} x {theoretical_patches_w:.1f} = {theoretical_patches_h * theoretical_patches_w:.0f} patches")
    print(f"    After merge: {theoretical_tokens_h:.1f} x {theoretical_tokens_w:.1f} = {theoretical_total:.0f} tokens")

    # Find annotation
    test_anns = [a for a in coco["annotations"] if a["image_id"] == test_img_info["id"]]
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    if test_anns:
        test_ann = max(test_anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        obj_name = cat_map[test_ann["category_id"]]
        bbox = test_ann["bbox"]
        print(f"Target object: {obj_name}, bbox: {bbox}")
    else:
        obj_name = "object"
        bbox = [img_w*0.25, img_h*0.25, img_w*0.5, img_h*0.5]
        print(f"No annotation, using center region")

    question = f"Is there a {obj_name} in this image? Answer yes or no."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": question},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Check image_grid_thw
    grid_thw = inputs.get("image_grid_thw", None)
    if grid_thw is not None:
        print(f"\nimage_grid_thw: {grid_thw}")
        grid_t, grid_h, grid_w = grid_thw[0].tolist()
        print(f"  temporal={grid_t}, height={grid_h}, width={grid_w}")
        print(f"  Total visual tokens (t*h*w) = {grid_t * grid_h * grid_w}")
    else:
        print("WARNING: image_grid_thw not found!")
        sys.exit(1)

    # Step 3: Hook test
    print("\n--- Step 3: Hook Mechanism Test ---")

    captured_output = {}

    def capture_hook(module, input, output):
        # output is a dataclass with pooler_output field
        captured_output["shape"] = output.pooler_output.shape
        captured_output["dtype"] = output.pooler_output.dtype
        captured_output["tensor"] = output.pooler_output.detach().clone()
        return output

    handle = model.model.visual.register_forward_hook(capture_hook)

    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    handle.remove()

    if "shape" not in captured_output:
        print("FAIL: Hook did not capture visual encoder output!")
        sys.exit(1)

    vis_shape = captured_output["shape"]
    print(f"Visual encoder output shape: {vis_shape}")
    print(f"Visual encoder output dtype: {captured_output['dtype']}")

    # Check shape vs grid_thw
    expected_tokens = grid_t * grid_h * grid_w
    actual_tokens = vis_shape[0] if len(vis_shape) == 2 else vis_shape[1]
    print(f"\nExpected visual tokens: {expected_tokens} (from grid_thw)")
    print(f"Actual visual tokens: {actual_tokens} (from hook output)")

    if actual_tokens == expected_tokens:
        print("✅ Visual token count matches!")
        merger_ratio = 1
    else:
        ratio = expected_tokens / actual_tokens if actual_tokens > 0 else 0
        print(f"⚠️  Mismatch! Ratio = {ratio:.1f}")
        merger_ratio = ratio

    hidden_dim = vis_shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    # Step 4: Replacement test
    print("\n--- Step 4: Replacement Test ---")

    # Calculate target token indices
    col_start = int(bbox[0] / img_w * grid_w)
    col_end = int(np.ceil((bbox[0] + bbox[2]) / img_w * grid_w))
    row_start = int(bbox[1] / img_h * grid_h)
    row_end = int(np.ceil((bbox[1] + bbox[3]) / img_h * grid_h))
    col_start = max(0, col_start)
    col_end = min(grid_w, col_end)
    row_start = max(0, row_start)
    row_end = min(grid_h, row_end)

    target_indices = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            target_indices.append(r * grid_w + c)

    # Enhanced bbox mapping logs
    print(f"\n--- BBox Mapping Validation ---")
    print(f"  Image size: {img_w} x {img_h} pixels")
    print(f"  Grid size: {grid_w} x {grid_h} tokens")
    print(f"  BBox (xywh): [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    print(f"  BBox center: ({bbox[0] + bbox[2]/2:.1f}, {bbox[1] + bbox[3]/2:.1f})")
    print(f"  Mapped token range: cols [{col_start}, {col_end}), rows [{row_start}, {row_end})")
    token_center_x = (col_start + col_end) / 2
    token_center_y = (row_start + row_end) / 2
    print(f"  Mapped token center: ({token_center_x:.1f}, {token_center_y:.1f})")
    print(f"  Target tokens: {len(target_indices)} / {grid_h * grid_w} total")
    print(f"  Target region ratio: {len(target_indices) / (grid_h * grid_w) * 100:.1f}%")

    # Compute replacement value
    vis_features = captured_output["tensor"]
    all_indices = set(range(vis_features.shape[0]))
    surr_indices = sorted(all_indices - set(target_indices))
    if surr_indices:
        replacement = vis_features[surr_indices].mean(dim=0)
    else:
        replacement = vis_features.mean(dim=0)

    # Modified version
    modified_features = vis_features.clone()
    for idx in target_indices:
        if idx < modified_features.shape[0]:
            modified_features[idx] = replacement

    def replace_hook(module, input, output):
        from copy import copy
        out = copy(output)
        out.pooler_output = modified_features.to(device=output.pooler_output.device, dtype=output.pooler_output.dtype)
        return out

    handle2 = model.model.visual.register_forward_hook(replace_hook)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle2.remove()

    # Compare logits
    orig_logits = orig_out.logits[0, -1].cpu().float()
    cf_logits = cf_out.logits[0, -1].cpu().float()
    logit_diff = (orig_logits - cf_logits).abs().mean().item()
    logit_max_diff = (orig_logits - cf_logits).abs().max().item()

    # JS divergence
    p = torch.softmax(orig_logits, dim=0).numpy()
    q = torch.softmax(cf_logits, dim=0).numpy()
    p = np.clip(p, 1e-12, None); p /= p.sum()
    q = np.clip(q, 1e-12, None); q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    print(f"\nLogits difference after replacement:")
    print(f"  Mean |Δlogits|: {logit_diff:.4f}")
    print(f"  Max  |Δlogits|: {logit_max_diff:.4f}")
    print(f"  JS divergence:  {js:.6f}")

    if js > 1e-6:
        print("✅ Replacing visual tokens affects output! JS > 0")
    else:
        print("❌ FAIL: Output barely changed after replacement")

    # Step 5: Control experiment
    print("\n--- Step 5: Control Comparison ---")

    corner_indices = []
    for r in range(min(2, grid_h)):
        for c in range(min(2, grid_w)):
            corner_indices.append(r * grid_w + c)

    ctrl_surr = sorted(set(range(vis_features.shape[0])) - set(corner_indices))
    ctrl_replacement = vis_features[ctrl_surr].mean(dim=0) if ctrl_surr else vis_features.mean(dim=0)

    ctrl_features = vis_features.clone()
    for idx in corner_indices:
        if idx < ctrl_features.shape[0]:
            ctrl_features[idx] = ctrl_replacement

    def ctrl_hook(module, input, output):
        return ctrl_features.to(device=output.device, dtype=output.dtype)

    handle3 = model.model.visual.register_forward_hook(ctrl_hook)
    with torch.no_grad():
        ctrl_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle3.remove()

    ctrl_logits = ctrl_out.logits[0, -1].cpu().float()
    p_ctrl = torch.softmax(ctrl_logits, dim=0).numpy()
    p_ctrl = np.clip(p_ctrl, 1e-12, None); p_ctrl /= p_ctrl.sum()
    m_ctrl = 0.5 * (p + p_ctrl)
    js_ctrl = 0.5 * entropy(p, m_ctrl, base=2) + 0.5 * entropy(p_ctrl, m_ctrl, base=2)

    print(f"Object region JS = {js:.6f}")
    print(f"Corner region JS = {js_ctrl:.6f}")
    if js > js_ctrl:
        print(f"✅ Object region JS > Corner region JS (ratio: {js/max(js_ctrl, 1e-12):.1f}x)")
    else:
        print(f"⚠️  Corner region JS >= Object region JS, signal may be weak")

    # Step 6: Hidden states check
    print("\n--- Step 6: Hidden States Structure ---")
    n_layers = len(orig_out.hidden_states)
    print(f"Total hidden states layers: {n_layers} (including embedding)")
    print(f"Available intermediate layers: 0 to {n_layers-1}")
    print(f"Planned probes: [16, 20, 24, 28, 32]", end=" ")
    if n_layers > 32:
        print("✅ All within range")
    else:
        safe_layers = [l for l in [16, 20, 24, 28, 32] if l < n_layers]
        print(f"⚠️  Actually available: {safe_layers}")

    # Summary
    print("\n" + "=" * 60)
    print("P0-a Probe Summary")
    print("=" * 60)

    checks = {
        "Hook capture normal": "shape" in captured_output,
        "Token count match": abs(actual_tokens - expected_tokens) < 10,
        "Replacement affects output": js > 1e-6,
        "Object region signal stronger": js > js_ctrl,
    }

    all_pass = True
    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("P0-a PASSED — Can proceed to P0-b")
    else:
        print("P0-a FAILED — Need to adjust hook position or replacement strategy")

    # Save probe info for P0-b
    probe_info = {
        "grid_h": grid_h, "grid_w": grid_w, "grid_t": grid_t,
        "hidden_dim": hidden_dim,
        "n_hidden_layers": n_layers,
        "visual_output_shape": list(vis_shape),
        "js_object_region": float(js),
        "js_control_region": float(js_ctrl),
        "spatial_merge_size": spatial_merge_size,
        "patch_size": patch_size,
        "all_passed": all_pass,
    }
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(f"{RESULT_DIR}/p0a_probe_info.json", "w") as f:
        json.dump(probe_info, f, indent=2)
    print(f"\nProbe info saved to {RESULT_DIR}/p0a_probe_info.json")

    return probe_info

# ============================================================
# Mode: Worker
# ============================================================

def run_worker(args):
    """Worker mode for parallel processing (sharded data)."""
    print("=" * 60)
    print(f"P0-b Worker: Shard {args.shard_idx}/{args.num_shards}")
    print("=" * 60)
    print(f"Pass criterion: correct_positive vs hallucination AUC > 0.85\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print(f"Loading model from {MODEL_PATH}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    hook = VisualTokenHook().register(model)

    # Load probe info
    probe_path = f"{RESULT_DIR}/p0a_probe_info.json"
    probe_info = None
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            probe_info = json.load(f)
        layers = [l for l in args.layers if l < probe_info["n_hidden_layers"]]
        print(f"P0-a probe info loaded:")
        print(f"  grid: {probe_info['grid_h']}x{probe_info['grid_w']} (temporal={probe_info.get('grid_t', 1)})")
        print(f"  spatial_merge_size: {probe_info.get('spatial_merge_size', 2)}")
        print(f"  patch_size: {probe_info.get('patch_size', 14)}")
        print(f"  hidden_dim: {probe_info['hidden_dim']}")
        print(f"  n_layers: {probe_info['n_hidden_layers']}, usable layers: {layers}")
    else:
        layers = args.layers
        print(f"Warning: P0-a probe info not found, using default layers: {layers}")

    # Load all samples and shard
    all_samples = load_samples(COCO_ANN_PATH, COCO_IMG_DIR,
                               n_samples=args.num_samples, seed=args.seed)
    shard_samples = all_samples[args.shard_idx::args.num_shards]
    print(f"\nShard {args.shard_idx}: processing {len(shard_samples)} samples "
          f"(from total {len(all_samples)})")

    results = []
    behavior_counts = defaultdict(int)

    for sample in tqdm(shard_samples, desc=f"Worker-{args.shard_idx}"):
        r = process_sample(sample, model, processor, hook, device,
                          layers, args.lambda_e_values)
        if r:
            results.append(r)
            behavior_counts[r["behavior"]] += 1

        # Real-time progress
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

    # Save shard results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = f"{args.output_dir}/p0b_shard_{args.shard_idx:03d}_of_{args.num_shards:03d}.csv"
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved {len(df)} results to {output_file}")
    print(f"Behavior distribution: {dict(behavior_counts)}")

    hook.remove()
    return df

# ============================================================
# Mode: Analyze
# ============================================================

def run_analyze(args):
    """Merge shards and analyze results."""
    print("=" * 60)
    print("P0-b Analysis: Merge Shards and Evaluate")
    print("=" * 60)

    # Find all shard files
    shard_pattern = f"{args.result_dir}/p0b_shard_*_of_*.csv"
    shard_files = sorted(glob.glob(shard_pattern))

    if not shard_files:
        print(f"Error: No shard files found matching {shard_pattern}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shard files:")
    for f in shard_files:
        print(f"  - {os.path.basename(f)}")

    # Merge all shards
    all_dfs = []
    for f in shard_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
            print(f"  Loaded {len(df)} rows from {os.path.basename(f)}")
        except Exception as e:
            print(f"  Warning: Failed to load {f}: {e}")

    if not all_dfs:
        print("Error: No valid data loaded from shards")
        sys.exit(1)

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal samples: {len(df)}")

    # Save merged results
    merged_file = f"{args.result_dir}/p0b_merged.csv"
    df.to_csv(merged_file, index=False)
    print(f"Merged results saved to {merged_file}")

    # Run analysis
    analyze_results(df, args.lambda_e_values, args.layers)
    plot_results(df, args.lambda_e_values, args.layers, args.result_dir)

def analyze_results(df, lambda_e_values, layers):
    """Analyze merged results."""
    print("\n" + "=" * 60)
    print("P0-b Results Analysis")
    print("=" * 60)

    # 1. Behavior distribution
    print("\n--- Behavior Distribution ---")
    for b in ["correct_positive", "hallucination", "correct_negative", "miss"]:
        sub = df[df["behavior"] == b]
        if len(sub) > 0:
            print(f"  {b:20s}: n={len(sub):4d}, JS={sub['js'].mean():.4f}±{sub['js'].std():.4f}")

    # 2. Core AUC: correct_positive vs hallucination
    print("\n--- Core AUC: correct_positive vs hallucination ---")
    cp = df[df["behavior"] == "correct_positive"]
    hal = df[df["behavior"] == "hallucination"]

    if len(cp) < 5 or len(hal) < 5:
        print(f"WARNING: Insufficient samples (cp={len(cp)}, hal={len(hal)})")
        print("  Too few hallucination samples — model doesn't make many mistakes on this data")
        print("  Suggestion: Increase sample size or use harder question templates")

        # Fallback: gt=1 vs gt=0
        print("\n--- Fallback: gt=1 vs gt=0 ---")
        labels = df["gt"]
        if labels.nunique() > 1:
            for metric in ["js"] + [f"ced_lam{l:.2f}" for l in lambda_e_values]:
                if metric in df.columns:
                    auc = safe_auc(labels, df[metric])
                    print(f"  {metric:25s}: AUC = {auc:.4f}")
        return

    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)])

    print("\n  [Logits Layer]")

    # A: Raw JS
    auc_js = safe_auc(sub["label"], sub["js"])
    print(f"  A. Raw JS divergence:      AUC = {auc_js:.4f}")

    # B: CED variants
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

    # F: Cosine
    if "cosine_dist" in sub.columns:
        auc_cos = safe_auc(sub["label"], sub["cosine_dist"])
        print(f"  F. Cosine distance:        AUC = {auc_cos:.4f}")
        if auc_cos > best_auc:
            best_auc, best_config = auc_cos, "cosine_dist"

    print(f"\n  → Best: {best_config}, AUC = {best_auc:.4f}")

    # 3. Intermediate layers
    print("\n  [Intermediate Layers]")
    for layer in layers:
        col = f"layer_{layer}_js"
        if col in sub.columns and sub[col].notna().sum() > 10:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  Layer {layer}: AUC = {auc:.4f}")

    # 4. Cross-task consistency
    print("\n--- Cross-task Consistency ---")
    for task in ["existence", "spatial", "counting"]:
        task_sub = df[df["task"] == task]
        cp_t = task_sub[task_sub["behavior"] == "correct_positive"]
        hal_t = task_sub[task_sub["behavior"] == "hallucination"]
        if len(cp_t) >= 3 and len(hal_t) >= 3:
            s = pd.concat([cp_t.assign(label=1), hal_t.assign(label=0)])
            auc = safe_auc(s["label"], s["js"])
            print(f"  {task:12s}: AUC = {auc:.4f} (cp={len(cp_t)}, hal={len(hal_t)})")
        else:
            print(f"  {task:12s}: insufficient samples (cp={len(cp_t)}, hal={len(hal_t)})")

    # 5. Control group comparison
    print("\n--- Control Group (correct_positive only) ---")
    ctrl_rows = cp[cp["ctrl_js"].notna()] if "ctrl_js" in cp.columns else pd.DataFrame()
    if len(ctrl_rows) > 0:
        print(f"  Object region JS:     {ctrl_rows['js'].mean():.4f} ± {ctrl_rows['js'].std():.4f}")
        print(f"  Non-object region JS: {ctrl_rows['ctrl_js'].mean():.4f} ± {ctrl_rows['ctrl_js'].std():.4f}")
        ratio = ctrl_rows['js'].mean() / max(ctrl_rows['ctrl_js'].mean(), 1e-8)
        print(f"  Ratio: {ratio:.1f}x")

    # 6. Entropy analysis
    print("\n--- Entropy Analysis ---")
    for b in ["correct_positive", "hallucination", "correct_negative"]:
        s = df[df["behavior"] == b]
        if len(s) > 0:
            print(f"  {b:20s}: H(P)={s['h_orig'].mean():.3f}, "
                  f"H(P^cf)={s['h_cf'].mean():.3f}, "
                  f"Δ_+={s['entropy_penalty'].mean():.4f}")

    # 7. Pass/Fail
    print("\n" + "=" * 60)
    if best_auc > 0.85:
        print(f"✅ P0-b PASSED: AUC = {best_auc:.4f} > 0.85 ({best_config})")
        print(f"   → Proceed to Phase 1 GRPO!")
    elif best_auc > 0.75:
        print(f"⚠️  P0-b MARGINAL: AUC = {best_auc:.4f}")
        print(f"   Suggestion: Adjust token replacement strategy or increase samples")
    else:
        print(f"❌ P0-b FAILED: AUC = {best_auc:.4f}")
        print(f"   Suggestion: Switch to VCD noise/MaskCD/PROJECTAWAY")
    print("=" * 60)

def plot_results(df, lambda_e_values, layers, result_dir):
    """Generate visualization plots."""
    os.makedirs(result_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    cp = df[df["behavior"] == "correct_positive"]
    hal = df[df["behavior"] == "hallucination"]
    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)]) if len(cp) > 0 and len(hal) > 0 else pd.DataFrame()

    # 1. JS distribution by behavior
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

    # 2. Formula ablation AUC
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

    # 3. AUC by layer
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

    # 4. ROC curve
    ax = axes[1, 0]
    if len(sub) > 0 and sub["label"].nunique() > 1:
        fpr, tpr, _ = roc_curve(sub["label"], sub["js"])
        ax.plot(fpr, tpr, label=f"JS (AUC={safe_auc(sub['label'], sub['js']):.3f})", lw=2)
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve"); ax.legend()

    # 5. Control group comparison
    ax = axes[1, 1]
    if "ctrl_js" in cp.columns and cp["ctrl_js"].notna().sum() > 5:
        ax.boxplot([cp["js"].dropna(), cp["ctrl_js"].dropna()],
                   labels=["Object Region", "Control Region"])
        ax.set_ylabel("JS Divergence")
        ax.set_title("Object vs Control Region (correct_positive only)")

    # 6. Cross-task consistency
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
    plot_file = f"{result_dir}/p0b_analysis.png"
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"\nCharts saved to {plot_file}")

# ============================================================
# Main Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="P0 Experiment: Unified Qwen3-VL CED validation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Probe mode (architecture detection)
  python p0_experiment.py --mode probe

  # Worker mode (parallel processing)
  python p0_experiment.py --mode worker --shard_idx 0 --num_shards 8

  # Analyze mode (merge and evaluate)
  python p0_experiment.py --mode analyze --result_dir /path/to/results
        """
    )

    parser.add_argument("--mode", type=str, required=True,
                        choices=["probe", "worker", "analyze"],
                        help="Execution mode: probe, worker, or analyze")

    # Common arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=[16, 20, 24, 28, 32],
                        help="Layers to probe (default: 16 20 24 28 32)")
    parser.add_argument("--lambda_e_values", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2, 0.5],
                        help="Lambda values for CED (default: 0.0 0.05 0.1 0.2 0.5)")

    # Worker mode arguments
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="Worker shard index (0-based)")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards")
    parser.add_argument("--num_samples", type=int, default=400,
                        help="Number of samples to process (default: 400)")
    parser.add_argument("--output_dir", type=str, default=RESULT_DIR,
                        help=f"Output directory for worker results (default: {RESULT_DIR})")

    # Analyze mode arguments
    parser.add_argument("--result_dir", type=str, default=RESULT_DIR,
                        help=f"Directory containing shard results (default: {RESULT_DIR})")

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dispatch to mode
    if args.mode == "probe":
        run_probe(args)
    elif args.mode == "worker":
        if args.num_shards < 1:
            print("Error: num_shards must be >= 1")
            sys.exit(1)
        if args.shard_idx < 0 or args.shard_idx >= args.num_shards:
            print(f"Error: shard_idx must be in [0, {args.num_shards-1}]")
            sys.exit(1)
        run_worker(args)
    elif args.mode == "analyze":
        run_analyze(args)
    else:
        print(f"Error: Unknown mode '{args.mode}'")
        sys.exit(1)

if __name__ == "__main__":
    main()
