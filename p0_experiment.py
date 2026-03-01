#!/usr/bin/env python3
"""
P0 Experiment: Unified script for Qwen3-VL CED validation

CED formula (per-sample, summed over T_key tokens):
  CED_i = sum_{t in T_key} [ D_JS(P(·|V) || P(·|V^cf)) + λ_e [H(P^cf) - H(P)]_+ ]

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
import hashlib
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
# CED Constants
# ============================================================
T_KEY_SIZE = 4          # Number of trailing tokens for CED summation
DEFAULT_LAMBDA_E = 0.1  # Default entropy penalty weight

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
    """Jensen-Shannon divergence (bits)."""
    p, q = _to_prob(p), _to_prob(q)
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def kl_divergence(p, q):
    """KL divergence from p to q (bits)."""
    p, q = _to_prob(p), _to_prob(q)
    return float(entropy(p, q, base=2))


def shannon_entropy(p):
    """Shannon entropy in bits."""
    return float(entropy(_to_prob(p), base=2))


def get_output_device_dtype(output):
    """Safely extract device and dtype from model output dataclass."""
    for attr_name in ['pooler_output', 'last_hidden_state', 'hidden_states']:
        if hasattr(output, attr_name):
            val = getattr(output, attr_name)
            if val is not None:
                if attr_name == 'hidden_states' and isinstance(val, (tuple, list)):
                    if len(val) > 0 and torch.is_tensor(val[-1]):
                        return val[-1].device, val[-1].dtype
                elif torch.is_tensor(val):
                    return val.device, val.dtype

    for attr_name in dir(output):
        if not attr_name.startswith('_'):
            try:
                val = getattr(output, attr_name)
                if torch.is_tensor(val):
                    return val.device, val.dtype
            except Exception:
                continue

    raise ValueError(f"Cannot find tensor in output of type {type(output)}")


def replace_visual_output(output, field_name, new_tensor):
    """Replace a field in a ModelOutput (OrderedDict subclass) correctly.

    ModelOutput is both a dataclass and OrderedDict.
    We must update BOTH the dict key and the attribute to ensure
    downstream code sees the replacement regardless of access method.
    """
    target_device, target_dtype = get_output_device_dtype(output)
    val = new_tensor.to(device=target_device, dtype=target_dtype)

    # Reconstruct the output with the replaced field
    # This is the most reliable way for ModelOutput subclasses
    new_kwargs = {}
    for key in output.keys():
        if key == field_name:
            new_kwargs[key] = val
        else:
            new_kwargs[key] = output[key]
    return output.__class__(**new_kwargs)


def find_main_visual_field(output):
    """Find the main tensor field in visual encoder output.

    Returns (field_name, tensor) for the primary visual feature tensor.
    """
    # Try common field names in order of preference
    for name in ['pooler_output', 'last_hidden_state']:
        if hasattr(output, name):
            val = getattr(output, name)
            if val is not None and torch.is_tensor(val):
                return name, val

    # Fallback: scan all fields for tensors
    if hasattr(output, 'keys'):
        for key in output.keys():
            val = output[key]
            if torch.is_tensor(val):
                return key, val

    return None, None


def debug_visual_output(output):
    """Print all fields and shapes of a visual encoder output for debugging."""
    print(f"  Output type: {type(output).__name__}")
    if hasattr(output, 'keys'):
        for key in output.keys():
            val = output[key]
            if torch.is_tensor(val):
                print(f"  Field '{key}': tensor {val.shape} {val.dtype}")
            elif val is None:
                print(f"  Field '{key}': None")
            elif isinstance(val, (tuple, list)):
                print(f"  Field '{key}': {type(val).__name__} len={len(val)}")
            else:
                print(f"  Field '{key}': {type(val).__name__}")
    else:
        for attr in ['pooler_output', 'last_hidden_state', 'hidden_states']:
            if hasattr(output, attr):
                val = getattr(output, attr)
                if torch.is_tensor(val):
                    print(f"  Attr '{attr}': tensor {val.shape} {val.dtype}")
                else:
                    print(f"  Attr '{attr}': {type(val).__name__ if val is not None else 'None'}")


def compute_tkey_ced(orig_logits_tkey, cf_logits_tkey, lambda_e_values):
    """Compute CED metrics over T_key tokens (per-token JS + entropy penalty, then sum).

    Args:
        orig_logits_tkey: np.ndarray of shape [T_KEY_SIZE, vocab_size] — original logits.
        cf_logits_tkey: np.ndarray of shape [T_KEY_SIZE, vocab_size] — counterfactual logits.
        lambda_e_values: list of lambda_e values to sweep.

    Returns:
        dict with ced, js_sum, entropy_penalty_sum, avg_ced_per_token, per-lambda variants, etc.
    """
    n_tokens = orig_logits_tkey.shape[0]
    js_per_token = []
    h_orig_per_token = []
    h_cf_per_token = []
    entropy_penalty_per_token = []

    for t in range(n_tokens):
        p_t = _softmax(orig_logits_tkey[t])
        q_t = _softmax(cf_logits_tkey[t])
        js_t = js_divergence(p_t, q_t)
        h_orig_t = shannon_entropy(p_t)
        h_cf_t = shannon_entropy(q_t)
        ep_t = max(0.0, h_cf_t - h_orig_t)

        js_per_token.append(js_t)
        h_orig_per_token.append(h_orig_t)
        h_cf_per_token.append(h_cf_t)
        entropy_penalty_per_token.append(ep_t)

    js_sum = sum(js_per_token)
    entropy_penalty_sum = sum(entropy_penalty_per_token)
    h_orig_mean = np.mean(h_orig_per_token)
    h_cf_mean = np.mean(h_cf_per_token)

    result = {
        "js_sum": js_sum,
        "entropy_penalty_sum": entropy_penalty_sum,
        "h_orig": h_orig_mean,
        "h_cf": h_cf_mean,
    }

    # CED for each lambda: sum over tokens of (js_t + lambda * ep_t)
    for lam in lambda_e_values:
        ced_val = sum(js_per_token[t] + lam * entropy_penalty_per_token[t] for t in range(n_tokens))
        result[f"ced_lam{lam:.2f}"] = ced_val
        # Absolute entropy difference variant
        ced_abs_val = sum(
            js_per_token[t] + lam * abs(h_cf_per_token[t] - h_orig_per_token[t])
            for t in range(n_tokens)
        )
        result[f"ced_abs_lam{lam:.2f}"] = ced_abs_val
        # H(P^cf) weighting variant
        ced_hcf_val = sum(
            js_per_token[t] + lam * h_cf_per_token[t]
            for t in range(n_tokens)
        )
        result[f"ced_hcf_lam{lam:.2f}"] = ced_hcf_val

    # Primary CED with default lambda
    ced_primary = sum(
        js_per_token[t] + DEFAULT_LAMBDA_E * entropy_penalty_per_token[t]
        for t in range(n_tokens)
    )
    result["ced"] = ced_primary
    result["avg_ced_per_token"] = ced_primary / n_tokens

    # Also keep KL sum for comparison
    kl_sum = 0.0
    for t in range(n_tokens):
        p_t = _softmax(orig_logits_tkey[t])
        q_t = _softmax(cf_logits_tkey[t])
        kl_sum += kl_divergence(p_t, q_t)
    result["kl_sum"] = kl_sum

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
    """Hook for capturing and modifying visual encoder output.

    IMPORTANT: Must be registered on model.model.visual (Qwen3-VL architecture).
    Uses find_main_visual_field() to detect the correct tensor field, and
    replace_visual_output() to properly reconstruct the ModelOutput.
    """

    def __init__(self):
        self.captured = None
        self._field_name = None  # which field contains visual features
        self._modifications = None
        self._handle = None

    def register(self, model):
        """Register hook on model.model.visual."""
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
        """Capture or replace visual features in ModelOutput."""
        field_name, tensor = find_main_visual_field(output)

        if self._modifications is None:
            # Capture mode
            if tensor is not None:
                self.captured = tensor.detach().clone()
                self._field_name = field_name
            return output
        else:
            # Replace mode
            if self.captured is None or self._field_name is None:
                return output
            target_device, target_dtype = get_output_device_dtype(output)
            mod = self.captured.clone().to(device=target_device, dtype=target_dtype)
            for idx, val in self._modifications.items():
                if 0 <= idx < mod.shape[0]:
                    mod[idx] = val.to(device=mod.device, dtype=mod.dtype)
            return replace_visual_output(output, self._field_name, mod)


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


def bbox_to_indices(bbox, img_w, img_h, grid_h, grid_w, spatial_merge_size=2, debug=False):
    """Map bbox coordinates to visual token indices (using merged grid)."""
    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size

    x, y, w, h = bbox
    col_start = max(0, int(x / img_w * merged_w))
    col_end = min(merged_w, int(np.ceil((x+w) / img_w * merged_w)))
    row_start = max(0, int(y / img_h * merged_h))
    row_end = min(merged_h, int(np.ceil((y+h) / img_h * merged_h)))
    indices = [r * merged_w + c for r in range(row_start, row_end) for c in range(col_start, col_end)]

    if debug:
        print(f"  [bbox] img({img_w}x{img_h}px) -> Grid({grid_w}x{grid_h} / merge={spatial_merge_size} -> {merged_w}x{merged_h})")
        print(f"  [bbox] bbox({x:.0f},{y:.0f},{w:.0f},{h:.0f}px) -> tokens[{col_start}:{col_end}, {row_start}:{row_end}]")
        print(f"  [bbox] {len(indices)} tokens ({len(indices)/(merged_h*merged_w)*100:.1f}%)")

    return indices


def get_control_indices(all_bboxes, img_w, img_h, grid_h, grid_w, n, spatial_merge_size=2):
    """Get control region indices (unoccupied by any bbox)."""
    merged_grid_h = grid_h // spatial_merge_size
    merged_grid_w = grid_w // spatial_merge_size
    occupied = set()
    for b in all_bboxes:
        occupied.update(bbox_to_indices(b, img_w, img_h, grid_h, grid_w, spatial_merge_size))
    free = sorted(set(range(merged_grid_h * merged_grid_w)) - occupied)
    if len(free) >= n:
        start = random.randint(0, len(free) - n)
        return free[start:start+n]
    return free if free else list(range(min(n, merged_grid_h * merged_grid_w)))


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


# ============================================================
# Counterfactual replacement helpers
# ============================================================

def _stable_u32(s: str) -> int:
    """Stable 32-bit hash (NOT Python's randomized hash)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def build_replacement_map(
    vis_feat: torch.Tensor,
    target_idx: List[int],
    mode: str = "moment_noise",
    noise_scale: float = 1.0,
    seed: int | None = None,
) -> Dict[int, torch.Tensor]:
    """
    Build a dict {token_index -> replacement_vector} for counterfactual visual-token replacement.

    mode:
      - "mean": use a single mean vector for all target tokens (baseline; tends to go OOD).
      - "moment_noise": match first+second moments of surrounding tokens (mu + sigma * eps).
                        Keeps replacement closer to the local feature distribution.

    Notes:
      - Stats are computed from surrounding tokens (non-target). If target covers all tokens,
        fall back to global stats.
      - Sampling is done in fp32 for stability, then cast back to vis_feat.dtype.
    """
    if not target_idx:
        return {}

    n_tokens = vis_feat.shape[0]
    all_idx = set(range(n_tokens))
    surr_idx = sorted(all_idx - set(target_idx))
    src = vis_feat[surr_idx] if surr_idx else vis_feat

    src_f = src.float()
    mu = src_f.mean(dim=0)

    if mode == "mean":
        rep = mu.to(dtype=vis_feat.dtype, device=vis_feat.device)
        return {i: rep for i in target_idx}

    if mode != "moment_noise":
        raise ValueError(f"Unknown replacement mode: {mode}")

    sigma = src_f.std(dim=0, unbiased=False).clamp_min(1e-6)

    gen = None
    if seed is not None:
        gen = torch.Generator(device=vis_feat.device)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)

    eps = torch.randn((len(target_idx), mu.shape[0]), generator=gen, device=vis_feat.device, dtype=torch.float32)
    reps = (mu.unsqueeze(0) + noise_scale * eps * sigma.unsqueeze(0)).to(dtype=vis_feat.dtype)
    return {i: reps[k] for k, i in enumerate(target_idx)}



def process_sample(sample, model, processor, hook, device, layers,
                   lambda_e_values, spatial_merge_size=2, t_key_size=T_KEY_SIZE,
                   cf_mode: str = 'moment_noise', noise_scale: float = 1.0, seed_base: int = 0):
    """Process a single sample through CED pipeline.

    Core logic:
      1. Get model answer (generate)
      2. Original forward → capture visual features + logits
      3. Replace object-region visual tokens → counterfactual forward
      4. For last t_key_size tokens: per-token JS + entropy penalty, then SUM
      5. Optionally compute control group (for correct_positive)
    """
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

    # Step 2: Construct inputs for CED (logits, no generation)
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

    # Step 3: Original forward (capture visual features)
    hook.reset()
    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    if hook.captured is None:
        return None

    # Step 4: Object region replacement
    target_idx = bbox_to_indices(sample["bbox"], sample["img_w"], sample["img_h"],
                                 grid_h, grid_w, spatial_merge_size)
    if not target_idx:
        return None

    vis_feat = hook.captured

    # Counterfactual replacement in visual feature space (reproducible seed per sample)
    s_seed = (seed_base ^ _stable_u32(sample["img_path"] + "|" + sample["question"])) & 0x7FFFFFFF
    mods = build_replacement_map(
        vis_feat,
        target_idx,
        mode=cf_mode,
        noise_scale=noise_scale,
        seed=s_seed,
    )
    hook.set_replace(mods)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)
    hook.reset()

    # Step 5: Compute T_key CED (last t_key_size tokens, per-token then sum)
    seq_len = orig_out.logits.shape[1]
    actual_tkey = min(t_key_size, seq_len)

    # Extract logits for last actual_tkey tokens: shape [actual_tkey, vocab_size]
    orig_logits_tkey = orig_out.logits[0, -actual_tkey:].cpu().float().numpy()
    cf_logits_tkey = cf_out.logits[0, -actual_tkey:].cpu().float().numpy()

    metrics = compute_tkey_ced(orig_logits_tkey, cf_logits_tkey, lambda_e_values)

    # Cosine distance (last hidden state, last token)
    h_orig_last = orig_out.hidden_states[-1][0, -1]
    h_cf_last = cf_out.hidden_states[-1][0, -1]
    metrics["cosine_dist"] = compute_cosine_dist(h_orig_last, h_cf_last)

    # Hidden layer JS (last token)
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
        "t_key_size": actual_tkey,
        **metrics,
    }

    # Step 6: Control group (compute for ALL samples; needed for delta/ratio readouts)
    n_ctrl = max(4, len(target_idx))
    ctrl_idx = get_control_indices(sample["all_bboxes"], sample["img_w"], sample["img_h"],
                                   grid_h, grid_w, n_ctrl, spatial_merge_size)
    # Control replacement uses the SAME counterfactual mode
    c_seed = (seed_base ^ _stable_u32(sample["img_path"] + "|CTRL|" + sample["question"])) & 0x7FFFFFFF
    ctrl_mods = build_replacement_map(
        vis_feat,
        ctrl_idx,
        mode=cf_mode,
        noise_scale=noise_scale,
        seed=c_seed,
    )

    # Need a fresh capture for control forward
    hook.reset()
    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False, return_dict=True)

    hook.set_replace(ctrl_mods)
    with torch.no_grad():
        ctrl_out = model(**inputs, output_hidden_states=False, return_dict=True)
    hook.reset()

    # Control T_key CED
    ctrl_logits_tkey = ctrl_out.logits[0, -actual_tkey:].cpu().float().numpy()
    ctrl_metrics = compute_tkey_ced(orig_logits_tkey, ctrl_logits_tkey, lambda_e_values)
    result["ctrl_ced"] = ctrl_metrics["ced"]
    result["ctrl_js_sum"] = ctrl_metrics["js_sum"]
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

    # Hook is on model.model.visual
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

    print(f"\nCED config: T_KEY_SIZE={T_KEY_SIZE}, DEFAULT_LAMBDA_E={DEFAULT_LAMBDA_E}")

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

    # Step 3: Hook test — registered on model.model.visual
    print("\n--- Step 3: Hook Mechanism Test ---")

    captured_output = {}

    def capture_hook(module, input, output):
        print("\n  --- Visual Encoder Output Structure ---")
        debug_visual_output(output)
        field_name, tensor = find_main_visual_field(output)
        if tensor is not None:
            captured_output["field_name"] = field_name
            captured_output["shape"] = tensor.shape
            captured_output["dtype"] = tensor.dtype
            captured_output["tensor"] = tensor.detach().clone()
            print(f"  -> Using field: '{field_name}', shape: {tensor.shape}")
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

    merger_ratio = expected_tokens / actual_tokens if actual_tokens > 0 else 1
    if actual_tokens == expected_tokens:
        print("OK: Visual token count matches (no merge)")
    elif abs(merger_ratio - spatial_merge_size ** 2) < 0.5:
        print(f"OK: pooler_output is post-merge ({actual_tokens} = {expected_tokens}/{spatial_merge_size}^2)")
        print(f"  This is expected: spatial_merge_size={spatial_merge_size}, ratio={merger_ratio:.0f}")
    else:
        print(f"WARNING: Unexpected token ratio = {merger_ratio:.1f}")

    hidden_dim = vis_shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    # Step 4: Replacement test
    print("\n--- Step 4: Replacement Test ---")

    spatial_merge_size = int(np.sqrt(merger_ratio)) if merger_ratio > 1 else 2
    merged_grid_h = grid_h // spatial_merge_size
    merged_grid_w = grid_w // spatial_merge_size

    col_start = int(bbox[0] / img_w * merged_grid_w)
    col_end = int(np.ceil((bbox[0] + bbox[2]) / img_w * merged_grid_w))
    row_start = int(bbox[1] / img_h * merged_grid_h)
    row_end = int(np.ceil((bbox[1] + bbox[3]) / img_h * merged_grid_h))
    col_start = max(0, col_start)
    col_end = min(merged_grid_w, col_end)
    row_start = max(0, row_start)
    row_end = min(merged_grid_h, row_end)

    target_indices = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            target_indices.append(r * merged_grid_w + c)

    print(f"\n--- BBox Mapping Validation ---")
    print(f"  Image size: {img_w} x {img_h} pixels")
    print(f"  Original grid: {grid_w} x {grid_h} tokens")
    print(f"  Spatial merge size: {spatial_merge_size}")
    print(f"  Merged grid: {merged_grid_w} x {merged_grid_h} tokens (actual visual tokens)")
    print(f"  BBox (xywh): [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
    print(f"  Mapped token range: cols [{col_start}, {col_end}), rows [{row_start}, {row_end})")
    print(f"  Target tokens: {len(target_indices)} / {merged_grid_h * merged_grid_w} total")
    print(f"  Target region ratio: {len(target_indices) / (merged_grid_h * merged_grid_w) * 100:.1f}%")

    vis_features = captured_output["tensor"]

    # Use the same counterfactual replacement strategy as P0-b
    p_seed = (args.seed ^ _stable_u32("PROBE|" + str(img_path))) & 0x7FFFFFFF
    mods = build_replacement_map(
        vis_features,
        target_indices,
        mode=args.cf_mode,
        noise_scale=args.noise_scale,
        seed=p_seed,
    )
    modified_features = vis_features.clone()
    for idx, rep in mods.items():
        if idx < modified_features.shape[0]:
            modified_features[idx] = rep

    vis_field_name = captured_output["field_name"]

    def replace_hook(module, input, output):
        return replace_visual_output(output, vis_field_name, modified_features)

    handle2 = model.model.visual.register_forward_hook(replace_hook)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle2.remove()

    # T_key logits comparison
    seq_len = orig_out.logits.shape[1]
    actual_tkey = min(T_KEY_SIZE, seq_len)
    orig_logits_tkey = orig_out.logits[0, -actual_tkey:].cpu().float().numpy()
    cf_logits_tkey = cf_out.logits[0, -actual_tkey:].cpu().float().numpy()

    tkey_metrics = compute_tkey_ced(orig_logits_tkey, cf_logits_tkey, [DEFAULT_LAMBDA_E])

    # Also compute single last-token JS for comparison
    p_last = _softmax(orig_logits_tkey[-1])
    q_last = _softmax(cf_logits_tkey[-1])
    js_last = js_divergence(p_last, q_last)

    print(f"\nT_key CED (last {actual_tkey} tokens, lambda={DEFAULT_LAMBDA_E}):")
    print(f"  ced (sum):         {tkey_metrics['ced']:.6f}")
    print(f"  js_sum:            {tkey_metrics['js_sum']:.6f}")
    print(f"  entropy_penalty:   {tkey_metrics['entropy_penalty_sum']:.6f}")
    print(f"  avg_ced_per_token: {tkey_metrics['avg_ced_per_token']:.6f}")
    print(f"  (last-token-only JS for ref: {js_last:.6f})")

    if tkey_metrics['js_sum'] > 1e-6:
        print("OK: Replacing visual tokens affects output! JS_sum > 0")
    else:
        print("FAIL: Output barely changed after replacement")

    # Step 5: Control experiment
    print("\n--- Step 5: Control Comparison ---")

    corner_indices = []
    for r in range(min(2, merged_grid_h)):
        for c in range(min(2, merged_grid_w)):
            corner_indices.append(r * merged_grid_w + c)

    ctrl_surr = sorted(set(range(vis_features.shape[0])) - set(corner_indices))
    ctrl_replacement = vis_features[ctrl_surr].mean(dim=0) if ctrl_surr else vis_features.mean(dim=0)

    ctrl_features = vis_features.clone()
    for idx in corner_indices:
        if idx < ctrl_features.shape[0]:
            ctrl_features[idx] = ctrl_replacement

    def ctrl_hook(module, input, output):
        return replace_visual_output(output, vis_field_name, ctrl_features)

    handle3 = model.model.visual.register_forward_hook(ctrl_hook)
    with torch.no_grad():
        ctrl_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle3.remove()

    ctrl_logits_tkey = ctrl_out.logits[0, -actual_tkey:].cpu().float().numpy()
    ctrl_metrics = compute_tkey_ced(orig_logits_tkey, ctrl_logits_tkey, [DEFAULT_LAMBDA_E])

    print(f"Object region CED = {tkey_metrics['ced']:.6f} (js_sum={tkey_metrics['js_sum']:.6f})")
    print(f"Corner region CED = {ctrl_metrics['ced']:.6f} (js_sum={ctrl_metrics['js_sum']:.6f})")
    if tkey_metrics['ced'] > ctrl_metrics['ced']:
        print(f"OK: Object region CED > Corner region CED (ratio: {tkey_metrics['ced']/max(ctrl_metrics['ced'], 1e-12):.1f}x)")
    else:
        print(f"Warning: Corner region CED >= Object region CED, signal may be weak")

    # Step 6: Hidden states check
    print("\n--- Step 6: Hidden States Structure ---")
    n_layers = len(orig_out.hidden_states)
    print(f"Total hidden states layers: {n_layers} (including embedding)")
    print(f"Available intermediate layers: 0 to {n_layers-1}")
    print(f"Planned probes: [16, 20, 24, 28, 32]", end=" ")
    if n_layers > 32:
        print("OK: All within range")
    else:
        safe_layers = [l for l in [16, 20, 24, 28, 32] if l < n_layers]
        print(f"Actually available: {safe_layers}")

    # Summary
    print("\n" + "=" * 60)
    print("P0-a Probe Summary")
    print("=" * 60)

    # Token count: either exact match or expected merge ratio
    token_match = (actual_tokens == expected_tokens) or \
                  (abs(merger_ratio - spatial_merge_size ** 2) < 0.5)

    checks = {
        "Hook capture normal": "shape" in captured_output,
        "Token count valid (exact or post-merge)": token_match,
        "Replacement affects output": tkey_metrics['js_sum'] > 1e-6,
        "Object region signal stronger": tkey_metrics['ced'] > ctrl_metrics['ced'],
    }

    all_pass = True
    for name, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
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
        "js_sum_object_region": float(tkey_metrics['js_sum']),
        "js_sum_control_region": float(ctrl_metrics['js_sum']),
        "ced_object_region": float(tkey_metrics['ced']),
        "ced_control_region": float(ctrl_metrics['ced']),
        "spatial_merge_size": spatial_merge_size,
        "patch_size": patch_size,
        "t_key_size": T_KEY_SIZE,
        "default_lambda_e": DEFAULT_LAMBDA_E,
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
    print(f"CED config: T_KEY_SIZE={T_KEY_SIZE}, DEFAULT_LAMBDA_E={DEFAULT_LAMBDA_E}")
    print(f"CF replacement: mode={args.cf_mode}, noise_scale={args.noise_scale}")
    print(f"Pass criterion: correct_positive vs hallucination AUC(ced) > 0.85\n")

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
    spatial_merge_size = 2  # default
    if os.path.exists(probe_path):
        with open(probe_path) as f:
            probe_info = json.load(f)
        layers = [l for l in args.layers if l < probe_info["n_hidden_layers"]]
        spatial_merge_size = probe_info.get("spatial_merge_size", 2)
        print(f"P0-a probe info loaded:")
        print(f"  grid: {probe_info['grid_h']}x{probe_info['grid_w']} (temporal={probe_info.get('grid_t', 1)})")
        print(f"  spatial_merge_size: {spatial_merge_size}")
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
                          layers, args.lambda_e_values, spatial_merge_size,
                          cf_mode=args.cf_mode, noise_scale=args.noise_scale, seed_base=args.seed)
        if r:
            results.append(r)
            behavior_counts[r["behavior"]] += 1

        # Real-time progress — use "ced" as primary metric
        if len(results) % 50 == 0 and len(results) > 0:
            df_t = pd.DataFrame(results)
            cp = df_t[df_t["behavior"] == "correct_positive"]
            hal = df_t[df_t["behavior"] == "hallucination"]
            if len(cp) > 5 and len(hal) > 5:
                sub = pd.concat([cp, hal])
                labels = (sub["behavior"] == "correct_positive").astype(int)
                auc_ced = roc_auc_score(labels, sub["ced"])
                auc_js = roc_auc_score(labels, sub["js_sum"])
                tqdm.write(f"  n={len(results)}, behaviors={dict(behavior_counts)}, "
                          f"cp_vs_hal AUC(ced)={auc_ced:.3f} AUC(js_sum)={auc_js:.3f}")

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
    """Analyze merged results — AUC primarily on 'ced' (label vs ced)."""
    print("\n" + "=" * 60)
    print("P0-b Results Analysis")
    print("=" * 60)

    # 1. Behavior distribution
    print("\n--- Behavior Distribution ---")
    for b in ["correct_positive", "hallucination", "correct_negative", "miss"]:
        sub = df[df["behavior"] == b]
        if len(sub) > 0:
            ced_col = "ced" if "ced" in sub.columns else "js_sum"
            print(f"  {b:20s}: n={len(sub):4d}, CED={sub[ced_col].mean():.4f}+-{sub[ced_col].std():.4f}, "
                  f"JS_sum={sub['js_sum'].mean():.4f}+-{sub['js_sum'].std():.4f}")

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
            for metric in ["ced", "js_sum"] + [f"ced_lam{l:.2f}" for l in lambda_e_values]:
                if metric in df.columns:
                    auc = safe_auc(labels, df[metric])
                    print(f"  {metric:25s}: AUC = {auc:.4f}")
        return

    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)])

    print(f"\n  Samples: cp={len(cp)}, hal={len(hal)}, total={len(sub)}")
    print(f"\n  [Primary: CED (sum over T_key={T_KEY_SIZE} tokens)]")

    # A: Primary CED (sum, default lambda)
    auc_ced = safe_auc(sub["label"], sub["ced"])
    print(f"  A. CED (lambda={DEFAULT_LAMBDA_E}):    AUC = {auc_ced:.4f}  <-- PRIMARY")

    # B: Raw JS sum
    auc_js = safe_auc(sub["label"], sub["js_sum"])
    print(f"  B. JS_sum (no penalty):     AUC = {auc_js:.4f}")

    # C: CED lambda sweep
    best_auc, best_config = auc_ced, f"ced (lambda={DEFAULT_LAMBDA_E})"
    print(f"\n  [Lambda Sweep]")
    for lam in lambda_e_values:
        col = f"ced_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            tag = " *" if auc > best_auc else ""
            print(f"  CED(lambda={lam:.2f}):           AUC = {auc:.4f}{tag}")
            if auc > best_auc:
                best_auc, best_config = auc, col

    # D: avg_ced_per_token
    if "avg_ced_per_token" in sub.columns:
        auc_avg = safe_auc(sub["label"], sub["avg_ced_per_token"])
        print(f"\n  avg_ced_per_token:           AUC = {auc_avg:.4f}")

    # E: ced_hcf variant
    for lam in [0.1]:
        col = f"ced_hcf_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  JS+lambda*H(P^cf) ({lam:.2f}):   AUC = {auc:.4f}")
            if auc > best_auc:
                best_auc, best_config = auc, col

    # F: ced_abs variant
    for lam in [0.1]:
        col = f"ced_abs_lam{lam:.2f}"
        if col in sub.columns:
            auc = safe_auc(sub["label"], sub[col])
            print(f"  JS+lambda*|dH| ({lam:.2f}):      AUC = {auc:.4f}")
            if auc > best_auc:
                best_auc, best_config = auc, col

    # G: KL sum
    if "kl_sum" in sub.columns:
        auc_kl = safe_auc(sub["label"], sub["kl_sum"])
        print(f"  KL_sum(P||P^cf):            AUC = {auc_kl:.4f}")
        if auc_kl > best_auc:
            best_auc, best_config = auc_kl, "kl_sum"

    # H: Cosine
    if "cosine_dist" in sub.columns:
        auc_cos = safe_auc(sub["label"], sub["cosine_dist"])
        print(f"  Cosine distance:            AUC = {auc_cos:.4f}")
        if auc_cos > best_auc:
            best_auc, best_config = auc_cos, "cosine_dist"

    print(f"\n  -> Best: {best_config}, AUC = {best_auc:.4f}")

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
            auc = safe_auc(s["label"], s["ced"])
            print(f"  {task:12s}: AUC(ced) = {auc:.4f} (cp={len(cp_t)}, hal={len(hal_t)})")
        else:
            print(f"  {task:12s}: insufficient samples (cp={len(cp_t)}, hal={len(hal_t)})")

    # 5. Control group comparison
    print("\n--- Control Group (correct_positive only) ---")
    ctrl_col = "ctrl_ced" if "ctrl_ced" in cp.columns else None
    if ctrl_col and cp[ctrl_col].notna().sum() > 0:
        ctrl_rows = cp[cp[ctrl_col].notna()]
        print(f"  Object region CED:     {ctrl_rows['ced'].mean():.4f} +- {ctrl_rows['ced'].std():.4f}")
        print(f"  Non-object region CED: {ctrl_rows[ctrl_col].mean():.4f} +- {ctrl_rows[ctrl_col].std():.4f}")
        ratio = ctrl_rows['ced'].mean() / max(ctrl_rows[ctrl_col].mean(), 1e-8)
        print(f"  Ratio: {ratio:.1f}x")
    if "ctrl_js_sum" in cp.columns and cp["ctrl_js_sum"].notna().sum() > 0:
        ctrl_js_rows = cp[cp["ctrl_js_sum"].notna()]
        print(f"  Object region JS_sum:     {ctrl_js_rows['js_sum'].mean():.4f} +- {ctrl_js_rows['js_sum'].std():.4f}")
        print(f"  Non-object region JS_sum: {ctrl_js_rows['ctrl_js_sum'].mean():.4f} +- {ctrl_js_rows['ctrl_js_sum'].std():.4f}")

    # 6. Entropy analysis
    print("\n--- Entropy Analysis ---")
    for b in ["correct_positive", "hallucination", "correct_negative"]:
        s = df[df["behavior"] == b]
        if len(s) > 0:
            print(f"  {b:20s}: H(P)={s['h_orig'].mean():.3f}, "
                  f"H(P^cf)={s['h_cf'].mean():.3f}, "
                  f"ep_sum={s['entropy_penalty_sum'].mean():.4f}")

    # 7. Pass/Fail
    print("\n" + "=" * 60)
    if best_auc > 0.85:
        print(f"P0-b PASSED: AUC = {best_auc:.4f} > 0.85 ({best_config})")
        print(f"   -> Proceed to Phase 1 GRPO!")
    elif best_auc > 0.75:
        print(f"P0-b MARGINAL: AUC = {best_auc:.4f}")
        print(f"   Suggestion: Adjust token replacement strategy or increase samples")
    else:
        print(f"P0-b FAILED: AUC = {best_auc:.4f}")
        print(f"   Suggestion: Switch to VCD noise/MaskCD/PROJECTAWAY")
    print("=" * 60)


def plot_results(df, lambda_e_values, layers, result_dir):
    """Generate visualization plots."""
    os.makedirs(result_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    cp = df[df["behavior"] == "correct_positive"]
    hal = df[df["behavior"] == "hallucination"]
    sub = pd.concat([cp.assign(label=1), hal.assign(label=0)]) if len(cp) > 0 and len(hal) > 0 else pd.DataFrame()

    primary_metric = "ced"

    # 1. CED distribution by behavior
    ax = axes[0, 0]
    colors = {"correct_positive": "green", "hallucination": "red",
              "correct_negative": "blue", "miss": "orange"}
    for b, c in colors.items():
        s = df[df["behavior"] == b]
        if len(s) > 2 and primary_metric in s.columns:
            ax.hist(s[primary_metric], bins=25, alpha=0.4, color=c, label=f"{b} (n={len(s)})", density=True)
    ax.set_xlabel(f"CED (T_key={T_KEY_SIZE}, lambda={DEFAULT_LAMBDA_E})")
    ax.set_title("CED Distribution by Behavior")
    ax.legend(fontsize=8)

    # 2. Formula ablation AUC
    ax = axes[0, 1]
    if len(sub) > 0 and sub["label"].nunique() > 1:
        names, aucs = [], []
        names.append("CED (primary)"); aucs.append(safe_auc(sub["label"], sub["ced"]))
        names.append("JS_sum"); aucs.append(safe_auc(sub["label"], sub["js_sum"]))
        for lam in lambda_e_values:
            col = f"ced_lam{lam:.2f}"
            if col in sub.columns:
                names.append(f"CED lam={lam}")
                aucs.append(safe_auc(sub["label"], sub[col]))
        if "kl_sum" in sub.columns:
            names.append("KL_sum"); aucs.append(safe_auc(sub["label"], sub["kl_sum"]))
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
        layer_names, layer_aucs = ["CED"], [safe_auc(sub["label"], sub["ced"])]
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
        fpr, tpr, _ = roc_curve(sub["label"], sub["ced"])
        ax.plot(fpr, tpr, label=f"CED (AUC={safe_auc(sub['label'], sub['ced']):.3f})", lw=2, color="red")
        if "js_sum" in sub.columns:
            fpr2, tpr2, _ = roc_curve(sub["label"], sub["js_sum"])
            ax.plot(fpr2, tpr2, label=f"JS_sum (AUC={safe_auc(sub['label'], sub['js_sum']):.3f})", lw=2, color="blue", ls="--")
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC Curve"); ax.legend()

    # 5. Control group comparison
    ax = axes[1, 1]
    ctrl_col = "ctrl_ced" if "ctrl_ced" in cp.columns else None
    if ctrl_col and cp[ctrl_col].notna().sum() > 5:
        ax.boxplot([cp["ced"].dropna(), cp[ctrl_col].dropna()],
                   labels=["Object Region", "Control Region"])
        ax.set_ylabel("CED")
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
            task_aucs[task] = safe_auc(s["label"], s["ced"])
    if task_aucs:
        ax.bar(task_aucs.keys(), task_aucs.values(),
               color=["green" if v > 0.85 else "orange" for v in task_aucs.values()])
        ax.axhline(0.85, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("AUC (CED)"); ax.set_title("Cross-task Consistency")
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

    # Counterfactual replacement config
    parser.add_argument("--cf_mode", type=str, default="moment_noise",
                        choices=["mean", "moment_noise"],
                        help="Visual-token replacement mode: mean (baseline) or moment_noise (recommended)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Noise scale for moment_noise replacement (default: 1.0)")

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
