#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py — P0 pipeline (probe/worker/analyze/summary) for Qwen3-VL visual-feature counterfactuals.

Key points:
  - Paths configurable via CLI; base_dir defaults to /mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl (override via env P0_BASE_DIR).
  - Control-region metrics (ctrl_*) computed for ALL samples (not cp-only).
  - Worker sharding can be explicit (--shard_idx/--num_shards) so you don't need torchrun.

"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch

warnings.filterwarnings("ignore")

# Optional plotting (safe on headless)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


# ============================================================
# Utils
# ============================================================

def _stable_u32(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) & 0xFFFFFFFF

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: dict, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def rank_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via rank statistic (no sklearn). y_true in {0,1}."""
    y = y_true.astype(np.int32)
    s = scores.astype(np.float64)

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)

    # tie-handling
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def entropy_np(p: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)

def js_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    """Jensen–Shannon divergence, base-e."""
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * (np.log(p) - np.log(m)), axis=axis) +
                  np.sum(q * (np.log(q) - np.log(m)), axis=axis))

def kl_divergence(p: np.ndarray, q: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=axis)


# ============================================================
# Paths
# ============================================================

@dataclass
class Paths:
    base_dir: str
    model_path: str
    coco_img_dir: str
    coco_ann_path: str
    result_dir: str
    log_dir: str

    @staticmethod
    def from_args(args: argparse.Namespace) -> "Paths":
        base_dir = args.base_dir
        model_path = args.model_path or os.path.join(base_dir, "data/models/Qwen3-VL-8B-Instruct")
        coco_img_dir = args.coco_img_dir or os.path.join(base_dir, "data/datasets/coco_val2017/val2017")
        coco_ann_path = args.coco_ann_path or os.path.join(base_dir, "data/datasets/coco_val2017/annotations/instances_val2017.json")
        result_dir = args.result_dir or os.path.join(base_dir, "results")
        log_dir = args.log_dir or os.path.join(base_dir, "logs")
        return Paths(base_dir, model_path, coco_img_dir, coco_ann_path, result_dir, log_dir)


# ============================================================
# Sample construction (COCO)
# ============================================================

SPATIAL_TEMPLATES = [
    ("Is the {obj} in the left half of the image?", lambda bbox, w, h: (bbox[0] + 0.5 * bbox[2]) < 0.5 * w),
    ("Is the {obj} in the right half of the image?", lambda bbox, w, h: (bbox[0] + 0.5 * bbox[2]) > 0.5 * w),
    ("Is the {obj} in the upper half of the image?", lambda bbox, w, h: (bbox[1] + 0.5 * bbox[3]) < 0.5 * h),
    ("Is the {obj} in the lower half of the image?", lambda bbox, w, h: (bbox[1] + 0.5 * bbox[3]) > 0.5 * h),
]

def load_samples(ann_path: str, img_dir: str, n_samples: int = 400,
                 min_bbox_ratio: float = 0.02, seed: int = 42) -> List[Dict]:
    """
    Build a mixed set of yes/no questions:
      - existence positive (gt=1)
      - existence negative (gt=0)
      - spatial (gt depends on bbox)
      - counting (gt depends on exact count)
    """
    print(f"Loading COCO from {ann_path} ...")
    random.seed(seed)

    coco = load_json(ann_path)
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {i["id"]: i for i in coco["images"]}
    all_cat_ids = set(cat_map.keys())

    img_anns = defaultdict(list)
    for a in coco["annotations"]:
        img_anns[a["image_id"]].append(a)

    img_cats = {iid: set(a["category_id"] for a in anns) for iid, anns in img_anns.items()}

    img_ids = list(img_anns.keys())
    random.shuffle(img_ids)

    samples: List[Dict] = []
    for img_id in img_ids:
        if len(samples) >= n_samples:
            break

        anns = img_anns[img_id]
        info = img_map[img_id]
        w, h = info["width"], info["height"]
        img_path = os.path.join(img_dir, info["file_name"])
        if not os.path.exists(img_path):
            continue

        valid = [a for a in anns
                 if (a["bbox"][2] * a["bbox"][3] / (w * h) > min_bbox_ratio) and not a.get("iscrowd", 0)]
        if not valid:
            continue

        target = max(valid, key=lambda a: a["bbox"][2] * a["bbox"][3])
        cat = cat_map[target["category_id"]]
        bbox = target["bbox"]
        all_bboxes = [a["bbox"] for a in anns]

        # existence positive
        samples.append({
            "task": "existence", "img_path": img_path, "img_w": w, "img_h": h,
            "question": f"Is there a {cat} in this image? Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": 1,
        })

        # existence negative (absent category)
        absent = all_cat_ids - img_cats.get(img_id, set())
        if absent:
            abs_name = cat_map[random.choice(list(absent))]
            samples.append({
                "task": "existence", "img_path": img_path, "img_w": w, "img_h": h,
                "question": f"Is there a {abs_name} in this image? Answer yes or no.",
                "object": abs_name, "bbox": bbox, "all_bboxes": all_bboxes,
                "gt": 0,
            })

        # spatial
        tmpl, check_fn = random.choice(SPATIAL_TEMPLATES)
        gt_spatial = 1 if check_fn(bbox, w, h) else 0
        samples.append({
            "task": "spatial", "img_path": img_path, "img_w": w, "img_h": h,
            "question": tmpl.format(obj=cat) + " Answer yes or no.",
            "object": cat, "bbox": bbox, "all_bboxes": all_bboxes,
            "gt": gt_spatial,
        })

        # counting
        same_cat = [a for a in valid if a["category_id"] == target["category_id"]]
        count = len(same_cat)
        wrong_count = count + random.choice([1, 2]) if random.random() < 0.5 else max(1, count - 1)
        ask_count = wrong_count if random.random() < 0.5 else count
        samples.append({
            "task": "counting", "img_path": img_path, "img_w": w, "img_h": h,
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
# Visual token indexing
# ============================================================

def bbox_to_indices(bbox, img_w, img_h, grid_h, grid_w, spatial_merge_size=2) -> List[int]:
    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size

    x, y, bw, bh = bbox
    col_start = max(0, int(x / img_w * merged_w))
    col_end = min(merged_w, int(np.ceil((x + bw) / img_w * merged_w)))
    row_start = max(0, int(y / img_h * merged_h))
    row_end = min(merged_h, int(np.ceil((y + bh) / img_h * merged_h)))

    return [r * merged_w + c for r in range(row_start, row_end) for c in range(col_start, col_end)]

def get_control_indices(all_bboxes, img_w, img_h, grid_h, grid_w, n: int, spatial_merge_size=2) -> List[int]:
    merged_h = grid_h // spatial_merge_size
    merged_w = grid_w // spatial_merge_size
    occupied = set()
    for b in all_bboxes:
        occupied.update(bbox_to_indices(b, img_w, img_h, grid_h, grid_w, spatial_merge_size))
    free = sorted(set(range(merged_h * merged_w)) - occupied)
    if len(free) >= n:
        start = random.randint(0, len(free) - n)
        return free[start:start + n]
    return free if free else list(range(min(n, merged_h * merged_w)))


# ============================================================
# Model / Hook
# ============================================================

def load_model_and_processor(model_path: str, device: torch.device):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def _find_main_visual_field(output) -> Tuple[str, Optional[torch.Tensor]]:
    if isinstance(output, dict):
        for k in ["pooler_output", "last_hidden_state"]:
            if k in output and isinstance(output[k], torch.Tensor):
                return k, output[k]
    for k in ["pooler_output", "last_hidden_state"]:
        if hasattr(output, k):
            t = getattr(output, k)
            if isinstance(t, torch.Tensor):
                return k, t
    return "unknown", None

class VisualHook:
    def __init__(self):
        self.captured: Optional[torch.Tensor] = None
        self._mods: Optional[Dict[int, torch.Tensor]] = None
        self._handle = None

    def register(self, module: torch.nn.Module):
        def hook_fn(_m, _inp, out):
            field, t = _find_main_visual_field(out)
            if t is None:
                return out
            self.captured = t.detach()
            if self._mods:
                t2 = t.clone()
                for idx, vec in self._mods.items():
                    if 0 <= idx < t2.shape[0]:
                        t2[idx] = vec.to(t2.device, dtype=t2.dtype)
                if isinstance(out, dict):
                    out[field] = t2
                    return out
                if hasattr(out, field):
                    setattr(out, field, t2)
                    return out
            return out
        self._handle = module.register_forward_hook(hook_fn)

    def set_replace(self, mods: Dict[int, torch.Tensor]):
        self._mods = mods

    def reset(self):
        self.captured = None
        self._mods = None

    def close(self):
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
            self._handle = None

def get_model_answer(model, processor, image, question, device) -> Tuple[int, str]:
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
    answer = processor.decode(gen_ids[0][input_len:], skip_special_tokens=True).strip().lower()
    if "yes" in answer:
        return 1, answer
    if "no" in answer:
        return 0, answer
    return -1, answer


# ============================================================
# Replacement maps
# ============================================================

def build_replacement_map(
    vis_feat: torch.Tensor,
    target_idx: List[int],
    mode: str,
    noise_scale: float,
    seed: int,
) -> Dict[int, torch.Tensor]:
    rng = np.random.default_rng(seed)
    idx = [i for i in target_idx if 0 <= i < vis_feat.shape[0]]
    if not idx:
        return {}
    feat = vis_feat.detach().cpu().float().numpy()  # [T, D]
    mods: Dict[int, torch.Tensor] = {}

    if mode == "mean":
        mu = feat.mean(axis=0)
        rep = torch.from_numpy(mu).float()
        for i in idx:
            mods[i] = rep
        return mods

    if mode == "moment_noise":
        mu = feat.mean(axis=0)
        std = feat.std(axis=0) + 1e-6
        for i in idx:
            eps = rng.standard_normal(size=mu.shape).astype(np.float32)
            vec = mu + noise_scale * std * eps
            mods[i] = torch.from_numpy(vec).float()
        return mods

    if mode == "patch_swap":
        all_ids = np.arange(feat.shape[0], dtype=np.int64)
        mask = np.ones_like(all_ids, dtype=bool)
        mask[idx] = False
        donors = all_ids[mask]
        if donors.size == 0:
            donors = all_ids
        chosen = rng.choice(donors, size=len(idx), replace=True)
        for ti, di in zip(idx, chosen):
            mods[ti] = torch.from_numpy(feat[int(di)]).float()
        return mods

    raise ValueError(f"Unknown replace mode: {mode}")


# ============================================================
# Metrics
# ============================================================

def compute_tkey_metrics(
    logits_a: np.ndarray,  # [T_key, V]
    logits_b: np.ndarray,  # [T_key, V]
    lambda_e_values: List[float],
) -> Dict[str, float]:
    p = softmax_np(logits_a, axis=-1)
    q = softmax_np(logits_b, axis=-1)

    js = js_divergence(p, q, axis=-1)              # [T_key]
    kl = kl_divergence(p, q, axis=-1)              # [T_key]
    ha = entropy_np(p, axis=-1)
    hb = entropy_np(q, axis=-1)
    dH_pos = np.maximum(0.0, hb - ha)              # [T_key]

    out: Dict[str, float] = {}
    out["js_sum"] = float(js.sum())
    out["kl_sum"] = float(kl.sum())
    out["entropy_penalty_sum"] = float(dH_pos.sum())

    default_lam = 0.1 if 0.1 in lambda_e_values else float(lambda_e_values[0]) if lambda_e_values else 0.1

    def ced(lam: float) -> float:
        return float((js + lam * dH_pos).sum())

    out["ced"] = ced(default_lam)
    for lam in lambda_e_values:
        out[f"ced_lam{lam:.2f}"] = ced(float(lam))
    return out


# ============================================================
# One sample run
# ============================================================

def run_one_sample(
    sample: Dict,
    model,
    processor,
    hook: VisualHook,
    device: torch.device,
    t_key_size: int,
    lambda_e_values: List[float],
    replace_mode: str,
    noise_scale: float,
    seed_base: int,
    spatial_merge_size: int,
) -> Optional[Dict]:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    try:
        image = Image.open(sample["img_path"]).convert("RGB")
    except Exception:
        return None

    pred, pred_text = get_model_answer(model, processor, image, sample["question"], device)
    if pred == -1:
        return None

    gt = int(sample["gt"])
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

    # forward inputs (no generation) for logits capture
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": sample["question"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    grid_thw = inputs.get("image_grid_thw", None)
    if grid_thw is None:
        return None
    grid_h = int(grid_thw[0, 1].item())
    grid_w = int(grid_thw[0, 2].item())

    # original forward
    hook.reset(),
    layers: 'Optional[List[int]]' = None,
    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=False, return_dict=True)
    if hook.captured is None:
        return None

    vis_feat = hook.captured
    logits = orig_out.logits[0]
    actual_t = min(t_key_size, logits.shape[0])
    orig_logits_tkey = logits[-actual_t:].detach().cpu().float().numpy()

    # target replacement
    target_idx = bbox_to_indices(sample["bbox"], sample["img_w"], sample["img_h"],
                                 grid_h, grid_w, spatial_merge_size)
    if not target_idx:
        return None

    s_seed = (seed_base ^ _stable_u32(sample["img_path"] + "|" + sample["question"])) & 0x7FFFFFFF
    mods = build_replacement_map(vis_feat, target_idx, replace_mode, noise_scale, s_seed)

    hook.set_replace(mods)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=False, return_dict=True)
    hook.reset()
    cf_logits_tkey = cf_out.logits[0, -actual_t:].detach().cpu().float().numpy()
    metrics = compute_tkey_metrics(orig_logits_tkey, cf_logits_tkey, lambda_e_values)

    # control region metrics (ALL samples)
    n_ctrl = max(4, len(target_idx))
    ctrl_idx = get_control_indices(sample["all_bboxes"], sample["img_w"], sample["img_h"],
                                   grid_h, grid_w, n_ctrl, spatial_merge_size)
    c_seed = (seed_base ^ _stable_u32(sample["img_path"] + "|CTRL|" + sample["question"])) & 0x7FFFFFFF
    ctrl_mods = build_replacement_map(vis_feat, ctrl_idx, replace_mode, noise_scale, c_seed)

    # fresh capture then control replacement forward
    hook.reset()
    with torch.no_grad():
        _ = model(**inputs, output_hidden_states=False, return_dict=True)

    hook.set_replace(ctrl_mods)
    with torch.no_grad():
        ctrl_out = model(**inputs, output_hidden_states=False, return_dict=True)
    hook.reset()

    ctrl_logits_tkey = ctrl_out.logits[0, -actual_t:].detach().cpu().float().numpy()
    ctrl_metrics = compute_tkey_metrics(orig_logits_tkey, ctrl_logits_tkey, lambda_e_values)

    row = {
        "task": sample["task"],
        "behavior": behavior,
        "gt": gt,
        "pred": int(pred),
        "pred_text": pred_text,
        "img_path": sample["img_path"],
        "question": sample["question"],
        "object": sample["object"],
        "ced": metrics["ced"],
        "js_sum": metrics["js_sum"],
        "kl_sum": metrics["kl_sum"],
        "entropy_penalty_sum": metrics["entropy_penalty_sum"],
        "ctrl_ced": ctrl_metrics["ced"],
        "ctrl_js_sum": ctrl_metrics["js_sum"],
        "ctrl_kl_sum": ctrl_metrics["kl_sum"],
        "ctrl_entropy_penalty_sum": ctrl_metrics["entropy_penalty_sum"],
        "t_key_size": actual_t,
        "replace_mode": replace_mode,
        "noise_scale": float(noise_scale),
    }
    for lam in lambda_e_values:
        row[f"ced_lam{lam:.2f}"] = metrics[f"ced_lam{lam:.2f}"]
        row[f"ctrl_ced_lam{lam:.2f}"] = ctrl_metrics[f"ced_lam{lam:.2f}"]
    return row


# ============================================================
# Modes
# ============================================================

def _infer_merge_size(model) -> int:
    merger = getattr(model.model.visual, "merger", None)
    spatial_merge_size = getattr(merger, "spatial_merge_size", None)
    if spatial_merge_size is None and hasattr(merger, "config"):
        spatial_merge_size = getattr(merger.config, "spatial_merge_size", None)
    return int(spatial_merge_size) if spatial_merge_size is not None else 2

def run_probe(args: argparse.Namespace, paths: Paths) -> None:
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}")
    model, processor = load_model_and_processor(paths.model_path, device)
    spatial_merge_size = _infer_merge_size(model)

    patch_size = getattr(model.model.visual, "patch_size", None)
    if patch_size is None and hasattr(model.model.visual, "config"):
        patch_size = getattr(model.model.visual.config, "patch_size", None)
    patch_size = int(patch_size) if patch_size is not None else 16

    print("model.model.visual type:", type(model.model.visual).__name__)
    print("--- Architecture Parameters ---")
    print(f"  spatial_merge_size: {spatial_merge_size}")
    print(f"  patch_size: {patch_size}")

    coco = load_json(paths.coco_ann_path)
    img_info = coco["images"][0]
    img_path = os.path.join(paths.coco_img_dir, img_info["file_name"])
    assert os.path.exists(img_path), f"Missing image: {img_path}"

    from PIL import Image
    image = Image.open(img_path).convert("RGB")
    w, h = image.size

    test_anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    if test_anns:
        ann = max(test_anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        obj = cat_map[ann["category_id"]]
        bbox = ann["bbox"]
    else:
        obj = "object"
        bbox = [w * 0.25, h * 0.25, w * 0.5, h * 0.5]

    question = f"Is there a {obj} in this image? Answer yes or no."

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

    grid_thw = inputs.get("image_grid_thw", None)
    assert grid_thw is not None, "image_grid_thw not found"
    grid_t, grid_h, grid_w = grid_thw[0].tolist()
    print(f"image_grid_thw: {grid_thw} -> total={grid_t*grid_h*grid_w}")

    hook = VisualHook()
    hook.register(model.model.visual)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=False, return_dict=True)
    assert hook.captured is not None, "Hook did not capture visual features"
    vis_feat = hook.captured
    expected_total = int(grid_t * grid_h * grid_w)
    actual_total = int(vis_feat.shape[0])
    ratio = expected_total / max(1, actual_total)
    print("captured visual feat:", tuple(vis_feat.shape), vis_feat.dtype)
    print(f"Expected {expected_total}, captured {actual_total}, ratio={ratio:.2f}")

    idx = bbox_to_indices(bbox, w, h, int(grid_h), int(grid_w), spatial_merge_size)
    s_seed = (args.seed ^ _stable_u32(img_path + "|" + question)) & 0x7FFFFFFF
    mods = build_replacement_map(vis_feat, idx, args.replace_mode, args.noise_scale, s_seed)
    hook.set_replace(mods)
    with torch.no_grad():
        out2 = model(**inputs, output_hidden_states=False, return_dict=True)
    hook.reset()

    logits_a = out.logits[0, -min(args.t_key_size, out.logits.shape[1]):].detach().cpu().float().numpy()
    logits_b = out2.logits[0, -min(args.t_key_size, out2.logits.shape[1]):].detach().cpu().float().numpy()
    m = compute_tkey_metrics(logits_a, logits_b, args.lambda_e_values)
    print(f"Replacement JS_sum={m['js_sum']:.6f} CED={m['ced']:.6f}")

    hook.close()

    probe_info = {
        "model_path": paths.model_path,
        "coco_img": img_info["file_name"],
        "grid_thw": [int(grid_t), int(grid_h), int(grid_w)],
        "captured_shape": list(vis_feat.shape),
        "spatial_merge_size": int(spatial_merge_size),
        "patch_size": int(patch_size),
        "t_key_size": int(args.t_key_size),
        "replace_mode": args.replace_mode,
        "noise_scale": float(args.noise_scale),
    }
    out_path = os.path.join(paths.result_dir, "p0a_probe_info.json")
    save_json(probe_info, out_path)
    print(f"[saved] {out_path}")

def run_worker(args: argparse.Namespace, paths: Paths) -> None:
    # ------------------------------------------------------------
    # Sharding
    # ------------------------------------------------------------
    if args.shard_idx is not None:
        shard_idx = int(args.shard_idx)
        num_shards = int(args.num_shards)
    else:
        shard_idx = int(os.environ.get("RANK", "0"))
        num_shards = int(os.environ.get("WORLD_SIZE", "1"))

    # ------------------------------------------------------------
    # Device binding (critical under torchrun)
    # ------------------------------------------------------------
    local_rank_env = (
        os.environ.get("LOCAL_RANK")
        or os.environ.get("SLURM_LOCALID")
        or os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
        or os.environ.get("MPI_LOCALRANKID")
    )
    global_rank_env = (
        os.environ.get("RANK")
        or os.environ.get("SLURM_PROCID")
        or os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMI_RANK")
    )

    device = torch.device("cpu")
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        if local_rank_env is not None:
            device_id = int(local_rank_env)
        elif global_rank_env is not None:
            device_id = int(global_rank_env) % n_gpu
        else:
            device_id = 0
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")

    print(
        f"[worker] shard={shard_idx}/{num_shards} "
        f"local_rank={local_rank_env} rank={global_rank_env} "
        f"device={device} n_gpu={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
    )

    ensure_dir(paths.result_dir)

    # ------------------------------------------------------------
    # Load model + hook (each process loads on its own GPU)
    # ------------------------------------------------------------
    model, processor = load_model_and_processor(paths.model_path, device)
    spatial_merge_size = _infer_merge_size(model)

    hook = VisualHook()
    hook.register(model.model.visual)

    try:
        # ------------------------------------------------------------
        # Load samples and take this shard's slice
        # ------------------------------------------------------------
        samples = load_samples(
            coco_img_dir=paths.coco_img_dir,
            coco_ann_path=paths.coco_ann_path,
            n_total=args.num_samples,
            seed=args.seed,
        )
        shard_samples = samples[shard_idx::num_shards]
        print(f"[worker] shard_samples={len(shard_samples)}")

        rows = []
        for ex in shard_samples:
            row = run_one_sample(
                ex=ex,
                model=model,
                processor=processor,
                device=device,
                hook=hook,
                spatial_merge_size=spatial_merge_size,
                layers=args.layers,
                lambda_e_values=args.lambda_e_values,
                replace_mode=args.replace_mode,
                noise_scale=args.noise_scale,
                t_key_size=args.t_key_size,
            )
            rows.append(row)

        out_csv = os.path.join(
            paths.result_dir, f"p0b_shard_{shard_idx:03d}_of_{num_shards:03d}.csv"
        )
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[saved] {out_csv} rows={len(rows)}")
    finally:
        hook.remove()

def run_analyze(args: argparse.Namespace, paths: Paths) -> None:
    import pandas as pd
    import numpy as np

    # Collect shards
    shard_paths = sorted(glob.glob(os.path.join(paths.result_dir, "p0b_shard_*_of_*.csv")))
    if not shard_paths:
        print(f"[analyze] no shard csv found under: {paths.result_dir}")
        return

    dfs = []
    for p in shard_paths:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"[analyze] failed to read {p}: {e}")

    if not dfs:
        print("[analyze] no readable shard csv.")
        return

    df = pd.concat(dfs, ignore_index=True)
    merged_csv = os.path.join(paths.result_dir, "p0b_merged.csv")
    df.to_csv(merged_csv, index=False)

    print("=" * 60)
    print("P0 Results Summary")
    print("=" * 60)
    print(f"merged_csv : {merged_csv}")
    print(f"rows       : {len(df)}")
    print(f"columns    : {len(df.columns)}")

    # Behavior stats
    if "behavior" in df.columns and "ced" in df.columns:
        stats = df.groupby("behavior")["ced"].agg(["count", "mean", "std", "min", "max"]).reset_index()
        print("\n--- Behavior Distribution ---")
        print(stats.set_index("behavior").to_string())
    else:
        stats = None
        print("\n[warn] missing columns to compute behavior distribution (need: behavior, ced)")

    # Ensure delta/ratio exist
    if "ctrl_ced" in df.columns and "ced" in df.columns:
        if "delta_ced" not in df.columns:
            df["delta_ced"] = df["ced"].astype(float) - df["ctrl_ced"].astype(float)
        if "ratio_ced" not in df.columns:
            df["ratio_ced"] = df["ced"].astype(float) / (df["ctrl_ced"].astype(float) + 1e-6)

    # Core AUCs: correct_positive vs hallucination
    sub = df[df["behavior"].isin(["correct_positive", "hallucination"])].copy() if "behavior" in df.columns else df.copy()
    cp_n = int((sub["behavior"] == "correct_positive").sum()) if "behavior" in sub.columns else 0
    hal_n = int((sub["behavior"] == "hallucination").sum()) if "behavior" in sub.columns else 0

    aucs: Dict[str, float] = {}
    if "behavior" in sub.columns:
        y = (sub["behavior"] == "correct_positive").astype(int).to_numpy()
        for col in ["ced", "delta_ced", "ratio_ced", "ctrl_ced", "js_sum", "kl_sum", "ctrl_js_sum", "ctrl_kl_sum"]:
            if col in sub.columns:
                ss = sub.dropna(subset=[col]).copy()
                if len(ss) < 10:
                    continue
                yy = (ss["behavior"] == "correct_positive").astype(int).to_numpy()
                vv = ss[col].to_numpy(dtype=float)
                a = rank_auc(yy, vv)
                if not np.isnan(a):
                    aucs[col] = float(a)

    # Choose score_mode for plotting / headline
    score_mode = getattr(args, "score_mode", "ratio")
    score_col = {"ced": "ced", "delta": "delta_ced", "ratio": "ratio_ced"}.get(score_mode, "ratio_ced")
    if score_col not in sub.columns:
        score_col = "ced" if "ced" in sub.columns else (list(aucs.keys())[0] if aucs else "")

    if "behavior" in sub.columns and score_col:
        ss = sub.dropna(subset=[score_col]).copy()
        yy = (ss["behavior"] == "correct_positive").astype(int).to_numpy()
        vv = ss[score_col].to_numpy(dtype=float)
        auc_head = rank_auc(yy, vv)
        print("\n--- Core AUC: correct_positive vs hallucination ---")
        print(f"  cp={cp_n} hal={hal_n} total={len(sub)}")
        print(f"  AUC({score_col}) = {auc_head:.4f}")
    else:
        auc_head = float("nan")

    # Plot histogram for chosen score
    fig_path = os.path.join(paths.result_dir, f"p0b_hist_cp_vs_hal_{score_col}.png" if score_col else "p0b_hist_cp_vs_hal.png")
    if "behavior" in sub.columns and score_col:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa
            cp = sub[sub["behavior"] == "correct_positive"][score_col].dropna().astype(float).to_numpy()
            hal = sub[sub["behavior"] == "hallucination"][score_col].dropna().astype(float).to_numpy()
            plt.figure(figsize=(8, 4))
            plt.hist(cp, bins=40, alpha=0.6, label="correct_positive")
            plt.hist(hal, bins=40, alpha=0.6, label="hallucination")
            plt.xlabel(score_col)
            plt.ylabel("count")
            title_auc = aucs.get(score_col, auc_head)
            plt.title(f"CP vs HAL ({score_col}) AUC={title_auc:.4f}" if not np.isnan(title_auc) else f"CP vs HAL ({score_col})")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_path, dpi=200)
            plt.close()
            print(f"[saved] {fig_path}")
        except Exception as e:
            print(f"[warn] matplotlib plot failed: {e}")

    # Write summary JSON
    summary = {
        "merged_csv": merged_csv,
        "rows": int(len(df)),
        "score_mode": score_mode,
        "score_col": score_col,
        "cp_n": cp_n,
        "hal_n": hal_n,
        "auc_headline": None if np.isnan(auc_head) else float(auc_head),
        "aucs": aucs,
        "behavior_stats": [],
    }
    if stats is not None:
        summary["behavior_stats"] = stats.to_dict(orient="records")

    out_json = os.path.join(paths.result_dir, "p0b_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[saved] {out_json}")
def run_summary(_args: argparse.Namespace, paths: Paths) -> None:
    merged_path = os.path.join(paths.result_dir, "p0b_merged.csv")
    if not os.path.exists(merged_path):
        raise RuntimeError(f"Missing {merged_path}, run analyze first.")

    df = pd.read_csv(merged_path)
    sub = df[df["behavior"].isin(["correct_positive", "hallucination"])].copy()
    y = (sub["behavior"] == "correct_positive").astype(int).to_numpy()

    # ctrl NaN check
    for c in ["ctrl_ced", "ctrl_js_sum"]:
        if c not in sub.columns:
            print(f"[missing] {c}")
            continue
        cp_nan = float(sub[sub["behavior"] == "correct_positive"][c].isna().mean())
        hal_nan = float(sub[sub["behavior"] == "hallucination"][c].isna().mean())
        print(f"{c}: cp_nan% {cp_nan:.3f} hal_nan% {hal_nan:.3f}")

    # AUC ranking for numeric cols
    auc_list = []
    for c in sub.columns:
        if c in ["behavior", "task", "img_path", "question", "object", "pred_text"]:
            continue
        v = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        if np.all(np.isnan(v)):
            continue
        vv = v[~np.isnan(v)]
        yy = y[~np.isnan(v)]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        auc_c = rank_auc(yy.astype(int), vv.astype(float))
        if np.isnan(auc_c):
            continue
        auc_list.append((auc_c, c, len(vv)))

    auc_list.sort(key=lambda x: x[0], reverse=True)
    print("\nTop 20 AUC cols (dropna):")
    for a, c, n in auc_list[:20]:
        print(f"{a:.4f}  {c:<18} (n={n})")

    if "ctrl_ced" in sub.columns:
        m = sub.dropna(subset=["ced", "ctrl_ced"]).copy()
        if not m.empty:
            m["delta_ced"] = m["ced"].astype(float) - m["ctrl_ced"].astype(float)
            m["ratio_ced"] = m["ced"].astype(float) / (m["ctrl_ced"].astype(float) + 1e-6)
            y2 = (m["behavior"] == "correct_positive").astype(int).to_numpy()
            print("\nDelta/Ratio AUC:")
            print("AUC ced       =", rank_auc(y2, m["ced"].to_numpy(dtype=float)))
            print("AUC delta_ced =", rank_auc(y2, m["delta_ced"].to_numpy(dtype=float)))
            print("AUC ratio_ced =", rank_auc(y2, m["ratio_ced"].to_numpy(dtype=float)))
        else:
            print("\n[warn] delta/ratio: no rows after dropna(ced, ctrl_ced)")


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", type=str, default=os.environ.get("P0_BASE_DIR", "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"),
                   help="shared base dir (override via env P0_BASE_DIR)")
    p.add_argument("--model_path", type=str, default=os.environ.get("P0_MODEL_PATH", ""), help="local model dir")
    p.add_argument("--coco_img_dir", type=str, default=os.environ.get("P0_COCO_IMG_DIR", ""), help="COCO val2017 image dir")
    p.add_argument("--coco_ann_path", type=str, default=os.environ.get("P0_COCO_ANN_PATH", ""), help="COCO instances_val2017.json")
    p.add_argument("--result_dir", type=str, default=os.environ.get("P0_RESULT_DIR", ""), help="results dir")
    p.add_argument("--log_dir", type=str, default=os.environ.get("P0_LOG_DIR", ""), help="logs dir")

    sp = p.add_subparsers(dest="cmd", required=True)

    def add_run_args(pp: argparse.ArgumentParser, device_default=None):
        pp.add_argument("--seed", type=int, default=42)
        pp.add_argument("--device", type=int, default=device_default)
        pp.add_argument("--t_key_size", type=int, default=4)
        pp.add_argument("--lambda_e_values", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20, 0.50])
        pp.add_argument("--replace_mode", type=str, default="moment_noise", choices=["moment_noise", "mean", "patch_swap"])
        pp.add_argument("--noise_scale", type=float, default=0.15)

    p_probe = sp.add_parser("probe", help="P0-a probe + sanity replacement")
    add_run_args(p_probe, device_default=0)

    p_worker = sp.add_parser("worker", help="P0-b worker shard")
    add_run_args(p_worker, device_default=None)
    p_worker.add_argument("--shard_idx", type=int, default=None)
    p_worker.add_argument("--num_shards", type=int, default=None)
    p_worker.add_argument("--num_samples", type=int, default=400)

    p_analyze = sp.add_parser("analyze", help="merge shards + summary")

    p_analyze.add_argument("--score_mode", type=str, default="ratio", choices=["ced","delta","ratio"],
                          help="which score to treat as headline (ced / delta / ratio)")

    p_summary = sp.add_parser("summary", help="AUC ranking + ctrl NaN check + delta/ratio")

    p_summary.add_argument("--score_mode", type=str, default="ratio", choices=["ced","delta","ratio"],
                          help="which score to treat as headline when printing (ced / delta / ratio)")


    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    # allow empty-string env overrides
    for k in ["model_path", "coco_img_dir", "coco_ann_path", "result_dir", "log_dir"]:
        if getattr(args, k) == "":
            setattr(args, k, None)

    paths = Paths.from_args(args)
    ensure_dir(paths.result_dir)
    ensure_dir(paths.log_dir)

    if args.cmd == "probe":
        run_probe(args, paths)
    elif args.cmd == "worker":
        run_worker(args, paths)
    elif args.cmd == "analyze":
        run_analyze(args, paths)
    elif args.cmd == "summary":
        run_summary(args, paths)
    else:
        raise ValueError(args.cmd)

if __name__ == "__main__":
    main()
