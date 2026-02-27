#!/usr/bin/env python3
"""
P0实验 —— CED公式验证 + JS散度视觉锚定验证（完全重写版）

=== 实验目的 ===
验证CED（Counterfactual Evidence Divergence）公式作为视觉锚定度物理测谎仪的可行性：
  CED_i = Σ_{t∈T_key} [ D_JS(P(·|V) || P(·|V^cf)) + λ_e · [H(P^cf) - H(P)]_+ ]

核心假设：
  1. 对目标物体对应的visual token做均值替换后，JS散度在logits层和中间层(16,20,24,28,32)
     能有效区分"模型真正使用了视觉信息"(正例) vs "模型依赖语言先验"(反例/控制组)
  2. 熵正则项 [H(P^cf)-H(P)]_+ 能进一步改善区分力
  3. 该信号跨任务(存在性/计数/空间/属性)具有一致性

=== 与原版代码的核心区别 ===
  - 在visual token特征空间做替换（非像素空间）—— 通过hook拦截视觉编码器输出
  - 使用COCO bbox精确定位目标物体对应的visual tokens
  - 三组对比：目标区域替换 / 无关区域替换(控制组) / 不存在物体探测(反例)
  - 验证完整CED公式（JS + 熵正则），扫描λ_e
  - 中间层hidden states用softmax转分布后再算JS（而非错误的L2归一化）

=== 通过标准 ===
  AUC > 0.85 → P0通过，可进入Phase 1 GRPO训练

模型：Qwen/Qwen3-VL-8B-Instruct (bf16)
环境：8×H200, torchrun --nproc_per_node=8
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
from sklearn.metrics import roc_auc_score, roc_curve

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

# ============================================================
# 路径配置（硬编码）
# ============================================================
BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH = f"{BASE}/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG_DIR = f"{BASE}/data/datasets/coco_val2017/val2017"
COCO_ANN_PATH = f"{BASE}/data/datasets/coco_val2017/annotations/instances_val2017.json"
HALLUSION_JSON = f"{BASE}/data/datasets/hallusion_bench/HallusionBench.json"
HALLUSION_IMG_DIR = f"{BASE}/data/datasets/hallusion_bench/images"
RESULT_DIR = f"{BASE}/results"
LOG_DIR = f"{BASE}/logs"


# ============================================================
# 1. CED 公式计算
# ============================================================

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability distributions (bits)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))


def shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy H(p) in bits."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    return float(entropy(p, base=2))


def compute_ced(
    orig_logits: np.ndarray,
    cf_logits: np.ndarray,
    lambda_e: float = 0.1,
    temperature: float = 1.0,
) -> Dict[str, float]:
    """
    计算CED公式的各项:
      CED = D_JS(P(·|V) || P(·|V^cf)) + λ_e · [H(P^cf) - H(P)]_+

    Returns dict with js, entropy_orig, entropy_cf, entropy_penalty, ced
    """
    p = _softmax(orig_logits / temperature)
    q = _softmax(cf_logits / temperature)

    js = js_divergence(p, q)
    h_orig = shannon_entropy(p)
    h_cf = shannon_entropy(q)
    entropy_penalty = max(0.0, h_cf - h_orig)  # [H(P^cf) - H(P)]_+
    ced = js + lambda_e * entropy_penalty

    return {
        "js": js,
        "entropy_orig": h_orig,
        "entropy_cf": h_cf,
        "entropy_penalty": entropy_penalty,
        "ced": ced,
    }


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def compute_hidden_divergence(
    h_orig: torch.Tensor, h_cf: torch.Tensor, top_k: int = 4096
) -> float:
    """
    对中间层hidden states计算JS散度。

    直接对全维度做softmax会过于平坦，
    所以先取绝对值最大的top-k维度的并集，再softmax转为伪概率分布。
    """
    h1 = h_orig.cpu().float().numpy()
    h2 = h_cf.cpu().float().numpy()

    # 取两个向量中活跃维度的并集
    top_dims = np.union1d(
        np.argsort(np.abs(h1))[-top_k:],
        np.argsort(np.abs(h2))[-top_k:]
    )
    p = _softmax(h1[top_dims])
    q = _softmax(h2[top_dims])
    return js_divergence(p, q)


# ============================================================
# 2. Visual Token Hook（特征空间替换的核心机制）
# ============================================================

class VisualTokenHook:
    """
    Hook on model.visual to:
    1. Capture mode: 保存原始visual token features
    2. Modify mode: 返回替换后的features（覆盖encoder实际输出）

    Qwen3-VL的visual encoder输出shape: (total_tokens, hidden_dim)
    对单张图片, total_tokens = grid_h * grid_w (merger后的grid)
    tokens按raster order排列: index = row * grid_w + col
    """

    def __init__(self):
        self.captured: Optional[torch.Tensor] = None
        self.modifications: Optional[Dict[int, torch.Tensor]] = None
        self._handle = None

    def register(self, model):
        """注册到model.visual的forward hook"""
        self._handle = model.visual.register_forward_hook(self._hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()

    def reset(self):
        """重置为capture mode"""
        self.captured = None
        self.modifications = None

    def set_modifications(self, mods: Dict[int, torch.Tensor]):
        """设置modifications后，下次forward会返回修改后的features"""
        self.modifications = mods

    def _hook_fn(self, module, input, output):
        if self.modifications is None:
            # Capture mode: 保存原始特征（不修改输出）
            self.captured = output.detach().clone()
            return output
        else:
            # Modify mode: 从captured构造修改版，直接覆盖encoder输出
            # （encoder实际跑了但结果被丢弃，对P0实验来说计算开销可接受）
            modified = self.captured.clone().to(device=output.device, dtype=output.dtype)
            for idx, replacement in self.modifications.items():
                if 0 <= idx < modified.shape[0]:
                    modified[idx] = replacement.to(device=modified.device, dtype=modified.dtype)
            return modified


# ============================================================
# 3. BBox → Visual Token Index 映射
# ============================================================

def bbox_to_token_indices(
    bbox_xywh: Tuple[float, float, float, float],
    img_w: int, img_h: int,
    grid_h: int, grid_w: int,
    padding: int = 0,
) -> List[int]:
    """
    将COCO格式bbox (x, y, w, h) 映射到visual token grid indices。

    Visual tokens按raster order排列: index = row * grid_w + col
    Grid均匀覆盖整张图片，按比例映射（无需知道具体resize尺寸）。
    """
    x, y, w, h = bbox_xywh

    col_start = int(x / img_w * grid_w) - padding
    col_end = int(np.ceil((x + w) / img_w * grid_w)) + padding
    row_start = int(y / img_h * grid_h) - padding
    row_end = int(np.ceil((y + h) / img_h * grid_h)) + padding

    col_start = max(0, col_start)
    col_end = min(grid_w, col_end)
    row_start = max(0, row_start)
    row_end = min(grid_h, row_end)

    indices = []
    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            indices.append(r * grid_w + c)
    return indices


def get_surrounding_mean(
    features: torch.Tensor, target_indices: List[int]
) -> torch.Tensor:
    """
    计算target indices之外所有visual token的均值，作为替换值。
    features shape: (total_tokens, hidden_dim)
    """
    all_indices = set(range(features.shape[0]))
    surr_indices = sorted(all_indices - set(target_indices))
    if len(surr_indices) == 0:
        return features.mean(dim=0)
    return features[surr_indices].mean(dim=0)


def get_control_indices(
    all_obj_bboxes: List,
    img_w: int, img_h: int,
    grid_h: int, grid_w: int,
    n_tokens: int,
) -> List[int]:
    """
    找到与所有物体不重叠的visual token indices作为控制组。
    """
    occupied = set()
    for bbox in all_obj_bboxes:
        occupied.update(bbox_to_token_indices(bbox, img_w, img_h, grid_h, grid_w, padding=1))

    all_tokens = set(range(grid_h * grid_w))
    free_tokens = sorted(all_tokens - occupied)

    if len(free_tokens) >= n_tokens:
        start = random.randint(0, len(free_tokens) - n_tokens)
        return free_tokens[start:start + n_tokens]
    elif len(free_tokens) > 0:
        return free_tokens
    else:
        # fallback: 选择图片角落token
        corner = []
        for r in range(min(3, grid_h)):
            for c in range(min(3, grid_w)):
                corner.append(r * grid_w + c)
        return corner[:n_tokens]


# ============================================================
# 4. 数据加载
# ============================================================

def load_coco_samples(
    ann_path: str, img_dir: str, n_samples: int = 400, min_bbox_ratio: float = 0.02
) -> List[dict]:
    """
    从COCO val2017构造三组样本：

    - positive: 物体存在，替换物体区域 → 期望CED高
    - negative: 图中不存在该物体 → 期望CED低
    - control:  物体存在，但替换无物体区域 → 期望CED低
    """
    print(f"Loading COCO annotations from {ann_path}...")
    with open(ann_path) as f:
        coco = json.load(f)

    cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # 按图片分组标注
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # 每张图的类别集合
    img_categories = {}
    for img_id, anns in img_anns.items():
        img_categories[img_id] = set(ann["category_id"] for ann in anns)

    all_category_ids = set(cat_map.keys())
    samples = []

    img_ids = list(img_anns.keys())
    random.shuffle(img_ids)

    for img_id in img_ids:
        if len(samples) >= n_samples:
            break

        anns = img_anns[img_id]
        img_info = img_map[img_id]
        img_w, img_h = img_info["width"], img_info["height"]
        img_area = img_w * img_h
        img_file = os.path.join(img_dir, img_info["file_name"])
        if not os.path.exists(img_file):
            continue

        # 筛选足够大的、非crowd物体
        valid_anns = [
            a for a in anns
            if a["bbox"][2] * a["bbox"][3] / img_area > min_bbox_ratio
            and not a.get("iscrowd", 0)
        ]
        if not valid_anns:
            continue

        # 选最大物体作为正例目标
        target_ann = max(valid_anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        target_cat = cat_map[target_ann["category_id"]]
        target_bbox = target_ann["bbox"]  # [x, y, w, h]
        all_bboxes = [a["bbox"] for a in anns]

        # === Positive: 物体存在 → 替换其区域 → CED应高 ===
        samples.append({
            "type": "positive",
            "img_path": img_file,
            "img_w": img_w, "img_h": img_h,
            "question": f"Is there a {target_cat} in this image? Answer yes or no.",
            "target_object": target_cat,
            "target_bbox": target_bbox,
            "all_bboxes": all_bboxes,
            "gt_answer": 1,
            "label": 1,  # 期望HIGH CED
        })

        # === Negative: 问不存在的物体 → CED应低 ===
        present_cats = img_categories.get(img_id, set())
        absent_cats = all_category_ids - present_cats
        if absent_cats:
            absent_id = random.choice(list(absent_cats))
            absent_name = cat_map[absent_id]
            # 该物体不存在，替换中心区域（替什么都不应有大CED）
            fake_bbox = [img_w * 0.25, img_h * 0.25, img_w * 0.5, img_h * 0.5]
            samples.append({
                "type": "negative",
                "img_path": img_file,
                "img_w": img_w, "img_h": img_h,
                "question": f"Is there a {absent_name} in this image? Answer yes or no.",
                "target_object": absent_name,
                "target_bbox": fake_bbox,
                "all_bboxes": all_bboxes,
                "gt_answer": 0,
                "label": 0,  # 期望LOW CED
            })

        # === Control: 物体存在，但替换无物体区域 → CED应低 ===
        samples.append({
            "type": "control",
            "img_path": img_file,
            "img_w": img_w, "img_h": img_h,
            "question": f"Is there a {target_cat} in this image? Answer yes or no.",
            "target_object": target_cat,
            "target_bbox": None,  # 在run时通过get_control_indices确定
            "all_bboxes": all_bboxes,
            "gt_answer": 1,
            "label": 0,  # 期望LOW CED
        })

    n_pos = sum(1 for s in samples if s["type"] == "positive")
    n_neg = sum(1 for s in samples if s["type"] == "negative")
    n_ctrl = sum(1 for s in samples if s["type"] == "control")
    print(f"Loaded {len(samples)} COCO samples (pos={n_pos}, neg={n_neg}, ctrl={n_ctrl})")
    return samples


# ============================================================
# 5. 单样本处理
# ============================================================

def process_single_sample(
    sample: dict,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    hook: VisualTokenHook,
    device: torch.device,
    layers_to_probe: List[int] = [16, 20, 24, 28, 32],
    lambda_e_values: List[float] = [0.0, 0.05, 0.1, 0.2, 0.5],
    n_agg_tokens: int = 3,
) -> Optional[dict]:
    """
    处理单个样本，返回CED各指标。

    流程:
    1. 正常forward → hook捕获原始visual features + 获取原始outputs
    2. 计算target tokens & 替换值
    3. Counterfactual forward → hook返回修改后的features → 获取CF outputs
    4. 在logits层和中间层计算CED各项
    """
    try:
        image = Image.open(sample["img_path"]).convert("RGB")
    except Exception as e:
        return None

    # === 构造模型输入 ===
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": sample["question"]},
        ]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # 获取visual token grid维度
    grid_thw = inputs.get("image_grid_thw", None)
    if grid_thw is None:
        return None
    grid_h = grid_thw[0, 1].item()
    grid_w = grid_thw[0, 2].item()
    total_vis_tokens = grid_h * grid_w

    # === 确定要替换的token indices ===
    if sample["type"] == "control":
        n_target = max(4, total_vis_tokens // 8)
        target_indices = get_control_indices(
            sample["all_bboxes"], sample["img_w"], sample["img_h"],
            grid_h, grid_w, n_target
        )
    else:
        target_indices = bbox_to_token_indices(
            sample["target_bbox"], sample["img_w"], sample["img_h"],
            grid_h, grid_w
        )

    if len(target_indices) == 0:
        return None

    # === Pass 1: 原始forward (capture mode) ===
    hook.reset()
    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    if hook.captured is None:
        return None

    # === 构造替换值 ===
    surrounding_mean = get_surrounding_mean(hook.captured, target_indices)

    # === Pass 2: Counterfactual forward (modify mode) ===
    modifications = {idx: surrounding_mean for idx in target_indices}
    hook.set_modifications(modifications)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)

    # 重置hook，防止影响后续样本
    hook.reset()

    # === 计算指标 ===
    result = {
        "type": sample["type"],
        "target_object": sample["target_object"],
        "gt_answer": sample["gt_answer"],
        "label": sample["label"],
        "n_target_tokens": len(target_indices),
        "n_total_vis_tokens": total_vis_tokens,
        "target_ratio": len(target_indices) / total_vis_tokens,
    }

    # --- Logits层CED（最后一个token，即generation position）---
    orig_logits_np = orig_out.logits[0, -1].cpu().float().numpy()
    cf_logits_np = cf_out.logits[0, -1].cpu().float().numpy()

    for lam in lambda_e_values:
        ced_dict = compute_ced(orig_logits_np, cf_logits_np, lambda_e=lam)
        result["logits_js"] = ced_dict["js"]
        result["logits_h_orig"] = ced_dict["entropy_orig"]
        result["logits_h_cf"] = ced_dict["entropy_cf"]
        result["logits_entropy_penalty"] = ced_dict["entropy_penalty"]
        result[f"logits_ced_lam{lam:.2f}"] = ced_dict["ced"]

    # --- 中间层JS散度 ---
    for layer_idx in layers_to_probe:
        if layer_idx < len(orig_out.hidden_states):
            h_orig = orig_out.hidden_states[layer_idx][0, -1]
            h_cf = cf_out.hidden_states[layer_idx][0, -1]
            result[f"layer_{layer_idx}_js"] = compute_hidden_divergence(h_orig, h_cf)

    # --- 多token聚合CED (T_key = 最后n个位置) ---
    seq_len = orig_out.logits.shape[1]
    multi_ced_sum = 0.0
    for t in range(min(n_agg_tokens, seq_len)):
        pos = seq_len - 1 - t
        p_np = orig_out.logits[0, pos].cpu().float().numpy()
        q_np = cf_out.logits[0, pos].cpu().float().numpy()
        multi_ced_sum += compute_ced(p_np, q_np, lambda_e=0.1)["ced"]
    result["multi_token_ced"] = multi_ced_sum

    return result


# ============================================================
# 6. 主实验循环
# ============================================================

def run_experiment(args):
    print("=" * 60)
    print("P0实验：CED公式验证 + JS散度视觉锚定验证")
    print("=" * 60)
    print(f"CED_i = Σ D_JS(P(·|V)||P(·|V^cf)) + λ_e·[H(P^cf)-H(P)]_+")
    print(f"通过标准：AUC > 0.85\n")

    # === 设备设置 ===
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    is_main = (local_rank == 0)
    if is_main:
        print(f"World size: {world_size}, Device: {device}")

    # === 加载模型 ===
    if is_main:
        print(f"Loading model: {MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 注册hook
    hook = VisualTokenHook().register(model)

    # === 加载数据 ===
    samples = load_coco_samples(COCO_ANN_PATH, COCO_IMG_DIR, n_samples=args.num_samples)

    # 多卡数据分片
    shard_size = len(samples) // max(world_size, 1)
    start = local_rank * shard_size
    end = start + shard_size if local_rank < world_size - 1 else len(samples)
    my_samples = samples[start:end]

    if is_main:
        print(f"Rank {local_rank}: {len(my_samples)} samples [{start}:{end}]")

    # === 运行 ===
    results = []
    failed = 0
    pbar = tqdm(my_samples, desc=f"[Rank {local_rank}]", disable=not is_main)

    for sample in pbar:
        result = process_single_sample(
            sample, model, processor, hook, device,
            layers_to_probe=args.layers,
            lambda_e_values=args.lambda_e_values,
        )
        if result is not None:
            results.append(result)
        else:
            failed += 1

        # 实时AUC更新
        if is_main and len(results) % 20 == 0 and len(results) > 20:
            df_temp = pd.DataFrame(results)
            if df_temp["label"].nunique() > 1:
                auc_t = roc_auc_score(df_temp["label"], df_temp["logits_js"])
                pbar.set_postfix(n=len(results), AUC=f"{auc_t:.3f}", fail=failed)

    # === 保存分片结果 ===
    os.makedirs(RESULT_DIR, exist_ok=True)
    shard_path = os.path.join(RESULT_DIR, f"js_results_rank{local_rank}.csv")
    df = pd.DataFrame(results)
    df.to_csv(shard_path, index=False)
    print(f"Rank {local_rank}: saved {len(results)} results (failed: {failed})")

    # 同步
    if world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # === 主进程合并 & 分析 ===
    if is_main:
        all_dfs = []
        for rank in range(world_size):
            path = os.path.join(RESULT_DIR, f"js_results_rank{rank}.csv")
            if os.path.exists(path):
                all_dfs.append(pd.read_csv(path))
        df_all = pd.concat(all_dfs, ignore_index=True)
        df_all.to_csv(os.path.join(RESULT_DIR, "js_results_all.csv"), index=False)

        print(f"\n=== Total: {len(df_all)} samples ===")
        analyze_results(df_all, args.lambda_e_values, args.layers)

    hook.remove()


# ============================================================
# 7. 结果分析
# ============================================================

def analyze_results(df, lambda_e_values, layers):
    print("\n" + "=" * 60)
    print("P0实验结果分析")
    print("=" * 60)

    # 7.1 按类型统计
    print("\n--- 各组JS散度统计 ---")
    for t in ["positive", "negative", "control"]:
        sub = df[df["type"] == t]
        if len(sub) > 0:
            print(f"  {t:10s}: n={len(sub):4d}, "
                  f"JS_mean={sub['logits_js'].mean():.4f} ± {sub['logits_js'].std():.4f}")

    if df["label"].nunique() < 2:
        print("ERROR: 标签只有一类，无法计算AUC！")
        return

    # 7.2 核心AUC
    print("\n--- Logits层 AUC ---")
    auc_js = roc_auc_score(df["label"], df["logits_js"])
    print(f"  裸JS散度:       AUC = {auc_js:.4f}")

    best_auc = auc_js
    best_lam = 0.0
    for lam in lambda_e_values:
        col = f"logits_ced_lam{lam:.2f}"
        if col in df.columns:
            auc = roc_auc_score(df["label"], df[col])
            tag = " ★" if auc > best_auc else ""
            print(f"  CED(λ_e={lam:.2f}): AUC = {auc:.4f}{tag}")
            if auc > best_auc:
                best_auc = auc
                best_lam = lam

    print(f"\n  → 最优: λ_e={best_lam:.2f}, AUC={best_auc:.4f}")

    # 7.3 中间层
    print("\n--- 中间层 AUC ---")
    for layer in layers:
        col = f"layer_{layer}_js"
        if col in df.columns and df[col].notna().sum() > 10:
            auc = roc_auc_score(df["label"], df[col])
            print(f"  Layer {layer}: AUC = {auc:.4f}")

    # 7.4 多token聚合
    if "multi_token_ced" in df.columns:
        auc_m = roc_auc_score(df["label"], df["multi_token_ced"])
        print(f"\n  Multi-token CED: AUC = {auc_m:.4f}")

    # 7.5 分组对比
    print("\n--- 两两对比 ---")
    for t1, t2 in [("positive", "negative"), ("positive", "control")]:
        sub = df[df["type"].isin([t1, t2])]
        if sub["label"].nunique() > 1 and len(sub) > 10:
            auc = roc_auc_score(sub["label"], sub["logits_js"])
            print(f"  {t1} vs {t2}: AUC = {auc:.4f}")

    # 7.6 熵项分析
    print("\n--- 熵项分析 (验证[H(P^cf)-H(P)]_+是否有区分力) ---")
    for t in ["positive", "negative", "control"]:
        sub = df[df["type"] == t]
        if len(sub) > 0:
            print(f"  {t:10s}: H(P)={sub['logits_h_orig'].mean():.3f}, "
                  f"H(P^cf)={sub['logits_h_cf'].mean():.3f}, "
                  f"Δ_+={sub['logits_entropy_penalty'].mean():.4f}")

    # 7.7 Pass/Fail
    print("\n" + "=" * 60)
    if best_auc > 0.85:
        print(f"✅ P0 PASSED: AUC = {best_auc:.4f} > 0.85")
        print(f"   最优λ_e = {best_lam:.2f}")
        print(f"   → 可进入 Phase 1 GRPO 训练！")
    elif best_auc > 0.75:
        print(f"⚠️  P0 MARGINAL: AUC = {best_auc:.4f}")
        print(f"   建议: 调整替换策略或增加样本")
    else:
        print(f"❌ P0 FAILED: AUC = {best_auc:.4f}")
        print(f"   建议: 切换备选方案 B/C/D")
    print("=" * 60)

    # 7.8 可视化
    plot_results(df, lambda_e_values, layers, RESULT_DIR)


def plot_results(df, lambda_e_values, layers, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: JS分布
    ax = axes[0, 0]
    for t, c in [("positive", "green"), ("negative", "red"), ("control", "gray")]:
        sub = df[df["type"] == t]
        if len(sub) > 0:
            ax.hist(sub["logits_js"], bins=30, alpha=0.5, color=c, label=t, density=True)
    ax.set_xlabel("JS Divergence (logits)")
    ax.set_ylabel("Density")
    ax.set_title("JS Divergence Distribution (positive should be rightmost)")
    ax.legend()

    # Plot 2: CED AUC vs λ_e
    ax = axes[0, 1]
    aucs = []
    for lam in lambda_e_values:
        col = f"logits_ced_lam{lam:.2f}"
        if col in df.columns and df["label"].nunique() > 1:
            aucs.append((lam, roc_auc_score(df["label"], df[col])))
    if aucs:
        lams, vals = zip(*aucs)
        ax.plot(lams, vals, "o-", linewidth=2)
        ax.axhline(0.85, color="red", linestyle="--", label="Threshold 0.85")
        ax.set_xlabel("λ_e")
        ax.set_ylabel("AUC")
        ax.set_title("CED AUC vs λ_e (entropy regularization effect)")
        ax.legend()

    # Plot 3: 各层AUC
    ax = axes[1, 0]
    layer_data = []
    if df["label"].nunique() > 1:
        layer_data.append(("logits", roc_auc_score(df["label"], df["logits_js"])))
    for layer in layers:
        col = f"layer_{layer}_js"
        if col in df.columns and df[col].notna().sum() > 10:
            layer_data.append((f"L{layer}", roc_auc_score(df["label"], df[col])))
    if layer_data:
        names, vals = zip(*layer_data)
        colors = ["green" if v > 0.85 else "orange" if v > 0.75 else "red" for v in vals]
        ax.bar(names, vals, color=colors)
        ax.axhline(0.85, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel("AUC")
        ax.set_title("AUC by Layer (find optimal layer)")
        ax.set_ylim(0.5, 1.0)

    # Plot 4: ROC
    ax = axes[1, 1]
    if df["label"].nunique() > 1:
        fpr, tpr, _ = roc_curve(df["label"], df["logits_js"])
        ax.plot(fpr, tpr, label=f"JS (AUC={roc_auc_score(df['label'], df['logits_js']):.3f})", lw=2)
        # Best CED
        best_lam, best_auc = 0.0, roc_auc_score(df["label"], df["logits_js"])
        for lam in lambda_e_values:
            col = f"logits_ced_lam{lam:.2f}"
            if col in df.columns:
                a = roc_auc_score(df["label"], df[col])
                if a > best_auc:
                    best_auc, best_lam = a, lam
        col = f"logits_ced_lam{best_lam:.2f}"
        if col in df.columns:
            fpr, tpr, _ = roc_curve(df["label"], df[col])
            ax.plot(fpr, tpr, label=f"CED λ={best_lam:.2f} (AUC={best_auc:.3f})", lw=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "p0_analysis.png"), dpi=150)
    plt.close()

    # 熵分解图
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    types = ["positive", "negative", "control"]
    metrics = ["logits_js", "logits_entropy_penalty"]
    matrix = []
    for t in types:
        sub = df[df["type"] == t]
        matrix.append([sub[m].mean() if m in sub.columns and len(sub) > 0 else 0 for m in metrics])
    matrix = np.array(matrix)
    im = ax2.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(2))
    ax2.set_xticklabels(["JS Divergence", "Entropy Penalty"])
    ax2.set_yticks(range(3))
    ax2.set_yticklabels(types)
    for i in range(3):
        for j in range(2):
            ax2.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", fontsize=12)
    ax2.set_title("CED Component Decomposition")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "p0_ced_decomposition.png"), dpi=150)
    plt.close()

    print(f"Charts saved to {save_dir}/")


# ============================================================
# 8. Entry
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P0: CED Formula Validation")
    parser.add_argument("--num_samples", type=int, default=600)
    parser.add_argument("--layers", type=int, nargs="+", default=[16, 20, 24, 28, 32])
    parser.add_argument("--lambda_e_values", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        torch.distributed.init_process_group(backend="nccl")

    run_experiment(args)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
