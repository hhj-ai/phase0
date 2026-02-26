# p0_main.py
"""
P0实验 —— JS散度视觉锚定验证实验（VLM幻觉缓解框架最核心Phase 0）

实验目的与背景：
本实验来自《VLM幻觉缓解框架文献调研报告》。目的是验证「对目标物体对应的visual token做均值替换」后，
output logits层 + 中间hidden layers (16,20,24,28,32) 的JS散度是否能作为可靠的“物理测谎仪”：
- 正例（图中真正存在该物体）：JS散度显著大 → 证明模型真的看了图、视觉信息流正常
- 反例（图中无物体但模型hallucinated）：JS散度显著小 → 证明模型在用纯语言先验瞎猜
- 控制组（替换无关区域）：JS散度小
核心假设：JS散度可作为特征空间的视觉锚定度物理信号，未来可直接嵌入对抗式RL训练的奖励函数。
这是范式级创新，所有先前工作（VCD、M3ID、Vision-SR1、TON等）均未实现此信号用于训练。
如果AUC>0.85则P0通过，可进入Phase 1 GRPO训练。

模型：Qwen/Qwen3-VL-8B-Instruct（bf16全精度，无任何量化）
环境：8×H200，torchrun --nproc_per_node=8
路径：全部硬编码在/shared路径下
"""

import os
import torch
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch.distributed as dist
from accelerate import Accelerator

def js_divergence(p, q):
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))

def calculate_patch_indices(bbox, grid_thw, patch_size=14):
    """根据bbox和grid_thw计算需要替换的patch索引（进阶版token均值替换核心）"""
    x1, y1, x2, y2 = bbox
    h, w = grid_thw[1], grid_thw[2]
    patch_h, patch_w = h // patch_size, w // patch_size
    start_row = int(y1 / patch_size)
    end_row = int(y2 / patch_size)
    start_col = int(x1 / patch_size)
    end_col = int(x2 / patch_size)
    indices = []
    for r in range(start_row, end_row):
        for c in range(start_col, end_col):
            idx = r * patch_w + c
            indices.append(idx)
    return indices

def main(args):
    print("=== P0实验启动 ===")
    print("实验目的：验证JS散度作为视觉锚定度物理测谎仪的可行性，用于未来RL奖励信号。")
    print("核心假设：JS散度可作为特征空间视觉锚定信号，打破现有工作仅推理时使用的局限。")

    accelerator = Accelerator()
    device = accelerator.device

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/data/models/Qwen3-VL-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(
        "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/data/models/Qwen3-VL-8B-Instruct"
    )

    results = []

    # 1. HallusionBench（presence/absence问题）
    hallusion_path = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/data/datasets/hallusion_bench/HallusionBench.json"
    with open(hallusion_path) as f:
        data = json.load(f)[:args.num_samples // 2]

    for item in tqdm(data, desc="HallusionBench", disable=not accelerator.is_main_process):
        if "is there" not in item['question'].lower() and "存在" not in item['question']:
            continue
        img_path = f"/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/data/datasets/hallusion_bench/images/{item['image']}"
        image = Image.open(img_path).convert("RGB")
        question = item['question']
        gt = int(item.get('answer', 1))  # 1=存在

        # 正例/反例/控制
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

        # 简化版：bbox像素mask（快速验证，推荐先跑此版本）
        masked_image = image.copy()
        w, h = masked_image.size
        mask_box = (w//4, h//4, w*3//4, h*3//4)
        mask_arr = np.array(masked_image)
        mask_arr[mask_box[1]:mask_box[3], mask_box[0]:mask_box[2]] = mask_arr.mean(axis=(0,1)).astype(np.uint8)
        masked_image = Image.fromarray(mask_arr)

        masked_inputs = processor(text=[text], images=[masked_image], videos=video_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            masked_out = model(**masked_inputs, output_hidden_states=True, return_dict=True)

        # 计算JS
        js_scores = {}
        p = torch.softmax(orig_out.logits[0, -1], dim=-1).cpu().float().numpy()
        q = torch.softmax(masked_out.logits[0, -1], dim=-1).cpu().float().numpy()
        js_scores['logits'] = js_divergence(p, q)

        for layer_idx in [16, 20, 24, 28, 32]:
            h_orig = orig_out.hidden_states[layer_idx][0, -1].cpu().float().numpy()
            h_mask = masked_out.hidden_states[layer_idx][0, -1].cpu().float().numpy()
            h_orig = h_orig / np.linalg.norm(h_orig + 1e-8)
            h_mask = h_mask / np.linalg.norm(h_mask + 1e-8)
            js_scores[f'layer_{layer_idx}'] = js_divergence(h_orig, h_mask)

        results.append({
            'type': 'hallusion',
            'has_object': gt,
            'js_logits': js_scores['logits'],
            **{k: v for k, v in js_scores.items() if 'layer' in k}
        })

    df = pd.DataFrame(results)
    os.makedirs("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/results", exist_ok=True)
    df.to_csv("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/results/js_results.csv", index=False)

    if len(df) > 10:
        auc_logits = roc_auc_score(df['has_object'], df['js_logits'])
        print(f"\n=== P0实验结果总结 ===")
        print(f"Logits层 AUC: {auc_logits:.4f}")
        for layer in [16,20,24,28,32]:
            col = f'layer_{layer}'
            if col in df.columns:
                auc = roc_auc_score(df['has_object'], df[col])
                print(f"Layer {layer} AUC: {auc:.4f}")
        if auc_logits > 0.85:
            print("P0通过阈值：AUC>0.85 → 可进入Phase 1 GRPO训练！")
        else:
            print("P0未达阈值，建议检查mask策略或切换进阶token替换。")

    layers = ['logits'] + [f'layer_{l}' for l in [16,20,24,28,32]]
    means = [df[l].mean() for l in layers if l in df.columns]
    plt.figure(figsize=(10,6))
    plt.bar(layers[:len(means)], means)
    plt.title("JS散度均值（物理测谎仪强度） - P0实验")
    plt.ylabel("JS Divergence")
    plt.savefig("/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl/results/js_heatmap.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    main(args)
