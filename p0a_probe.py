#!/usr/bin/env python3
"""
P0-a: 架构探测脚本（单卡，5分钟内完成）

在正式跑P0-b之前，必须先确认：
1. model.visual的输出shape和hook机制是否正常工作
2. Qwen3-VL的merger降采样比例（影响bbox→token映射）
3. image_grid_thw与visual token数量的关系
4. 替换目标区域token后，输出是否真的发生了变化

任何一项失败都意味着P0-b的实验设计需要调整。
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image

# 路径
BASE = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"
MODEL_PATH = f"{BASE}/data/models/Qwen3-VL-8B-Instruct"
COCO_IMG_DIR = f"{BASE}/data/datasets/coco_val2017/val2017"
COCO_ANN_PATH = f"{BASE}/data/datasets/coco_val2017/annotations/instances_val2017.json"


def main():
    print("=" * 60)
    print("P0-a: Qwen3-VL 架构探测")
    print("=" * 60)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ============================================================
    # Step 1: 加载模型，检查visual encoder结构
    # ============================================================
    print("\n--- Step 1: 加载模型 ---")
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 打印visual encoder结构概要
    print(f"\nmodel.visual 类型: {type(model.visual).__name__}")
    if hasattr(model.visual, 'merger'):
        print(f"model.visual.merger 类型: {type(model.visual.merger).__name__}")
    # 统计visual encoder参数量
    vis_params = sum(p.numel() for p in model.visual.parameters())
    print(f"Visual encoder参数量: {vis_params / 1e6:.1f}M")

    # ============================================================
    # Step 2: 单张图片前向，检查visual输出shape
    # ============================================================
    print("\n--- Step 2: 单张图片前向传播 ---")

    # 找一张COCO图
    with open(COCO_ANN_PATH) as f:
        coco = json.load(f)
    test_img_info = coco["images"][0]
    test_img_path = os.path.join(COCO_IMG_DIR, test_img_info["file_name"])
    img = Image.open(test_img_path).convert("RGB")
    img_w, img_h = img.size
    print(f"测试图片: {test_img_info['file_name']}, 尺寸: {img_w}x{img_h}")

    # 找该图的一个标注
    test_anns = [a for a in coco["annotations"] if a["image_id"] == test_img_info["id"]]
    cat_map = {c["id"]: c["name"] for c in coco["categories"]}
    if test_anns:
        test_ann = max(test_anns, key=lambda a: a["bbox"][2] * a["bbox"][3])
        obj_name = cat_map[test_ann["category_id"]]
        bbox = test_ann["bbox"]  # [x, y, w, h]
        print(f"目标物体: {obj_name}, bbox: {bbox}")
    else:
        obj_name = "object"
        bbox = [img_w*0.25, img_h*0.25, img_w*0.5, img_h*0.5]
        print(f"无标注，使用中心区域")

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

    # 检查image_grid_thw
    grid_thw = inputs.get("image_grid_thw", None)
    if grid_thw is not None:
        print(f"\nimage_grid_thw: {grid_thw}")
        grid_t, grid_h, grid_w = grid_thw[0].tolist()
        print(f"  temporal={grid_t}, height={grid_h}, width={grid_w}")
        print(f"  总visual tokens (t*h*w) = {grid_t * grid_h * grid_w}")
    else:
        print("WARNING: image_grid_thw 不存在！")
        sys.exit(1)

    # ============================================================
    # Step 3: Hook测试 - 捕获visual encoder输出
    # ============================================================
    print("\n--- Step 3: Hook机制测试 ---")

    captured_output = {}

    def capture_hook(module, input, output):
        captured_output["shape"] = output.shape
        captured_output["dtype"] = output.dtype
        captured_output["tensor"] = output.detach().clone()
        return output

    handle = model.visual.register_forward_hook(capture_hook)

    with torch.no_grad():
        orig_out = model(**inputs, output_hidden_states=True, return_dict=True)

    handle.remove()

    if "shape" not in captured_output:
        print("FAIL: Hook没有捕获到visual encoder输出！")
        print("  可能原因: model.visual不是标准的nn.Module")
        print("  需要检查Qwen3-VL的visual pipeline结构")
        sys.exit(1)

    vis_shape = captured_output["shape"]
    print(f"Visual encoder输出 shape: {vis_shape}")
    print(f"Visual encoder输出 dtype: {captured_output['dtype']}")

    # 关键检查：shape与grid_thw的关系
    expected_tokens = grid_t * grid_h * grid_w
    actual_tokens = vis_shape[0] if len(vis_shape) == 2 else vis_shape[1]
    print(f"\n期望visual tokens: {expected_tokens} (from grid_thw)")
    print(f"实际visual tokens: {actual_tokens} (from hook output dim 0)")

    if actual_tokens == expected_tokens:
        print("✅ visual tokens数量匹配！")
        merger_ratio = 1
    else:
        ratio = expected_tokens / actual_tokens if actual_tokens > 0 else 0
        print(f"⚠️  不匹配！比例 = {ratio:.1f}")
        print(f"  如果ratio≈1说明hook抓对了位置")
        print(f"  如果ratio>1说明hook在merger之前，需要调整hook位置")
        print(f"  如果ratio<1说明有temporal维度的影响")
        merger_ratio = ratio

    hidden_dim = vis_shape[-1]
    print(f"Hidden dim: {hidden_dim}")

    # ============================================================
    # Step 4: 替换测试 - 验证修改visual tokens能影响输出
    # ============================================================
    print("\n--- Step 4: 替换测试 ---")

    # 计算目标token indices
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

    print(f"目标区域: rows [{row_start},{row_end}), cols [{col_start},{col_end})")
    print(f"目标token数: {len(target_indices)} / {grid_h * grid_w} 总tokens")
    print(f"目标区域占比: {len(target_indices) / (grid_h * grid_w) * 100:.1f}%")

    # 计算替换值（周围token均值）
    vis_features = captured_output["tensor"]
    all_indices = set(range(vis_features.shape[0]))
    surr_indices = sorted(all_indices - set(target_indices))
    if surr_indices:
        replacement = vis_features[surr_indices].mean(dim=0)
    else:
        replacement = vis_features.mean(dim=0)

    # 构造修改版
    modified_features = vis_features.clone()
    for idx in target_indices:
        if idx < modified_features.shape[0]:
            modified_features[idx] = replacement

    # 用修改版做forward
    def replace_hook(module, input, output):
        return modified_features.to(device=output.device, dtype=output.dtype)

    handle2 = model.visual.register_forward_hook(replace_hook)
    with torch.no_grad():
        cf_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle2.remove()

    # 比较logits差异
    orig_logits = orig_out.logits[0, -1].cpu().float()
    cf_logits = cf_out.logits[0, -1].cpu().float()
    logit_diff = (orig_logits - cf_logits).abs().mean().item()
    logit_max_diff = (orig_logits - cf_logits).abs().max().item()

    # 计算JS散度
    from scipy.stats import entropy
    p = torch.softmax(orig_logits, dim=0).numpy()
    q = torch.softmax(cf_logits, dim=0).numpy()
    p = np.clip(p, 1e-12, None); p /= p.sum()
    q = np.clip(q, 1e-12, None); q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2)

    print(f"\n替换前后logits差异:")
    print(f"  Mean |Δlogits|: {logit_diff:.4f}")
    print(f"  Max  |Δlogits|: {logit_max_diff:.4f}")
    print(f"  JS divergence:  {js:.6f}")

    if js > 1e-6:
        print("✅ 替换visual tokens确实影响了输出！JS > 0")
    else:
        print("❌ FAIL: 替换后输出几乎没变，hook可能没有正确覆盖encoder输出")
        print("  需要检查hook是否在正确的位置")

    # ============================================================
    # Step 5: 控制实验 - 替换无物体区域应该影响更小
    # ============================================================
    print("\n--- Step 5: 控制对比 ---")

    # 选择角落tokens（大概率无物体）
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

    handle3 = model.visual.register_forward_hook(ctrl_hook)
    with torch.no_grad():
        ctrl_out = model(**inputs, output_hidden_states=True, return_dict=True)
    handle3.remove()

    ctrl_logits = ctrl_out.logits[0, -1].cpu().float()
    p_ctrl = torch.softmax(ctrl_logits, dim=0).numpy()
    p_ctrl = np.clip(p_ctrl, 1e-12, None); p_ctrl /= p_ctrl.sum()
    m_ctrl = 0.5 * (p + p_ctrl)
    js_ctrl = 0.5 * entropy(p, m_ctrl, base=2) + 0.5 * entropy(p_ctrl, m_ctrl, base=2)

    print(f"替换物体区域 JS = {js:.6f}")
    print(f"替换角落区域 JS = {js_ctrl:.6f}")
    if js > js_ctrl:
        print(f"✅ 物体区域JS > 角落区域JS (比例: {js/max(js_ctrl, 1e-12):.1f}x)")
    else:
        print(f"⚠️  角落区域JS >= 物体区域JS，信号可能不够强")

    # ============================================================
    # Step 6: Hidden states层数检查
    # ============================================================
    print("\n--- Step 6: Hidden states结构 ---")
    n_layers = len(orig_out.hidden_states)
    print(f"总hidden states层数: {n_layers} (包括embedding层)")
    print(f"可探测的中间层: 0 到 {n_layers-1}")
    print(f"计划探测: [16, 20, 24, 28, 32]", end=" ")
    if n_layers > 32:
        print("✅ 均在范围内")
    else:
        safe_layers = [l for l in [16, 20, 24, 28, 32] if l < n_layers]
        print(f"⚠️  实际可用: {safe_layers}")

    # ============================================================
    # 总结
    # ============================================================
    print("\n" + "=" * 60)
    print("P0-a 探测总结")
    print("=" * 60)

    checks = {
        "Hook捕获正常": "shape" in captured_output,
        "Token数量匹配": abs(actual_tokens - expected_tokens) < 10,
        "替换影响输出": js > 1e-6,
        "物体区域信号更强": js > js_ctrl,
    }

    all_pass = True
    for name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("P0-a PASSED — 可以运行P0-b")
    else:
        print("P0-a FAILED — 需要调整hook位置或替换策略后重试")

    # 保存关键参数供P0-b使用
    probe_info = {
        "grid_h": grid_h, "grid_w": grid_w, "grid_t": grid_t,
        "hidden_dim": hidden_dim,
        "n_hidden_layers": n_layers,
        "visual_output_shape": list(vis_shape),
        "js_object_region": float(js),
        "js_control_region": float(js_ctrl),
        "all_passed": all_pass,
    }
    os.makedirs(f"{BASE}/results", exist_ok=True)
    with open(f"{BASE}/results/p0a_probe_info.json", "w") as f:
        json.dump(probe_info, f, indent=2)
    print(f"\n探测信息已保存到 {BASE}/results/p0a_probe_info.json")


if __name__ == "__main__":
    main()
