import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

"""
实验名称：P0实验 —— JS散度视觉锚定验证实验
核心假设：验证视觉Token均值替换后的Hidden Layers JS散度是否能区分【真视觉流入】与【语言先验幻觉】。
模型要求：Qwen3-VL-8B (BF16, No Quantization)
物理测谎仪逻辑：
- Positive (Exist): High JS Divergence (Model relies on visual tokens)
- Negative (Hallucination): Low JS Divergence (Model ignores visual tokens)
"""

SHARED_ROOT = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/native_mm/zhangmanyuan/zhangquan/agent/xl/hhj-train/data/p0_qwen3vl"

def print_p0_header():
    header = """
    ****************************************************************
    * P0 实验启动: JS 散度视觉锚定物理测谎仪验证                     *
    * 框架阶段: Phase 0 (信号有效性验证)                            *
    * 关键指标: AUC-ROC > 0.85 则通过，进入 GRPO 训练阶段           *
    * 运行模式: 8-GPU BF16 全精度分布式推理                         *
    ****************************************************************
    """
    print(header)

def calculate_js_divergence(p_logits, q_logits):
    """计算两个分布之间的JS散度"""
    p = F.softmax(p_logits, dim=-1).detach().cpu().float().numpy()
    q = F.softmax(q_logits, dim=-1).detach().cpu().float().numpy()
    # 为计算方便，取平均
    m = 0.5 * (p + q)
    return 0.5 * (jensenshannon(p, m, axis=-1)**2 + jensenshannon(q, m, axis=-1)**2)

class P0Dataset(Dataset):
    def __init__(self, data_root, num_samples=800):
        # 简单实现：混合 HallusionBench 和 COCO 样本
        self.samples = [] 
        # 此处省略具体解析逻辑，预设格式：{'image_path': str, 'prompt': str, 'bbox': [y1, x1, y2, x2], 'label': 1/0}
        # 生产级代码应从 self.data_root 加载真实标注
        for i in range(num_samples):
            self.samples.append({
                "id": i,
                "image_path": f"{data_root}/coco_val2017/val2017/000000000139.jpg", # 示例路径
                "prompt": "Is there a person in the image?",
                "bbox": [100, 100, 300, 300], # 模拟bbox
                "is_hallucination": i % 2 # 模拟构造正负例
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def get_visual_token_indices(grid_thw, bbox):
    """
    根据Qwen3-VL的image_grid_thw(T, H, W)和归一化bbox计算对应的visual token索引
    逻辑：根据网格划分，定位bbox覆盖的patch。
    """
    t, h_grid, w_grid = grid_thw[0]
    y1, x1, y2, x2 = bbox # 0-1000 scale
    
    start_h, end_h = int(y1 * h_grid / 1000), int(y2 * h_grid / 1000)
    start_w, end_w = int(x1 * w_grid / 1000), int(x2 * w_grid / 1000)
    
    indices = []
    for h in range(start_h, min(end_h + 1, h_grid)):
        for w in range(start_w, min(end_w + 1, w_grid)):
            indices.append(h * w_grid + w)
    return indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    
    if local_rank == 0: print_p0_header()

    # 1. 加载模型 (BF16 全精度)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": local_rank}
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # 2. 钩子函数获取中间层 (16, 20, 24, 28, 32)
    target_layers = [16, 20, 24, 28, 32]
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # Qwen2.5-VL output is (batch, seq, hidden)
            activation[name] = output[0].detach()
        return hook

    for l_idx in target_layers:
        model.model.layers[l_idx].register_forward_hook(get_activation(f"layer_{l_idx}"))

    # 3. 数据准备
    dataset = P0Dataset(args.data_root, args.num_samples)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    results = []

    # 4. 推理循环
    model.eval()
    for batch in tqdm(dataloader, disable=(local_rank != 0)):
        # 预处理
        messages = [{"role": "user", "content": [{"type": "image", "image": batch['image_path'][0]}, {"type": "text", "text": batch['prompt'][0]}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)

        # A. 原始前向传播
        with torch.no_grad():
            outputs_orig = model(**inputs)
            orig_logits = outputs_orig.logits[:, -1, :]
            orig_hiddens = {name: act[:, -1, :] for name, act in activation.items()}

        # B. 视觉Token均值替换 (核心介入)
        # 寻找视觉token在sequence中的位置
        # 注意：Qwen3-VL视觉token通常在特定占位符之后
        grid_thw = inputs['image_grid_thw']
        patch_indices = get_visual_token_indices(grid_thw, batch['bbox'])
        
        # 简易实现：找到第一张图的起始offset (根据 <|vision_start|> 标签)
        input_ids = inputs['input_ids'][0]
        vision_start_idx = (input_ids == processor.tokenizer.convert_tokens_to_ids("<|visual_pad|>")).nonzero(as_tuple=True)[0]
        
        if len(vision_start_idx) > 0:
            offset = vision_start_idx[0]
            # 这里的输入嵌入层替换
            modified_inputs = inputs.copy()
            inputs_embeds = model.get_input_embeddings()(inputs['input_ids']).clone()
            
            # 对目标patch执行均值替换 (物理测谎仪核心步骤)
            for idx in patch_indices:
                if offset + idx < inputs_embeds.shape[1]:
                    # 取周围token均值或全局均值
                    inputs_embeds[0, offset + idx] = torch.mean(inputs_embeds[0, offset], dim=0) 
            
            # C. 介入后前向传播
            outputs_mod = model(inputs_embeds=inputs_embeds, attention_mask=inputs['attention_mask'])
            mod_logits = outputs_mod.logits[:, -1, :]
            mod_hiddens = {name: act[:, -1, :] for name, act in activation.items()}

            # 5. 计算各层 JS 散度
            sample_res = {"id": batch['id'].item(), "is_hallucination": batch['is_hallucination'].item()}
            sample_res["js_logits"] = calculate_js_divergence(orig_logits, mod_logits)
            for l_idx in target_layers:
                js_l = calculate_js_divergence(orig_hiddens[f"layer_{l_idx}"], mod_hiddens[f"layer_{l_idx}"])
                sample_res[f"js_layer_{l_idx}"] = js_l
            
            results.append(sample_res)

    # 6. 聚合结果与分析
    df = pd.DataFrame(results)
    # 分布式收集 (省略逻辑，假设主卡汇总)
    if local_rank == 0:
        df.to_csv(f"{args.output_dir}/js_results.csv", index=False)
        
        # 计算针对幻觉检测的 AUC
        # 注意：逻辑是 JS散度越大 -> 越不是幻觉
        auc_logits = roc_auc_score(1 - df['is_hallucination'], df['js_logits'])
        auc_layer_24 = roc_auc_score(1 - df['is_hallucination'], df['js_layer_24'])
        
        summary = f"P0 实验总结:\nLogits AUC: {auc_logits:.4f}\nLayer 24 AUC: {auc_layer_24:.4f}\n"
        status = "PASSED" if auc_layer_24 > 0.85 else "FAILED"
        summary += f"结论: {status} (阈值 0.85)\n"
        
        with open(f"{args.output_dir}/summary.log", "w") as f:
            f.write(summary)
        print(summary)

        # 绘制层级 JS 散度趋势图
        plt.figure(figsize=(10, 6))
        plt.plot(target_layers, [df[f"js_layer_{l}"].mean() for l in target_layers], marker='o')
        plt.title("Visual Anchoring Signal (JS-Div) across Layers")
        plt.xlabel("Layer Index")
        plt.ylabel("Mean JS Divergence")
        plt.savefig(f"{args.output_dir}/js_trend.png")

    destroy_process_group()

if __name__ == "__main__":
    main()
