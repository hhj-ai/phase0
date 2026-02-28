import os
import argparse
import torch
import torch.distributed as dist
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import pandas as pd
import random

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    else:
        return 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def load_model(model_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    ).cuda()
    model.eval()
    return model

def run_probe(args):
    model = load_model(args.model_path)
    print("✓ Probe success, model loaded.")

def run_worker(args):
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    model = load_model(args.model_path).to(device)

    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    results = []

    for i in range(10):  # 示例任务
        x = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            _ = model

        results.append({"rank": local_rank, "sample": i})

    if not dist.is_initialized() or dist.get_rank() == 0:
        df = pd.DataFrame(results)
        df.to_csv("worker_output.csv", index=False)

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "probe":
        run_probe(args)
    elif args.mode == "worker":
        run_worker(args)
    else:
        raise ValueError("Unknown mode")

if __name__ == "__main__":
    main()
