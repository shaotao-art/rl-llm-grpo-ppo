"""
eval.py — evaluate pass@1 (and pass@n when num_samples > 1).
Supports single-GPU and multi-GPU (torchrun) inference.

Usage:
    python eval.py --config configs/eval.yaml
    torchrun --nproc_per_node=4 eval.py --config configs/eval.yaml
"""

import torch
import torch.distributed as dist
import argparse
import yaml
import json
import re
import os
import math
from math import comb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler


def setup_dist():
    is_dist = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, f"cuda:{local_rank}"
    else:
        return 0, 'cuda:0'

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, data_p, split="test"):
        self.data  = load_dataset(data_p, "main")
        self.split = split

    def __getitem__(self, index):
        sample = self.data[self.split][index]
        q, a   = sample["question"], sample["answer"]
        prompt = (
            "Answer the question in the following format:\n"
            "```\n## Reasoning\nyour reasoning process\n\n## Answer\n\\boxed{your answer}\n\n```\n"
            f"# Question\n{q}"
        )
        return prompt, a.split("####")[-1].strip()

    def __len__(self):
        return len(self.data[self.split])


def collect_fn(batch):
    return [b[0] for b in batch], [b[1] for b in batch]



# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, texts):
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for t in texts
    ]


def is_correct(pred, gt):
    if "## Answer" not in pred:
        return False
    m = re.search(r"\\boxed{(.*?)}", pred.split("## Answer")[-1])
    return (m.group(1).strip() if m else "") == gt.strip()

def extract_ans(pred):
    m = re.search(r"\\boxed{(.*?)}", pred.split("## Answer")[-1])
    return m.group(1).strip() if m else ""

def repeat_lst(lst, num_gen):
    """repeat input prompt for num_gen [1, 1, 1, 1, 2, 2, 2, 2, ...]"""
    return [item for item in lst for _ in range(num_gen)]



def pass_at_k(n, c, k):
    """Unbiased estimator: pass@k = 1 - C(n-c, k) / C(n, k)  (Chen et al. 2021)."""
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, tokenizer, dataloader, cfg, device):
    model.eval()
    n = cfg["num_samples"]
    T = cfg["temperature"]
    do_sample = T > 0

    out = []
    for batch_idx, (q_lst, a_lst) in enumerate(tqdm(dataloader)):
        q_lst = build_prompt(tokenizer, q_lst)
        inp  = tokenizer(q_lst, return_tensors="pt", padding=True, padding_side="left").to(device)
        seq_ids = model.generate(
            **inp,
            do_sample    = do_sample,
            temperature  = T,
            max_new_tokens  = cfg["max_new_tokens"],
            num_return_sequences = n,
        )
        inp_len = inp["input_ids"].shape[1]
        pred = tokenizer.batch_decode(seq_ids[:, inp_len:],
                                       skip_special_tokens=True)
        for q, a, p in zip(repeat_lst(q_lst, n), repeat_lst(a_lst, n), pred):
            out.append({
                "question": q,
                "answer": a,
                "pred": p,
                "extracted_ans": extract_ans(p),
                "is_correct": is_correct(p, a)
            })
    return out
# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, device = setup_dist()
    is_main_process = (rank == 0)
    is_dist = int(os.environ.get('WORLD_SIZE', 1)) > 1

    if is_main_process:
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # model
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=torch.bfloat16)
    if cfg.get("checkpoint_path"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["checkpoint_path"]).merge_and_unload()
        if is_main_process:
            print(f"LoRA merged: {cfg['checkpoint_path']}")
    model.to(device).eval()

    # dataset
    val_dataset = PromptDataset(cfg["data_p"], split=cfg.get("eval_split", "test"))
    if cfg.get("max_problems"):
        val_dataset = torch.utils.data.Subset(val_dataset, range(cfg["max_problems"]))

    if is_dist:
        val_dist_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg['batch_size'], collate_fn=collect_fn, sampler=val_dist_sampler, drop_last=True
        )
    else:
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg['batch_size'], collate_fn=collect_fn, shuffle=False, drop_last=True
        )

    if is_main_process:
        print(f"{len(val_dataset)} problems | {cfg['num_samples']} sample(s) | T={cfg['temperature']}\n")

    if is_dist:
        dist.barrier()
        
        
    result = evaluate(model, tokenizer, val_dataloader, cfg, device)

    if is_dist:
        gather_lst = [None for _ in range(dist.get_world_size())] if is_main_process else None
        dist.gather_object(result, gather_lst, dst=0)
        if is_main_process:
            result = [item for rank_data in gather_lst for item in rank_data]

    if is_main_process:
        with open(cfg["output_path"], "w") as f:
            json.dump({"config": cfg, "result": result}, f, indent=2)
        print(f"\nSaved (Distributed) → {cfg['output_path']}")
