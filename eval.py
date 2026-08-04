"""
eval.py — evaluate pass@1 and pass@k under different sampling configurations.
Supports single-GPU and multi-GPU (torchrun) inference.

Unbiased pass@k estimator (Chen et al. 2021):
    pass@k = 1 - C(n-c, k) / C(n, k)
where n = samples per problem, c = correct samples, k = pass@k target.

Usage:
    # single GPU
    python eval.py --model /path/to/model --num_samples 8 --temperatures 0.0 0.6 0.95

    # multi-GPU (4 GPUs), each rank evaluates its own slice of problems
    torchrun --nproc_per_node=4 eval.py --model /path/to/model --num_samples 8

    # LoRA checkpoint
    python eval.py --model /path/to/base --checkpoint /path/to/lora_ckpt --num_samples 8

    # yaml config
    python eval.py --config configs/eval.yaml
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


# ──────────────────────────────────────────────────────────────────────────────
# Distributed setup
# ──────────────────────────────────────────────────────────────────────────────

def setup_dist():
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, f"cuda:{local_rank}", True, dist.get_world_size()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return 0, device, False, 1


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    def __init__(self, data_p, split="test"):
        self.data = load_dataset(data_p, "main")
        self.split = split

    def __getitem__(self, index):
        sample = self.data[self.split][index]
        q, a = sample["question"], sample["answer"]
        prompt = f"""Answer the question in the following format:
```
## Reasoning
your reasoning process

## Answer
\\boxed{{your answer}}

```
# Question
{q}"""
        return prompt, a.split("####")[-1].strip()

    def __len__(self):
        return len(self.data[self.split])


class SubsetDataset(Dataset):
    def __init__(self, base, n):
        self.base = base
        self.n = min(n, len(base))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.base[idx]


def collect_fn(batch):
    return [item[0] for item in batch], [item[1] for item in batch]


# ──────────────────────────────────────────────────────────────────────────────
# Contiguous distributed sampler
# ──────────────────────────────────────────────────────────────────────────────

class ContiguousDistributedSampler(Sampler):
    """
    Each rank gets a contiguous slice of indices: [r*chunk, (r+1)*chunk).
    No interleaving and no padding — concatenating results from rank 0..W-1
    reconstructs the full dataset in order.
    """
    def __init__(self, dataset, rank, world_size):
        n = len(dataset)
        chunk = math.ceil(n / world_size)
        self.indices = list(range(rank * chunk, min((rank + 1) * chunk, n)))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def build_prompt(tokenizer, text_lst):
    result = []
    for text in text_lst:
        messages = [{"role": "user", "content": text}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        result.append(formatted)
    return result


def extract_ans(text):
    match = re.search(r"\\boxed{(.*?)}", text)
    return match.group(1).strip() if match else ""


def is_correct(pred, gt):
    if "## Answer" not in pred:
        return False
    return extract_ans(pred.split("## Answer")[-1]).strip() == gt.strip()


def has_format(pred):
    return "## Reasoning" in pred and "## Answer" in pred


def repeat_lst(lst, n):
    return [item for item in lst for _ in range(n)]


def pass_at_k(n, c, k):
    """Unbiased pass@k estimator (Chen et al. 2021)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


# ──────────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_samples(model, tokenizer, prompts, num_samples, max_new_tokens,
                     temperature, top_p, gen_batch_size, device):
    """Generate num_samples completions per prompt. Returns list[list[str]]."""
    expanded = repeat_lst(prompts, num_samples)
    all_texts = []
    for start in range(0, len(expanded), gen_batch_size):
        batch = expanded[start: start + gen_batch_size]
        inputs = tokenizer(
            batch, padding=True, padding_side="left", return_tensors="pt",
        ).to(device)
        inp_len = inputs["input_ids"].shape[1]
        greedy = (temperature == 0.0)
        if greedy and num_samples > 1:
            # greedy is deterministic; warn once per run (handled at call site)
            pass
        seq_ids = model.generate(
            **inputs,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            top_k=0,
            max_new_tokens=max_new_tokens,
        )
        texts = tokenizer.batch_decode(seq_ids[:, inp_len:], skip_special_tokens=True)
        all_texts.extend(texts)
    return [all_texts[i * num_samples: (i + 1) * num_samples] for i in range(len(prompts))]


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop for one sampling configuration
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sampling_config(
    model, tokenizer, dataloader, num_samples, max_new_tokens,
    temperature, top_p, pass_k_values, gen_batch_size, device,
    rank, is_dist, is_main_process,
):
    """
    Each rank evaluates its own slice of problems, then all_gather_object
    collects results on every rank. Returns the merged result dict on rank 0,
    None on other ranks.
    """
    if is_main_process and temperature == 0.0 and num_samples > 1:
        print(f"  [note] greedy decoding (T=0) with num_samples={num_samples}: "
              f"all samples are identical, pass@k > pass@1 won't improve.")

    model.eval()
    per_problem_correct = []
    per_problem_fmt     = []

    for batch_idx, (q_lst, a_lst) in enumerate(
        tqdm(dataloader, desc=f"rank{rank} T={temperature:.2f}", disable=not is_main_process)
    ):
        formatted = build_prompt(tokenizer, q_lst)
        sample_groups = generate_samples(
            model, tokenizer, formatted, num_samples, max_new_tokens,
            temperature, top_p, gen_batch_size, device,
        )

        if is_main_process and batch_idx == 0:
            print(f"\n[sample output T={temperature}]")
            for q, preds, gt in zip(q_lst[:2], sample_groups[:2], a_lst[:2]):
                print(f"  Q  : {q[:80]}...")
                print(f"  GT : {gt}")
                print(f"  P0 : {preds[0][:100]}...")
            print()

        for preds, gt in zip(sample_groups, a_lst):
            per_problem_correct.append(sum(is_correct(p, gt) for p in preds))
            per_problem_fmt.append(sum(has_format(p) for p in preds))

    # ── gather from all ranks ──────────────────────────────────────────────
    if is_dist:
        # all_gather_object handles variable-length lists (last rank may have fewer items)
        gathered_correct = [None] * dist.get_world_size()
        gathered_fmt     = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_correct, per_problem_correct)
        dist.all_gather_object(gathered_fmt,     per_problem_fmt)
        # ContiguousDistributedSampler ensures concatenation = full dataset in order
        all_correct = [x for sub in gathered_correct for x in sub]
        all_fmt     = [x for sub in gathered_fmt     for x in sub]
    else:
        all_correct = per_problem_correct
        all_fmt     = per_problem_fmt

    if not is_main_process:
        return None

    n_problems = len(all_correct)
    n = num_samples
    return {
        "temperature":            temperature,
        "top_p":                  top_p,
        "num_samples_per_problem": n,
        "num_problems":           n_problems,
        "pass@1":     sum(pass_at_k(n, c, 1) for c in all_correct) / n_problems,
        "format_acc": sum(pass_at_k(n, f, 1) for f in all_fmt)     / n_problems,
        "pass@k": {
            k: sum(pass_at_k(n, c, k) for c in all_correct) / n_problems
            for k in pass_k_values if k <= n
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name, checkpoint_path, device, is_main_process):
    if is_main_process:
        print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if checkpoint_path:
        if is_main_process:
            print(f"Loading LoRA adapter: {checkpoint_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        if is_main_process:
            print("LoRA merged into base model.")
    model.to(device)
    model.eval()
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        type=str,   default=None)
    p.add_argument("--model",         type=str,   default=None)
    p.add_argument("--checkpoint",    type=str,   default=None,
                   help="LoRA checkpoint dir (optional)")
    p.add_argument("--data_p",        type=str,   default=None)
    p.add_argument("--eval_split",    type=str,   default=None)
    p.add_argument("--num_samples",   type=int,   default=None,
                   help="n samples per problem (must be >= max pass_k_values)")
    p.add_argument("--max_new_tokens",type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None,
                   help="problems per DataLoader batch")
    p.add_argument("--gen_batch_size",type=int,   default=None,
                   help="generation mini-batch size to avoid OOM")
    p.add_argument("--temperatures",  type=float, nargs="+", default=None,
                   help="temperatures to sweep; 0.0 = greedy")
    p.add_argument("--top_p",         type=float, default=None)
    p.add_argument("--pass_k_values", type=int,   nargs="+", default=None)
    p.add_argument("--max_problems",  type=int,   default=None,
                   help="limit eval to first N problems (default: all)")
    p.add_argument("--output",        type=str,   default=None,
                   help="path to save JSON results")
    return p.parse_args()


def build_cfg(args):
    cfg = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    defaults = dict(
        model_name       = "/tmp/model_cache/Qwen3-0.6B",
        checkpoint_path  = None,
        data_p           = "data/gsm8k",
        eval_split       = "test",
        num_samples      = 8,
        max_new_tokens   = 512,
        batch_size       = 16,
        gen_batch_size   = 16,
        temperatures     = [0.0, 0.6, 0.95, 1.2],
        top_p            = 1.0,
        pass_k_values    = [1, 2, 4, 8],
        max_problems     = None,
        output_path      = "./eval_results.json",
    )
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    if args.model:             cfg["model_name"]      = args.model
    if args.checkpoint:        cfg["checkpoint_path"] = args.checkpoint
    if args.data_p:            cfg["data_p"]          = args.data_p
    if args.eval_split:        cfg["eval_split"]      = args.eval_split
    if args.num_samples:       cfg["num_samples"]     = args.num_samples
    if args.max_new_tokens:    cfg["max_new_tokens"]  = args.max_new_tokens
    if args.batch_size:        cfg["batch_size"]      = args.batch_size
    if args.gen_batch_size:    cfg["gen_batch_size"]  = args.gen_batch_size
    if args.temperatures:      cfg["temperatures"]    = args.temperatures
    if args.top_p is not None: cfg["top_p"]           = args.top_p
    if args.pass_k_values:     cfg["pass_k_values"]   = args.pass_k_values
    if args.max_problems:      cfg["max_problems"]    = args.max_problems
    if args.output:            cfg["output_path"]     = args.output
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rank, device, is_dist, world_size = setup_dist()
    is_main_process = (rank == 0)

    args = parse_args()
    cfg  = build_cfg(args)

    if is_main_process:
        print("=" * 60)
        print(f"Eval config  (world_size={world_size}, device={device}):")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
        print("=" * 60)

    model, tokenizer = load_model(
        cfg["model_name"], cfg["checkpoint_path"], device, is_main_process
    )

    # dataset & dataloader
    base_dataset = PromptDataset(cfg["data_p"], split=cfg["eval_split"])
    eval_dataset = SubsetDataset(base_dataset, cfg["max_problems"]) \
                   if cfg["max_problems"] else base_dataset

    if is_dist:
        sampler    = ContiguousDistributedSampler(eval_dataset, rank=rank, world_size=world_size)
        dataloader = DataLoader(eval_dataset, batch_size=cfg["batch_size"],
                                collate_fn=collect_fn, sampler=sampler)
    else:
        dataloader = DataLoader(eval_dataset, batch_size=cfg["batch_size"],
                                collate_fn=collect_fn, shuffle=False)

    if is_main_process:
        n_per_rank = math.ceil(len(eval_dataset) / world_size)
        print(f"Dataset: {len(eval_dataset)} problems  |  "
              f"~{n_per_rank} per rank  |  "
              f"{cfg['num_samples']} samples each\n")

    n = cfg["num_samples"]
    pass_k_values = [k for k in cfg["pass_k_values"] if k <= n]
    if is_main_process and len(pass_k_values) < len(cfg["pass_k_values"]):
        skipped = [k for k in cfg["pass_k_values"] if k > n]
        print(f"[warn] skipping pass@k for k={skipped} (> num_samples={n})")

    all_results = []

    for temp in cfg["temperatures"]:
        if is_dist:
            dist.barrier()   # synchronise before each temperature sweep

        if is_main_process:
            print(f"\n{'─'*60}")
            print(f"temperature={temp}  top_p={cfg['top_p']}")

        result = evaluate_sampling_config(
            model=model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            num_samples=n,
            max_new_tokens=cfg["max_new_tokens"],
            temperature=temp,
            top_p=cfg["top_p"],
            pass_k_values=pass_k_values,
            gen_batch_size=cfg["gen_batch_size"],
            device=device,
            rank=rank,
            is_dist=is_dist,
            is_main_process=is_main_process,
        )

        if is_main_process and result is not None:
            all_results.append(result)
            print(f"  format_acc : {result['format_acc']:.4f}")
            print(f"  pass@1     : {result['pass@1']:.4f}")
            for k, v in sorted(result["pass@k"].items()):
                print(f"  pass@{k:<4d}  : {v:.4f}")

    # ── summary table & save ───────────────────────────────────────────────
    if is_main_process:
        cols = sorted(pass_k_values)
        header  = f"{'temp':>6}  {'top_p':>5}  {'format':>7}  {'pass@1':>7}"
        header += "".join(f"  {'pass@'+str(k):>8}" for k in cols)

        print(f"\n{'='*60}\nSummary\n{'='*60}")
        print(header)
        print("─" * len(header))
        for r in all_results:
            row  = f"{r['temperature']:>6.2f}  {r['top_p']:>5.2f}  "
            row += f"{r['format_acc']:>7.4f}  {r['pass@1']:>7.4f}"
            row += "".join(f"  {r['pass@k'].get(k, float('nan')):>8.4f}" for k in cols)
            print(row)

        out_path = cfg["output_path"]
        out_dir  = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"config": cfg, "results": all_results}, f, indent=2)
        print(f"\nResults saved to {out_path}")

    if is_dist:
        dist.destroy_process_group()
