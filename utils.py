import os

import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from core import build_prompt, get_generated_text_lst
from gsm_8k_dataset import cal_reward


def resolve_resume_path(resume_from, ckp_dir):
    """Turn the `resume_from` config value into a concrete checkpoint directory.
    - "latest": pick the step_<N> dir with the largest N under ckp_dir
    - an existing path (abs or relative): use as-is
    - otherwise: treat as a name under ckp_dir (e.g. "step_912")
    """
    if resume_from == "latest":
        candidates = []
        if os.path.isdir(ckp_dir):
            for name in os.listdir(ckp_dir):
                if name.startswith("step_") and os.path.isdir(
                    os.path.join(ckp_dir, name)
                ):
                    try:
                        candidates.append((int(name.split("_")[1]), name))
                    except (ValueError, IndexError):
                        pass
        if not candidates:
            raise FileNotFoundError(
                f"resume_from='latest' but no step_* checkpoint under {ckp_dir}"
            )
        return os.path.join(ckp_dir, max(candidates)[1])
    if os.path.isabs(resume_from) or os.path.exists(resume_from):
        return resume_from
    return os.path.join(ckp_dir, resume_from)


def load_checkpoint(ckp_path, lora_model, opt, device):
    """Load LoRA adapter weights + optimizer state + training counters from a checkpoint dir.
    Returns the saved training_state dict."""
    from safetensors.torch import load_file

    adapter_sd = load_file(
        os.path.join(ckp_path, "adapter_model.safetensors"), device=str(device)
    )
    set_peft_model_state_dict(lora_model, adapter_sd)
    state = torch.load(os.path.join(ckp_path, "training_state.pt"), map_location=device)
    opt.load_state_dict(state["optimizer_state_dict"])
    # make sure optimizer state tensors live on the current device
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    return state


def setup_dist():
    """return is_dist, world_size, local_rank, device"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return True, world_size, local_rank, f"cuda:{local_rank}"
    else:
        return False, 1, 0, "cuda:0"


@torch.no_grad
def evaluate(model, tokenizer, val_dataloader, max_new_tokens, is_dist, is_main_process):
    model.eval()
    device = next(model.parameters()).device
    format_r_sum = torch.tensor([0], device=device, dtype=torch.float)
    content_r_sum = torch.tensor([0], device=device, dtype=torch.float)
    num_sample_sum = torch.tensor([0], device=device, dtype=torch.float)
    for d_idx, (q_lst, a_lst) in enumerate(
        tqdm(val_dataloader, desc=f"val", total=len(val_dataloader))
    ):
        prompt_lst = build_prompt(tokenizer, q_lst)
        # tokenize
        model_inputs = tokenizer(
            prompt_lst, padding=True, padding_side="left", return_tensors="pt"
        ).to(model.device)
        inp_len = model_inputs["input_ids"].shape[1]
        seq_ids = model.generate(
            **model_inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
        generated_ids = seq_ids[:, inp_len:]
        generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)

        if is_main_process and d_idx == 0:
            for inp, pred, gt in zip(q_lst, generated_text_lst, a_lst):
                print(f">>> Input: {inp}")
                print(f">>> Generated: {pred}")
                print(f">>> Ground Truth: {gt}")
                print("-" * 50)

        reward_lst_ = [
            cal_reward(gen_pred, gt) for gen_pred, gt in zip(generated_text_lst, a_lst)
        ]
        format_r_lst = [x[0] for x in reward_lst_]
        content_r_lst = [x[1] for x in reward_lst_]
        format_r_sum += sum(format_r_lst)
        content_r_sum += sum(content_r_lst)
        num_sample_sum += len(a_lst)
    if is_dist:
        dist.reduce(format_r_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(content_r_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(num_sample_sum, dst=0, op=dist.ReduceOp.SUM)
    model.train()
    if is_main_process:
        mean_f_r = format_r_sum / num_sample_sum
        mean_c_r = content_r_sum / num_sample_sum
        return mean_f_r.item(), mean_c_r.item()
    else:
        return None, None

def rank_zero_print(*args):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1
    if is_dist:
        local_rank = int(os.environ["LOCAL_RANK"])
        is_main_process = local_rank == 0
        if is_main_process:
            print(*args)
    else:
        print(*args)


def load_model_tokenier(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch infer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer  


def get_eval_steps(eval_ratio, len_dataloader):
    if eval_ratio > 0:
        n_evals_per_epoch = max(1, round(1.0 / eval_ratio))
        _total_steps = len_dataloader
        eval_steps = {
            round(_total_steps * (i + 1) / n_evals_per_epoch)
            for i in range(n_evals_per_epoch)
        }
        eval_steps.add(_total_steps)  # always include last step of epoch
    else:
        eval_steps = set()
    if eval_steps:
        rank_zero_print(
            f"eval_steps per epoch: {sorted(eval_steps)} (total {len_dataloader} steps/epoch, eval_ratio={eval_ratio})"
        )
    return eval_steps

def resume_from_ckp(resume_from, ckp_dir, model, opt, device):
    start_epoch = 0
    resume_skip_batches = 0  # number of batches to skip at the start of start_epoch
    if resume_from:
        ckp_path = resolve_resume_path(resume_from, ckp_dir)
        rank_zero_print(f"[resume] loading checkpoint from {ckp_path}")
        state = load_checkpoint(ckp_path, model, opt, device)
        g_step = state["global_step"]
        start_epoch = state["epoch"]
        resume_skip_batches = (
            state["b_idx"] + 1
        )  # continue at the batch after the saved one
        rank_zero_print(
                f"[resume] start_epoch={start_epoch}, skip first {resume_skip_batches} "
                f"batches, g_step={g_step}"
            )
    return start_epoch, g_step, resume_skip_batches



if __name__ == "__main__":
    a = "world"
    rank_zero_print("hello", a)
    
    steps = get_eval_steps(0.1, 98)
    print(steps)
