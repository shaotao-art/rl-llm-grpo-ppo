import os

import torch
import torch.distributed as dist
from peft import set_peft_model_state_dict
from tqdm import tqdm

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

def rank_zero_print(is_main_process, *args):
    if is_main_process:
        print(*args)
