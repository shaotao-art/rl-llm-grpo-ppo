import argparse
import os

import torch
import torch.distributed as dist
import yaml
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from core import build_prompt, get_generated_text_lst
from gsm_8k_dataset import cal_reward


def load_config(default_config="configs/default.yaml"):
    """Parse --config from cli, load the yaml and echo it (rank 0 only)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rank_zero_print(f"[config] loaded from {args.config}:")
    rank_zero_print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


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


def all_gather_cat(tensor, is_dist):
    """Gather a (…)-shaped tensor from all ranks and concat on dim 0 → (world*…,).
    Single-process run (is_dist=False) returns the tensor unchanged."""
    if not is_dist:
        return tensor
    lst = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(lst, tensor)
    return torch.cat(lst, dim=0)


def all_reduce_sum(tensor, is_dist):
    """Sum a scalar tensor across all ranks; result is meaningful only on rank 0.
    Single-process run (is_dist=False) returns the tensor unchanged."""
    if not is_dist:
        return tensor
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad
def evaluate(
    model,
    tokenizer,
    val_dataloader,
    max_new_tokens,
    is_dist,
    is_main_process,
    cal_reward_fn=None,
):
    reward_fn = cal_reward_fn if cal_reward_fn is not None else cal_reward
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
            reward_fn(gen_pred, gt) for gen_pred, gt in zip(generated_text_lst, a_lst)
        ]
        format_r_lst = [x[0] for x in reward_lst_]
        content_r_lst = [x[1] for x in reward_lst_]
        format_r_sum += sum(format_r_lst)
        content_r_sum += sum(content_r_lst)
        num_sample_sum += len(a_lst)
    format_r_sum = all_reduce_sum(format_r_sum, is_dist)
    content_r_sum = all_reduce_sum(content_r_sum, is_dist)
    num_sample_sum = all_reduce_sum(num_sample_sum, is_dist)
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


def load_model_tokenier(model_name, device=None):
    """Load tokenizer + model. device defaults to the current cuda device
    (setup_dist has already called torch.cuda.set_device by the time this runs).
    Returns (tokenizer, model)."""
    if device is None:
        device = (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch infer
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


def build_lora_model(model, lora_rank, lora_alpha, is_main_process=True):
    """Wrap a base model with LoRA adapters and print trainable params on rank 0."""
    peft_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, task_type="CAUSAL_LM")
    lora_model = get_peft_model(model, peft_config)
    if is_main_process:
        lora_model.print_trainable_parameters()
    return lora_model


def update_lr(opt, g_step, total_train_steps, max_lr, min_lr):
    """Step-wise linear decay from max_lr to min_lr over optimizer updates.
    Returns the current lr."""
    progress = min(g_step / max(1, total_train_steps), 1.0)
    cur_lr = max_lr - (max_lr - min_lr) * progress
    for p_g in opt.param_groups:
        p_g["lr"] = cur_lr
    return cur_lr


def log_gen_len_stats(
    logger, gen_len, max_new_tokens, g_step, is_dist, is_main_process
):
    """All-gather per-sample generated lengths across ranks and log stats to tensorboard."""
    gen_len = all_gather_cat(
        gen_len, is_dist
    ).float()  # cast: some callers pass int64 lengths
    if is_main_process:
        mean_gen_len = gen_len.mean().item()
        max_gen_len = gen_len.max().item()
        min_gen_len = gen_len.min().item()
        # a sequence is "clipped" when it used up the whole length budget (no natural EOS stop)
        gen_len_clip_frac = (
            torch.sum((gen_len == max_new_tokens).float()) / gen_len.shape[0]
        )
        logger.add_scalar("rollout/mean_gen_len", mean_gen_len, global_step=g_step)
        logger.add_scalar("rollout/max_gen_len", max_gen_len, global_step=g_step)
        logger.add_scalar("rollout/min_gen_len", min_gen_len, global_step=g_step)
        logger.add_scalar(
            "rollout/gen_len_clip_frac", gen_len_clip_frac, global_step=g_step
        )


def save_checkpoint(
    ckp_dir,
    lora_model,
    tokenizer,
    opt,
    ep,
    b_idx,
    g_step,
    is_dist,
    is_main_process,
    extra_state=None,
):
    """Save LoRA adapter + tokenizer + training state under ckp_dir/step_<g_step>.
    extra_state: optional dict merged into training_state.pt (e.g. {"micro_step": ...}).
    Returns ckp_path on the main process, None elsewhere."""
    if is_dist:
        dist.barrier()
    if is_main_process:
        ckp_path = os.path.join(ckp_dir, f"step_{g_step}")
        lora_model.save_pretrained(ckp_path)
        tokenizer.save_pretrained(ckp_path)
        state = {
            "epoch": ep,
            "b_idx": b_idx,
            "global_step": g_step,
            "optimizer_state_dict": opt.state_dict(),
        }
        if extra_state:
            state.update(extra_state)
        torch.save(state, os.path.join(ckp_path, "training_state.pt"))
        rank_zero_print(f"Checkpoint saved to {ckp_path}")
        return ckp_path
    return None


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


def resume_from_ckp(resume_from, ckp_dir, model, opt, device, load_fn=None):
    """Optionally resume from a checkpoint.

    load_fn: custom loader with the same signature as load_checkpoint (e.g. ppo's
        value-aware loader, partially bound with value_head/value_model).
    Returns (start_epoch, g_step, resume_skip_batches, state); state is None when
    not resuming — callers can pull extra counters from it, e.g. state["micro_step"].
    """
    start_epoch = 0
    g_step = 0
    resume_skip_batches = 0  # number of batches to skip at the start of start_epoch
    state = None
    if resume_from:
        loader = load_fn or load_checkpoint
        ckp_path = resolve_resume_path(resume_from, ckp_dir)
        rank_zero_print(f"[resume] loading checkpoint from {ckp_path}")
        state = loader(ckp_path, model, opt, device)
        g_step = state["global_step"]
        start_epoch = state["epoch"]
        resume_skip_batches = (
            state["b_idx"] + 1
        )  # continue at the batch after the saved one
        rank_zero_print(
            f"[resume] start_epoch={start_epoch}, skip first {resume_skip_batches} "
            f"batches, g_step={g_step}"
        )
    return start_epoch, g_step, resume_skip_batches, state


if __name__ == "__main__":
    a = "world"
    rank_zero_print("hello", a)

    steps = get_eval_steps(0.1, 98)
    print(steps)
