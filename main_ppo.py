"""
main_ppo.py — PPO (Proximal Policy Optimization) for LLM fine-tuning.

Architecture differences vs main.py (GRPO):
  - ValueHead:  a linear probe on LM hidden states to estimate V(s_t) per token
  - GAE:        Generalized Advantage Estimation instead of group-relative normalization
  - Reward:     per-token KL penalty (InstructGPT style) + terminal scalar reward at EOS
  - Loss:       pg_loss + vf_coef * value_loss - entropy_coef * entropy

Usage:
    python main_ppo.py --config configs/ppo_default.yaml
    torchrun --nproc_per_node=4 main_ppo.py --config configs/ppo_default.yaml
"""

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import re
from contextlib import nullcontext
import torch.distributed as dist
import argparse
import yaml
import contextlib

# ──────────────────────────────────────────────────────────────────────────────
# Distributed setup  (identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────


def setup_dist():
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, f"cuda:{local_rank}"
    return 0, "cuda:0"


# ──────────────────────────────────────────────────────────────────────────────
# Value head
# ──────────────────────────────────────────────────────────────────────────────


class ValueHead(nn.Module):
    """Per-token scalar value estimator (single linear layer on LM hidden states)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.linear.weight)  # start from V ≈ 0

    def forward(self, hidden_states):
        # hidden_states: (..., hidden_size)  →  (...,)
        return self.linear(hidden_states).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset / prompt helpers  (identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────


def build_prompt(tokenizer, text_lst: list):
    prompt_lst = []
    for prompt in text_lst:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_lst.append(text)
    return prompt_lst


def repeat_lst(lst, num_gen):
    return [item for item in lst for _ in range(num_gen)]


def get_generated_text_lst(generated_ids, tokenizer):
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


class PromptDataset(Dataset):
    def __init__(self, data_p, split="train"):
        super().__init__()
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


def collect_fn(batch):
    return [_[0] for _ in batch], [_[1] for _ in batch]


def extract_ans(text):
    match = re.search(r"\\boxed{(.*?)}", text)
    return match.group(1).strip() if match else ""


def cal_reward(pred: str, gt: str):
    if "## Reasoning" not in pred or "## Answer" not in pred:
        return 0, 0
    extracted = extract_ans(pred.split("## Answer")[-1])
    return 1, int(extracted.strip() == gt.strip())


def make_loss_mask(x, pad_token_id):
    N, l = x.shape
    mask = x != pad_token_id
    for i in range(N):
        found = False
        for j in range(l - 1, 0, -1):
            if j == l - 1 and x[i][j] != pad_token_id:
                found = True
                break
            if x[i][j] == pad_token_id and x[i][j - 1] != pad_token_id:
                mask[i, j] = True
                found = True
                break
        if not found:
            mask[i, 0] = True
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Log-prob helpers  (identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────


def get_log_probs(logits, generated_ids, temperature):
    """logits (N,T,V), generated_ids (N,T) → log_probs (N,T)"""
    logits = logits / temperature
    log_probs_all = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs_all, -1, generated_ids.unsqueeze(-1)).squeeze(-1)


def forward_get_log_probs(
    sequence_ids, model, attn_mask, inp_len, generated_ids, temperature
):
    """Full-sequence forward, returns log_prob for generated positions only."""
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)
    logits = model(sequence_ids, attention_mask=attn_mask, position_ids=pos_ids).logits
    gen_logits = logits[:, inp_len - 1 : -1]  # (N, gen_len, V)
    return get_log_probs(gen_logits, generated_ids, temperature)


# ──────────────────────────────────────────────────────────────────────────────
# Rollout  (generation + old log probs; identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout(model_inputs, model, max_new_tokens, temperature, rollout_bs, pad_token_id):
    model.eval()
    device = next(model.parameters()).device
    N, inp_len = model_inputs["attention_mask"].shape
    all_seq_ids = torch.full(
        (N, inp_len + max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    all_gen_ids = torch.full(
        (N, max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    all_lp_old = torch.zeros((N, max_new_tokens), dtype=torch.float, device=device)
    assert N % rollout_bs == 0
    for i in range(N // rollout_bs):
        s, e = i * rollout_bs, (i + 1) * rollout_bs
        batch = {k: v[s:e] for k, v in model_inputs.items()}
        out = model.generate(
            **batch,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
        )
        seq_ids = out.sequences
        logits_old = torch.stack(out.logits, dim=1)  # (bs, actual_gen, V)
        gen_ids = seq_ids[:, inp_len:]
        lp_old = get_log_probs(logits_old, gen_ids, temperature)
        del logits_old
        actual = gen_ids.shape[1]
        all_seq_ids[s:e, : inp_len + actual] = seq_ids
        all_gen_ids[s:e, :actual] = gen_ids
        all_lp_old[s:e, :actual] = lp_old
    model.train()
    return {
        "sequence_ids": all_seq_ids,
        "log_probs_old": all_lp_old,
        "generated_ids": all_gen_ids,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Value rollout  (separate no-grad forward to get V(s_0)..V(s_T) for GAE)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout_values(
    sequence_ids, lm_model, value_head, seq_mask, inp_len, gen_len, rollout_bs
):
    """
    Returns old_values: (N, gen_len)
      col t   = V(s_t) for t = 0..gen_len-1  (state before generating token t)
      col T   = V(s_T) = bootstrap value after the full generated sequence
    """
    lm_model.eval()
    device = sequence_ids.device
    N = sequence_ids.shape[0]
    values = torch.zeros(N, gen_len, device=device)

    for i in range(N // rollout_bs):
        s, e = i * rollout_bs, (i + 1) * rollout_bs
        batch_seq = sequence_ids[s:e]
        batch_mask = seq_mask[s:e]
        pos_ids = batch_mask.long().cumsum(-1) - 1
        pos_ids.masked_fill_(batch_mask == 0, 1)
        out = lm_model(
            batch_seq,
            attention_mask=batch_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]  # (bs, inp_len + gen_len, H)
        # positions inp_len-1 .. inp_len+gen_len-1 → states s_0 .. s_T
        vals = value_head(hidden[:, inp_len - 1: -1])  # (bs, gen_len)
        values[s:e] = vals

    lm_model.train()
    return values  # (N, gen_len)


# ──────────────────────────────────────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────────────────────────────────────


def compute_gae(per_token_rewards, values, gamma, gae_lambda, gen_mask):
    """
    per_token_rewards : (N, T)  — reward at each gen position (0 except EOS + KL)
    values            : (N, T)  — V(s_0)..V(s_T)
    Returns:
        advantages (N, T),  returns (N, T)
    """
    N, T = per_token_rewards.shape
    adv = torch.zeros(N, T, device=per_token_rewards.device)
    last_g = torch.zeros(N, device=per_token_rewards.device)
    last_idx_mask = gen_mask[:, -2:-1]
    last_values = values[:, -2:-1] * last_idx_mask
    values = torch.cat([values, last_values], dim=1)  # (N, gen_len + 1)

    for t in reversed(range(T)):
        delta = per_token_rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        last_g = delta + gamma * gae_lambda * last_g
        adv[:, t] = last_g

    returns = adv + values[:, :T]
    return adv, returns


# ──────────────────────────────────────────────────────────────────────────────
# Combined policy + value forward (used during PPO training)
# ──────────────────────────────────────────────────────────────────────────────


def forward_ppo(
    sequence_ids, lm_model, value_head, attn_mask, inp_len, generated_ids, temperature
):
    """
    Single forward pass returning:
        log_prob : (N, gen_len)
        values   : (N, gen_len)   — col gen_len is the bootstrap value
    """
    pos_ids = attn_mask.long().cumsum(-1) - 1
    pos_ids.masked_fill_(attn_mask == 0, 1)
    out = lm_model(
        sequence_ids,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        output_hidden_states=True,
    )
    logits = out.logits
    hidden = out.hidden_states[-1]  # (N, inp_len+gen_len, H)

    gen_logits = logits[:, inp_len - 1 : -1]  # (N, gen_len, V)
    log_prob = get_log_probs(gen_logits, generated_ids, temperature)  # (N, gen_len)
    values = value_head(hidden[:, inp_len - 1 : -1])  # (N, gen_len)

    return log_prob, values


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation  (identical to main.py)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def evaluate(model, tokenizer, val_dataloader, max_new_tokens):
    model.eval()
    device = next(model.parameters()).device
    fmt_sum = torch.tensor([0.0], device=device)
    cnt_sum = torch.tensor([0.0], device=device)
    n_sum = torch.tensor([0.0], device=device)

    for d_idx, (q_lst, a_lst) in enumerate(
        tqdm(val_dataloader, desc="val", total=len(val_dataloader))
    ):
        inputs = tokenizer(
            build_prompt(tokenizer, q_lst),
            padding=True,
            padding_side="left",
            return_tensors="pt",
        ).to(device)
        inp_len = inputs["input_ids"].shape[1]
        seq_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=max_new_tokens
        )
        texts = get_generated_text_lst(seq_ids[:, inp_len:], tokenizer)

        if is_main_process and d_idx == 0:
            for inp, pred, gt in zip(q_lst, texts, a_lst):
                print(f">>> Input: {inp}")
                print(f">>> Generated: {pred}")
                print(f">>> GT: {gt}")
                print("-" * 50)

        rewards = [cal_reward(p, a) for p, a in zip(texts, a_lst)]
        fmt_sum += sum(r[0] for r in rewards)
        cnt_sum += sum(r[1] for r in rewards)
        n_sum += len(a_lst)

    if is_dist:
        dist.reduce(fmt_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(cnt_sum, dst=0, op=dist.ReduceOp.SUM)
        dist.reduce(n_sum, dst=0, op=dist.ReduceOp.SUM)
    model.train()
    if is_main_process:
        return (fmt_sum / n_sum).item(), (cnt_sum / n_sum).item()
    return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────


def resolve_resume_path(resume_from, ckp_dir):
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
            raise FileNotFoundError(f"No step_* checkpoint under {ckp_dir}")
        return os.path.join(ckp_dir, max(candidates)[1])
    if os.path.isabs(resume_from) or os.path.exists(resume_from):
        return resume_from
    return os.path.join(ckp_dir, resume_from)


def load_checkpoint(ckp_path, lora_model, value_head, opt, device):
    from safetensors.torch import load_file

    adapter_sd = load_file(
        os.path.join(ckp_path, "adapter_model.safetensors"), device=str(device)
    )
    set_peft_model_state_dict(lora_model, adapter_sd)
    state = torch.load(os.path.join(ckp_path, "training_state.pt"), map_location=device)
    opt.load_state_dict(state["optimizer_state_dict"])
    value_head.load_state_dict(state["value_head_state_dict"])
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo_default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── config ────────────────────────────────────────────────────────────────
    model_name = cfg["model_name"]
    data_p = cfg["data_p"]
    lora_rank = cfg["lora_rank"]
    lora_alpha = cfg["lora_alpha"]
    max_lr = cfg["max_lr"]
    min_lr = cfg["min_lr"]
    weight_decay = cfg["weight_decay"]
    max_grad_norm = cfg["max_grad_norm"]
    save_root = cfg["save_root"]
    temperature = cfg["temperature"]
    max_new_tokens = cfg["max_new_tokens"]
    w_format_r = cfg["w_format_r"]
    use_ref_model = cfg["use_ref_model"]
    kl_weight = cfg["kl_weight"]
    # PPO-specific
    vf_coef = cfg.get("vf_coef", 0.1)
    gamma = cfg.get("gamma", 1.0)
    gae_lambda = cfg.get("gae_lambda", 0.95)
    entropy_coef = cfg.get("entropy_coef", 0.0)
    eps_clip = cfg.get("eps", 0.2)
    value_clip_eps = cfg.get("value_clip_eps", 0.2)
    normalize_adv = cfg.get("normalize_adv", True)
    # training sizes
    train_prompt_size = cfg["train_prompt_size"]
    num_gen = cfg["num_gen"]
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]
    ppo_train_mini_bs = cfg["ppo_train_mini_bs"]
    ppo_num_epoch = cfg["ppo_num_epoch"]
    rollout_mini_bs = cfg["rollout_mini_bs"]
    val_bs = cfg["val_bs"]
    num_epoch = cfg["num_epoch"]
    eval_ratio = cfg.get("eval_ratio", 0.0)
    resume_from = cfg.get("resume_from", None)

    # ── dist ──────────────────────────────────────────────────────────────────
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, device = setup_dist()
    is_main_process = rank == 0

    if is_main_process:
        print(f"[config] {args.config}:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # ── model ─────────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    peft_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, task_type="CAUSAL_LM")
    lora_model = get_peft_model(model, peft_config)
    if is_main_process:
        lora_model.print_trainable_parameters()

    value_head = ValueHead(model.config.hidden_size).to(device, dtype=model.dtype)

    ref_model = None
    if use_ref_model:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── DDP ───────────────────────────────────────────────────────────────────
    ddp_model = None
    ddp_value_head = None
    if is_dist:
        ddp_model = DDP(lora_model, device_ids=[rank], output_device=rank)
        ddp_value_head = DDP(value_head, device_ids=[rank], output_device=rank)
        opt = optim.AdamW(
            list(ddp_model.parameters()) + list(ddp_value_head.parameters()),
            lr=max_lr,
            weight_decay=weight_decay,
        )
    else:
        opt = optim.AdamW(
            list(lora_model.parameters()) + list(value_head.parameters()),
            lr=max_lr,
            weight_decay=weight_decay,
        )

    if is_main_process:
        logger = SummaryWriter(save_root)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_dataset = PromptDataset(data_p)
    if is_dist:
        train_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_prompt_size,
            collate_fn=collect_fn,
            sampler=train_sampler,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_prompt_size,
            collate_fn=collect_fn,
            shuffle=True,
            drop_last=True,
        )

    if eval_ratio > 0:
        n_evals = max(1, round(1.0 / eval_ratio))
        _tot = len(train_dataloader)
        eval_steps = {round(_tot * (i + 1) / n_evals) for i in range(n_evals)}
        eval_steps.add(_tot)
    else:
        eval_steps = set([len(train_dataloader)])

    val_dataset = PromptDataset(data_p, split="test")
    if is_dist:
        val_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            collate_fn=collect_fn,
            sampler=val_sampler,
            drop_last=True,
        )
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            collate_fn=collect_fn,
            shuffle=False,
            drop_last=True,
        )

    ckp_dir = os.path.join(save_root, "checkpoints")
    if is_main_process:
        os.makedirs(ckp_dir, exist_ok=True)

    opt.zero_grad()
    g_step = 0
    micro_step = 0

    samples_per_rollout = num_gen * train_prompt_size
    micro_per_rollout = ppo_num_epoch * (samples_per_rollout // ppo_train_mini_bs)
    total_micro_steps = num_epoch * len(train_dataloader) * micro_per_rollout
    total_train_steps = total_micro_steps // gradient_accumulation_steps
    if is_main_process:
        print(f"total optimizer updates: {total_train_steps}")

    start_epoch = 0
    resume_skip_batches = 0
    if resume_from:
        ckp_path = resolve_resume_path(resume_from, ckp_dir)
        if is_main_process:
            print(f"[resume] {ckp_path}")
        state = load_checkpoint(ckp_path, lora_model, value_head, opt, device)
        g_step = state["global_step"]
        micro_step = state["micro_step"]
        start_epoch = state["epoch"]
        resume_skip_batches = state["b_idx"] + 1

    # if not resume_from:
    #     val_f_r, val_c_r = evaluate(
    #         lora_model, tokenizer, val_dataloader, max_new_tokens
    #     )
    #     if is_main_process:
    #         print(f"init eval: val_f_r={val_f_r:.4f}  val_c_r={val_c_r:.4f}")
    #         logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
    #         logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

    # ── Training ──────────────────────────────────────────────────────────────
    for ep in range(start_epoch, num_epoch):
        if is_dist:
            train_dataloader.sampler.set_epoch(ep)

        for b_idx, (q_lst, a_lst) in enumerate(
            tqdm(
                train_dataloader,
                desc=f"ep[{ep}/{num_epoch}]",
                total=len(train_dataloader),
            )
        ):
            if ep == start_epoch and b_idx < resume_skip_batches:
                continue

            prompt_lst = build_prompt(tokenizer, q_lst)
            prompt_lst_expand = repeat_lst(prompt_lst, num_gen)
            a_lst_expand = repeat_lst(a_lst, num_gen)

            model_inputs = tokenizer(
                prompt_lst_expand,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(device)
            N, inp_len = model_inputs["attention_mask"].shape

            # ── Rollout: generate + old log probs ─────────────────────────────
            rollout_res = rollout(
                model_inputs,
                lora_model,
                max_new_tokens,
                temperature,
                rollout_mini_bs,
                tokenizer.pad_token_id,
            )
            sequence_ids = rollout_res["sequence_ids"]  # (N, inp_len + gen_len)
            generated_ids = rollout_res["generated_ids"]  # (N, gen_len)
            log_probs_old = rollout_res["log_probs_old"]  # (N, gen_len)

            seq_mask = make_loss_mask(
                sequence_ids, tokenizer.pad_token_id
            )  # (N, inp_len + gen_len)
            gen_mask = make_loss_mask(generated_ids, tokenizer.pad_token_id) # cover <eos> 0 -> inp_len
            real_len = gen_mask.sum(dim=-1)  # (N,)
            gen_len = gen_mask.shape[-1]

            # log gen-length stats
            if is_dist:
                rl_g = [
                    torch.zeros_like(real_len) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(rl_g, real_len)
                real_len_all = torch.cat(rl_g)
            else:
                real_len_all = real_len
            if is_main_process:
                logger.add_scalar(
                    "rollout/mean_gen_len", real_len_all.float().mean().item(), g_step
                )
                logger.add_scalar(
                    "rollout/gen_len_clip_frac",
                    (real_len_all == gen_len).float().mean().item(), # TODO: x x x <eos> last token == <eos> should not be counted
                    g_step,
                )

            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)

            # ── Compute rewards ───────────────────────────────────────────────
            with torch.no_grad():
                reward_lst_ = [
                    cal_reward(p, gt) for p, gt in zip(generated_text_lst, a_lst_expand)
                ]
                format_r_lst = [r[0] for r in reward_lst_]
                content_r_lst = [r[1] for r in reward_lst_]
                scalar_reward = torch.tensor(
                    [f * w_format_r + c for f, c in zip(format_r_lst, content_r_lst)],
                    device=device,
                    dtype=torch.float,
                )  # (N,)

                # log scalar reward
                f_sum = torch.tensor([float(sum(format_r_lst))], device=device)
                c_sum = torch.tensor([float(sum(content_r_lst))], device=device)
                n_t = torch.tensor([float(N)], device=device)
                if is_dist:
                    dist.reduce(f_sum, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(c_sum, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(n_t, dst=0, op=dist.ReduceOp.SUM)
                if is_main_process:
                    logger.add_scalar("rollout/f_r", (f_sum / n_t).item(), g_step)
                    logger.add_scalar("rollout/c_r", (c_sum / n_t).item(), g_step)
                    
                # log rollout reward std as proxy for advantage spread
                if is_dist:
                    scalar_reward_lst = [torch.zeros_like(scalar_reward) for _ in range(dist.get_world_size())]
                    dist.all_gather(scalar_reward_lst, scalar_reward)
                    scalar_reward_log = torch.cat(scalar_reward_lst)
                else:
                    scalar_reward_log = scalar_reward
                if is_main_process:
                    logger.add_scalar(
                        "rollout/reward_mean", scalar_reward_log.mean().item(), g_step
                    )
                    logger.add_scalar(
                        "rollout/reward_std",
                        scalar_reward_log.std(unbiased=False).item(),
                        g_step,
                    )

                # Per-token rewards: KL penalty at every token + scalar reward at EOS
                # r_t = -kl_weight * KL(policy||ref)_t
                # r_{t_eos} += scalar_reward
                per_token_rewards = torch.zeros(N, gen_len, device=device) # (N, gen_len)

                if use_ref_model:
                    ref_log_prob = forward_get_log_probs(
                        sequence_ids,
                        ref_model,
                        seq_mask,
                        inp_len,
                        generated_ids,
                        temperature,
                    )  # (N, gen_len) first -> inp_len - 1
                    per_token_kl = (log_probs_old - ref_log_prob) * gen_mask  # KL ≥ 0 on average
                    per_token_rewards = -kl_weight * per_token_kl # (N, gen_len) first -> inp_len - 1

                # Add scalar reward at the last real token of each sample
                last_real_idx = (gen_mask.sum(dim=-1) - 1).long().clamp(min=0)  # (N,) -> <eos>
                per_token_rewards[torch.arange(N, device=device), last_real_idx] += scalar_reward
                per_token_rewards = per_token_rewards * gen_mask # mask out padding

            # ── Old values (no grad) ──────────────────────────────────────────
            with torch.no_grad():
                old_values = rollout_values(
                    sequence_ids,
                    lora_model,
                    value_head,
                    seq_mask,
                    inp_len,
                    gen_len,
                    rollout_mini_bs,
                )  # (N, gen_len) 0 -> inp_len
                old_values = old_values * gen_mask # mask out padding

            # ── GAE ───────────────────────────────────────────────────────────
            with torch.no_grad():
                advantages, returns = compute_gae(
                    per_token_rewards, old_values, gamma, gae_lambda, gen_mask
                )
                # (N, gen_len), (N, gen_len)

                # Normalize advantages over the full rollout batch (masked positions excluded)
                if normalize_adv:
                    if is_dist:
                        advantages_lst = [torch.zeros_like(advantages) for _ in range(dist.get_world_size())]
                        gen_mask_lst = [torch.zeros_like(gen_mask) for _ in range(dist.get_world_size())]
                        dist.all_gather(advantages_lst, advantages)
                        dist.all_gather(gen_mask_lst, gen_mask)
                        advantages_to_cal_mean_std = torch.cat(advantages_lst)[torch.cat(gen_mask_lst)]
                    else:
                        advantages_to_cal_mean_std = advantages[gen_mask]
                    adv_mean = advantages_to_cal_mean_std.mean()
                    adv_std = advantages_to_cal_mean_std.std(unbiased=False)
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                    advantages = advantages * gen_mask # (N, gen_len)

                adv_for_train = advantages.unsqueeze(-1)  # (N, T, 1) — broadcast with mask


            # ── PPO training epochs ────────────────────────────────────────────
            total_samples = N
            assert total_samples % ppo_train_mini_bs == 0
            world_size = 1 if not is_dist else dist.get_world_size()
            assert (total_samples * ppo_num_epoch) % (
                ppo_train_mini_bs * gradient_accumulation_steps
            ) == 0

            for ppo_epoch in range(ppo_num_epoch):
                batch_indices = torch.randperm(total_samples, device=device)
                num_batches = total_samples // ppo_train_mini_bs

                for mb_idx in range(num_batches):
                    # step-wise linear LR decay
                    progress = min(g_step / max(1, total_train_steps), 1.0)
                    cur_lr = max_lr - (max_lr - min_lr) * progress
                    for pg in opt.param_groups:
                        pg["lr"] = cur_lr

                    idx = batch_indices[
                        mb_idx * ppo_train_mini_bs : (mb_idx + 1) * ppo_train_mini_bs
                    ]

                    mb_seq_ids = sequence_ids[idx]  # (mb, inp_len + gen_len)
                    mb_gen_ids = generated_ids[idx]  # (mb, gen_len)
                    mb_seq_mask = seq_mask[idx]  # (mb, inp_len + gen_len)
                    mb_lp_old = log_probs_old[idx]  # (mb, gen_len)
                    mb_adv = adv_for_train[idx]  # (mb, gen_len, 1)
                    mb_returns = returns[idx]  # (mb, gen_len)
                    mb_val_old = old_values[idx, :gen_len]  # (mb, gen_len)

                    # token mask for loss computation
                    mask = make_loss_mask(mb_gen_ids, tokenizer.pad_token_id).to(
                        mb_lp_old.dtype
                    )  # (mb, gen_len)

                    # ── Forward ──────────────────────────────────────────────
                    active_lm = ddp_model if is_dist else lora_model
                    active_vh = ddp_value_head if is_dist else value_head

                    mb_log_prob, mb_new_values = forward_ppo(
                        mb_seq_ids,
                        active_lm,
                        active_vh,
                        mb_seq_mask,
                        inp_len,
                        mb_gen_ids,
                        temperature,
                    )  # (mb, gen_len), (mb, gen_len)

                    # ── Policy loss (clipped surrogate) ───────────────────────
                    log_ratio = (mb_log_prob - mb_lp_old.detach()) * mask
                    ratio = torch.exp(log_ratio.clamp(-20, 20))
                    pg1 = ratio * mb_adv.detach().squeeze(-1)
                    pg2 = torch.clamp(
                        ratio, 1 - eps_clip, 1 + eps_clip
                    ) * mb_adv.detach().squeeze(-1)
                    pg_loss_matrix = torch.min(pg1, pg2) * mask
                    pg_loss_sample = pg_loss_matrix.sum(dim=-1) / (
                        mask.sum(dim=-1) + 1e-5
                    )
                    pg_loss = -pg_loss_sample.mean()

                    # ── Value loss (clipped MSE) ──────────────────────────────
                    tgt = mb_returns.detach()
                    val_unclipped = (mb_new_values - tgt) ** 2
                    val_clipped = (mb_val_old+ (mb_new_values - mb_val_old).clamp(-value_clip_eps, value_clip_eps) # new value clip
                        - tgt
                    ) ** 2
                    val_loss_matrix = torch.max(val_unclipped, val_clipped) * mask
                    val_loss_sample = val_loss_matrix.sum(dim=-1) / (
                        mask.sum(dim=-1) + 1e-5
                    )
                    val_loss = val_loss_sample.mean()

                    # ── Entropy bonus ─────────────────────────────────────────
                    entropy_loss = torch.tensor(0.0, device=device)
                    if entropy_coef > 0:
                        entropy_matrix = -mb_log_prob * mask
                        entropy_sample = entropy_matrix.sum(dim=-1) / (
                            mask.sum(dim=-1) + 1e-5
                        )
                        entropy_loss = entropy_sample.mean()

                    # ── Total loss ────────────────────────────────────────────
                    loss = pg_loss + vf_coef * val_loss - entropy_coef * entropy_loss

                    is_accumulating = (
                        micro_step + 1
                    ) % gradient_accumulation_steps != 0

                    sync_ctx = contextlib.ExitStack()
                    if is_dist and is_accumulating:
                        sync_ctx.enter_context(ddp_model.no_sync())
                        sync_ctx.enter_context(ddp_value_head.no_sync())
                    with sync_ctx:
                        (loss / gradient_accumulation_steps).backward()

                    if (micro_step + 1) % gradient_accumulation_steps == 0:
                        all_params_for_clip = (
                            list(ddp_model.parameters())
                            + list(ddp_value_head.parameters())
                            if is_dist
                            else list(lora_model.parameters())
                            + list(value_head.parameters())
                        )
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            all_params_for_clip, max_grad_norm
                        )
                        if is_main_process:
                            logger.add_scalar(
                                "train/grad_norm", grad_norm.item(), g_step
                            )
                            logger.add_scalar("lr", cur_lr, g_step)

                        if not torch.isfinite(grad_norm):
                            if is_main_process:
                                print(
                                    f"[warn] non-finite grad_norm at g_step {g_step}, skip"
                                )
                            opt.zero_grad(set_to_none=True)
                            micro_step += 1
                            continue

                        opt.step()
                        opt.zero_grad()

                        with torch.no_grad():
                            # log losses
                            if is_main_process:
                                logger.add_scalar(
                                    "train/pg_loss", pg_loss.item(), g_step
                                )
                                logger.add_scalar(
                                    "train/val_loss", val_loss.item(), g_step
                                )
                                if entropy_coef > 0:
                                    logger.add_scalar(
                                        "train/entropy", entropy_loss.item(), g_step
                                    )

                            # clip fraction
                            is_clipped = (pg1 > pg2).to(mask.dtype)
                            clip_mat = is_clipped * mask
                            if is_dist:
                                cf_g = [
                                    torch.zeros_like(clip_mat)
                                    for _ in range(dist.get_world_size())
                                ]
                                gm_g = [
                                    torch.zeros_like(mask)
                                    for _ in range(dist.get_world_size())
                                ]
                                dist.all_gather(cf_g, clip_mat)
                                dist.all_gather(gm_g, mask)
                                clip_frac = torch.cat(cf_g).sum() / (
                                    torch.cat(gm_g).sum() + 1e-5
                                )
                            else:
                                clip_frac = clip_mat.sum() / (mask.sum() + 1e-5)
                            if is_main_process:
                                logger.add_scalar(
                                    "train/clip_frac", clip_frac.item(), g_step
                                )

                            # entropy (for logging even when coef=0)
                            ent_mat = -mb_log_prob.detach() * mask
                            ent_sample = ent_mat.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5)
                            if is_dist:
                                eg = [
                                    torch.zeros_like(ent_sample)
                                    for _ in range(dist.get_world_size())
                                ]
                                dist.all_gather(eg, ent_sample)
                                ent_sample = torch.cat(eg)
                            if is_main_process:
                                logger.add_scalar(
                                    "train/entropy", ent_sample.mean().item(), g_step
                                )

                        g_step += 1
                    micro_step += 1

            # ── eval + checkpoint ─────────────────────────────────────────────
            if (b_idx + 1) in eval_steps:
                val_f_r, val_c_r = evaluate(
                    lora_model, tokenizer, val_dataloader, max_new_tokens
                )
                if is_main_process:
                    frac = (b_idx + 1) / len(train_dataloader)
                    print(
                        f"eval ep{ep} step{b_idx+1} ({frac:.0%}): f_r={val_f_r:.4f}  c_r={val_c_r:.4f}"
                    )
                    logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
                    logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

            if (b_idx + 1) in eval_steps:
                if is_dist:
                    dist.barrier()
                if is_main_process:
                    ckp_path = os.path.join(ckp_dir, f"step_{g_step}")
                    lora_model.save_pretrained(ckp_path)
                    tokenizer.save_pretrained(ckp_path)
                    torch.save(
                        {
                            "epoch": ep,
                            "b_idx": b_idx,
                            "global_step": g_step,
                            "micro_step": micro_step,
                            "optimizer_state_dict": opt.state_dict(),
                            "value_head_state_dict": value_head.state_dict(),
                        },
                        os.path.join(ckp_path, "training_state.pt"),
                    )
                    print(f"checkpoint saved: {ckp_path}")
