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

import os
import contextlib
from functools import partial

import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import set_peft_model_state_dict

from core import (
    build_prompt,
    repeat_lst,
    get_generated_text_lst,
    rollout,
    get_log_probs,
    forward_get_log_probs,
    left_pad_position_ids,
    make_loss_mask,
)
from gsm_8k_dataset import get_train_dataloader, get_val_dataloader, cal_reward
from utils import (
    setup_dist,
    all_gather_cat,
    all_reduce_sum,
    evaluate,
    rank_zero_print,
    load_config,
    load_model_tokenier,
    build_lora_model,
    update_lr,
    log_gen_len_stats,
    save_checkpoint,
    get_eval_steps,
    load_checkpoint,
    resume_from_ckp,
)

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
# Value rollout  (separate no-grad forward to get V(s_0)..V(s_T) for GAE)
# ──────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def rollout_values(
    sequence_ids,
    lm_model,
    value_head,
    seq_mask,
    inp_len,
    gen_len,
    rollout_bs,
    value_model=None,
):
    """
    Returns old_values: (N, gen_len)
      col t   = V(s_t) for t = 0..gen_len-1  (state before generating token t)
      col T   = V(s_T) = bootstrap value after the full generated sequence

    value_model : backbone used for the value head.
        None                 → share the policy backbone (lm_model)
        an AutoModelForCausalLM → independent value backbone (indep_value_model=True)
    """
    lm_model.eval()
    if value_model is not None:
        value_model.eval()
    device = sequence_ids.device
    N = sequence_ids.shape[0]
    values = torch.zeros(N, gen_len + 1, device=device)

    for i in range(N // rollout_bs):
        s, e = i * rollout_bs, (i + 1) * rollout_bs
        batch_seq = sequence_ids[s:e]
        batch_mask = seq_mask[s:e]
        pos_ids = left_pad_position_ids(batch_mask)
        if value_model is None:
            # shared backbone: hidden states from the policy model itself
            out = lm_model(
                batch_seq,
                attention_mask=batch_mask,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]  # (bs, inp_len + gen_len, H)
        else:
            # independent value backbone: the policy forward would be discarded, skip it
            v_out = value_model(
                batch_seq,
                attention_mask=batch_mask,
                position_ids=pos_ids,
                output_hidden_states=True,
            )
            hidden = v_out.hidden_states[-1]  # (bs, inp_len + gen_len, H)
        # positions inp_len-1 .. inp_len+gen_len-1 → states s_0 .. s_T
        vals = value_head(hidden[:, inp_len - 1 :])  # (bs, gen_len + 1)
        values[s:e] = vals

    lm_model.train()
    if value_model is not None:
        value_model.train()
    return values  # (N, gen_len + 1)


# ──────────────────────────────────────────────────────────────────────────────
# GAE
# ──────────────────────────────────────────────────────────────────────────────


def compute_gae(per_token_rewards, values, gamma, gae_lambda):
    """
    per_token_rewards : (N, T)  — reward at each gen position (0 except EOS + KL)
    values            : (N, T + 1)  input_len - 1 -> <eos>
    Returns:
        advantages (N, T),  returns (N, T)
    """
    N, T = per_token_rewards.shape
    adv = torch.zeros(N, T, device=per_token_rewards.device)
    last_g = torch.zeros(N, device=per_token_rewards.device)

    for t in reversed(range(T)):
        delta = per_token_rewards[:, t] + gamma * values[:, t + 1] - values[:, t]
        last_g = delta + gamma * gae_lambda * last_g
        adv[:, t] = last_g

    returns = adv + values[:, :T]  # Q(s, a) = Adv(s, a) + V(s)
    return adv, returns


# ──────────────────────────────────────────────────────────────────────────────
# Combined policy + value forward (used during PPO training)
# ──────────────────────────────────────────────────────────────────────────────


def forward_ppo(
    sequence_ids,
    lm_model,
    value_head,
    attn_mask,
    inp_len,
    generated_ids,
    value_model=None,
):
    """
    Single forward pass returning:
        log_prob : (N, gen_len)
        values   : (N, gen_len)   — col gen_len is the bootstrap value

    value_model : backbone used for the value head.
        None                 → share the policy backbone (hidden states of lm_model)
        an AutoModelForCausalLM → independent value backbone (indep_value_model=True)
    """
    pos_ids = left_pad_position_ids(attn_mask)
    out = lm_model(
        sequence_ids,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        output_hidden_states=(value_model is None),
    )
    logits = out.logits
    gen_logits = logits[:, inp_len - 1 : -1]  # (N, gen_len, V)
    log_prob = get_log_probs(gen_logits, generated_ids)  # (N, gen_len)

    if value_model is None:
        # shared backbone: values from the same pass as the policy
        hidden = out.hidden_states[-1]  # (N, inp_len+gen_len, H)
        values = value_head(hidden[:, inp_len - 1 : -1])  # (N, gen_len)
    else:
        # independent value backbone: separate forward for values
        v_out = value_model(
            sequence_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            output_hidden_states=True,
        )
        v_hidden = v_out.hidden_states[-1]
        values = value_head(v_hidden[:, inp_len - 1 : -1])  # (N, gen_len)

    return log_prob, values


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_ppo_checkpoint(
    ckp_path, lora_model, opt, device, value_head=None, value_model=None
):
    """utils.load_checkpoint + PPO extras: value head weights and, when an
    independent value backbone is used, its LoRA adapter."""
    from safetensors.torch import load_file

    state = load_checkpoint(ckp_path, lora_model, opt, device)
    value_head.load_state_dict(state["value_head_state_dict"])
    if value_model is not None:
        # independent critic is a PeftModel → restore its LoRA adapter
        val_adapter_sd = load_file(
            os.path.join(ckp_path, "value_adapter", "adapter_model.safetensors"),
            device=str(device),
        )
        set_peft_model_state_dict(value_model, val_adapter_sd)
    return state


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ------------- Config ---------------- #
    cfg = load_config("configs/ppo_default.yaml")

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
    shuffle_rollout_samples = cfg.get(
        "shuffle_rollout_samples", True
    )  # shuffle samples before each ppo epoch
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
    # value backbone: whether the value model has an independent backbone (default True)
    indep_value_model = cfg.get("indep_value_model", False)

    # ── dist ──────────────────────────────────────────────────────────────────
    is_dist, world_size, rank, device = setup_dist()
    is_main_process = rank == 0

    # ── model ─────────────────────────────────────────────────────────────────
    tokenizer, model = load_model_tokenier(model_name, device)

    lora_model = build_lora_model(model, lora_rank, lora_alpha, is_main_process)

    # value backbone + value head
    if indep_value_model:
        # independent backbone (own parameters, separate from the policy LoRA model),
        # also wrapped in LoRA so only its adapters are trainable
        value_base = AutoModelForCausalLM.from_pretrained(model_name).to(
            device, dtype=model.dtype
        )
        value_model = build_lora_model(
            value_base, lora_rank, lora_alpha, is_main_process
        )
        rank_zero_print("[value model] trainable parameters (independent LoRA critic)")
    else:
        # shared backbone: value head reads hidden states of the policy model
        value_model = None
    value_head = ValueHead(model.config.hidden_size).to(device, dtype=model.dtype)

    ref_model = None
    if use_ref_model:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    # ── DDP ───────────────────────────────────────────────────────────────────
    ddp_model = None
    ddp_value_head = None
    ddp_value_model = None
    if is_dist:
        ddp_model = DDP(lora_model, device_ids=[rank], output_device=rank)
        ddp_value_head = DDP(value_head, device_ids=[rank], output_device=rank)
        if value_model is not None:
            ddp_value_model = DDP(value_model, device_ids=[rank], output_device=rank)
        opt = optim.AdamW(
            list(ddp_model.parameters())
            + list(ddp_value_head.parameters())
            + (
                list(ddp_value_model.parameters())
                if ddp_value_model is not None
                else []
            ),
            lr=max_lr,
            weight_decay=weight_decay,
        )
    else:
        opt = optim.AdamW(
            list(lora_model.parameters())
            + list(value_head.parameters())
            + (list(value_model.parameters()) if value_model is not None else []),
            lr=max_lr,
            weight_decay=weight_decay,
        )

    if is_main_process:
        logger = SummaryWriter(save_root)

    # ── datasets ──────────────────────────────────────────────────────────────
    train_dataloader = get_train_dataloader(
        data_p, train_prompt_size, is_dist, rank, is_main_process
    )
    val_dataloader = get_val_dataloader(data_p, val_bs, is_dist, rank, is_main_process)

    # ppo always evals at the end of each epoch, even when eval_ratio == 0
    eval_steps = get_eval_steps(eval_ratio, len(train_dataloader))
    eval_steps.add(len(train_dataloader))

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
    rank_zero_print(f"total optimizer updates: {total_train_steps}")

    start_epoch, g_step, resume_skip_batches, state = resume_from_ckp(
        resume_from,
        ckp_dir,
        lora_model,
        opt,
        device,
        load_fn=partial(
            load_ppo_checkpoint, value_head=value_head, value_model=value_model
        ),
    )
    if state:
        micro_step = state["micro_step"]

    if not resume_from:
        val_f_r, val_c_r = evaluate(
            lora_model,
            tokenizer,
            val_dataloader,
            max_new_tokens,
            is_dist,
            is_main_process,
        )
        if is_main_process:
            # val_f_r/val_c_r are None on other ranks — keep the %.4f formatting guarded
            rank_zero_print(f"init eval: val_f_r={val_f_r:.4f}  val_c_r={val_c_r:.4f}")
            logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
            logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

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

            # Rollout: generate + old log probs
            rollout_res = rollout(
                model_inputs,
                lora_model,
                max_new_tokens,
                temperature,
                rollout_mini_bs,
                tokenizer.pad_token_id,
                return_log_probs=True,
            )
            sequence_ids = rollout_res["sequence_ids"]  # (N, inp_len + gen_len)
            generated_ids = rollout_res["generated_ids"]  # (N, gen_len)
            log_probs_old = rollout_res["log_probs_old"]  # (N, gen_len)

            seq_mask = make_loss_mask(
                sequence_ids, tokenizer.pad_token_id
            )  # (N, inp_len + gen_len)
            gen_mask = make_loss_mask(
                generated_ids, tokenizer.pad_token_id
            )  # cover the first <eos>
            real_len = gen_mask.sum(dim=-1)  # (N,)
            gen_len = gen_mask.shape[-1]

            # log gen-length stats (must be called on every rank: all_gather inside)
            log_gen_len_stats(
                logger if is_main_process else None,
                real_len,
                max_new_tokens,
                g_step,
                is_dist,
                is_main_process,
            )

            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)

            # Compute final rewards
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
                f_sum = all_reduce_sum(f_sum, is_dist)
                c_sum = all_reduce_sum(c_sum, is_dist)
                n_t = all_reduce_sum(n_t, is_dist)
                if is_main_process:
                    logger.add_scalar("rollout/f_r", (f_sum / n_t).item(), g_step)
                    logger.add_scalar("rollout/c_r", (c_sum / n_t).item(), g_step)

                # log rollout reward std as proxy for advantage spread
                scalar_reward_log = all_gather_cat(scalar_reward, is_dist)
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
                per_token_rewards = torch.zeros(
                    N, gen_len, device=device
                )  # (N, gen_len)

                if use_ref_model:
                    ref_log_prob = forward_get_log_probs(
                        sequence_ids,
                        ref_model,
                        seq_mask,
                        inp_len,
                        generated_ids,
                    )  # (N, gen_len) first -> inp_len - 1
                    per_token_kl = (
                        log_probs_old - ref_log_prob
                    ) * gen_mask  # KL(p||q) = p * (log p - log q), wish kl smaller
                    per_token_kl_to_log = all_gather_cat(per_token_kl, is_dist)
                    all_gen_mask = all_gather_cat(gen_mask, is_dist)
                    per_token_kl_all = per_token_kl_to_log[all_gen_mask]
                    if is_main_process:
                        logger.add_scalar(
                            "rollout/per_token_kl/mean",
                            per_token_kl_all.mean().item(),
                            g_step,
                        )
                        logger.add_scalar(
                            "rollout/per_token_kl/std",
                            per_token_kl_all.std(unbiased=False).item(),
                            g_step,
                        )
                        logger.add_scalar(
                            "rollout/per_token_kl/max",
                            per_token_kl_all.max().item(),
                            g_step,
                        )
                        logger.add_scalar(
                            "rollout/per_token_kl/min",
                            per_token_kl_all.min().item(),
                            g_step,
                        )
                    per_token_rewards = (
                        -kl_weight * per_token_kl
                    )  # (N, gen_len) first -> inp_len - 1

                # Add scalar reward at the last real token of each sample
                last_real_idx = (
                    (gen_mask.sum(dim=-1) - 1).long().clamp(min=0)
                )  # (N,) -> <eos>
                per_token_rewards[
                    torch.arange(N, device=device), last_real_idx
                ] += scalar_reward
                per_token_rewards = per_token_rewards * gen_mask  # mask out padding

            # Old values
            active_value_model = ddp_value_model if is_dist else value_model
            with torch.no_grad():
                old_values = rollout_values(
                    sequence_ids,
                    lora_model,
                    value_head,
                    seq_mask,
                    inp_len,
                    gen_len,
                    rollout_mini_bs,
                    active_value_model,
                )  # (N, gen_len + 1)
                tmp_value_mask = generated_ids != tokenizer.pad_token_id  # (N, gen_len)
                old_values[:, 1:] = (
                    old_values[:, 1:] * tmp_value_mask
                )  # mask out padding

            # GAE
            with torch.no_grad():
                advantages, returns = compute_gae(
                    per_token_rewards, old_values, gamma, gae_lambda
                )
                # (N, gen_len), (N, gen_len)

                # Normalize advantages over the full rollout batch (masked positions excluded)
                if normalize_adv:
                    adv_all = all_gather_cat(advantages, is_dist)
                    mask_all = all_gather_cat(gen_mask, is_dist)
                    advantages_to_cal_mean_std = adv_all[mask_all]
                    adv_mean = advantages_to_cal_mean_std.mean()
                    adv_std = advantages_to_cal_mean_std.std(unbiased=False)
                    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
                    advantages = advantages * gen_mask  # (N, gen_len)

                adv_for_train = advantages.unsqueeze(
                    -1
                )  # (N, T, 1) — broadcast with mask

            # ── PPO training epochs ────────────────────────────────────────────
            total_samples = N
            assert total_samples % ppo_train_mini_bs == 0
            assert (total_samples * ppo_num_epoch) % (
                ppo_train_mini_bs * gradient_accumulation_steps
            ) == 0

            for ppo_epoch in range(ppo_num_epoch):
                # shuffle batch
                if shuffle_rollout_samples:
                    batch_indices = torch.randperm(total_samples, device=device)
                else:
                    batch_indices = torch.arange(total_samples, device=device)
                num_batches = total_samples // ppo_train_mini_bs

                for mb_idx in range(num_batches):
                    # step-wise linear LR decay
                    cur_lr = update_lr(opt, g_step, total_train_steps, max_lr, min_lr)

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
                        active_value_model,
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
                    # value head should in clip range to avoid aggressive update to explode
                    tgt = mb_returns.detach()
                    val_unclipped = (mb_new_values - tgt) ** 2
                    val_clipped = (
                        mb_val_old
                        + (mb_new_values - mb_val_old).clamp(
                            -value_clip_eps, value_clip_eps
                        )  # new value clip
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
                        if ddp_value_model is not None:
                            sync_ctx.enter_context(ddp_value_model.no_sync())
                    with sync_ctx:
                        (loss / gradient_accumulation_steps).backward()

                    if (micro_step + 1) % gradient_accumulation_steps == 0:
                        all_params_for_clip = (
                            list(ddp_model.parameters())
                            + list(ddp_value_head.parameters())
                            + (
                                list(ddp_value_model.parameters())
                                if ddp_value_model is not None
                                else []
                            )
                            if is_dist
                            else list(lora_model.parameters())
                            + list(value_head.parameters())
                            + (
                                list(value_model.parameters())
                                if value_model is not None
                                else []
                            )
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
                            rank_zero_print(
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
                            clip_mat_all = all_gather_cat(clip_mat, is_dist)
                            mask_all = all_gather_cat(mask, is_dist)
                            clip_frac = clip_mat_all.sum() / (mask_all.sum() + 1e-5)
                            if is_main_process:
                                logger.add_scalar(
                                    "train/pg_clip_frac", clip_frac.item(), g_step
                                )

                            # entropy (for logging even when coef=0)
                            ent_mat = -mb_log_prob.detach() * mask
                            ent_sample = ent_mat.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5)
                            ent_sample = all_gather_cat(ent_sample, is_dist)
                            if is_main_process:
                                logger.add_scalar(
                                    "train/entropy", ent_sample.mean().item(), g_step
                                )

                        g_step += 1
                    micro_step += 1

            # ── eval + checkpoint ─────────────────────────────────────────────
            if (b_idx + 1) in eval_steps:
                val_f_r, val_c_r = evaluate(
                    lora_model,
                    tokenizer,
                    val_dataloader,
                    max_new_tokens,
                    is_dist,
                    is_main_process,
                )
                if is_main_process:
                    frac = (b_idx + 1) / len(train_dataloader)
                    rank_zero_print(
                        f"eval ep{ep} step{b_idx+1} ({frac:.0%}): f_r={val_f_r:.4f}  c_r={val_c_r:.4f}"
                    )
                    logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
                    logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

            if (b_idx + 1) in eval_steps:
                ckp_path = save_checkpoint(
                    ckp_dir,
                    lora_model,
                    tokenizer,
                    opt,
                    ep,
                    b_idx,
                    g_step,
                    is_dist,
                    is_main_process,
                    extra_state={
                        "micro_step": micro_step,
                        "value_head_state_dict": value_head.state_dict(),
                    },
                )
                if is_main_process and value_model is not None:
                    value_model.save_pretrained(os.path.join(ckp_path, "value_adapter"))
