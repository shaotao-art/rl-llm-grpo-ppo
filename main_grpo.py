import os
from contextlib import nullcontext
from functools import partial

import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from core import (
    build_prompt,
    repeat_lst,
    get_generated_text_lst,
    rollout,
    forward_get_log_probs,
    make_loss_mask,
)
from gsm_8k_dataset import (
    get_train_dataloader,
    get_val_dataloader,
    cal_reward as _cal_reward,
)
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
    resume_from_ckp,
)

# grpo uses the strict reward: both "## Reasoning" and "## Answer" required, otherwise (0, 0);
# answer extracted only from the text after "## Answer"
cal_reward = partial(_cal_reward, strict=True)


if __name__ == "__main__":
    # ------------- Config ---------------- #
    cfg = load_config("configs/grpo_default.yaml")

    # 模型 / 数据
    model_name = cfg["model_name"]
    data_p = cfg["data_p"]
    # LoRA
    lora_rank = cfg["lora_rank"]
    lora_alpha = cfg["lora_alpha"]
    # 优化器
    max_lr = cfg["max_lr"]
    min_lr = cfg["min_lr"]
    weight_decay = cfg["weight_decay"]
    max_grad_norm = cfg["max_grad_norm"]
    # 训练 / rollout
    save_root = cfg["save_root"]
    temperature = cfg["temperature"]
    max_new_tokens = cfg["max_new_tokens"]
    shuffle_rollout_samples = cfg.get(
        "shuffle_rollout_samples", True
    )  # shuffle samples before each ppo epoch
    eps = cfg.get("eps", 0.2)
    # asymmetric clip: lower bound uses (1 - eps_low), upper bound uses (1 + eps_high);
    # both default to eps so symmetric configs keep working
    eps_low = cfg.get("eps_low", eps)
    eps_high = cfg.get("eps_high", eps)
    w_format_r = cfg["w_format_r"]
    use_ref_model = cfg["use_ref_model"]
    kl_weight = cfg["kl_weight"]
    train_prompt_size = cfg[
        "train_prompt_size"
    ]  # per_device, train_len = len(train set) / (train_prompt_size * world_size)
    num_gen = cfg["num_gen"]  # each rollout samples train_prompt_size * num_gen
    gradient_accumulation_steps = cfg[
        "gradient_accumulation_steps"
    ]  # gradient accumulation
    ppo_train_mini_bs = cfg[
        "ppo_train_mini_bs"
    ]  # effective train_bs = ppo_train_mini_bs * world_size * gradient_accumulation_steps
    ppo_num_epoch = cfg["ppo_num_epoch"]  # each rollout train x ep
    rollout_mini_bs = cfg["rollout_mini_bs"]  # bs during rollout to avoid oom
    val_bs = cfg[
        "val_bs"
    ]  # per device eval batch size -> val_len = len(val set) / (val_bs * world_size)
    num_epoch = cfg["num_epoch"]
    eval_ratio = cfg.get(
        "eval_ratio", 0.0
    )  # eval (1/eval_ratio) times per epoch; 0 = disable mid-epoch eval
    resume_from = cfg.get(
        "resume_from", None
    )  # None=fresh; "latest"; or a checkpoint dir/name to resume from

    loss_type = cfg.get("loss_type", "grpo")
    loss_aggregate_type = cfg.get("loss_aggregate_type", "sample")
    # token-level (Dr.GRPO style) loss: normalize by a fixed constant instead of the actual token count,
    # which removes per-sample length bias and needs no cross-device communication. defaults to max_new_tokens.
    token_norm_const = cfg.get("token_norm_const", max_new_tokens)

    # set up dist env
    is_dist, world_size, rank, device = setup_dist()
    is_main_process = rank == 0

    # load model
    tokenizer, model = load_model_tokenier(model_name, device)

    # get lora model
    lora_model = build_lora_model(model, lora_rank, lora_alpha, is_main_process)

    # get ref model, cal kl, avoid policy model get too far from "safe" ref model
    # in lora, can use lora_model.disable_adapter() to decrease memory usage, not used here
    ref_model = None
    if use_ref_model:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    # setup ddp model
    ddp_model = None
    if is_dist:
        ddp_model = DDP(lora_model, device_ids=[rank], output_device=rank)
        opt = optim.AdamW(ddp_model.parameters(), lr=max_lr, weight_decay=weight_decay)
    else:
        opt = optim.AdamW(lora_model.parameters(), lr=max_lr, weight_decay=weight_decay)

    # get tensorboard logger
    if is_main_process:
        logger = SummaryWriter(save_root)

    # get datasets
    train_dataloader = get_train_dataloader(
        data_p, train_prompt_size, is_dist, rank, is_main_process
    )
    val_dataloader = get_val_dataloader(data_p, val_bs, is_dist, rank, is_main_process)

    # get eval steps
    eval_steps = get_eval_steps(eval_ratio, len(train_dataloader))

    ckp_dir = os.path.join(save_root, "checkpoints")
    if is_main_process:
        os.makedirs(ckp_dir, exist_ok=True)

    opt.zero_grad()
    g_step = 0  # number of actual optimizer updates (opt.step calls)
    micro_step = (
        0  # number of ppo mini-batches processed (drives gradient accumulation)
    )

    # total number of optimizer updates over the whole run, for step-wise lr decay
    samples_per_rollout = num_gen * train_prompt_size  # per device
    micro_steps_per_rollout = ppo_num_epoch * (samples_per_rollout // ppo_train_mini_bs)
    total_micro_steps = num_epoch * len(train_dataloader) * micro_steps_per_rollout
    total_train_steps = total_micro_steps // gradient_accumulation_steps
    rank_zero_print(
        f"total_train_steps (optimizer updates, for lr decay): {total_train_steps}"
    )

    # ---- optionally resume from a checkpoint ----
    start_epoch, g_step, resume_skip_batches, state = resume_from_ckp(
        resume_from, ckp_dir, lora_model, opt, device
    )
    if state:
        micro_step = state["micro_step"]

    # init eval (skip when resuming: the loaded model has already been evaluated)
    if not resume_from:
        rank_zero_print(f"init eval start")
        val_f_r, val_c_r = evaluate(
            lora_model,
            tokenizer,
            val_dataloader,
            max_new_tokens,
            is_dist,
            is_main_process,
            cal_reward_fn=cal_reward,
        )
        rank_zero_print(f"init eval end, val_f_r: {val_f_r}, val_c_r: {val_c_r}\n\n\n")
        if is_main_process:
            logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
            logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

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
            # fast-forward past batches already trained before the checkpoint
            if ep == start_epoch and b_idx < resume_skip_batches:
                continue
            prompt_lst = build_prompt(tokenizer, q_lst)
            prompt_lst_expand = repeat_lst(prompt_lst, num_gen)
            a_lst_expand = repeat_lst(a_lst, num_gen)
            # tokenize
            model_inputs = tokenizer(
                prompt_lst_expand,
                padding=True,
                padding_side="left",
                return_tensors="pt",
            ).to(lora_model.device)
            N, inp_len = model_inputs["attention_mask"].shape
            rollout_res = rollout(
                model_inputs,
                lora_model,
                max_new_tokens,
                temperature,
                rollout_mini_bs,
                tokenizer.pad_token_id,
                return_log_probs=True,
            )
            sequence_ids = rollout_res["sequence_ids"]
            seq_mask = make_loss_mask(sequence_ids, tokenizer.pad_token_id)
            log_probs_old = rollout_res["log_probs_old"]
            generated_ids = rollout_res["generated_ids"]  # (N, gen_len)
            gen_mask = (generated_ids != tokenizer.pad_token_id).to(log_probs_old.dtype)
            gen_len = gen_mask.sum(dim=-1)  # (N, )
            # log gen_len (must be called on every rank: all_gather inside)
            log_gen_len_stats(
                logger if is_main_process else None,
                gen_len,
                max_new_tokens,
                g_step,
                is_dist,
                is_main_process,
            )

            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)

            # cal adv
            with torch.no_grad():
                f_r_sum_tensor = torch.tensor([0.0], device=device)
                c_r_sum_tensor = torch.tensor([0.0], device=device)
                num_sample_tensor = torch.tensor([0.0], device=device)
                reward_lst_ = [
                    cal_reward(gen_pred, gt)
                    for gen_pred, gt in zip(generated_text_lst, a_lst_expand)
                ]
                format_r_lst = [x[0] for x in reward_lst_]
                content_r_lst = [x[1] for x in reward_lst_]
                reward_lst = [
                    f_r * w_format_r + c_r
                    for f_r, c_r in zip(format_r_lst, content_r_lst)
                ]
                f_r_sum_tensor += sum(format_r_lst)
                c_r_sum_tensor += sum(content_r_lst)
                num_sample_tensor += len(a_lst_expand)
                f_r_sum_tensor = all_reduce_sum(f_r_sum_tensor, is_dist)
                c_r_sum_tensor = all_reduce_sum(c_r_sum_tensor, is_dist)
                num_sample_tensor = all_reduce_sum(num_sample_tensor, is_dist)
                if is_main_process:
                    logger.add_scalar(
                        tag="rollout/f_r",
                        scalar_value=(
                            f_r_sum_tensor / num_sample_tensor
                        ).item(),  # mean f_r across all gpu
                        global_step=g_step,
                    )
                    logger.add_scalar(
                        tag="rollout/c_r",
                        scalar_value=(c_r_sum_tensor / num_sample_tensor).item(),
                        global_step=g_step,
                    )
                reward = torch.tensor(reward_lst, device=device)  # (N, )
                mean = (
                    reward.reshape(-1, num_gen)  # (b, num_gen)
                    .mean(dim=-1)  # (b)
                    .repeat_interleave(num_gen)
                )  # (N, )
                std = (
                    reward.reshape(-1, num_gen)
                    .std(dim=-1, unbiased=False)
                    .repeat_interleave(num_gen)
                )
                adv = (reward - mean) / (std + 1e-5)
                adv = adv.unsqueeze(-1)  # (N, 1)

                # log std
                std_to_log = reward.reshape(-1, num_gen).std(dim=-1, unbiased=False)
                std_to_log = all_gather_cat(std_to_log, is_dist)
                if is_main_process:
                    logger.add_scalar(
                        tag="rollout/std",
                        scalar_value=std_to_log.mean().item(),
                        global_step=g_step,
                    )

            total_samples = len(generated_text_lst)
            assert total_samples % ppo_train_mini_bs == 0
            assert (total_samples * ppo_num_epoch) % (
                ppo_train_mini_bs * gradient_accumulation_steps * world_size
            ) == 0

            # ref log probs are loop-invariant across ppo epochs: compute once per rollout
            ref_log_prob_all = None
            if use_ref_model:
                with torch.no_grad():
                    ref_log_prob_all = forward_get_log_probs(
                        sequence_ids, ref_model, seq_mask, inp_len, generated_ids
                    )  # (N, gen_len)

            for ppo_epoch in range(ppo_num_epoch):
                num_batches = total_samples // ppo_train_mini_bs
                # shuffle batch
                if shuffle_rollout_samples:
                    batch_indices = torch.randperm(total_samples, device=device)
                else:
                    batch_indices = torch.arange(total_samples, device=device)
                for mb_idx in range(num_batches):
                    # step-wise lr scheduler (linear decay from max_lr to min_lr over optimizer updates)
                    cur_lr = update_lr(opt, g_step, total_train_steps, max_lr, min_lr)

                    str_idx = mb_idx * ppo_train_mini_bs
                    end_idx = (mb_idx + 1) * ppo_train_mini_bs
                    idx = batch_indices[str_idx:end_idx]

                    mb_seq_ids = sequence_ids[idx]
                    mb_old_log_prob = log_probs_old[idx]
                    mb_gen_ids = generated_ids[idx]
                    mb_seq_mask = seq_mask[idx]
                    mb_adv = adv[idx]
                    # cur policy log prob
                    mb_log_prob = forward_get_log_probs(
                        mb_seq_ids,
                        ddp_model if is_dist else lora_model,
                        mb_seq_mask,
                        inp_len,
                        mb_gen_ids,
                    )

                    if use_ref_model:
                        ref_log_prob = ref_log_prob_all[idx]  # (mb, gen_len)

                    # loss mask: which generated positions are real tokens (not padding)
                    mask = make_loss_mask(mb_gen_ids, tokenizer.pad_token_id).to(
                        mb_log_prob.dtype
                    )

                    # PPO ratio: zero out padding positions before exp, and clamp to avoid fp32 overflow
                    log_ratio = (mb_log_prob - mb_old_log_prob.detach()) * mask
                    if loss_type == "grpo":
                        ratio = torch.exp(
                            log_ratio.clamp(-20, 20)
                        )  # (mb, gen_len) IMPORTANT, clamp when use exp, solution to model.generate() prob inf,nan error
                        sour1 = (
                            torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
                            * mb_adv.detach()
                        )
                        sour2 = ratio * mb_adv.detach()

                        pg_loss_matrix = torch.min(sour1, sour2) * mask
                    elif loss_type == "gspo":
                        log_seq_ratio = torch.sum(log_ratio, dim=-1, keepdim=True) / (
                            mask.sum(dim=-1, keepdim=True) + 1e-5
                        )  # (mb, 1)
                        seq_ratio = torch.exp(log_seq_ratio.clamp(-20, 20))  # (mb, 1)
                        sour1 = (
                            torch.clamp(seq_ratio, 1 - eps_low, 1 + eps_high)
                            * mb_adv.detach()
                        )  # (mb, 1)
                        sour2 = seq_ratio * mb_adv.detach()  # (mb, 1)
                        pg_loss_matrix = (
                            torch.min(sour1, sour2) * mask
                        )  # (mb, 1) * (mb, gen_len) -> (mb, gen_len)

                    elif loss_type == "cispo":
                        ratio = torch.exp(log_ratio.clamp(-20, 20))
                        clamped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
                        pg_loss_matrix = (
                            clamped_ratio.detach()
                            * mb_adv.detach()
                            * (mb_log_prob * mask)
                        )
                    else:
                        raise ValueError(f"Invalid loss_type: {loss_type}")

                    if loss_aggregate_type == "sample":
                        pg_loss_samplewise = pg_loss_matrix.sum(dim=-1) / (
                            mask.sum(dim=-1) + 1e-5
                        )
                        pg_loss = -pg_loss_samplewise.mean()
                    elif loss_aggregate_type == "token":
                        # Dr.GRPO style: normalize by a fixed constant (default max_new_tokens) rather than the
                        # actual token count. this gives every token equal weight, removes per-sample length bias,
                        # and combines correctly across ranks / grad-accum with no extra communication.
                        pg_loss = -pg_loss_matrix.sum(dim=-1).mean() / token_norm_const
                    else:
                        raise ValueError(
                            f"Invalid loss_aggregate_type: {loss_aggregate_type}"
                        )

                    if use_ref_model:
                        # mask BEFORE exp so padding positions give kl(0)=0 (avoids inf*0 -> NaN);
                        # clamp keeps exp() from overflowing when policy drifts away from ref.
                        # k3 estimator: kl(p||q) ≈ exp(log q - log p) - 1 - (log q - log p)
                        log_diff = ((ref_log_prob - mb_log_prob) * mask).clamp(-20, 20)
                        kl_loss_matrix = (torch.exp(log_diff) - 1 - log_diff) * mask
                        kl_loss = kl_loss_matrix.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5)
                        kl_loss = kl_loss.mean()

                    if use_ref_model:
                        loss = pg_loss + kl_loss * kl_weight
                    else:
                        loss = pg_loss

                    is_accumulating = (
                        micro_step + 1
                    ) % gradient_accumulation_steps != 0
                    sync_context = (
                        ddp_model.no_sync()
                        if (is_dist and is_accumulating)
                        else nullcontext()
                    )

                    with sync_context:
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                    if (micro_step + 1) % gradient_accumulation_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            (
                                ddp_model.parameters()
                                if is_dist
                                else lora_model.parameters()
                            ),
                            max_grad_norm,
                        )
                        if is_main_process:
                            logger.add_scalar(
                                "train/grad_norm", grad_norm.item(), global_step=g_step
                            )
                            logger.add_scalar("lr", cur_lr, global_step=g_step)
                        # safety net: never let a non-finite grad poison the weights (would crash rollout sampling)
                        if not torch.isfinite(grad_norm):
                            rank_zero_print(
                                f"[warn] non-finite grad_norm at g_step {g_step}, skipping optimizer step"
                            )
                            opt.zero_grad(set_to_none=True)
                            micro_step += 1
                            continue
                        opt.step()
                        opt.zero_grad()
                        with torch.no_grad():
                            # log loss
                            pg_loss_log = all_reduce_sum(
                                pg_loss.detach().clone(), is_dist
                            )
                            if use_ref_model:
                                kl_loss_log = all_reduce_sum(
                                    kl_loss.detach().clone(), is_dist
                                )

                            if is_main_process:
                                logger.add_scalar(
                                    tag="train/pg_loss",
                                    scalar_value=pg_loss_log.item() / world_size,
                                    global_step=g_step,
                                )
                                if use_ref_model:
                                    logger.add_scalar(
                                        tag="train/kl_loss",
                                        scalar_value=kl_loss_log.item() / world_size,
                                        global_step=g_step,
                                    )

                            # log entropy
                            entropy = -mb_log_prob * mask.to(
                                mb_log_prob.dtype
                            )  # -plogp
                            entropy = entropy.sum(dim=-1) / (
                                mask.sum(dim=-1) + 1e-5
                            )  # per_sample (N, )
                            entropy = all_gather_cat(entropy, is_dist).mean()

                            if is_main_process:
                                logger.add_scalar(
                                    tag="train/entropy",
                                    scalar_value=entropy.item(),
                                    global_step=g_step,
                                )

                            # log clip ratio
                            with torch.no_grad():
                                if loss_type == "cispo":
                                    is_clipped = ratio != clamped_ratio
                                else:
                                    is_clipped = (sour1 < sour2).to(mask.dtype)
                                # two scalar sums carry everything the full-matrix gather did
                                clip_num = (is_clipped * mask).sum()
                                clip_den = mask.sum()
                                if is_dist:
                                    dist.all_reduce(clip_num, op=dist.ReduceOp.SUM)
                                    dist.all_reduce(clip_den, op=dist.ReduceOp.SUM)
                                clip_frac = clip_num / (clip_den + 1e-5)
                                if is_main_process:
                                    logger.add_scalar(
                                        tag="train/clip_frac",
                                        scalar_value=clip_frac.item(),
                                        global_step=g_step,
                                    )
                        g_step += 1  # count only actual optimizer updates
                    micro_step += 1

            # eval at configured fractions of epoch
            if (b_idx + 1) in eval_steps:
                val_f_r, val_c_r = evaluate(
                    lora_model,
                    tokenizer,
                    val_dataloader,
                    max_new_tokens,
                    is_dist,
                    is_main_process,
                    cal_reward_fn=cal_reward,
                )
                if is_main_process:
                    frac = (b_idx + 1) / len(train_dataloader)
                    rank_zero_print(
                        f"eval ep {ep} step {b_idx+1}/{len(train_dataloader)} ({frac:.0%}): val_f_r={val_f_r:.4f}, val_c_r={val_c_r:.4f}"
                    )
                    logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
                    logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

            # save checkpoint after each eval
            if (b_idx + 1) in eval_steps:
                save_checkpoint(
                    ckp_dir,
                    lora_model,
                    tokenizer,
                    opt,
                    ep,
                    b_idx,
                    g_step,
                    is_dist,
                    is_main_process,
                    extra_state={"micro_step": micro_step},
                )
