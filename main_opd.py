import torch
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from peft import LoraConfig, get_peft_model
import torch.distributed as dist
import argparse
import yaml

from utils import setup_dist, resolve_resume_path, load_checkpoint, evaluate
from gsm_8k_dataset import get_train_dataloader, get_val_dataloader
from core import (
    build_prompt,
    repeat_lst,
    rollout,
    make_loss_mask,
    get_generated_text_lst,
    forward_get_log_probs,
)


# device 由分布式环境决定（见下方 setup_dist），不从配置读取





if __name__ == "__main__":
    # ------------- Config ---------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 模型 / 数据
    student_model_name = cfg["student_model_name"]
    teacher_model_name = cfg["teacher_model_name"]
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
    shuffle_rollout_samples = cfg.get("shuffle_rollout_samples", False)
    train_prompt_size = cfg[
        "train_prompt_size"
    ]  # per_device, train_len = len(train set) / (train_prompt_size * world_size)
    num_gen = 1  # 1 for opd
    ppo_train_mini_bs = cfg[
        "ppo_train_mini_bs"
    ]  # effective train_bs = ppo_train_mini_bs * world_size * gradient_accumulation_steps
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

    # set up dist env
    is_dist, world_size, rank, device = setup_dist()
    is_main_process = rank == 0
    if is_main_process:
        print(f"[config] loaded from {args.config}:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # load model
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch infer
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    student_model.to(device)

    # get lora model
    peft_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, task_type="CAUSAL_LM")
    lora_model = get_peft_model(student_model, peft_config)
    if is_main_process:
        lora_model.print_trainable_parameters()

    # load teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)
    teacher_model.to(device)
    teacher_model.requires_grad_(False)

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

    if eval_ratio > 0:
        n_evals_per_epoch = max(1, round(1.0 / eval_ratio))
        _total_steps = len(train_dataloader)
        eval_steps = {
            round(_total_steps * (i + 1) / n_evals_per_epoch)
            for i in range(n_evals_per_epoch)
        }
        eval_steps.add(_total_steps)  # always include last step of epoch
    else:
        eval_steps = set()
    if is_main_process and eval_steps:
        print(
            f"eval_steps per epoch: {sorted(eval_steps)} (total {len(train_dataloader)} steps/epoch, eval_ratio={eval_ratio})"
        )

    ckp_dir = os.path.join(save_root, "checkpoints")
    if is_main_process:
        os.makedirs(ckp_dir, exist_ok=True)

    # total number of optimizer updates over the whole run, for step-wise lr decay
    total_train_steps = num_epoch * len(train_dataloader)
    if is_main_process:
        print(
            f"total_train_steps (optimizer updates, for lr decay): {total_train_steps}"
        )

    opt.zero_grad()
    g_step = 0  # number of actual optimizer updates (opt.step calls)

    # ---- optionally resume from a checkpoint ----
    start_epoch = 0
    resume_skip_batches = 0  # number of batches to skip at the start of start_epoch
    if resume_from:
        ckp_path = resolve_resume_path(resume_from, ckp_dir)
        if is_main_process:
            print(f"[resume] loading checkpoint from {ckp_path}")
        state = load_checkpoint(ckp_path, lora_model, opt, device)
        g_step = state["global_step"]
        start_epoch = state["epoch"]
        resume_skip_batches = (
            state["b_idx"] + 1
        )  # continue at the batch after the saved one
        if is_main_process:
            print(
                f"[resume] start_epoch={start_epoch}, skip first {resume_skip_batches} "
                f"batches, g_step={g_step}"
            )

    # # init eval (skip when resuming: the loaded model has already been evaluated)
    # if not resume_from:
    #     if is_main_process:
    #         print(f"init eval start")
    #     val_f_r, val_c_r = evaluate(
    #         lora_model,
    #         tokenizer,
    #         val_dataloader,
    #         max_new_tokens,
    #         is_dist,
    #         is_main_process,
    #     )
    #     if is_main_process:
    #         print(f"init eval end, val_f_r: {val_f_r}, val_c_r: {val_c_r}\n\n\n")
    #         logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
    #         logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

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
            )
            sequence_ids = rollout_res["sequence_ids"]
            seq_mask = make_loss_mask(sequence_ids, tokenizer.pad_token_id)
            gen_mask = seq_mask[:, inp_len:]
            gen_len = gen_mask.float().sum(dim=-1)  # (N, )
            # log gen_len
            if is_dist:
                gen_len_gather_lst = [
                    torch.zeros_like(gen_len) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gen_len_gather_lst, gen_len)
                gen_len = torch.cat(gen_len_gather_lst, dim=0)
            if is_main_process:
                mean_gen_len = gen_len.mean().item()
                max_gen_len = gen_len.max().item()
                min_gen_len = gen_len.min().item()
                # a sequence is "clipped" when it used up the whole length budget (no natural EOS stop)
                gen_len_clip_frac = (
                    torch.sum((gen_len == max_new_tokens).float()) / gen_len.shape[0]
                )
                logger.add_scalar(
                    "rollout/mean_gen_len", mean_gen_len, global_step=g_step
                )
                logger.add_scalar(
                    "rollout/max_gen_len", max_gen_len, global_step=g_step
                )
                logger.add_scalar(
                    "rollout/min_gen_len", min_gen_len, global_step=g_step
                )
                logger.add_scalar(
                    "rollout/gen_len_clip_frac", gen_len_clip_frac, global_step=g_step
                )

            generated_ids = rollout_res["generated_ids"]
            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)
            total_samples = len(generated_text_lst)
            assert (
                total_samples % ppo_train_mini_bs == 0
            ), f"total_samples: {total_samples}, ppo_train_mini_bs: {ppo_train_mini_bs}, cannot be divided"
            num_batches = total_samples // ppo_train_mini_bs
            if shuffle_rollout_samples:
                batch_indices = torch.randperm(total_samples, device=device)
            else:
                batch_indices = torch.arange(total_samples, device=device)
            for mb_idx in range(num_batches):
                # step-wise lr scheduler (linear decay from max_lr to min_lr over optimizer updates)
                progress = min(g_step / max(1, total_train_steps), 1.0)
                cur_lr = max_lr - (max_lr - min_lr) * progress
                for p_g in opt.param_groups:
                    p_g["lr"] = cur_lr
                if is_main_process:
                    logger.add_scalar("lr", cur_lr, global_step=g_step)

                str_idx = mb_idx * ppo_train_mini_bs
                end_idx = min((mb_idx + 1) * ppo_train_mini_bs, total_samples)
                idx = batch_indices[str_idx:end_idx]

                mb_seq_ids = sequence_ids[idx]
                mb_gen_ids = generated_ids[idx]
                mb_seq_mask = seq_mask[idx]
                # cur policy log prob
                policy_log_prob = forward_get_log_probs(
                    mb_seq_ids,
                    ddp_model if is_dist else lora_model,
                    mb_seq_mask,
                    inp_len,
                    mb_gen_ids,
                    temperature,
                )

                with torch.no_grad():
                    teacher_log_prob = forward_get_log_probs(
                        mb_seq_ids,
                        teacher_model,
                        mb_seq_mask,
                        inp_len,
                        mb_gen_ids,
                        temperature,
                    )  # (N, gen_len)

                # loss mask: which generated positions are real tokens (not padding)
                mask = mb_seq_mask[:, inp_len:]
                # PPO ratio: zero out padding positions before exp, and clamp to avoid fp32 overflow
                with torch.no_grad():
                    adv = -(policy_log_prob - teacher_log_prob) * mask
                reverse_kl = -policy_log_prob * adv.detach() * mask
                mean_reverse_kl = reverse_kl.sum() / mask.sum().item()

                if is_dist:
                    if mb_idx == num_batches - 1:
                        mean_reverse_kl.backward()
                    else:
                        with ddp_model.no_sync():
                            mean_reverse_kl.backward()
                else:
                    mean_reverse_kl.backward()
            # log kl
            if is_dist:
                dist.all_reduce(mean_reverse_kl, op=dist.ReduceOp.SUM)
                mean_reverse_kl = mean_reverse_kl / world_size
            if is_main_process:
                logger.add_scalar(
                    "train/mean_reverse_kl", mean_reverse_kl.item(), global_step=g_step
                )
            # log grad norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters() if is_dist else lora_model.parameters(),
                max_grad_norm,
            )
            if is_main_process:
                logger.add_scalar(
                    "train/grad_norm", grad_norm.item(), global_step=g_step
                )

            # safety net: never let a non-finite grad poison the weights (would crash rollout sampling)
            if not torch.isfinite(grad_norm):
                if is_main_process:
                    print(
                        f"[warn] non-finite grad_norm at g_step {g_step}, skipping optimizer step"
                    )
                opt.zero_grad(set_to_none=True)
                continue
            opt.step()
            opt.zero_grad()
            g_step += 1  # count only actual optimizer updates

            # eval at configured fractions of epoch
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
                    print(
                        f"eval ep {ep} step {b_idx+1}/{len(train_dataloader)} ({frac:.0%}): val_f_r={val_f_r:.4f}, val_c_r={val_c_r:.4f}"
                    )
                    logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
                    logger.add_scalar("val/c_r", val_c_r, global_step=g_step)

            # save checkpoint after each eval
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
                            "optimizer_state_dict": opt.state_dict(),
                        },
                        os.path.join(ckp_path, "training_state.pt"),
                    )
                    print(f"Checkpoint saved to {ckp_path}")
