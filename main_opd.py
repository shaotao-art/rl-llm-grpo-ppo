import torch
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


# device 由分布式环境决定（见下方 setup_dist），不从配置读取
def setup_dist():
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, f"cuda:{local_rank}"
    else:
        return 0, "cuda:0"


def build_prompt(tokenizer, text_lst: list[str]):
    """build prompt for input text"""
    # prepare the model input
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
    """repeat input prompt for num_gen [1, 1, 1, 1, 2, 2, 2, 2, ...]"""
    return [item for item in lst for _ in range(num_gen)]


def get_generated_text_lst(generated_ids, tokenizer):
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text


### ------------- Rollout ---------------- ###
@torch.no_grad
def rollout(model_inputs, model, max_new_tokens, temperature, rollout_bs, pad_token_id):
    model.eval()
    device = next(model.parameters()).device
    N, inp_len = model_inputs["attention_mask"].shape
    all_sequence_ids = torch.full(
        (N, inp_len + max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    all_generated_ids = torch.full(
        (N, max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    assert N % rollout_bs == 0, "N must be divisible by rollout_bs"
    num_batches = N // rollout_bs
    # use mini batch to avoid oom
    for i in range(num_batches):
        batch_start = i * rollout_bs
        batch_end = (i + 1) * rollout_bs
        batch_model_inputs = {
            k: v[batch_start:batch_end] for k, v in model_inputs.items()
        }
        output = model.generate(
            **batch_model_inputs,
            return_dict_in_generate=True,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
        )
        sequence_ids = output.sequences
        generated_ids = sequence_ids[:, inp_len:]
        actual_gen_len = generated_ids.shape[1]

        all_sequence_ids[batch_start:batch_end, : inp_len + actual_gen_len] = (
            sequence_ids
        )
        all_generated_ids[batch_start:batch_end, :actual_gen_len] = generated_ids
    model.train()
    return {
        "sequence_ids": all_sequence_ids,
        "generated_ids": all_generated_ids,
    }


### ------------- Froward ---------------- ###
def get_log_probs(logits, generated_ids, temperature):
    """
    logits: (N, gen_len, vocab)  -- 每一步生成前的 logits
    generated_ids: (N, gen_len)  -- 实际采样出的 token id
    return: (N, gen_len)  -- 每个生成 token 的 log prob
    """
    # logits = logits / temperature
    log_probs_all = torch.log_softmax(logits, dim=-1)  # (N, gen_len, vocab)
    log_probs = torch.gather(
        log_probs_all, dim=-1, index=generated_ids.unsqueeze(-1)  # (N, gen_len, 1)
    ).squeeze(
        -1
    )  # (N, gen_len)
    return log_probs


def forward_get_log_probs(
    sequence_ids, model, attn_mask, inp_len, generated_ids, temperature
):
    """whole seq forward and get log prob for gen ids"""
    position_ids = attn_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn_mask == 0, 1)
    logits = model(
        sequence_ids, attention_mask=attn_mask, position_ids=position_ids
    ).logits  # (N, inp_len + gen_len, vocab)
    gen_logits = logits[:, inp_len - 1 : -1]  # (N, gen_len, vocab)
    gen_log_prob = get_log_probs(gen_logits, generated_ids, temperature)
    return gen_log_prob


### ------------- Dataset ---------------- ###
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
        # if self.split == 'train': # debug only
        #     return 200
        # else:
        #     return 100
        return len(self.data[self.split])


def collect_fn(batch):
    prompts = [_[0] for _ in batch]
    ans = [_[1] for _ in batch]
    return prompts, ans


### ------------- RLVR ---------------- ###
def extract_ans(text):
    match = re.search(r"\\boxed{(.*?)}", text)
    if match:
        result = match.group(1)
        return result.strip()
    else:
        return ""


def cal_reward(pred: str, gt: str):
    format_r, content_r = 0, 0
    if "## Reasoning" in pred and "## Answer" in pred:
        format_r = 1
    else:
        return 0, 0
    pred_ans = pred.split("## Answer")[-1]
    extracted_ans = extract_ans(pred_ans)
    if extracted_ans.strip() == gt.strip():  # only for simple case, assume int answer
        content_r = 1
    return format_r, content_r


@torch.no_grad
def evaluate(model, tokenizer, val_dataloader, max_new_tokens):
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


def kl_div(x, is_log=True):
    if is_log:
        return torch.exp(x) - 1 - x
    return x - 1 - torch.log(x)


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
    is_dist = int(os.environ.get("WORLD_SIZE", 1)) > 1
    rank, device = setup_dist()
    is_main_process = rank == 0
    if is_main_process:
        print(f"[config] loaded from {args.config}:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # load model
    tokenizer = AutoTokenizer.from_pretrained(student_model_name)
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

    # setting up tokenizer, assume tokenizer is the same as teacher tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # for batch infer

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
    train_dataset = PromptDataset(data_p)
    if is_dist:
        train_dist_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_prompt_size,
            collate_fn=collect_fn,
            sampler=train_dist_sampler,
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
    if is_main_process:
        print("len train dataset: ", len(train_dataset))
        print("len train dataloader: ", len(train_dataloader))

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

    val_dataset = PromptDataset(data_p, split="test")
    if is_dist:
        val_dist_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            collate_fn=collect_fn,
            sampler=val_dist_sampler,
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

    if is_main_process:
        print("len val dataset: ", len(val_dataset))
        print("len val dataloader: ", len(val_dataloader))

    ckp_dir = os.path.join(save_root, "checkpoints")
    if is_main_process:
        os.makedirs(ckp_dir, exist_ok=True)

    opt.zero_grad()
    g_step = 0  # number of actual optimizer updates (opt.step calls)

    # total number of optimizer updates over the whole run, for step-wise lr decay
    total_train_steps = num_epoch * len(train_dataloader)
    if is_main_process:
        print(
            f"total_train_steps (optimizer updates, for lr decay): {total_train_steps}"
        )

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

    # init eval (skip when resuming: the loaded model has already been evaluated)
    if not resume_from:
        if is_main_process:
            print(f"init eval start")
        val_f_r, val_c_r = evaluate(
            lora_model, tokenizer, val_dataloader, max_new_tokens
        )
        if is_main_process:
            print(f"init eval end, val_f_r: {val_f_r}, val_c_r: {val_c_r}\n\n\n")
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
            generated_ids = rollout_res["generated_ids"]  # (N, gen_len)
            gen_mask = (generated_ids != tokenizer.pad_token_id).to(lora_model.dtype)
            gen_len = gen_mask.sum(dim=-1)  # (N, )
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

            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)
            total_samples = len(generated_text_lst)
            assert total_samples % ppo_train_mini_bs == 0
            world_size = 1 if not is_dist else dist.get_world_size()
            num_batches = total_samples // ppo_train_mini_bs
            # shuffle batch
            batch_indices = torch.arange(total_samples, device=device)
            for mb_idx in range(num_batches):
                # step-wise lr scheduler (linear decay from max_lr to min_lr over optimizer updates)
                progress = min(g_step / max(1, total_train_steps), 1.0)
                cur_lr = max_lr - (max_lr - min_lr) * progress
                for p_g in opt.param_groups:
                    p_g["lr"] = cur_lr

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
                mask = make_loss_mask(mb_gen_ids, tokenizer.pad_token_id).to(
                    lora_model.dtype
                )
                # PPO ratio: zero out padding positions before exp, and clamp to avoid fp32 overflow
                with torch.no_grad():
                    adv =  - (policy_log_prob - teacher_log_prob) * mask
                reverse_kl = -policy_log_prob * adv.detach() * mask
                mean_reverse_kl = reverse_kl.sum() / mask.sum().item()
                
                if mb_idx == num_batches - 1:
                    mean_reverse_kl.backward()
                else:
                    if is_dist:
                        with ddp_model.no_sync():
                            mean_reverse_kl.backward()
                    else:
                        mean_reverse_kl.backward()

            if is_dist:
                dist.all_reduce(mean_reverse_kl, op=dist.ReduceOp.SUM)
                mean_reverse_kl = mean_reverse_kl / world_size
            if is_main_process:
                logger.add_scalar(
                    "train/mean_reverse_kl", mean_reverse_kl.item(), global_step=g_step
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters() if is_dist else lora_model.parameters(),
                max_grad_norm,
            )
            if is_main_process:
                logger.add_scalar(
                    "train/grad_norm", grad_norm.item(), global_step=g_step
                )
                logger.add_scalar("lr", cur_lr, global_step=g_step)
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
                    lora_model, tokenizer, val_dataloader, max_new_tokens
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
