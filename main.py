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
    is_dist = int(os.environ.get('WORLD_SIZE', 1)) > 1
    if is_dist:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, f"cuda:{local_rank}"
    else:
        return 0, 'cuda:0'

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
    all_sequence_ids = torch.full((N, inp_len + max_new_tokens), pad_token_id, dtype=torch.long, device=device)
    all_generated_ids = torch.full((N, max_new_tokens), pad_token_id, dtype=torch.long, device=device)
    all_log_probs_old = torch.zeros((N, max_new_tokens), dtype=torch.float, device=device)
    assert N % rollout_bs == 0, "N must be divisible by rollout_bs"
    num_batches = N // rollout_bs
    # use mini batch to avoid oom
    for i in range(num_batches):
        batch_start = i * rollout_bs
        batch_end = (i + 1) * rollout_bs
        batch_model_inputs = {k: v[batch_start:batch_end] for k, v in model_inputs.items()}
        output = model.generate(
            **batch_model_inputs,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
        )
        sequence_ids, logits_old = (
            output.sequences,
            output.logits,
        )  # seq_ids: (N, inp_len + gen_len)
        logits_old = torch.stack(logits_old, dim=1)  # (N, gen_len, vocab)
        generated_ids = sequence_ids[:, inp_len:]
        log_probs_old = get_log_probs(logits_old, generated_ids, temperature)
        del logits_old
        actual_gen_len = generated_ids.shape[1]
        
        all_sequence_ids[batch_start:batch_end, :inp_len + actual_gen_len] = sequence_ids
        all_generated_ids[batch_start:batch_end, :actual_gen_len] = generated_ids
        all_log_probs_old[batch_start:batch_end, :actual_gen_len] = log_probs_old
    model.train()
    return {
        "sequence_ids": all_sequence_ids,
        "log_probs_old": all_log_probs_old,
        "generated_ids": all_generated_ids,
    }


### ------------- Froward ---------------- ###
def get_log_probs(logits, generated_ids, temperature):
    """
    logits: (N, gen_len, vocab)  -- 每一步生成前的 logits
    generated_ids: (N, gen_len)  -- 实际采样出的 token id
    return: (N, gen_len)  -- 每个生成 token 的 log prob
    """
    logits = logits / temperature
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
    if extracted_ans.strip() == gt.strip(): # only for simple case, assume int answer
        content_r = 1
    return format_r, content_r


@torch.no_grad
def evaluate(model, tokenizer, val_dataloader, max_new_tokens):
    model.eval()
    device = next(model.parameters()).device
    format_r_sum = torch.tensor([0], device=device, dtype=torch.float)
    content_r_sum = torch.tensor([0], device=device, dtype=torch.float)
    num_sample_sum = torch.tensor([0], device=device, dtype=torch.float)
    for d_idx, (q_lst, a_lst) in enumerate(tqdm(val_dataloader, desc=f"val", total=len(val_dataloader))):
        prompt_lst = build_prompt(tokenizer, q_lst)
        # tokenize
        model_inputs = tokenizer(
            prompt_lst, padding=True, padding_side="left", return_tensors="pt"
        ).to(model.device)
        inp_len = model_inputs["input_ids"].shape[1]
        seq_ids = model.generate(
            **model_inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens
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
    mask = (x != pad_token_id)
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
                if name.startswith("step_") and os.path.isdir(os.path.join(ckp_dir, name)):
                    try:
                        candidates.append((int(name.split("_")[1]), name))
                    except (ValueError, IndexError):
                        pass
        if not candidates:
            raise FileNotFoundError(f"resume_from='latest' but no step_* checkpoint under {ckp_dir}")
        return os.path.join(ckp_dir, max(candidates)[1])
    if os.path.isabs(resume_from) or os.path.exists(resume_from):
        return resume_from
    return os.path.join(ckp_dir, resume_from)


def load_checkpoint(ckp_path, lora_model, opt, device):
    """Load LoRA adapter weights + optimizer state + training counters from a checkpoint dir.
    Returns the saved training_state dict."""
    from safetensors.torch import load_file
    adapter_sd = load_file(os.path.join(ckp_path, "adapter_model.safetensors"), device=str(device))
    set_peft_model_state_dict(lora_model, adapter_sd)
    state = torch.load(os.path.join(ckp_path, "training_state.pt"), map_location=device)
    opt.load_state_dict(state["optimizer_state_dict"])
    # make sure optimizer state tensors live on the current device
    for st in opt.state.values():
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    return state


if __name__ == '__main__':
    # ------------- Config ---------------- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

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
    eps = cfg.get("eps", 0.2)
    # asymmetric clip: lower bound uses (1 - eps_low), upper bound uses (1 + eps_high);
    # both default to eps so symmetric configs keep working
    eps_low = cfg.get("eps_low", eps)
    eps_high = cfg.get("eps_high", eps)
    w_format_r = cfg["w_format_r"]
    use_ref_model = cfg["use_ref_model"]
    kl_weight = cfg["kl_weight"]
    train_prompt_size = cfg["train_prompt_size"]  # per_device, train_len = len(train set) / (train_prompt_size * world_size)
    num_gen = cfg["num_gen"]  # each rollout samples train_prompt_size * num_gen
    gradient_accumulation_steps = cfg["gradient_accumulation_steps"]  # gradient accumulation
    ppo_train_mini_bs = cfg["ppo_train_mini_bs"]  # effective train_bs = ppo_train_mini_bs * world_size * gradient_accumulation_steps
    ppo_num_epoch = cfg["ppo_num_epoch"]  # each rollout train x ep
    rollout_mini_bs = cfg["rollout_mini_bs"]  # bs during rollout to avoid oom
    val_bs = cfg["val_bs"]  # per device eval batch size -> val_len = len(val set) / (val_bs * world_size)
    num_epoch = cfg["num_epoch"]
    eval_ratio = cfg.get("eval_ratio", 0.0)  # eval (1/eval_ratio) times per epoch; 0 = disable mid-epoch eval
    resume_from = cfg.get("resume_from", None)  # None=fresh; "latest"; or a checkpoint dir/name to resume from
    
    loss_type = cfg.get("loss_type", "grpo")
    loss_aggregate_type = cfg.get("loss_aggregate_type", "sample")
    # token-level (Dr.GRPO style) loss: normalize by a fixed constant instead of the actual token count,
    # which removes per-sample length bias and needs no cross-device communication. defaults to max_new_tokens.
    token_norm_const = cfg.get("token_norm_const", max_new_tokens)
    

    # set up dist env
    is_dist = int(os.environ.get('WORLD_SIZE', 1)) > 1
    rank, device = setup_dist()
    is_main_process = rank == 0
    if is_main_process:
        print(f"[config] loaded from {args.config}:")
        print(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))

    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)

    # get lora model
    peft_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, task_type="CAUSAL_LM"
    )
    lora_model = get_peft_model(model, peft_config)
    if is_main_process:
        lora_model.print_trainable_parameters()

    # get ref model, cal kl, avoid policy model get too far from "safe" ref model
    # in lora, can use lora_model.disable_adapter() to decrease memory usage, not used here
    ref_model = None
    if use_ref_model:
        ref_model = AutoModelForCausalLM.from_pretrained(model_name)
        ref_model.to(device)
        ref_model.eval()
        ref_model.requires_grad_(False)

    # setting up tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # for batch infer

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
            train_dataset, batch_size=train_prompt_size, collate_fn=collect_fn, sampler=train_dist_sampler, drop_last=True
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=train_prompt_size, collate_fn=collect_fn, shuffle=True, drop_last=True
        )
    if is_main_process:
        print("len train dataset: ", len(train_dataset))
    if eval_ratio > 0:
        n_evals_per_epoch = max(1, round(1.0 / eval_ratio))
        _total_steps = len(train_dataloader)
        eval_steps = {round(_total_steps * (i + 1) / n_evals_per_epoch) for i in range(n_evals_per_epoch)}
        eval_steps.add(_total_steps)  # always include last step of epoch
    else:
        eval_steps = set()
    if is_main_process and eval_steps:
        print(f"eval_steps per epoch: {sorted(eval_steps)} (total {len(train_dataloader)} steps/epoch, eval_ratio={eval_ratio})")

    val_dataset = PromptDataset(data_p, split="test")
    if is_dist:
        val_dist_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset, batch_size=val_bs, collate_fn=collect_fn, sampler=val_dist_sampler, drop_last=True
        )
    else:
        val_dataloader = DataLoader(
            val_dataset, batch_size=val_bs, collate_fn=collect_fn, shuffle=False, drop_last=True
        )
        
    if is_main_process:
        print("len val dataset: ", len(val_dataset))

    ckp_dir = os.path.join(save_root, "checkpoints")
    if is_main_process:
        os.makedirs(ckp_dir, exist_ok=True)

    opt.zero_grad()
    g_step = 0        # number of actual optimizer updates (opt.step calls)
    micro_step = 0    # number of ppo mini-batches processed (drives gradient accumulation)

    # total number of optimizer updates over the whole run, for step-wise lr decay
    samples_per_rollout = num_gen * train_prompt_size  # per device
    micro_steps_per_rollout = ppo_num_epoch * (samples_per_rollout // ppo_train_mini_bs)
    total_micro_steps = num_epoch * len(train_dataloader) * micro_steps_per_rollout
    total_train_steps = total_micro_steps // gradient_accumulation_steps
    if is_main_process:
        print(f"total_train_steps (optimizer updates, for lr decay): {total_train_steps}")

    # ---- optionally resume from a checkpoint ----
    start_epoch = 0
    resume_skip_batches = 0  # number of batches to skip at the start of start_epoch
    if resume_from:
        ckp_path = resolve_resume_path(resume_from, ckp_dir)
        if is_main_process:
            print(f"[resume] loading checkpoint from {ckp_path}")
        state = load_checkpoint(ckp_path, lora_model, opt, device)
        g_step = state["global_step"]
        micro_step = state["micro_step"]
        start_epoch = state["epoch"]
        resume_skip_batches = state["b_idx"] + 1  # continue at the batch after the saved one
        if is_main_process:
            print(f"[resume] start_epoch={start_epoch}, skip first {resume_skip_batches} "
                  f"batches, g_step={g_step}, micro_step={micro_step}")

    # init eval (skip when resuming: the loaded model has already been evaluated)
    if not resume_from:
        if is_main_process:
            print(f'init eval start')
        val_f_r, val_c_r = evaluate(lora_model, tokenizer, val_dataloader, max_new_tokens)
        if is_main_process:
            print(f'init eval end, val_f_r: {val_f_r}, val_c_r: {val_c_r}\n\n\n')
            logger.add_scalar("val/f_r", val_f_r, global_step=g_step)
            logger.add_scalar("val/c_r", val_c_r, global_step=g_step)


    for ep in range(start_epoch, num_epoch):
        if is_dist:
            train_dataloader.sampler.set_epoch(ep)

        for b_idx, (q_lst, a_lst) in enumerate(
            tqdm(
                train_dataloader, desc=f"ep[{ep}/{num_epoch}]", total=len(train_dataloader)
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
                prompt_lst_expand, padding=True, padding_side="left", return_tensors="pt"
            ).to(lora_model.device)
            N, inp_len = model_inputs["attention_mask"].shape
            rollout_res = rollout(model_inputs, lora_model, max_new_tokens, temperature, rollout_mini_bs, tokenizer.pad_token_id)
            sequence_ids = rollout_res["sequence_ids"]
            seq_mask = make_loss_mask(sequence_ids, tokenizer.pad_token_id)
            log_probs_old = rollout_res["log_probs_old"]
            generated_ids = rollout_res["generated_ids"] # (N, gen_len)
            gen_mask = (generated_ids != tokenizer.pad_token_id).to(log_probs_old.dtype)
            gen_len = gen_mask.sum(dim=-1) # (N, )
            # log gen_len
            if is_dist:
                gen_len_gather_lst = [torch.zeros_like(gen_len) for _ in range(dist.get_world_size())]
                dist.all_gather(gen_len_gather_lst, gen_len)
                gen_len = torch.cat(gen_len_gather_lst, dim=0)
            if is_main_process:
                mean_gen_len = gen_len.mean().item()
                max_gen_len = gen_len.max().item()
                min_gen_len = gen_len.min().item()
                # a sequence is "clipped" when it used up the whole length budget (no natural EOS stop)
                gen_len_clip_frac = torch.sum((gen_len == max_new_tokens).float()) / gen_len.shape[0]
                logger.add_scalar("rollout/mean_gen_len", mean_gen_len, global_step=g_step)
                logger.add_scalar("rollout/max_gen_len", max_gen_len, global_step=g_step)
                logger.add_scalar("rollout/min_gen_len", min_gen_len, global_step=g_step)
                logger.add_scalar("rollout/gen_len_clip_frac", gen_len_clip_frac, global_step=g_step)
            
            
            generated_text_lst = get_generated_text_lst(generated_ids, tokenizer)

            # cal adv
            with torch.no_grad():
                f_r_sum_tensor = torch.tensor([0], device=device)
                c_r_sum_tensor = torch.tensor([0], device=device)
                num_sample_tensor = torch.tensor([0], device=device)
                reward_lst_ = [
                    cal_reward(gen_pred, gt)
                    for gen_pred, gt in zip(generated_text_lst, a_lst_expand)
                ]
                format_r_lst = [x[0] for x in reward_lst_]
                content_r_lst = [x[1] for x in reward_lst_]
                reward_lst = [
                    f_r * w_format_r + c_r for f_r, c_r in zip(format_r_lst, content_r_lst)
                ]
                f_r_sum_tensor += sum(format_r_lst)
                c_r_sum_tensor += sum(content_r_lst)
                num_sample_tensor += len(a_lst_expand)
                if is_dist:
                    dist.reduce(f_r_sum_tensor, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(c_r_sum_tensor, dst=0, op=dist.ReduceOp.SUM)
                    dist.reduce(num_sample_tensor, dst=0, op=dist.ReduceOp.SUM)
                if is_main_process:
                    logger.add_scalar(
                        tag="rollout/f_r",
                        scalar_value=(f_r_sum_tensor / num_sample_tensor).item(), # mean f_r across all gpu
                        global_step=g_step,
                    )
                    logger.add_scalar(
                        tag="rollout/c_r",
                        scalar_value=(c_r_sum_tensor / num_sample_tensor).item(),
                        global_step=g_step,
                    )
                reward = torch.tensor(reward_lst, device=device)  # (N, )
                mean = (
                    reward.reshape(-1, num_gen) # (b, num_gen)
                    .mean(dim=-1) # (b)
                    .repeat_interleave(num_gen)
                )  # (N, )
                std = reward.reshape(-1, num_gen).std(dim=-1, unbiased=False).repeat_interleave(num_gen)
                adv = (reward - mean) / (std + 1e-5)
                adv = adv.unsqueeze(-1)  # (N, 1)
                
                # log std
                std_to_log = reward.reshape(-1, num_gen).std(dim=-1, unbiased=False)
                if is_dist:
                    std_gather = [torch.zeros_like(std_to_log) for _ in range(dist.get_world_size())]
                    dist.all_gather(std_gather, std_to_log)
                    std_to_log = torch.cat(std_gather, dim=0)
                    if is_main_process:
                        logger.add_scalar(
                            tag="rollout/std",
                            scalar_value=std_to_log.mean().item(),
                            global_step=g_step,
                        )
                else:
                    logger.add_scalar(
                        tag="rollout/std",
                        scalar_value=std_to_log.mean().item(),
                        global_step=g_step,
                    )

            total_samples = len(generated_text_lst)
            assert total_samples % ppo_train_mini_bs == 0
            world_size = 1 if not is_dist else dist.get_world_size()
            assert (total_samples * ppo_num_epoch) % (ppo_train_mini_bs * gradient_accumulation_steps * world_size) == 0
            for ppo_epoch in range(ppo_num_epoch):
                num_batches = total_samples // ppo_train_mini_bs
                # shuffle batch
                batch_indices = torch.randperm(total_samples, device=device)
                for mb_idx in range(num_batches):
                    # step-wise lr scheduler (linear decay from max_lr to min_lr over optimizer updates)
                    progress = min(g_step / max(1, total_train_steps), 1.0)
                    cur_lr = max_lr - (max_lr - min_lr) * progress
                    for p_g in opt.param_groups:
                        p_g["lr"] = cur_lr

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
                        temperature,
                    )
                    
                    if use_ref_model:
                        with torch.no_grad():
                            ref_log_prob = forward_get_log_probs(
                                mb_seq_ids,
                                ref_model,
                                mb_seq_mask,
                                inp_len,
                                mb_gen_ids,
                                temperature,
                            ) # (N, gen_len)
                    

                    # loss mask: which generated positions are real tokens (not padding)
                    mask = make_loss_mask(mb_gen_ids, tokenizer.pad_token_id).to(mb_log_prob.dtype)

                    # PPO ratio: zero out padding positions before exp, and clamp to avoid fp32 overflow
                    log_ratio = (mb_log_prob - mb_old_log_prob.detach()) * mask
                    if loss_type == 'grpo':
                        ratio = torch.exp(log_ratio.clamp(-20, 20))  # (mb, gen_len) IMPORTANT, clamp when use exp, solution to model.generate() prob inf,nan error
                        sour1 = torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * mb_adv.detach()
                        sour2 = ratio * mb_adv.detach()

                        pg_loss_matrix = torch.min(sour1, sour2) * mask
                    elif loss_type == 'gspo':
                        log_seq_ratio = torch.sum(log_ratio, dim=-1, keepdim=True) / (mask.sum(dim=-1, keepdim=True) + 1e-5)  # (mb, 1)
                        seq_ratio = torch.exp(log_seq_ratio.clamp(-20, 20))  # (mb, 1)
                        sour1 = torch.clamp(seq_ratio, 1 - eps_low, 1 + eps_high) * mb_adv.detach()  # (mb, 1)
                        sour2 = seq_ratio * mb_adv.detach()  # (mb, 1)
                        pg_loss_matrix = torch.min(sour1, sour2) * mask  # (mb, 1) * (mb, gen_len) -> (mb, gen_len)

                    elif loss_type == 'cispo':
                        ratio = torch.exp(log_ratio.clamp(-20, 20)) 
                        clamped_ratio = torch.clamp(ratio, 1 - eps_low, 1 + eps_high)
                        pg_loss_matrix = clamped_ratio.detach() * mb_adv.detach() * (mb_log_prob * mask)
                    else:
                        raise ValueError(f"Invalid loss_type: {loss_type}")
                    
                    if loss_aggregate_type == 'sample':
                        pg_loss_samplewise = pg_loss_matrix.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5)
                        pg_loss = - pg_loss_samplewise.mean()
                    elif loss_aggregate_type == 'token':
                        # Dr.GRPO style: normalize by a fixed constant (default max_new_tokens) rather than the
                        # actual token count. this gives every token equal weight, removes per-sample length bias,
                        # and combines correctly across ranks / grad-accum with no extra communication.
                        pg_loss = - pg_loss_matrix.sum(dim=-1).mean() / token_norm_const
                    else:
                        raise ValueError(f"Invalid loss_aggregate_type: {loss_aggregate_type}")


                    if use_ref_model:
                        # mask BEFORE exp so padding positions give kl_div(0)=0 (avoids inf*0 -> NaN);
                        # clamp keeps exp() from overflowing when policy drifts away from ref
                        log_diff = ((ref_log_prob - mb_log_prob) * mask).clamp(-20, 20)
                        kl_loss_matrix = kl_div(log_diff, is_log=True) * mask
                        kl_loss = kl_loss_matrix.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5)
                        kl_loss = kl_loss.mean()
                    
                    if use_ref_model:
                        loss = pg_loss + kl_loss * kl_weight
                    else:
                        loss = pg_loss
                        
                    is_accumulating = (micro_step + 1) % gradient_accumulation_steps != 0
                    sync_context = ddp_model.no_sync() if (is_dist and is_accumulating) else nullcontext()

                    with sync_context:
                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                    if (micro_step + 1) % gradient_accumulation_steps == 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters() if is_dist else lora_model.parameters(), max_grad_norm)
                        if is_main_process:
                            logger.add_scalar("train/grad_norm", grad_norm.item(), global_step=g_step)
                            logger.add_scalar("lr", cur_lr, global_step=g_step)
                        # safety net: never let a non-finite grad poison the weights (would crash rollout sampling)
                        if not torch.isfinite(grad_norm):
                            if is_main_process:
                                print(f"[warn] non-finite grad_norm at g_step {g_step}, skipping optimizer step")
                            opt.zero_grad(set_to_none=True)
                            micro_step += 1
                            continue
                        opt.step()
                        opt.zero_grad()
                        with torch.no_grad():
                            # log loss
                            pg_loss_log = pg_loss.detach().clone()
                            if use_ref_model:
                                kl_loss_log = kl_loss.detach().clone()
                            if is_dist:
                                dist.reduce(pg_loss_log, dst=0, op=dist.ReduceOp.SUM)
                                if use_ref_model:
                                    dist.reduce(kl_loss_log, dst=0, op=dist.ReduceOp.SUM)

                            if is_main_process:
                                logger.add_scalar(
                                    tag="train/pg_loss", 
                                    scalar_value=pg_loss_log.item() / dist.get_world_size() if is_dist else pg_loss_log.item(), 
                                    global_step=g_step
                                )
                                if use_ref_model:
                                    logger.add_scalar(
                                        tag="train/kl_loss", 
                                        scalar_value=kl_loss_log.item() / dist.get_world_size() if is_dist else kl_loss_log.item(), 
                                        global_step=g_step
                                    )
                                    
                            # log entropy
                            entropy = -mb_log_prob * mask.to(mb_log_prob.dtype)  # -plogp
                            entropy = entropy.sum(dim=-1) / (mask.sum(dim=-1) + 1e-5) # per_sample (N, )
                            
                            if is_dist:
                                entropy_gather = [torch.zeros_like(entropy) for _ in range(dist.get_world_size())]
                                dist.all_gather(entropy_gather, entropy)
                                entropy = torch.cat(entropy_gather, dim=0)
                            entropy = entropy.mean()
                            
                            if is_main_process:
                                logger.add_scalar(
                                    tag="train/entropy",
                                    scalar_value=entropy.item(),
                                    global_step=g_step,
                                )
                                
                        
                            # log clip ratio
                            with torch.no_grad():
                                if loss_type == 'cispo':
                                    is_clipped = ratio != clamped_ratio
                                    clip_frac_matrix = is_clipped * mask # (N, gen_len)
                                else:
                                    is_clipped = (sour1 < sour2).to(mask.dtype)
                                    clip_frac_matrix = is_clipped * mask # (N, gen_len)
                                
                                if is_dist:
                                    clip_frac_gather = [torch.zeros_like(clip_frac_matrix) for _ in range(dist.get_world_size())]
                                    gen_mask_gather = [torch.zeros_like(mask) for _ in range(dist.get_world_size())]
                                    dist.all_gather(gen_mask_gather, mask)
                                    gen_mask_gather = torch.cat(gen_mask_gather, dim=0)
                                    dist.all_gather(clip_frac_gather, clip_frac_matrix)
                                    clip_frac_matrix = torch.cat(clip_frac_gather, dim=0)
                                    clip_frac = clip_frac_matrix.sum() / (gen_mask_gather.sum() + 1e-5)
                                    if is_main_process:
                                        logger.add_scalar(
                                            tag="train/clip_frac",
                                            scalar_value=clip_frac.item(),
                                            global_step=g_step,
                                        )
                                else:
                                    clip_frac = clip_frac_matrix.sum() / (mask.sum() + 1e-5)
                                    logger.add_scalar(
                                        tag="train/clip_frac",
                                        scalar_value=clip_frac.item(),
                                        global_step=g_step,
                                    )
                        g_step += 1  # count only actual optimizer updates
                    micro_step += 1
                    
                
            # eval at configured fractions of epoch
            if (b_idx + 1) in eval_steps:
                val_f_r, val_c_r = evaluate(lora_model, tokenizer, val_dataloader, max_new_tokens)
                if is_main_process:
                    frac = (b_idx + 1) / len(train_dataloader)
                    print(f'eval ep {ep} step {b_idx+1}/{len(train_dataloader)} ({frac:.0%}): val_f_r={val_f_r:.4f}, val_c_r={val_c_r:.4f}')
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
                    torch.save({
                        "epoch": ep,
                        "b_idx": b_idx,
                        "global_step": g_step,
                        "micro_step": micro_step,
                        "optimizer_state_dict": opt.state_dict(),
                    }, os.path.join(ckp_path, "training_state.pt"))
                    print(f"Checkpoint saved to {ckp_path}")
