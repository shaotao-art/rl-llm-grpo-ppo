import torch


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


def repeat_lst(lst: list, num_gen: int):
    """repeat input prompt for num_gen [1, 1, 1, 1, 2, 2, 2, 2, ...]"""
    return [item for item in lst for _ in range(num_gen)]


def get_generated_text_lst(generated_ids: torch.Tensor, tokenizer):
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_text


@torch.no_grad
def rollout(
    model_inputs,
    model,
    max_new_tokens,
    temperature,
    rollout_bs,
    pad_token_id,
    return_log_probs=False,
):
    """
    rollout model
    Args:
        model_inputs: dict of model inputs, paded batch input seq, (N, inp_len)
        model: model to rollout
        max_new_tokens: max new tokens to generate
        temperature: temperature for sampling
        rollout_bs: batch size for rollout
        pad_token_id: pad token id
        return_log_probs: whether to return log probs
    Returns:
        dict of rollout results
        - sequence_ids: (N, inp_len + max_new_tokens)
        - log_probs_old: (N, max_new_tokens)
        - generated_ids: (N, max_new_tokens)
    """
    model.eval()
    device = next(model.parameters()).device
    N, inp_len = model_inputs["attention_mask"].shape
    all_sequence_ids = torch.full(
        (N, inp_len + max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    all_generated_ids = torch.full(
        (N, max_new_tokens), pad_token_id, dtype=torch.long, device=device
    )
    if return_log_probs:
        all_log_probs_old = torch.zeros(
            (N, max_new_tokens), dtype=torch.float, device=device
        )
    else:
        all_log_probs_old = None
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
            output_logits=return_log_probs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=1.0,
            top_k=0,
        )
        sequence_ids = output.sequences  # seq_ids: (N, inp_len + gen_len)
        generated_ids = sequence_ids[:, inp_len:]
        actual_gen_len = generated_ids.shape[1]
        all_sequence_ids[batch_start:batch_end, : inp_len + actual_gen_len] = (
            sequence_ids
        )
        all_generated_ids[batch_start:batch_end, :actual_gen_len] = generated_ids

        if return_log_probs:
            logits_old = output.logits
            logits_old = torch.stack(logits_old, dim=1)  # (N, gen_len, vocab)
            log_probs_old = get_log_probs(logits_old, generated_ids)
            del logits_old
            all_log_probs_old[batch_start:batch_end, :actual_gen_len] = log_probs_old
    model.train()
    return {
        "sequence_ids": all_sequence_ids,
        "log_probs_old": all_log_probs_old,
        "generated_ids": all_generated_ids,
    }


def get_log_probs(logits, generated_ids):
    """get log probs for generated ids

    logits: (N, gen_len, vocab)  -- 每一步生成前的 logits
    generated_ids: (N, gen_len)  -- 实际采样出的 token id
    return: (N, gen_len)  -- 每个生成 token 的 log prob
    """
    log_probs_all = torch.log_softmax(logits, dim=-1)  # (N, gen_len, vocab)
    log_probs = torch.gather(
        log_probs_all, dim=-1, index=generated_ids.unsqueeze(-1)  # (N, gen_len, 1)
    ).squeeze(
        -1
    )  # (N, gen_len)
    return log_probs


def left_pad_position_ids(attn_mask):
    """position_ids for a left-padded batch: real tokens get 0..len-1, pads get 1."""
    position_ids = attn_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn_mask == 0, 1)
    return position_ids


def forward_get_log_probs(sequence_ids, model, attn_mask, inp_len, generated_ids):
    """whole seq forward and get log prob for gen ids"""
    logits = model(
        sequence_ids,
        attention_mask=attn_mask,
        position_ids=left_pad_position_ids(attn_mask),
    ).logits  # (N, inp_len + gen_len, vocab)
    gen_logits = logits[:, inp_len - 1 : -1]  # (N, gen_len, vocab)
    gen_log_prob = get_log_probs(gen_logits, generated_ids)
    return gen_log_prob


def make_loss_mask(x, pad_token_id):
    """(N, l) token ids -> (N, l) bool mask.

    Real tokens are True; padding is False, EXCEPT that the first pad token after
    the real content is kept True — since pad_token == eos_token, that position is
    the natural <eos> and must contribute to the loss. Truncated rows (no trailing
    pad) are left untouched; all-pad rows get only their first token set.
    """
    mask = x != pad_token_id
    if x.shape[1] == 1:  # degenerate length: original loop marks the only token
        return torch.ones_like(mask)
    # transition: pad at j following a real token at j-1
    trans = (x[:, 1:] == pad_token_id) & mask[:, :-1]  # (N, l-1), index j-1
    has_trans = trans.any(dim=1)
    last_is_pad = x[:, -1] == pad_token_id
    # largest such j (loop in the original impl scanned from the end)
    last_trans = trans.shape[1] - 1 - trans.flip(dims=[1]).float().argmax(dim=1)
    rows = last_is_pad & has_trans
    mask[rows, last_trans[rows] + 1] = True
    # no transition found (e.g. all-pad row): mark the first token
    rows_nf = last_is_pad & ~has_trans
    mask[rows_nf, 0] = True
    return mask
