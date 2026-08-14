import re

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


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
    prompts = [_[0] for _ in batch]
    ans = [_[1] for _ in batch]
    return prompts, ans


def get_train_dataloader(
    data_p,
    batch_size,
    is_dist=False,
    rank=0,
    is_main_process=True,
):
    """Build the training dataloader (prompt + answer pairs).

    Distributed: uses a shuffling DistributedSampler; otherwise local shuffle.
    """
    train_dataset = PromptDataset(data_p)
    if is_dist:
        train_dist_sampler = DistributedSampler(train_dataset, rank=rank, shuffle=True)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collect_fn,
            sampler=train_dist_sampler,
            drop_last=True,
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collect_fn,
            shuffle=True,
            drop_last=True,
        )
    if is_main_process:
        print("len train dataset: ", len(train_dataset))
        print("len train dataloader: ", len(train_dataloader))
    return train_dataloader


def get_val_dataloader(
    data_p,
    batch_size,
    is_dist=False,
    rank=0,
    is_main_process=True,
):
    """Build the validation dataloader (no shuffle)."""
    val_dataset = PromptDataset(data_p, split="test")
    if is_dist:
        val_dist_sampler = DistributedSampler(val_dataset, rank=rank, shuffle=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collect_fn,
            sampler=val_dist_sampler,
            drop_last=True,
        )
    else:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collect_fn,
            shuffle=False,
            drop_last=True,
        )
    if is_main_process:
        print("len val dataset: ", len(val_dataset))
        print("len val dataloader: ", len(val_dataloader))
    return val_dataloader


### ------------- RLVR ---------------- ###
def extract_ans(text):
    match = re.search(r"\\boxed{(.*?)}", text)
    if match:
        result = match.group(1)
        return result.strip()
    else:
        return ""


def cal_reward(pred: str, gt: str):
    format_r, answer_r = 0.0, 0.0
    if "## Reasoning" in pred:
        format_r += 0.5
    if "## Answer" in pred:
        format_r += 0.5
    extracted_ans = extract_ans(pred)
    if extracted_ans.strip() == gt.strip() and len(extracted_ans) > 0:
        answer_r = 1.0
    return format_r, answer_r
