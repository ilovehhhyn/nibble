"""Shared fork prompt membership/order and unchanged mathematical reward.

The fork uses model seed 1 but DistributedSampler's default shuffle seed 0.
Keep these separate. Evaluation always uses official test indices 0..99.
"""

import hashlib
import json

PROMPT_SUFFIX = "\nPlease put your final answer within \\boxed{}."


def math_reward(completion: str, answer: str) -> float:
    """Exactly the fork MathAgent parser/verifier, with its original defaults."""
    from math_verify import parse, verify

    return float(verify(parse(completion), parse(answer)))


def prepare_training(tokenizer):
    """Return full filtered train set, fork epoch-zero order and content hashes."""
    from datasets import load_dataset
    from torch.utils.data import DistributedSampler

    dataset = load_dataset("openai/gsm8k", "main", split="train")

    def process(row):
        return {
            "messages": [{"role": "user", "content": row["question"] + PROMPT_SUFFIX}]
        }

    dataset = dataset.map(process).remove_columns(["question"])
    dataset = dataset.filter(
        lambda row: len(tokenizer.encode(row["messages"][0]["content"])) <= 1024
    )
    order = list(
        DistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, drop_last=True
        )
    )
    membership = [
        {"messages": row["messages"], "answer": row["answer"]} for row in dataset
    ]
    digest = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    selected = order[: 15 * 256]
    metadata = {
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "filtered_count": len(dataset),
        "membership_sha256": digest,
        "model_seed": 1,
        "sampler_seed": 0,
        "sampler_epoch": 0,
        "requested_prompt_order": selected,
        "prompts_per_update": 256,
        "completions_per_prompt": 4,
        "updates": 15,
        "reward": "math_verify.verify(parse(completion), parse(answer)); fork defaults",
        "test_indices": list(range(100)),
    }
    return dataset, selected, metadata
