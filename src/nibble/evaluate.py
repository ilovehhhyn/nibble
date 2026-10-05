"""Common native-cache held-out evaluation, independent of each training arm."""

import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .native import strict_gsm8k_reward
    from .report import atomic_json

    start = time.perf_counter()
    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    test = load_dataset("openai/gsm8k", "main", split="test")
    records = []
    for begin in range(0, args.count, 4):
        rows = [test[i] for i in range(begin, min(begin + 4, args.count))]
        texts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": r["question"]
                        + "\nPut the final answer in \\boxed{}.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for r in rows
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for offset, (row, tokens) in enumerate(
            zip(rows, output[:, inputs["input_ids"].shape[1] :])
        ):
            ids = tokens.tolist()
            if tokenizer.eos_token_id in ids:
                ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
            text = tokenizer.decode(ids)
            reward = strict_gsm8k_reward(None, text, None, ids, row["answer"])
            records.append(
                {
                    "test_index": begin + offset,
                    "reward": float(reward),
                    "output_tokens": len(ids),
                    "truncated": len(ids) >= args.max_new_tokens,
                    "completion": text,
                }
            )
    atomic_json(
        args.output,
        {
            "checkpoint": args.checkpoint,
            "count": len(records),
            "accuracy": sum(r["reward"] for r in records) / len(records),
            "seconds": time.perf_counter() - start,
            "records": records,
            "scope": "first official GSM8K test prompts; deterministic native HF cache; one seed",
        },
    )


if __name__ == "__main__":
    main()
