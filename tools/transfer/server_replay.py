"""Replay AReaL's interrupt-update-resume step in the real SGLang server and compare full re-prefill with KV transfer.

For one policy transition (source checkpoint -> target checkpoint) the server:
  1. loads the source weights and starts sampled GSM8K rollouts (temperature 1.0, as AReaL does);
  2. is interrupted by loading the target weights with abort_all_requests, which retains every running request's cache;
  3. receives each interrupted request again (prompt + partial output) twice, both greedy so the pair is comparable:
       transfer  the first send finds the retained cache, which the server translates with the fitted map;
       baseline  the second send finds nothing and prefills the whole prefix, which is what AReaL does today.
Per request it records time to first token (first streamed chunk), the completed answer and its GSM8K reward under
both treatments, and whether the two continuations agree. Sends are concurrent (``--send concurrent``, how AReaL
resumes) or one at a time (``--send sequential``, latency without queueing).
"""

import argparse
import asyncio
import json
import pathlib
import statistics
import sys
import time

import aiohttp

GSM8K_SUFFIX = "\nPlease put your final answer within \\boxed{}."  # areal/dataset/gsm8k.py


async def post_json(session, url, path, payload):
    async with session.post(f"{url}{path}", json=payload) as response:
        body = await response.json()
        if response.status != 200:
            raise RuntimeError(f"{path} returned {response.status}: {body}")
        return body


def sampling(max_new_tokens: int, temperature: float, *, ignore_eos: bool = False) -> dict:
    return {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": 1.0, "ignore_eos": ignore_eos}


async def generate(session, url, input_ids, *, max_new_tokens, temperature, ignore_eos=False):
    return await post_json(session, url, "/generate", {"input_ids": input_ids, "sampling_params": sampling(max_new_tokens, temperature, ignore_eos=ignore_eos), "return_logprob": True, "stream": False})


async def generate_streaming(session, url, input_ids, *, max_new_tokens):
    """Greedy streaming generate; returns (ttft_ms, total_ms, output token ids, final meta_info)."""
    payload = {"input_ids": input_ids, "sampling_params": sampling(max_new_tokens, 0.0), "return_logprob": True, "stream": True}
    start = time.perf_counter()
    ttft_ms, last = None, None
    async with session.post(f"{url}/generate", json=payload) as response:
        if response.status != 200:
            raise RuntimeError(f"/generate stream returned {response.status}: {await response.text()}")
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - start) * 1000.0
            last = json.loads(line[len("data:"):])
    if last is None:
        raise RuntimeError("stream produced no data chunk")
    meta = last["meta_info"]
    tokens = [item[1] for item in meta["output_token_logprobs"]]
    return ttft_ms, (time.perf_counter() - start) * 1000.0, tokens, meta


async def resume_all(session, url, prefixes, budgets, *, send: str):
    """Resume every prefix greedily; concurrent sends go together, sequential ones wait for each other."""
    async def one(prefix, budget):
        return await generate_streaming(session, url, prefix, max_new_tokens=budget)
    if send == "concurrent":
        return await asyncio.gather(*(one(p, b) for p, b in zip(prefixes, budgets)))
    return [await one(p, b) for p, b in zip(prefixes, budgets)]


def quantile_summary(values: list[float]) -> dict:
    ordered = sorted(values)
    return {"n": len(ordered), "p50": statistics.median(ordered), "p90": ordered[int(0.9 * (len(ordered) - 1))], "mean": statistics.fmean(ordered)}


async def run(args) -> dict:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from areal.reward.gsm8k import gsm8k_reward_fn

    tokenizer = AutoTokenizer.from_pretrained(args.target)
    split = json.loads(args.splits.read_text())["indices"][args.split]
    indices = split[args.offset : args.offset + args.count]
    train = load_dataset(path="openai/gsm8k", name="main", split="train")
    prompts = []
    for index in indices:
        text = tokenizer.apply_chat_template([{"role": "user", "content": train[index]["question"] + GSM8K_SUFFIX}], add_generation_prompt=True, tokenize=False)
        prompts.append(list(tokenizer(text, add_special_tokens=False)["input_ids"]))
    timeout = aiohttp.ClientTimeout(total=1800)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        await post_json(session, args.url, "/update_weights_from_disk", {"model_path": str(args.source), "abort_all_requests": False, "weight_version": str(args.source_version)})
        # --ignore-eos keeps every rollout generating past its answer: synthetic long prefixes for latency only.
        rollouts = [asyncio.create_task(generate(session, args.url, p, max_new_tokens=args.max_new_tokens, temperature=1.0, ignore_eos=args.ignore_eos)) for p in prompts]
        await asyncio.sleep(args.abort_after_s)
        update_start = time.perf_counter()
        await post_json(session, args.url, "/update_weights_from_disk", {"model_path": str(args.target), "abort_all_requests": True, "weight_version": str(args.target_version)})
        update_s = time.perf_counter() - update_start
        first = await asyncio.gather(*rollouts)
        interrupted = []
        for index, prompt, result in zip(indices, prompts, first):
            meta = result["meta_info"]
            partial = [item[1] for item in meta["output_token_logprobs"]] if meta.get("output_token_logprobs") else []
            if meta["finish_reason"]["type"] == "abort" and len(partial) >= 1:
                interrupted.append({"dataset_index": index, "prompt": prompt, "partial": partial})
        prefixes = [r["prompt"] + r["partial"] for r in interrupted]
        budgets = [args.resume_new_tokens if args.resume_new_tokens is not None else max(1, args.max_new_tokens - len(r["partial"])) for r in interrupted]
        transfer = await resume_all(session, args.url, prefixes, budgets, send=args.send)
        baseline = await resume_all(session, args.url, prefixes, budgets, send=args.send)
        # Noise floor: re-prefill the same prefixes once more; any difference from the first re-prefill is not the transfer's doing.
        baseline_again = await resume_all(session, args.url, prefixes, budgets, send=args.send)
    rows = []
    for record, (t_ttft, t_total, t_tokens, t_meta), (b_ttft, b_total, b_tokens, b_meta), (_, _, b2_tokens, _) in zip(interrupted, transfer, baseline, baseline_again):
        answer = train[record["dataset_index"]]["answer"]
        def reward(tokens):
            return float(gsm8k_reward_fn(None, tokenizer.decode(record["partial"] + tokens, skip_special_tokens=True), None, None, answer))
        divergence = next((i for i, (a, b) in enumerate(zip(t_tokens, b_tokens)) if a != b), None)
        rows.append({
            "dataset_index": record["dataset_index"], "prefix_tokens": len(record["prompt"]) + len(record["partial"]),
            "transfer_cached_tokens": t_meta.get("cached_tokens"), "baseline_cached_tokens": b_meta.get("cached_tokens"),
            "transfer_ttft_ms": t_ttft, "baseline_ttft_ms": b_ttft, "transfer_total_ms": t_total, "baseline_total_ms": b_total,
            "transfer_reward": reward(t_tokens), "baseline_reward": reward(b_tokens),
            "continuation_equal": t_tokens == b_tokens, "reprefill_repeat_equal": b2_tokens == b_tokens, "reprefill_repeat_reward": reward(b2_tokens), "first_divergence": divergence if divergence is not None else (None if len(t_tokens) == len(b_tokens) else min(len(t_tokens), len(b_tokens))),
            "transfer_new_tokens": len(t_tokens), "baseline_new_tokens": len(b_tokens),
        })
    used = [r for r in rows if r["transfer_cached_tokens"] and r["transfer_cached_tokens"] > 0]
    summary = {
        "source_version": args.source_version, "target_version": args.target_version, "send": args.send,
        "ignore_eos": args.ignore_eos, "abort_after_s": args.abort_after_s,
        "requested": len(prompts), "interrupted": len(interrupted), "transfer_used_cache": len(used), "weight_update_s": update_s,
    }
    if len(rows) != 0:
        summary.update(
            prefix_tokens=quantile_summary([r["prefix_tokens"] for r in rows]),
            transfer_ttft_ms=quantile_summary([r["transfer_ttft_ms"] for r in rows]),
            baseline_ttft_ms=quantile_summary([r["baseline_ttft_ms"] for r in rows]),
            transfer_accuracy=statistics.fmean(r["transfer_reward"] for r in rows),
            baseline_accuracy=statistics.fmean(r["baseline_reward"] for r in rows),
            reward_agreement=statistics.fmean(float(r["transfer_reward"] == r["baseline_reward"]) for r in rows),
            continuation_equal_rate=statistics.fmean(float(r["continuation_equal"]) for r in rows),
            reprefill_repeat_equal_rate=statistics.fmean(float(r["reprefill_repeat_equal"]) for r in rows),
            reprefill_repeat_accuracy=statistics.fmean(r["reprefill_repeat_reward"] for r in rows),
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rows.json").write_text(json.dumps(rows) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--target", type=pathlib.Path, required=True)
    parser.add_argument("--source-version", type=int, required=True)
    parser.add_argument("--target-version", type=int, required=True)
    parser.add_argument("--splits", type=pathlib.Path, required=True)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--abort-after-s", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--send", choices=("concurrent", "sequential"), required=True)
    parser.add_argument("--ignore-eos", action="store_true", help="synthetic long prefixes: rollouts never stop at EOS; accuracy is meaningless in this mode")
    parser.add_argument("--resume-new-tokens", type=int, default=None, help="tokens to generate after the resume; default finishes the answer")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    summary = asyncio.run(run(parser.parse_args(argv)))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
