"""Isolated score-centering loss overhead on cached causal-LM checkpoints."""

from __future__ import annotations
import argparse
import json
import os
import statistics
from pathlib import Path


def _records(path, count=4, max_tokens=512):
    selected = []
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            p = row.get("prompt_ids")
            c = row.get("completion_ids")
            if not p or not c:
                continue
            p = list(map(int, p))
            c = list(map(int, c))
            if len(p) + len(c) > max_tokens:
                continue
            selected.append((p, c))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(
            f"need {count} prefix records of <= {max_tokens} tokens; found {len(selected)}"
        )
    return selected


def _summary(v):
    return {
        "milliseconds": [round(x, 4) for x in v],
        "mean": round(statistics.mean(v), 4),
        "median": round(statistics.median(v), 4),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint")
    ap.add_argument("output", type=Path)
    ap.add_argument("--prefixes", type=Path, required=True)
    ap.add_argument("--topk", type=int, default=128)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    a = ap.parse_args(argv)
    if a.repeats != 3:
        ap.error("protocol uses exactly three paired repeats")
    import torch
    from transformers import AutoModelForCausalLM
    from .areal_loss import loss_from_logits

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("score-centering GPU benchmark requires Slurm")
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark must run on a GPU inside Slurm")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = (
        AutoModelForCausalLM.from_pretrained(
            a.checkpoint,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .to("cuda")
        .train()
    )
    examples = _records(a.prefixes, max_tokens=a.max_tokens)
    lengths = [len(p) + len(c) for p, c in examples]
    width = max(lengths)
    ids = torch.zeros((4, width), dtype=torch.long, device="cuda")
    mask = torch.zeros((4, width), dtype=torch.bool, device="cuda")
    attention = torch.zeros((4, width), dtype=torch.long, device="cuda")
    for i, (p, c) in enumerate(examples):
        seq = p + c
        ids[i, : len(seq)] = torch.tensor(seq, device="cuda")
        attention[i, : len(seq)] = 1
        mask[i, len(p) - 1 : len(seq) - 1] = True
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        init = model(input_ids=ids, attention_mask=attention, use_cache=False).logits
        active = init.reshape(-1, init.shape[-1])[mask.reshape(-1)].float()
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        sampler_logp = torch.log_softmax(active.detach(), dim=-1)
        q, qids = torch.topk(sampler_logp, a.topk, dim=-1)
        e1.record()
        torch.cuda.synchronize()
        head_ms = e0.elapsed_time(e1)
    flat = mask.reshape(-1)
    counts = [int(mask[i].sum()) for i in range(4)]
    adv = torch.tensor([-1.5, -0.5, 0.5, 1.5], device="cuda").repeat_interleave(
        torch.tensor(counts, device="cuda")
    )
    advantages = torch.zeros((4, width), device="cuda")
    advantages.reshape(-1)[flat] = adv
    data = {
        "input_ids": ids,
        "loss_mask": mask,
        "logprobs": torch.zeros((4, width), device="cuda"),
        "advantages": advantages,
        "sampler_topk_ids": torch.zeros(
            (4, width, a.topk), dtype=torch.long, device="cuda"
        ),
        "sampler_topk_logprobs": torch.zeros((4, width, a.topk), device="cuda"),
    }
    data["sampler_topk_ids"].reshape(-1, a.topk)[flat] = qids
    data["sampler_topk_logprobs"].reshape(-1, a.topk)[flat] = q
    target = ids.roll(-1, dims=1).reshape(-1)[flat]
    data["logprobs"].reshape(-1)[flat] = sampler_logp.gather(
        1, target[:, None]
    ).squeeze(1)

    def trial(enabled):
        os.environ["NIBBLE_SCORE_CENTERING"] = "1" if enabled else "0"
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        st = torch.cuda.Event(enable_timing=True)
        en = torch.cuda.Event(enable_timing=True)
        st.record()
        out = model(input_ids=ids, attention_mask=attention, use_cache=False)
        loss = loss_from_logits(out.logits, data)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        loss.backward()
        en.record()
        torch.cuda.synchronize()
        if any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        ):
            raise FloatingPointError("non-finite gradient")
        return st.elapsed_time(en), float(loss.detach())

    trial(False)
    trial(True)
    timing = {"off": [], "on": []}
    losses = {"off": [], "on": []}
    for i in range(3):
        for enabled in (False, True) if i % 2 == 0 else (True, False):
            ms, val = trial(enabled)
            key = "on" if enabled else "off"
            timing[key].append(ms)
            losses[key].append(val)
    result = {
        "checkpoint": a.checkpoint,
        "prefixes": str(a.prefixes),
        "prefix_count": 4,
        "sequence_lengths": lengths,
        "active_response_tokens": int(mask.sum()),
        "topk": a.topk,
        "sampler_head_logsoftmax_topk_ms": round(head_ms, 4),
        "loss_forward_backward_ms": {k: _summary(v) for k, v in timing.items()},
        "paired_overhead_ms": [
            round(x - y, 4) for x, y in zip(timing["on"], timing["off"])
        ],
        "loss_values": losses,
        "scope": "isolated HF forward plus existing AReaL loss backward; deterministic artificial group advantages; excludes SGLang top-k transport, end-to-end training throughput, accuracy, and variance",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
