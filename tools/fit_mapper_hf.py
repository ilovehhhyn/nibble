"""Fit a KV mapper from one checkpoint to another on recorded prefixes and record every cost.

For each prefix record, the calibration sequence is ``prompt + completion[:cut]``. Both models
prefill the same right-padded batch; the native caches and the validity mask stream into the
ridge accumulators. Timing is split into checkpoint hashing, model loading, cache extraction,
solving, and serialization, because the proposal charges each of them to the live path.
"""

import argparse
import json
import pathlib
import sys
import time

import torch


def load_sequences(prefixes: pathlib.Path, *, limit: int | None) -> list[list[int]]:
    sequences = []
    with prefixes.open() as handle:
        for line in handle:
            if line.strip() == "":
                continue
            record = json.loads(line)
            sequences.append(record["prompt_ids"] + record["completion_ids"][: record["cut"]])
    if len(sequences) == 0:
        raise ValueError(f"{prefixes} holds no records")
    return sequences if limit is None else sequences[:limit]


def right_pad(sequences: list[list[int]], *, pad_id: int, max_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``input_ids`` and a prefix ``attention_mask`` ``[batch, tokens]`` (sequences are truncated to ``max_tokens``)."""
    clipped = [s[:max_tokens] for s in sequences]
    width = max(len(s) for s in clipped)
    input_ids = torch.full((len(clipped), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(clipped):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        mask[row, : len(ids)] = 1
    return input_ids, mask


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, name: str, start: float) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds[name] = self.seconds.get(name, 0.0) + (time.perf_counter() - start)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--target", type=pathlib.Path, required=True)
    parser.add_argument("--prefixes", type=pathlib.Path, required=True)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N records")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lambda", dest="lambda_", type=float, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--selection", choices=("same_layer", "top_k"), required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--out", type=pathlib.Path, required=True, help="artifact directory; must not exist")
    parser.add_argument("--provenance", type=str, default="{}", help="JSON merged into the artifact provenance")
    args = parser.parse_args(argv)
    if args.selection == "top_k" and (args.k is None or args.k < 1):
        raise SystemExit("--selection top_k requires --k >= 1")
    if args.selection == "same_layer" and args.k is not None:
        raise SystemExit("--k is only meaningful with --selection top_k")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from areal.experimental.kvmap.contract import FeatureContract
    from areal.experimental.kvmap.fit import CachePair, FitOptions, fit_mapper
    from areal.experimental.kvmap.hf import extract_cache, hash_checkpoint_directory, model_identity_from_config
    from areal.experimental.kvmap.select import select_source_layers_by_r_squared

    timer = Timer()
    dtype = getattr(torch, args.dtype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = time.perf_counter()
    source_sha = hash_checkpoint_directory(args.source)
    target_sha = hash_checkpoint_directory(args.target)
    timer.add("hash_checkpoints", start)
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.target)
    source_model = AutoModelForCausalLM.from_pretrained(args.source, dtype=dtype).to(device).eval()
    target_model = AutoModelForCausalLM.from_pretrained(args.target, dtype=dtype).to(device).eval()
    timer.add("load_models", start)
    source = model_identity_from_config(source_model.config, checkpoint_sha256=source_sha, cache_dtype=dtype)
    target = model_identity_from_config(target_model.config, checkpoint_sha256=target_sha, cache_dtype=dtype)
    sequences = load_sequences(args.prefixes, limit=args.limit)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def pairs():
        for begin in range(0, len(sequences), args.batch_size):
            batch_start = time.perf_counter()
            input_ids, mask = right_pad(sequences[begin : begin + args.batch_size], pad_id=pad_id, max_tokens=args.max_tokens)
            input_ids, mask = input_ids.to(device), mask.to(device)
            source_cache, positions = extract_cache(source_model, input_ids=input_ids, attention_mask=mask)
            target_cache, _ = extract_cache(target_model, input_ids=input_ids, attention_mask=mask)
            timer.add("extract_caches", batch_start)
            yield CachePair(source=source_cache, target=target_cache, positions=positions, valid=mask.bool())

    if args.selection == "same_layer":
        contract = FeatureContract.same_layer(num_layers=source.num_layers, kv_heads=source.num_kv_heads, head_dim=source.head_dim)
    else:
        start = time.perf_counter()
        contract = select_source_layers_by_r_squared(list(pairs()), source=source, target=target, k=args.k, lambda_=args.lambda_, stride=args.stride)
        timer.add("select_layers_total", start)
    provenance = {
        "source_checkpoint": str(args.source), "target_checkpoint": str(args.target), "prefixes": str(args.prefixes),
        "sequences": len(sequences), "max_tokens": args.max_tokens, "selection": args.selection, "k": args.k,
        "device": device, "torch": torch.__version__, **json.loads(args.provenance),
    }
    start = time.perf_counter()
    artifact = fit_mapper(pairs(), FitOptions(contract=contract, source=source, target=target, lambda_=args.lambda_, stride=args.stride, provenance=provenance))
    timer.add("fit_total_including_extraction", start)
    start = time.perf_counter()
    sha = artifact.save(args.out)
    timer.add("serialize", start)
    report = {
        "artifact": str(args.out), "artifact_sha256": sha, "source_sha256": source_sha, "target_sha256": target_sha,
        "diagnostics": artifact.diagnostics, "seconds": timer.seconds,
        "peak_memory_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "contract": contract.to_json(),
    }
    (args.out / "fit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "contract"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
