"""Check that the keys and values SGLang keeps in its KV pool equal what Hugging Face computes for the same tokens.

The mapper is fitted on Hugging Face caches and applied to SGLang's pool, so the two must store the same thing:
keys after the model's key normalization (if any) and after the rotary transform, values as projected. This
loads caches dumped by the retention layer (``SGLANG_KV_RETAIN_DUMP_DIR``), recomputes them with the given
checkpoint, and reports the relative error per layer. Decode-time and prefill-time kernels differ slightly in
bf16, so small errors are expected; a wrong convention shows up as errors near 1.
"""

import argparse
import json
import pathlib
import sys

import torch


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dump-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True, help="checkpoint the server held when the caches were retained")
    parser.add_argument("--version", type=int, required=True, help="only dumps retained under this policy version are checked")
    args = parser.parse_args(argv)
    from transformers import AutoModelForCausalLM

    from areal.experimental.kvmap.hf import extract_cache

    dumps = [torch.load(path) for path in sorted(args.dump_dir.glob("retained-*.pt"))]
    dumps = [d for d in dumps if d["version"] == args.version]
    if len(dumps) == 0:
        raise SystemExit(f"no dump for version {args.version} under {args.dump_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=torch.bfloat16).to(device).eval()
    report = []
    for dump in dumps:
        ids = torch.tensor([dump["token_ids"]], device=device)
        cache, _ = extract_cache(model, input_ids=ids, attention_mask=torch.ones_like(ids))
        keys = torch.stack([k[0].permute(1, 0, 2) for k in cache.keys]).float().cpu()  # [L, T, kv, d]
        values = torch.stack([v[0].permute(1, 0, 2) for v in cache.values]).float().cpu()
        pool_keys, pool_values = dump["keys"].float(), dump["values"].float()
        key_error = ((pool_keys - keys).flatten(1).norm(dim=1) / keys.flatten(1).norm(dim=1)).tolist()
        value_error = ((pool_values - values).flatten(1).norm(dim=1) / values.flatten(1).norm(dim=1)).tolist()
        report.append({"tokens": len(dump["token_ids"]), "key_relative_error_max": max(key_error), "key_relative_error_median": sorted(key_error)[len(key_error) // 2],
                       "value_relative_error_max": max(value_error), "value_relative_error_median": sorted(value_error)[len(value_error) // 2]})
    print(json.dumps({"layout_check": report}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
