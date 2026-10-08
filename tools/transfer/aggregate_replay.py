"""Pool the per-transition replay rows of one run into the two headline comparisons: GSM8K accuracy and time to first token."""

import json
import pathlib
import statistics
import sys


def bootstrap_interval(differences: list[float], *, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile interval of the mean paired difference, resampling requests."""
    import random

    generator = random.Random(seed)
    means = sorted(statistics.fmean(generator.choices(differences, k=len(differences))) for _ in range(draws))
    return means[int(0.025 * draws)], means[int(0.975 * draws)]


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[0])
    rows = []
    for directory in sorted(out.glob("v*-v*")):
        if not (directory / "rows.json").is_file():
            continue
        summary = json.loads((directory / "summary.json").read_text())
        for row in json.loads((directory / "rows.json").read_text()):
            rows.append({**row, "transition": directory.name, "send": summary["send"]})
    if len(rows) == 0:
        raise SystemExit(f"no replay rows under {out}")
    used = [r for r in rows if r["transfer_cached_tokens"] and r["transfer_cached_tokens"] > 0]
    report = {"requests": len(rows), "requests_where_transfer_used_the_cache": len(used), "transitions": len({r["transition"] for r in rows})}
    differences = [r["transfer_reward"] - r["baseline_reward"] for r in used]
    low, high = bootstrap_interval(differences)
    report["accuracy"] = {
        "baseline": statistics.fmean(r["baseline_reward"] for r in used), "transfer": statistics.fmean(r["transfer_reward"] for r in used),
        "mean_paired_difference": statistics.fmean(differences), "difference_95_interval_over_requests": [low, high],
        "same_reward_rate": statistics.fmean(float(d == 0) for d in differences), "identical_continuation_rate": statistics.fmean(float(r["continuation_equal"]) for r in used),
        "noise_floor_reprefill_twice_identical_rate": statistics.fmean(float(r["reprefill_repeat_equal"]) for r in used) if "reprefill_repeat_equal" in used[0] else None,
        "noise_floor_reprefill_twice_accuracy": statistics.fmean(r["reprefill_repeat_reward"] for r in used) if "reprefill_repeat_reward" in used[0] else None,
    }
    report["time_to_first_token_ms"] = {}
    for send in sorted({r["send"] for r in used}):
        subset = [r for r in used if r["send"] == send]
        report["time_to_first_token_ms"][send] = {
            "n": len(subset), "prefix_tokens_median": statistics.median(r["prefix_tokens"] for r in subset),
            "baseline_median": statistics.median(r["baseline_ttft_ms"] for r in subset), "transfer_median": statistics.median(r["transfer_ttft_ms"] for r in subset),
            "baseline_p90": sorted(r["baseline_ttft_ms"] for r in subset)[int(0.9 * (len(subset) - 1))], "transfer_p90": sorted(r["transfer_ttft_ms"] for r in subset)[int(0.9 * (len(subset) - 1))],
        }
    fits = [json.loads(p.read_text())["seconds"] for p in sorted((out / "registry").glob("v*-v*/fit_report.json"))]
    if fits:
        report["map_fit_seconds_per_transition"] = {key: statistics.fmean(f.get(key, 0.0) for f in fits) for key in sorted(fits[0])}
    (out / "replay_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
