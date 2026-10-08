"""Fit a mapper for every new adjacent checkpoint pair an AReaL run writes, into the registry the server reads.

The service polls the run's checkpoint directory. AReaL saves the checkpoint of global step ``g``
after publishing policy version ``g + 1``, so the pair of checkpoints (g-1, g) is the transition
v<g> -> v<g+1>. For each new complete pair it fits ``v<g>-v<g+1>`` from the fixed calibration
prefixes with the same code path as ``fit_mapper_hf.py`` and records, per artifact, when the
target checkpoint appeared and when the artifact became loadable: that gap is the readiness
latency the proposal charges to the live path. It never deletes checkpoints and stops when a
stop file appears or the run's final version is fitted.
"""

import argparse
import json
import pathlib
import re
import shlex
import subprocess
import sys
import time

STEP_PATTERN = re.compile(r"globalstep(\d+)$")
COMPLETE_MARKER = "tokenizer_config.json"  # FSDPEngine saves weights, then config, then the tokenizer
POLL_SECONDS = 10.0


def directory_size(directory: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in directory.iterdir() if p.is_file())


def list_complete_steps(checkpoint_root: pathlib.Path, *, sizes_seen: dict[pathlib.Path, int] | None = None) -> dict[int, pathlib.Path]:
    """Map global step -> checkpoint directory for directories whose last file exists and whose size stopped changing.

    ``sizes_seen`` carries the previous poll's sizes; a directory counts as complete only when the
    marker file exists and its total size equals the size recorded on the previous call.
    """
    steps: dict[int, pathlib.Path] = {}
    if not checkpoint_root.is_dir():
        return steps
    for directory in sorted(checkpoint_root.iterdir()):
        match = STEP_PATTERN.search(directory.name)
        if match is None or not (directory / COMPLETE_MARKER).is_file():
            continue
        size = directory_size(directory)
        if sizes_seen is None or sizes_seen.get(directory) == size:
            steps[int(match.group(1))] = directory
        if sizes_seen is not None:
            sizes_seen[directory] = size
    return steps


def pending_pairs(steps: dict[int, pathlib.Path], registry: pathlib.Path) -> list[tuple[int, int]]:
    """Adjacent (g-1, g) pairs that exist on disk and have no artifact ``v<g>-v<g+1>`` yet, ascending."""
    pairs = []
    for step in sorted(steps):
        previous = step - 1
        if previous in steps and not (registry / f"v{step}-v{step + 1}" / "mapper.json").is_file():
            pairs.append((previous, step))
    return pairs


def fit_pair(*, source: pathlib.Path, target: pathlib.Path, registry: pathlib.Path, source_version: int, target_version: int, fit_args: list[str], python: list[str], tools: pathlib.Path) -> dict:
    """Run fit_mapper_hf.py into a staging name and rename on success so readers never see a partial artifact."""
    final = registry / f"v{source_version}-v{target_version}"
    staging = registry / f".staging-v{source_version}-v{target_version}"
    if staging.exists():
        subprocess.run(["rm", "-rf", str(staging)], check=True)
    started = time.time()
    command = [*python, str(tools / "fit_mapper_hf.py"), "--source", str(source), "--target", str(target), "--out", str(staging),
               "--provenance", json.dumps({"fitter": True, "source_version": source_version, "target_version": target_version}), *fit_args]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        subprocess.run(["rm", "-rf", str(staging)], check=False)
        raise RuntimeError(f"fit for v{source_version}-v{target_version} failed:\n{completed.stderr[-4000:]}")
    staging.rename(final)
    return {"pair": final.name, "target_checkpoint_mtime": target.stat().st_mtime, "fit_started": started, "artifact_ready": time.time()}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint-root", type=pathlib.Path, required=True, help="<fileroot>/checkpoints/<user>/<exp>/<trial>/default")
    parser.add_argument("--registry", type=pathlib.Path, required=True)
    parser.add_argument("--tools", type=pathlib.Path, required=True, help="directory holding fit_mapper_hf.py")
    parser.add_argument("--python", type=str, default="python", help='interpreter command as one string, e.g. "uv run --offline python"')
    parser.add_argument("--final-version", type=int, required=True, help="stop after v<final-1>-v<final> is fitted")
    parser.add_argument("--stop-file", type=pathlib.Path, required=True)
    parser.add_argument("--ledger", type=pathlib.Path, required=True, help="JSONL of readiness records")
    parser.add_argument("fit_args", nargs=argparse.REMAINDER, help="arguments passed to fit_mapper_hf.py after --")
    args = parser.parse_args(argv)
    fit_args = [a for a in args.fit_args if a != "--"]
    if "--prefixes" not in fit_args:
        raise SystemExit("pass fit_mapper_hf.py arguments after --, including --prefixes")
    args.registry.mkdir(parents=True, exist_ok=True)
    sizes_seen: dict[pathlib.Path, int] = {}
    while not args.stop_file.exists():
        steps = list_complete_steps(args.checkpoint_root, sizes_seen=sizes_seen)
        for previous, step in pending_pairs(steps, args.registry):
            record = fit_pair(source=steps[previous], target=steps[step], registry=args.registry, source_version=step, target_version=step + 1,
                              fit_args=fit_args, python=shlex.split(args.python), tools=args.tools)
            record["readiness_s"] = record["artifact_ready"] - record["target_checkpoint_mtime"]
            with args.ledger.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            if step + 1 >= args.final_version:
                return 0
        time.sleep(POLL_SECONDS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
