"""Run three matched arms sequentially within one four-GPU Slurm allocation.

An allocation is required. This avoids comparing different GPU models across
queued jobs. Failure stops remaining arms; no automatic retry spends more GPU time.
"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import secrets
import subprocess
import time
from .report import atomic_json
from .runtime import Runtime
from .runner import execute, terminate_group
from .sdk import Client
from .spec import TrainSpec

ARM_TIMEOUT_SECONDS = 900
EVALUATION_TIMEOUT_SECONDS = 180


def evaluate(runtime, checkpoint, output, tokens):
    env = runtime.environment(output.parent)
    env["CUDA_VISIBLE_DEVICES"] = runtime.gpu_ids[3]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(
            [
                str(runtime.python),
                "-m",
                "nibble.evaluate",
                str(checkpoint),
                str(output),
                "--max-new-tokens",
                str(tokens),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=EVALUATION_TIMEOUT_SECONDS,
        )


def summarize(root):
    manifest = json.loads((root / "campaign.json").read_text())
    rows = []
    for arm in manifest["arms"]:
        run = Path(arm["run_dir"])
        if not (run / "report.json").exists():
            rows.append({"arm": arm["name"], "status": "not_finished"})
            continue
        report = json.loads((run / "report.json").read_text())
        native = report.get("native") or {}
        evaluation = root / (arm["name"] + "-evaluation.json")
        rows.append(
            {
                "arm": arm["name"],
                "status": report["status"],
                "wall_seconds": report["wall_seconds"],
                "training_seconds": native.get("training_seconds"),
                "heldout_accuracy": json.loads(evaluation.read_text())["accuracy"]
                if evaluation.exists()
                else None,
                "audit": report["audit"],
                "training_evidence": report["training_evidence"],
                "kv": report.get("kv"),
                "report": str(run / "report.json"),
            }
        )
    atomic_json(
        root / "comparison.json",
        {
            "arms": rows,
            "limitations": [
                "One seed and short run: no convergence or time-to-target-quality conclusion.",
                "Arm 3 bundles API, mapping, and score-centering; no separate causal attribution.",
                "Raw centered/uncentered losses differ; compare common diagnostics and heldout accuracy.",
                "All four allocated GPUs are charged; fitting barrier cost stays in training time.",
            ],
        },
    )
    lines = [
        "# Four-GPU comparison",
        "",
        "| Arm | Status | Training s | Full run s | Heldout accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['arm']} | {r['status']} | {r.get('training_seconds')} | {r.get('wall_seconds')} | {r.get('heldout_accuracy')} |"
        )
    lines += [
        "",
        "See comparison.json for Cado records, loss metrics, mapping usage and limitations.",
    ]
    (root / "comparison.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.root)
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("campaign requires a Slurm GPU allocation")
    args.root.mkdir(parents=True, exist_ok=False)
    runtime = Runtime.load(require_token=False)
    if len(runtime.gpu_ids) != 4:
        raise ValueError("campaign needs exactly four allocated GPU identifiers")
    spec = TrainSpec(
        model="qwen25-1.5b",
        dataset="gsm8k-benchmark",
        reward="gsm8k-strict",
        target="slurm",
        max_gpus=4,
        max_steps=15,
        prompts_per_step=8,
        rollouts_per_prompt=4,
        max_new_tokens=1024,
        max_prompt_tokens=1024,
        max_concurrent_rollouts=64,
        max_staleness=2,
        timeout_seconds=ARM_TIMEOUT_SECONDS,
        objective="group_pg_tis",
    )
    # Cost-bounded default: original 1.5B workload, 15 updates, one seed.
    arms = [
        (
            "standalone-sync",
            dataclasses.replace(spec, execution="standalone_sync", max_staleness=0),
        ),
        ("areal-async", spec),
        (
            "nibble-map-sc",
            dataclasses.replace(
                spec, score_centering=True, cache_mode="map", fit_mappers=True
            ),
        ),
    ]
    manifest = {
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "started_at": time.time(),
        "arms": [],
        "allocation_gpus": runtime.gpu_ids,
        "status": "running",
    }
    atomic_json(args.root / "campaign.json", manifest)
    hardware = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    (args.root / "hardware.txt").write_text(hardware)
    evaluate(
        runtime,
        runtime.models[spec.model],
        args.root / "initial-evaluation.json",
        spec.max_new_tokens,
    )
    for name, arm_spec in arms:
        if name != "nibble-map-sc":
            run_dir = args.root / name
            manifest["arms"].append(
                {"name": name, "run_dir": str(run_dir), "spec": arm_spec.to_dict()}
            )
            atomic_json(args.root / "campaign.json", manifest)
            report = execute(runtime, run_dir, arm_spec)
        else:
            token = secrets.token_urlsafe(32)
            api_root = args.root / "api"
            config = json.loads(Path(os.environ["NIBBLE_RUNNER_CONFIG"]).read_text())
            config["root"] = str(api_root)
            # Tools and data remain immutable inputs; adapter uses NIBBLE_PROJECT_ROOT.
            api_root.mkdir()
            api_config = api_root / "runner.json"
            atomic_json(api_config, config)
            env = os.environ.copy()
            env["NIBBLE_RUNNER_CONFIG"] = str(api_config)
            env["NIBBLE_TOKEN"] = token
            port = 18000 + int(os.environ["SLURM_JOB_ID"]) % 10000
            with (api_root / "server.log").open("w") as log:
                server = subprocess.Popen(
                    [
                        str(runtime.python),
                        "-m",
                        "uvicorn",
                        "nibble.server:create_app",
                        "--factory",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        str(port),
                    ],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    # No GPU work is dispatched until the authenticated API is ready.
                    import urllib.request
                    import urllib.error

                    deadline = time.monotonic() + 30
                    while True:
                        if server.poll() is not None:
                            raise RuntimeError("API exited; see server.log")
                        try:
                            with urllib.request.urlopen(
                                f"http://127.0.0.1:{port}/health", timeout=2
                            ):
                                break
                        except urllib.error.URLError:
                            if time.monotonic() > deadline:
                                raise TimeoutError("API startup exceeded 30s")
                            time.sleep(0.25)
                    client = Client(f"http://127.0.0.1:{port}", token=token)
                    submitted = time.perf_counter()
                    job = client.train(**arm_spec.to_dict())
                    run_dir = api_root / "state/runs" / job.id
                    manifest["arms"].append(
                        {
                            "name": name,
                            "run_dir": str(run_dir),
                            "spec": arm_spec.to_dict(),
                            "submission_seconds": time.perf_counter() - submitted,
                        }
                    )
                    atomic_json(args.root / "campaign.json", manifest)
                    job.wait(timeout=arm_spec.timeout_seconds + 30, poll_seconds=2)
                    report = job.report()
                finally:
                    terminate_group(server, {})
        summarize(args.root)
        if report["status"] != "succeeded":
            raise RuntimeError(
                f"{name} failed; remaining arms not launched; see {run_dir}/worker.log"
            )
        if name == "standalone-sync":
            checkpoint = run_dir / "checkpoints" / f"step-{spec.max_steps:04d}"
        else:
            candidates = list(
                (run_dir / "areal/checkpoints").rglob(
                    f"*globalstep{spec.max_steps - 1}/config.json"
                )
            )
            if len(candidates) != 1:
                raise RuntimeError("final AReaL checkpoint is missing or ambiguous")
            checkpoint = candidates[0].parent
        evaluate(
            runtime,
            checkpoint,
            args.root / (name + "-evaluation.json"),
            spec.max_new_tokens,
        )
        summarize(args.root)
    manifest["status"] = "completed"
    manifest["ended_at"] = time.time()
    atomic_json(args.root / "campaign.json", manifest)


if __name__ == "__main__":
    main()
