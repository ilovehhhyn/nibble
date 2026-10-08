"""One source-faithful fork arm per allocation, with common 100-question evaluation."""

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

from .runtime import Runtime
from .spec import TrainSpec
from .runner import execute, terminate_group
from .report import atomic_json
from .sdk import Client


def api_run(runtime, root, spec):
    """Submit the same fork execution through HTTP; preserve launch overhead."""
    api_root = root / "api"
    api_root.mkdir()
    config = json.loads(Path(os.environ["NIBBLE_RUNNER_CONFIG"]).read_text())
    config["root"] = str(api_root)
    api_config = api_root / "runner.json"
    atomic_json(api_config, config)
    token = secrets.token_urlsafe(32)
    env = os.environ.copy()
    env.update(NIBBLE_TOKEN=token, NIBBLE_RUNNER_CONFIG=str(api_config))
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
            deadline = time.monotonic() + 30
            while subprocess.run(
                [
                    "curl",
                    "--silent",
                    "--fail",
                    "--max-time",
                    "2",
                    f"http://127.0.0.1:{port}/health",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("API startup failed; see server.log")
                time.sleep(0.25)
            client = Client(f"http://127.0.0.1:{port}", token=token)
            started = time.perf_counter()
            job = client.train(**spec.to_dict())
            submission_seconds = time.perf_counter() - started
            run_dir = api_root / "state/runs" / job.id
            atomic_json(
                root / "submission.json",
                dict(
                    run_dir=str(run_dir),
                    spec=spec.to_dict(),
                    submission_seconds=submission_seconds,
                ),
            )
            job.wait(timeout=spec.timeout_seconds + 30, poll_seconds=3)
            return run_dir, job.report(), submission_seconds
        finally:
            terminate_group(server, {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument(
        "--arm", choices=("fork-exact", "api-exact", "api-map"), required=True
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("fork comparison requires Slurm allocation")
    args.root.mkdir(parents=True, exist_ok=False)
    runtime = Runtime.load(require_token=False)
    mapping = args.arm == "api-map"
    spec = TrainSpec(
        model="qwen25-1.5b",
        dataset="gsm8k",
        reward="gsm8k",
        algorithm="ppo",
        target="slurm",
        max_gpus=4 if mapping else 3,
        max_steps=15,
        prompts_per_step=256,
        rollouts_per_prompt=4,
        max_new_tokens=1024,
        max_prompt_tokens=1024,
        max_concurrent_rollouts=256,
        max_staleness=2,
        timeout_seconds=20000,
        learning_rate=1.7e-5,
        objective="fork_ppo",
        execution="fork_arm",
        cache_mode="map" if mapping else "recompute",
        fit_mappers=mapping,
    )
    started = time.time()
    atomic_json(
        args.root / "campaign.json",
        dict(
            status="running",
            arm=args.arm,
            slurm_job=os.environ["SLURM_JOB_ID"],
            started_at=started,
            spec=spec.to_dict(),
        ),
    )
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
    if args.arm == "fork-exact":
        initial_env = runtime.environment(args.root)
        initial_env["CUDA_VISIBLE_DEVICES"] = runtime.gpu_ids[0]
        with (args.root / "initial-evaluation.log").open("w") as log:
            subprocess.run(
                [
                    str(runtime.python),
                    "-m",
                    "nibble.evaluate",
                    runtime.models[spec.model],
                    str(args.root / "initial-evaluation-100.json"),
                    "--count",
                    "100",
                    "--max-new-tokens",
                    "1024",
                ],
                env=initial_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=480,
            )
    if args.arm == "fork-exact":
        run_dir = args.root / args.arm
        report = execute(runtime, run_dir, spec)
        submission = 0.0
    else:
        run_dir, report, submission = api_run(runtime, args.root, spec)
    atomic_json(
        args.root / "arm-result.json",
        dict(run_dir=str(run_dir), report=report, submission_seconds=submission),
    )
    if report["status"] != "succeeded":
        raise RuntimeError(
            f"{args.arm} failed: {report['error']}; see {run_dir}/worker.log"
        )
    checkpoints = list(
        (run_dir / "areal/checkpoints").rglob("*globalstep14/config.json")
    )
    if len(checkpoints) != 1:
        raise RuntimeError("final fork checkpoint missing or ambiguous")
    checkpoint = checkpoints[0].parent
    eval_env = runtime.environment(args.root)
    eval_env["CUDA_VISIBLE_DEVICES"] = runtime.gpu_ids[0]
    with (args.root / "evaluation.log").open("w") as log:
        subprocess.run(
            [
                str(runtime.python),
                "-m",
                "nibble.evaluate",
                str(checkpoint),
                str(args.root / "evaluation-100.json"),
                "--count",
                "100",
                "--max-new-tokens",
                "1024",
            ],
            env=eval_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=480,
        )
    evaluation = json.loads((args.root / "evaluation-100.json").read_text())
    if (
        evaluation["count"] != 100
        or report["training_evidence"]["completed_steps"] != 15
    ):
        raise RuntimeError("requested train/evaluation count did not complete")
    atomic_json(
        args.root / "campaign.json",
        dict(
            status="completed",
            arm=args.arm,
            slurm_job=os.environ["SLURM_JOB_ID"],
            started_at=started,
            ended_at=time.time(),
            run_dir=str(run_dir),
            checkpoint=str(checkpoint),
            evaluation=evaluation["accuracy"],
            spec=spec.to_dict(),
            submission_seconds=submission,
        ),
    )


if __name__ == "__main__":
    main()
