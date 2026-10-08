"""Actual Cado execution boundaries, separate from optimizer-step telemetry."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess


def create_run(directory):
    import tplane

    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not ids or not all(value.isdigit() for value in ids):
        raise RuntimeError(
            "Cado requires the allocation's actual numeric GPU identifiers"
        )
    reader = tplane.create_nvml_gpu_reader(indices=tuple(map(int, ids)))
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split(":", 2)[-1].lstrip("/")
    return tplane.Run(
        run_id=Path(directory).parent.name,
        store=tplane.FileRecordStore(Path(directory)),
        options=tplane.RunOptions(
            cpus_held=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
            gpus_held=len(reader.total_bytes),
            policy=tplane.Policy(
                on_agent=tplane.Action.ABORT,
                on_grader=tplane.Action.ABORT,
                on_infra=tplane.Action.ABORT,
                on_timeout=tplane.Action.ABORT,
                max_attempts=1,
            ),
        ),
        read_host=tplane.cgroup_memory_reader(cgroup),
        read_gpu=reader.read,
        gpu_total_bytes=reader.total_bytes,
    )


@contextmanager
def audited_run(directory):
    run = create_run(directory)
    try:
        yield run
    finally:
        run.finish()


def execute(run, command, env):
    import tplane

    # Cado has no allocation kind. This explicitly tagged record is NOT an
    # optimizer update and must not be counted with the inner train-step units.
    with run.unit("execution", kind=tplane.UnitKind.TRAIN_STEP) as unit:
        unit.tag("granularity", "process")
        unit.tag("command", command[0])
        unit.tag("optimizer_step", "false")
        with unit.stage(tplane.Stage.TOOL):
            result = subprocess.run(command, env=env, check=False)
        if result.returncode:
            unit.fail(
                tplane.Failure(
                    kind=(
                        tplane.FailureKind.PROCESS_KILLED
                        if result.returncode < 0
                        else tplane.FailureKind.AGENT_EXCEPTION
                    ),
                    stage=tplane.Stage.TOOL,
                    retryable=False,
                    message=f"audited process exited with code {result.returncode}; inspect its original log",
                    evidence=(f"returncode={result.returncode}",),
                )
            )
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    args.directory.mkdir(parents=True, exist_ok=True)
    (args.directory.parent / "audit-open.json").write_text(
        json.dumps(
            {
                "command": command,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "scope": "process boundary; inner trajectory/evaluation records remain separate",
            }
        )
        + "\n"
    )
    with audited_run(args.directory) as run:
        execute(run, command, dict(os.environ, NIBBLE_CADO_OUTER="1"))
    (args.directory.parent / "audit-complete.json").write_text(
        '{"status":"completed"}\n'
    )


if __name__ == "__main__":
    main()
