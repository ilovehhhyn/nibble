"""Manage the independent adjacent-checkpoint mapper fitter."""

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import time

from .report import event


@contextmanager
def fitter_service(runtime, run_dir, max_steps):
    """Run the original checkpoint-polling fitter alongside training.

    The service polls complete adjacent checkpoint pairs every ten seconds. Its
    readiness delay is recorded in mapper-readiness.jsonl and does not block the
    trainer. Shutdown requests the service stop file and waits at most 180s.
    """
    run_dir = Path(run_dir)
    project_root = Path(__file__).resolve().parents[2]
    service = project_root / "tools" / "fitter_service.py"
    calibration = Path(
        "/scratch/gpfs/ARORA/hh9077/areal-kv/results/calibration-v0.jsonl"
    )
    checkpoint_root = (
        run_dir
        / "areal"
        / "checkpoints"
        / os.environ["USER"]
        / "nibble"
        / run_dir.name
        / "default"
    )
    registry = run_dir / "registry"
    registry.mkdir(parents=True, exist_ok=True)
    stop_file = run_dir / "mapper-service.stop"
    stop_file.unlink(missing_ok=True)
    ledger = run_dir / "mapper-readiness.jsonl"
    log_path = run_dir / "mapper-service.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = runtime.gpu_ids[3]
    command = [
        str(runtime.python),
        str(service),
        "--checkpoint-root",
        str(checkpoint_root),
        "--registry",
        str(registry),
        "--tools",
        str(project_root / "tools"),
        "--python",
        str(runtime.python),
        "--final-version",
        str(max_steps + 1),
        "--stop-file",
        str(stop_file),
        "--ledger",
        str(ledger),
        "--",
        "--prefixes",
        str(calibration),
        "--max-tokens",
        "2048",
        "--batch-size",
        "16",
        "--lambda",
        "0.01",
        "--stride",
        "4",
        "--selection",
        "same_layer",
    ]
    log = log_path.open("a")
    started = time.monotonic()
    proc = subprocess.Popen(
        command,
        cwd=project_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    event(run_dir, "mapper_service_started", pid=proc.pid, gpu=runtime.gpu_ids[3])
    try:
        yield
    finally:
        stop_file.touch()
        try:
            proc.wait(timeout=180)
            event(
                run_dir,
                "mapper_service_stopped",
                returncode=proc.returncode,
                seconds=time.monotonic() - started,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"mapper service exited {proc.returncode}; see {log_path}"
                )
        finally:
            log.close()
