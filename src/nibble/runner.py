"""One job at a time, durable status and bounded process-tree cleanup."""

import fcntl
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import psutil
from .report import atomic_json, build_report, event
from .runtime import check_idle
from .spec import TrainSpec


def terminate_group(proc, descendants):
    # Track descendants even when AReaL gives them separate sessions.
    alive = []
    for child in descendants.values():
        try:
            if child.is_running():
                alive.append(child)
                child.terminate()
        except psutil.Error:
            pass
    for sig, grace in ((signal.SIGTERM, 5), (signal.SIGKILL, 2)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            break
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            continue
    _, remaining = psutil.wait_procs(alive, timeout=2)
    for child in remaining:
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass


def execute(runtime, run_dir, spec, cancelled=lambda: False, check_gpu=True):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(run_dir / "spec.json", spec.to_dict())
    start = time.monotonic()
    proc = None
    descendants = {}
    gpu_lock = None
    status, error = "failed", None
    try:
        lock_path = runtime.root / "state" / "gpu.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        gpu_lock = lock_path.open("a")
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = runtime.plan(spec)
        atomic_json(run_dir / "plan.json", plan)
        if check_gpu:
            check_idle(plan["gpu_ids"])
        event(run_dir, "launch", plan=plan)
        env = runtime.environment(run_dir)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(plan["gpu_ids"])
        with (run_dir / "worker.log").open("w") as log:
            proc = subprocess.Popen(
                [str(runtime.python), "-m", "nibble.worker", str(run_dir)],
                cwd=runtime.areal_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            while proc.poll() is None:
                try:
                    for child in psutil.Process(proc.pid).children(recursive=True):
                        descendants[(child.pid, child.create_time())] = child
                except psutil.Error:
                    pass
                if cancelled():
                    status, error = "cancelled", "cancel requested"
                    break
                if time.monotonic() - start >= spec.timeout_seconds:
                    status, error = (
                        "failed",
                        f"wall-time budget of {spec.timeout_seconds}s exceeded",
                    )
                    break
                time.sleep(0.25)
            else:
                if proc.returncode == 0 and (run_dir / "native-result.json").exists():
                    status = "succeeded"
                else:
                    error = f"worker exited {proc.returncode}; see worker.log"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            terminate_group(proc, descendants)
        if gpu_lock is not None:
            gpu_lock.close()
        elapsed = time.monotonic() - start
        event(run_dir, "finished", status=status, error=error, wall_seconds=elapsed)
        report = build_report(run_dir, status, error, elapsed)
    return report


class Runner:
    def __init__(self, runtime, store):
        self.runtime, self.store = runtime, store
        self.stop = threading.Event()
        self.thread = None
        self.lock = None

    def start(self):
        self.lock = (self.store.root / "runner.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.store.recover()
        self.thread = threading.Thread(
            target=self.loop, name="nibble-runner", daemon=True
        )
        self.thread.start()

    def close(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=15)
        if self.lock:
            self.lock.close()

    def loop(self):
        while not self.stop.is_set():
            job = self.store.claim()
            if job is None:
                self.stop.wait(0.25)
                continue
            try:
                spec = TrainSpec.from_dict(job["spec"])
                result = execute(
                    self.runtime,
                    self.store.root / "runs" / job["id"],
                    spec,
                    cancelled=lambda: self.stop.is_set()
                    or self.store.get(job["id"])["cancel_requested"],
                )
                self.store.finish(job["id"], result["status"], result["error"])
            except Exception as exc:
                self.store.finish(job["id"], "failed", f"{type(exc).__name__}: {exc}")
