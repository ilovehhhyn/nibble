"""Watch one AReaL trial with cado (the tplane library) from a side process, without touching the trainer or the server.

AReaL has no tplane adapter, so this reads what the trial already writes: one JSONL record per finished rollout under
``<log dir>/rollout/<version>/<task>.jsonl`` and the role logs. For each rollout it writes one cado trajectory record
(reward, tokens, tags ``step`` = the policy version that finished it and ``group`` = the prompt), so cado's live watch
can score rewards and failure rates per training step. One cado train-step unit stays open per policy version, so its
record carries the peak GPU memory and the peak host memory of the whole login session (cgroup) during that step.
Out-of-memory events come from three places: cado's own error classifier applied to error lines in the logs, the
cgroup's oom_kill counter, and two server texts cado does not know yet (listed in KNOWN_UNCLASSIFIED).
"""

import argparse
import json
import os
import pathlib
import re
import signal
import sys
import time

import tplane
from tplane.classify import classify_error_name
from tplane.resources import read_cgroup_oom_kill_count

ERROR_LINE = re.compile(r"(\w*(?:Error|Exception))\b:? ?(.*)")
# Messages seen in this project that cado's classifier labels as an agent error; reported upstream as a gap.
KNOWN_UNCLASSIFIED = (
    ("Not enough memory", tplane.FailureKind.OOM_KV_CACHE, tplane.Stage.SETUP),
    ("scheduler died during initialization (exit code: -9)", tplane.FailureKind.PROCESS_KILLED, tplane.Stage.SETUP),
)
POLL_S = 2.0
DUPLICATE_WINDOW_S = 60.0


def new_lines(path: pathlib.Path, offsets: dict[pathlib.Path, int]) -> list[str]:
    """Return complete lines appended to ``path`` since the last call."""
    position = offsets.get(path, 0)
    with path.open("rb") as handle:
        handle.seek(position)
        data = handle.read()
    end = data.rfind(b"\n") + 1
    offsets[path] = position + end
    return data[:end].decode(errors="replace").splitlines()


def classify_log_line(line: str, *, stage: tplane.Stage) -> tuple[tplane.Failure | None, str]:
    """Return (failure, who decided): cado's classifier first, then this project's known gaps; None for a non-infra line."""
    match = ERROR_LINE.search(line)
    if match is not None:
        failure = classify_error_name(match.group(1), match.group(2) or match.group(1), stage=stage)
        if failure.failure_class is tplane.FailureClass.INFRA:
            return failure, "cado"
    for text, kind, known_stage in KNOWN_UNCLASSIFIED:
        if text in line:
            return tplane.Failure(kind=kind, stage=known_stage, retryable=False, message=text, evidence=(line.strip()[:300],)), "sidecar"
    return None, ""


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--log-dir", type=pathlib.Path, required=True, help="AReaL <fileroot>/logs/<user>/<experiment>/<trial>")
    parser.add_argument("--extra-log", type=pathlib.Path, action="append", default=[])
    parser.add_argument("--run-dir", type=pathlib.Path, required=True, help="cado record store (must not exist)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", default="gsm8k")
    parser.add_argument("--cgroup-dir", type=pathlib.Path, required=True)
    parser.add_argument("--gpu", choices=("nvml", "none"), required=True)
    parser.add_argument("--action", choices=("warn", "stop"), required=True)
    parser.add_argument("--stop-pid", type=int, default=None, help="process to SIGTERM when --action stop raises an alarm")
    parser.add_argument("--done-file", type=pathlib.Path, required=True, help="the watcher exits once this file exists and no data is pending")
    args = parser.parse_args(argv)
    if args.action == "stop" and args.stop_pid is None:
        raise SystemExit("--action stop requires --stop-pid, the process this watcher may stop")
    gpu = tplane.create_nvml_gpu_reader(indices=tuple(int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(","))) if args.gpu == "nvml" else tplane.no_gpu_reader()
    run = tplane.Run(
        run_id=args.run_id,
        store=tplane.FileRecordStore(args.run_dir),
        options=tplane.RunOptions(gpus_held=len(gpu.total_bytes)),
        read_host=tplane.cgroup_memory_reader(args.cgroup_dir),
        read_gpu=gpu.read,
        gpu_total_bytes=gpu.total_bytes,
        watch=tplane.WatchOptions(action=tplane.WatchAction(args.action)),
    )
    offsets: dict[pathlib.Path, int] = {}
    counts = {"trajectories": 0, "infra_events": 0, "events_cado_classified": 0, "events_sidecar_classified": 0, "steps": 0}
    oom_kills = read_cgroup_oom_kill_count(args.cgroup_dir / "memory.events")
    step_unit, step_version, event_index = None, None, 0
    started = time.monotonic()

    def close_step() -> None:
        if step_unit is not None:
            step_unit.__exit__(None, None, None)

    def open_step(version: int):
        unit = run.unit(f"step-{version:05d}", kind=tplane.UnitKind.TRAIN_STEP)
        unit.__enter__()
        unit.tag("step", str(version))
        counts["steps"] += 1
        return unit

    seen_events: dict[tuple, float] = {}

    def record_event(failure: tplane.Failure, who: str, stage_kind: tplane.UnitKind) -> None:
        nonlocal event_index
        # AReaL repeats one error in several role logs and tracebacks; count it once per minute.
        key = (failure.kind, failure.message[:120])
        now = time.monotonic()
        if key in seen_events and now - seen_events[key] < DUPLICATE_WINDOW_S:
            counts["duplicate_log_lines"] = counts.get("duplicate_log_lines", 0) + 1
            return
        seen_events[key] = now
        event_index += 1
        counts["infra_events"] += 1
        counts[f"events_{who}_classified"] += 1
        try:
            with run.unit(f"event-{event_index:04d}", kind=stage_kind, source=args.source) as unit:
                unit.tag("step", str(step_version if step_version is not None else 0))
                unit.tag("classified_by", who)
                unit.fail(failure)
        except tplane.UnitFailed:
            pass

    try:
        while True:
            pending = False
            rollout_dir = args.log_dir / "rollout"
            files = sorted(rollout_dir.glob("*/*.jsonl"), key=lambda p: (int(p.parent.name), p.name)) if rollout_dir.is_dir() else []
            for file in files:
                for line in new_lines(file, offsets):
                    record = json.loads(line)
                    version = int(record["tail_version"])
                    if step_version is None or version > step_version:
                        close_step()
                        step_unit, step_version = open_step(version), version
                    with run.unit(f"t{record['task_id']}-s{record['sample_idx']}-v{version}-n{counts['trajectories']}", kind=tplane.UnitKind.TRAJECTORY, source=args.source) as unit:
                        unit.tag("step", str(version))
                        unit.tag("group", str(record["task_id"]))
                        unit.tokens(tokens_in=int(record["prompt_len"]), tokens_out=int(record["seqlen"]) - int(record["prompt_len"]))
                        unit.reward(float(record["reward"]))
                    counts["trajectories"] += 1
                    pending = True
            for log in sorted(args.log_dir.glob("*.log")) + list(args.extra_log):
                if not log.is_file() or log.name == "merged.log":
                    continue
                stage = tplane.Stage.ROLLOUT if log.name.startswith("rollout") else tplane.Stage.TRAIN
                kind = tplane.UnitKind.TRAJECTORY if stage is tplane.Stage.ROLLOUT else tplane.UnitKind.TRAIN_STEP
                for line in new_lines(log, offsets):
                    failure, who = classify_log_line(line, stage=stage)
                    if failure is not None:
                        record_event(failure, who, kind)
                        pending = True
            now_kills = read_cgroup_oom_kill_count(args.cgroup_dir / "memory.events")
            if now_kills > oom_kills:
                record_event(tplane.Failure(kind=tplane.FailureKind.OOM_HOST, stage=tplane.Stage.TRAIN, retryable=False,
                                            message="the session cgroup recorded an OOM kill", evidence=(f"oom_kill={now_kills}",)), "cado", tplane.UnitKind.TRAIN_STEP)
                oom_kills = now_kills
            if args.done_file.exists() and not pending:
                break
            time.sleep(POLL_S)
        close_step()
        alarms = run.finish()
    except tplane.IntegrityStop as stop:
        alarms = run.finish()
        print(f"cado stop: {stop}", file=sys.stderr)
        if args.stop_pid is not None:
            os.kill(args.stop_pid, signal.SIGTERM)
    summary = {**counts, "alarms": len(alarms), "watcher_cpu_seconds": round(time.process_time(), 2), "watcher_wall_seconds": round(time.monotonic() - started, 1)}
    (args.run_dir / "cado_watch_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
