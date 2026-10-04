"""Small evidence report. Training quality and speed claims require a proper benchmark."""

from pathlib import Path
import json
import os
import re
import time


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def event(run_dir, kind, **fields):
    with (Path(run_dir) / "events.jsonl").open("a") as f:
        f.write(
            json.dumps({"time": time.time(), "event": kind, **fields}, allow_nan=False)
            + "\n"
        )


def build_report(run_dir, status, error=None, elapsed=None):
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    result_path = run_dir / "native-result.json"
    report = {
        "job_id": run_dir.name,
        "status": status,
        "error": error,
        "wall_seconds": elapsed,
        "spec": spec,
        "native": json.loads(result_path.read_text()) if result_path.exists() else None,
        "audit": {"enabled": spec["audit_on"], "record_files": 0},
        "checkpoint_paths": [
            str(p.parent.relative_to(run_dir))
            for p in (run_dir / "areal").rglob("config.json")
        ],
        "limitations": [
            "Tiny interface pilot; no convergence or time-to-target-quality claim.",
            "Single login GPU shares trainer and rollout compute; no separate-GPU async speed claim.",
            "Score centering is not enabled; no sampler top-k transport in this adapter.",
            "Cado records outcomes; live change detection needs more steps and is not enabled.",
            "GPU busy checks do not reserve hardware against other cluster users.",
        ],
    }
    audit_files = list((run_dir / "audit" / "records").glob("*.json"))
    report["audit"]["record_files"] = len(audit_files)
    records = [json.loads(p.read_text()) for p in audit_files]
    rewards = [r["reward"] for r in records if r.get("reward") is not None]
    report["audit"].update(
        successful_trajectories=sum(r.get("outcome") == "ok" for r in records),
        failed_trajectories=sum(r.get("outcome") == "failed" for r in records),
        rewards=rewards,
        output_tokens=sum(r.get("cost", {}).get("tokens_out", 0) for r in records),
    )
    log_path = run_dir / "worker.log"
    log = log_path.read_text(errors="replace") if log_path.exists() else ""

    def metric(name):
        return [
            float(x) for x in re.findall(re.escape(name) + r"\s*│\s*([+\-0-9.e]+)", log)
        ]

    report["training_evidence"] = {
        "completed_steps": len(re.findall(r"Train step [0-9]+/[0-9]+ done", log)),
        "gradient_norms": metric("ppo_actor/update/grad_norm"),
        "update_successful": metric("ppo_actor/update/update_successful"),
    }
    if records and rewards and len(set(rewards)) == 1:
        report["limitations"].append(
            "All recorded rewards are equal; inspect GRPO advantages and gradient norm."
        )
    atomic_json(run_dir / "report.json", report)
    text = f"# Nibble feasibility run\n\nStatus: **{status}**\n\n"
    if error:
        text += f"Error: {error}\n\n"
    if report["native"]:
        text += "| Phase | Seconds |\n|---|---:|\n"
        for key, value in report["native"].items():
            if key.endswith("_seconds"):
                text += f"| {key.removesuffix('_seconds')} | {value:.3f} |\n"
    text += "\nThis run uses the real AReaL trainer. It does not establish a training speedup or equivalent final quality.\n"
    text += f"\nCado record files: {len(audit_files)}.\n"
    (run_dir / "report.md").write_text(text)
    return report
