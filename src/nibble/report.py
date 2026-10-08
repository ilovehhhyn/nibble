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
            str(p.parent.relative_to(run_dir)) for p in run_dir.rglob("config.json")
        ],
        "limitations": [
            "Tiny interface pilot; no convergence or time-to-target-quality claim.",
            "Single seed; short-run comparison cannot establish convergence.",
            "Centered and uncentered loss values are different objectives; use heldout reward for quality.",
            "Cado records outcomes; live change detection needs more steps and is not enabled.",
            "Hardware is reserved only for jobs run inside a Slurm allocation.",
        ],
    }
    audit_files = list((run_dir / "audit" / "records").glob("*.json"))
    report["audit"]["record_files"] = len(audit_files)
    all_records = [json.loads(p.read_text()) for p in audit_files]
    records = [r for r in all_records if r.get("kind") == "trajectory"]
    report["audit"]["trajectory_records"] = len(records)
    report["audit"]["non_trajectory_records"] = len(all_records) - len(records)
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
            float(x)
            for x in re.findall(re.escape(name) + r"(?:/avg)?\s*│\s*([+\-0-9.e]+)", log)
        ]

    report["training_evidence"] = {
        "completed_steps": len(re.findall(r"Train step [0-9]+/[0-9]+ done", log)),
        "gradient_norms": metric("ppo_actor/update/grad_norm"),
        "update_successful": metric("ppo_actor/update/update_successful"),
    }
    report["training_evidence"].update(
        policy_losses=metric("ppo_actor/update/nibble_policy_loss"),
        original_policy_losses=metric("ppo_actor/update/loss"),
        critic_losses=metric("ppo_critic/update/loss"),
        uncentered_tis_losses=metric("ppo_actor/update/nibble_uncentered_tis_loss"),
        train_sampler_logprob_gaps=metric("ppo_actor/update/nibble_abs_logp_gap"),
        score_centering_token_fraction=metric("ppo_actor/update/nibble_sc_tokens"),
        score_centering_correction=metric("ppo_actor/update/nibble_sc_correction"),
        score_centering_q_head_mass=metric("ppo_actor/update/nibble_sc_q_head_mass"),
        score_centering_p_head_mass=metric("ppo_actor/update/nibble_sc_p_head_mass"),
        score_centering_tail_rho=metric("ppo_actor/update/nibble_sc_tail_rho"),
        score_centering_behavior_gap=metric("ppo_actor/update/nibble_sc_abs_logp_gap"),
        score_centering_proximal_gap=metric("ppo_actor/update/nibble_sc_prox_logp_gap"),
        score_centering_valid_fraction=metric(
            "ppo_actor/update/nibble_sc_prox_gap_within_tolerance"
        ),
    )
    report["audit"]["policy_versions"] = [r.get("tags", {}) for r in records]
    event_path = run_dir / "events.jsonl"
    events = (
        [json.loads(line) for line in event_path.read_text().splitlines() if line]
        if event_path.exists()
        else []
    )
    report["mapper_fitting_seconds"] = sum(
        e.get("seconds", 0) for e in events if e.get("event") == "mapper_ready"
    )
    ledger = run_dir / "mapper-readiness.jsonl"
    if ledger.exists():
        readiness = [
            json.loads(line) for line in ledger.read_text().splitlines() if line
        ]
        report["mapper_service"] = {
            "mode": "background",
            "readiness_records": readiness,
            "fitting_work_seconds": sum(
                x["artifact_ready"] - x["fit_started"] for x in readiness
            ),
            "note": "Concurrent fitter work is not an additive trainer barrier cost; full reserved-GPU elapsed remains charged.",
        }
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        steps = [
            json.loads(line) for line in metrics_path.read_text().splitlines() if line
        ]
        report["training_evidence"].update(
            completed_steps=len(steps),
            steps=steps,
            gradient_norms=[step["grad_norm"] for step in steps],
            policy_losses=[step["train_loss"] for step in steps],
        )
    kv_events = []
    for path in list(run_dir.glob("kv-events*")) + list(run_dir.glob("kv_events*")):
        paths = list(path.rglob("*.jsonl")) if path.is_dir() else [path]
        for file in paths:
            kv_events.extend(
                json.loads(line) for line in file.read_text().splitlines() if line
            )
    report["kv"] = {
        "map_resumes": sum(
            e.get("event") == "resume" and e.get("method") == "map" for e in kv_events
        ),
        "rebuild_resumes": sum(
            e.get("event") == "resume" and e.get("method") == "rebuild"
            for e in kv_events
        ),
        "translation_batches": sum(
            e.get("event") == "translate_batch" for e in kv_events
        ),
        "events": kv_events,
    }
    if spec.get("cache_mode") == "map" and not report["kv"]["map_resumes"]:
        report["limitations"].append(
            "No mapped resume was observed: cache speedup was not exercised."
        )
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
    text += "\nTrainer identity and objective are in native-result.json. A short run does not establish equivalent final quality.\n"
    text += f"\nCado record files: {len(audit_files)}.\n"
    (run_dir / "report.md").write_text(text)
    return report
