"""Summarize only completed, inspected matrix reports; never infer cache speedup."""

import argparse
import json
import math
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()
    manifest = load(args.submission)
    project = args.submission.parent.parent
    rows = []
    hashes = set()
    for job in manifest["training_jobs"]:
        campaign_dir = project / "state" / f"comparison-{job['job_id']}"
        campaign = load(campaign_dir / "campaign.json")
        if campaign["status"] != "completed":
            raise ValueError(f"{job['job_id']}: campaign has not completed")
        arm = load(campaign_dir / "arm-result.json")
        report = arm["report"]
        evidence = report["training_evidence"]
        evaluation = load(campaign_dir / "evaluation-100.json")
        data = load(Path(campaign["run_dir"]) / "data-manifest.json")
        hashes.add(data["membership_sha256"])
        if report["status"] != "succeeded" or evidence["completed_steps"] != 15:
            raise ValueError(f"{job['job_id']}: incomplete training")
        if evaluation["count"] != 100 or evaluation["test_indices"] != list(range(100)):
            raise ValueError(f"{job['job_id']}: evaluation split/count mismatch")
        gradients = evidence["gradient_norms"]
        if len(gradients) < 15 or not all(math.isfinite(x) for x in gradients):
            raise ValueError(f"{job['job_id']}: missing or nonfinite gradient evidence")
        if report["audit"]["trajectory_records"] == 0:
            raise ValueError(f"{job['job_id']}: Cado has no training trajectories")
        kv = report["kv"]
        fitting = report.get("mapper_service", {})
        row = {
            **job,
            "correct": sum(r["reward"] for r in evaluation["records"]),
            "accuracy": evaluation["accuracy"],
            "training_seconds": report["native"]["training_seconds"],
            "worker_seconds": report["wall_seconds"],
            "reserved_gpu_worker_seconds": job["gpus"] * report["wall_seconds"],
            "evaluation_seconds": evaluation["seconds"],
            "submission_seconds": arm["submission_seconds"],
            "map_resumes": kv["map_resumes"],
            "rebuild_resumes": kv["rebuild_resumes"],
            "translation_batches": kv["translation_batches"],
            "background_fit_work_seconds": fitting.get("fitting_work_seconds", 0),
            "cado_trajectories": report["audit"]["trajectory_records"],
            "gradient_norms": gradients,
            "score_centering_metrics": {
                key: value
                for key, value in evidence.items()
                if "centering" in key or "sc_" in key
            },
            "campaign_path": str(campaign_dir),
            "isolated_score_centering_overhead": load(
                campaign_dir / "score-centering-overhead.json"
            )
            if job["arm"] == "api-map-sc"
            else None,
        }
        rows.append(row)
    if len(rows) != 8 or len(hashes) != 1:
        raise ValueError("expected eight arms on identical training membership")
    lines = [
        "# Eight-arm Nibble feasibility comparison",
        "",
        "Frozen sources (per-arm revisions are recorded in the submission manifest): "
        + ", ".join(
            sorted(
                {
                    job.get("source_revision", manifest["source_revision"])
                    for job in manifest["training_jobs"]
                }
            )
        )
        + ".",
        "",
        "15 updates × 256 prompts × 4 completions; model seed 1; same 100 official GSM8K test questions. Accuracy is deterministic greedy completion correctness under the original MathAgent parser/verifier.",
        "",
        "| Algorithm | Arm | Job | Correct / 100 | Train s | Worker s | Maps / rebuilds | Fit work s | Cado trajectories |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['algorithm']} | {row['arm']} | {row['job_id']} | "
            f"{row['correct']:g} | {row['training_seconds']:.2f} | "
            f"{row['worker_seconds']:.2f} | {row['map_resumes']} / "
            f"{row['rebuild_resumes']} | {row['background_fit_work_seconds']:.2f} | "
            f"{row['cado_trajectories']} |"
        )
    lines += [
        "",
        "Fitting work runs concurrently; it is not added to trainer time. Worker time includes initialization, checkpointing and fitter shutdown; GPU accounting also charges the dedicated mapper GPU. Slurm allocation elapsed must be reported separately because evaluation and launcher work lie outside the worker.",
        "",
        "No cache benefit can be inferred from fallback-only runs. A latency change is descriptive when completion lengths, accepted stale samples or training paths differ. Compare PPO and GRPO within their four-arm sets; raw loss values across implementations or centered objectives are not comparable.",
        "",
        "Score-centering is the paper's weighted head/tail correction composed with the fork's token-level upper-five rejection mask, under one pre-update optimizer step. The paper's experiments use REINFORCE; this clipped-PPO/GRPO composition is a tested analytic extension.",
        "",
        "TRL is an independent synchronous implementation. Its documented normalization/value-loss details differ from AReaL, so these are framework comparisons rather than an isolated scheduling ablation. One 15-update seed does not establish convergence or time to target accuracy.",
    ]
    output = (
        project / "docs" / f"comparison-matrix-{manifest['training_jobs'][0]['job_id']}"
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {"rows": rows, "source_revision": manifest["source_revision"]}, indent=2
        )
        + "\n"
    )
    output.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(output.with_suffix(".md"))


if __name__ == "__main__":
    main()
