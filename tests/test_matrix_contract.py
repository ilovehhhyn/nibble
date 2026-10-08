"""The HTTP contract must represent all eight comparisons without silently changing requests."""

from pathlib import Path
from fastapi.testclient import TestClient
from nibble.server import create_app
from nibble.runtime import Runtime
from nibble.spec import TrainSpec


def test_eight_comparison_contracts_preserve_algorithm_mapping_and_score_switch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SLURM_JOB_ID", "test-allocation")
    (tmp_path / "NIBBLE_PATCHED").write_text("{}")
    runtime = Runtime(
        tmp_path,
        tmp_path,
        tmp_path,
        Path("/usr/bin/python3"),
        {"qwen": str(tmp_path)},
        {"gsm8k": "registered"},
        {"gsm8k": "registered"},
        {"math": "registered"},
        ["0", "1", "2", "3"],
        "a" * 32,
    )
    client = TestClient(create_app(runtime, start_runner=False))
    client.headers["Authorization"] = "Bearer " + runtime.token
    ids = []
    for algorithm in ("ppo", "grpo"):
        for arm in ("sync", "areal", "api-map", "api-map-sc"):
            mapping = arm.startswith("api-map")
            spec = TrainSpec(
                model="qwen",
                algorithm=algorithm,
                target="slurm",
                execution="trl_sync" if arm == "sync" else "fork_arm",
                objective=f"fork_{algorithm}",
                score_centering=arm == "api-map-sc",
                max_gpus=4 if mapping else 3,
                max_steps=15,
                prompts_per_step=256,
                rollouts_per_prompt=4,
                max_new_tokens=1024,
                max_prompt_tokens=1024,
                max_concurrent_rollouts=256,
                max_staleness=0 if arm == "sync" else 2,
                learning_rate=1.7e-5 if algorithm == "ppo" else 6e-6,
                cache_mode="map" if mapping else "recompute",
                fit_mappers=mapping,
                timeout_seconds=20000,
            )
            created = client.post("/v1/jobs", json=spec.to_dict())
            assert created.status_code == 202, created.text
            job_id = created.json()["id"]
            stored = client.get(f"/v1/jobs/{job_id}").json()
            assert stored["spec"] == spec.to_dict()
            assert (
                client.post("/v1/plan", json=spec.to_dict()).json()["physical_gpus"]
                == spec.max_gpus
            )
            ids.append(job_id)
    assert len(set(ids)) == 8
