import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from nibble.runtime import Runtime
from nibble.server import create_app
from nibble.spec import TrainSpec
from nibble.store import Store


@pytest.fixture
def runtime(tmp_path):
    return Runtime(
        tmp_path,
        tmp_path,
        tmp_path,
        Path("/usr/bin/python3"),
        {"tiny": str(tmp_path)},
        {"gsm8k": "nibble.native:gsm8k_dataset"},
        {"gsm8k": "areal.reward.gsm8k:gsm8k_reward_fn"},
        {"math": "nibble.workflow:AuditedRLVRWorkflow"},
        ["0"],
        "x" * 32,
    )


def test_invalid_compute_and_algorithm_fail_before_launch(runtime):
    for options in (
        {"dataset": []},
        {"mapper_registry": 123},
        {"max_gpus": 0},
        {"max_steps": True},
        {"learning_rate": float("nan")},
        {"score_centering": True},
        {"max_staleness": 1},
        {"cache_mode": "map"},
        {"timeout_seconds": 999},
    ):
        with pytest.raises(ValueError):
            TrainSpec(model="tiny", **options)
    with pytest.raises(ValueError):
        runtime.plan(TrainSpec(model="not-registered"))
    with pytest.raises(ValueError):
        runtime.plan(TrainSpec(model="tiny", target="local", max_gpus=2))


def test_cancel_and_restart_do_not_replay_optimizer_steps(tmp_path):
    store = Store(tmp_path)
    first = store.create(TrainSpec(model="tiny").to_dict())
    second = store.create(TrainSpec(model="tiny").to_dict())
    store.cancel(second["id"])
    assert store.claim()["id"] == first["id"]
    store.recover()
    assert store.get(first["id"])["status"] == "interrupted"
    assert store.get(second["id"])["status"] == "cancelled"
    assert store.claim() is None


def test_authenticated_api_submit_cancel_and_validation(runtime):
    client = TestClient(create_app(runtime, start_runner=False))
    body = TrainSpec(model="tiny").to_dict()
    assert client.post("/v1/jobs", json=body).status_code == 401
    client.headers["Authorization"] = "Bearer " + runtime.token
    assert client.post("/v1/plan", json=body).json()["physical_gpus"] == 1
    assert client.post("/v1/jobs", json={**body, "model": "unknown"}).status_code == 422
    created = client.post("/v1/jobs", json=body)
    assert created.status_code == 202
    job_id = created.json()["id"]
    assert client.get(f"/v1/jobs/{job_id}/report").status_code == 409
    assert client.post(f"/v1/jobs/{job_id}/cancel").json()["status"] == "cancelled"
    assert client.get("/v1/jobs/not-a-job").status_code == 404


def test_venv_interpreter_is_not_resolved_out_of_environment(tmp_path, monkeypatch):
    import sys

    fake_python = tmp_path / "venv" / "bin" / "python"
    fake_python.parent.mkdir(parents=True)
    fake_python.symlink_to(sys.executable)
    config = {
        "root": str(tmp_path),
        "areal_root": str(tmp_path),
        "cado_src": str(tmp_path),
        "python": str(fake_python),
        "models": {},
        "datasets": {},
        "rewards": {},
        "environments": {},
        "gpu_ids": ["0"],
    }
    cfg = tmp_path / "runner.json"
    cfg.write_text(json.dumps(config))
    monkeypatch.setenv("NIBBLE_RUNNER_CONFIG", str(cfg))
    runtime = Runtime.load(require_token=False)
    assert runtime.python == fake_python
    assert runtime.environment(tmp_path)["PATH"].split(":")[0] == str(
        fake_python.parent
    )


def test_cancel_cleans_owned_child_with_separate_session(runtime, tmp_path):
    import os
    import sys
    import time
    import psutil
    from nibble.runner import execute

    fake = tmp_path / "fake" / "nibble"
    fake.mkdir(parents=True)
    (fake / "__init__.py").write_text("")
    (fake / "worker.py").write_text(
        "import subprocess,sys,time,pathlib\n"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)\n"
        "pathlib.Path(sys.argv[1],'child.pid').write_text(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    runtime.python = Path(sys.executable)
    runtime.environment = lambda run_dir: {**os.environ, "PYTHONPATH": str(fake.parent)}
    run_dir = tmp_path / "cancelled"
    start = time.monotonic()
    report = execute(
        runtime,
        run_dir,
        TrainSpec(model="tiny"),
        cancelled=lambda: time.monotonic() - start > 0.75,
        check_gpu=False,
    )
    assert report["status"] == "cancelled"
    pid = int((run_dir / "child.pid").read_text())
    assert (
        not psutil.pid_exists(pid)
        or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    )


def test_toy_reward_supplies_signal_for_saved_fixture():
    from nibble.native import digit_reward

    a = digit_reward(None, "Choose one digit from 0 through 9.", None, None, "9")
    b = digit_reward(None, "Choose 1 digit from 0 through 9.", None, None, "9")
    assert (a, b) == (0.0, 1 / 9)
