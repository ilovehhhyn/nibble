import json
import os
import sys
from pathlib import Path

import pytest
import tplane

from nibble import audit


def cpu_run(directory):
    return tplane.Run(
        run_id="cpu-regression",
        store=tplane.FileRecordStore(directory),
        options=tplane.RunOptions(
            policy=tplane.Policy(
                on_agent=tplane.Action.ABORT,
                on_grader=tplane.Action.ABORT,
                on_infra=tplane.Action.ABORT,
                on_timeout=tplane.Action.ABORT,
            )
        ),
        read_host=tplane.proc_status_reader(os.getpid()),
        read_gpu=tplane.no_gpu_reader().read,
        gpu_total_bytes=(),
    )


def records(directory):
    return [json.loads(p.read_text()) for p in directory.rglob("*.json")]


def test_process_success_preserves_stdout_and_is_not_optimizer(tmp_path, capfd):
    run = cpu_run(tmp_path)
    assert (
        audit.execute(run, [sys.executable, "-c", "print('child-log')"], os.environ)
        == 0
    )
    run.finish()
    assert "child-log" in capfd.readouterr().out
    (record,) = records(tmp_path)
    assert record["outcome"] == "ok"
    assert dict(record["tags"])["optimizer_step"] == "false"
    assert record["resources"]["sample_count"] > 0


def test_process_failure_is_recorded_and_not_retried(tmp_path):
    run = cpu_run(tmp_path)
    with pytest.raises(tplane.UnitFailed):
        audit.execute(run, [sys.executable, "-c", "raise SystemExit(7)"], os.environ)
    run.finish()
    (record,) = records(tmp_path)
    assert record["outcome"] == "failed"
    assert record["failure"]["evidence"] == ["returncode=7"]
    assert record["attempt"] == 1


def test_evaluation_batch_records_actual_rewards_and_tokens(tmp_path, monkeypatch):
    import torch
    import datasets
    import transformers
    from nibble import comparison_data, evaluate

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, texts, **kwargs):
            class Inputs(dict):
                def to(self, device):
                    return self

            return Inputs(
                input_ids=torch.tensor([[1, 3], [1, 3]]),
                attention_mask=torch.ones(2, 2, dtype=torch.long),
            )

        def decode(self, ids):
            return str(ids[0])

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            assert kwargs["do_sample"] is False
            return torch.tensor([[1, 3, 7, 2], [1, 3, 8, 2]])

    monkeypatch.setattr(audit, "create_run", cpu_run)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer()
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: Model()
    )
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *a, **k: [
            {"question": "q0", "answer": "7"},
            {"question": "q1", "answer": "9"},
        ],
    )
    monkeypatch.setattr(
        comparison_data, "math_reward", lambda text, answer: float(text == answer)
    )
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys, "argv", ["evaluate", "checkpoint", str(output), "--count", "2"]
    )
    evaluate.main()
    result = json.loads(output.read_text())
    assert result["accuracy"] == 0.5
    (record,) = records(tmp_path / "evaluation-cado" / "audit")
    assert record["kind"] == "eval" and record["reward"] == 0.5
    assert record["cost"]["tokens_in"] == 4
    assert record["cost"]["tokens_out"] == 4
    assert dict(record["tags"])["test_start"] == "0"


def test_trl_policy_uses_supported_cado_actions():
    import inspect
    from nibble.trl_sync import _audit_run

    source = inspect.getsource(_audit_run)
    assert "Action.WARN" not in source
    tplane.Policy(on_grader=tplane.Action.ABORT)


def test_gpu_allocation_reader_and_actual_trl_run(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from nibble.trl_sync import _audit_run

    seen = []

    def reader(*, indices):
        seen.append(indices)
        return SimpleNamespace(
            total_bytes=(100,) * len(indices), read=lambda: (0,) * len(indices)
        )

    monkeypatch.setattr(tplane, "create_nvml_gpu_reader", reader)
    monkeypatch.setattr(tplane, "cgroup_memory_reader", lambda path: lambda: 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,3,5,7")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
    run = audit.create_run(tmp_path / "allocation" / "audit")
    assert run.options.gpus_held == 4 and run.options.cpus_held == 16
    run.finish()
    trl = _audit_run(tmp_path / "trl", True, SimpleNamespace(gpu_ids=["1", "3", "5"]))
    assert trl.options.policy.on_grader is tplane.Action.ABORT
    trl.finish()
    assert seen == [(1, 3, 5, 7), (1, 3, 5)]


def test_replay_validation_rejects_partial_transition(tmp_path):
    from nibble.validate_transfer import validate

    directory = tmp_path / "v0-v1"
    directory.mkdir()
    (directory / "summary.json").write_text(
        json.dumps(
            dict(source_version=0, target_version=1, requested=64, interrupted=0)
        )
    )
    (directory / "rows.json").write_text("[]")
    with pytest.raises(ValueError, match="no complete"):
        validate(tmp_path, 1, 64)
    (directory / "summary.json").write_text(
        json.dumps(
            dict(source_version=0, target_version=1, requested=64, interrupted=1)
        )
    )
    (directory / "rows.json").write_text('[{"transfer_cached_tokens":0}]')
    validate(tmp_path, 1, 64)
    with pytest.raises(FileNotFoundError):
        validate(tmp_path, 2, 64)
