"""CPU reference checks for the observed Della compiler and child-exit failures."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from nibble.cuda_build import build_environment
from nibble.runner import WorkerFailureMonitor


def test_cuda_build_uses_existing_headers_and_scratch_caches(tmp_path, monkeypatch):
    package = tmp_path / "nvidia/curand"
    include = package / "include"
    include.mkdir(parents=True)
    for name in ("curand.h", "curand_kernel.h"):
        (include / name).touch()
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(submodule_search_locations=[str(package)]),
    )
    monkeypatch.setenv("FLASHINFER_EXTRA_CUDAFLAGS", "-DKEEP_EXISTING=1")
    env = build_environment(tmp_path)
    assert env["FLASHINFER_EXTRA_CUDAFLAGS"].split() == [
        "-DKEEP_EXISTING=1",
        f"-I{include}",
    ]
    for key in (
        "FLASHINFER_WORKSPACE_BASE",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "XDG_CACHE_HOME",
        "CUDA_CACHE_PATH",
    ):
        assert Path(env[key]).is_relative_to(tmp_path / "work")
    monkeypatch.setenv("FLASHINFER_EXTRA_CUDAFLAGS", env["FLASHINFER_EXTRA_CUDAFLAGS"])
    assert (
        build_environment(tmp_path)["FLASHINFER_EXTRA_CUDAFLAGS"]
        == env["FLASHINFER_EXTRA_CUDAFLAGS"]
    )


def test_fatal_scheduler_exit_is_detected_across_log_chunks(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text("normal initialization\n")
    monitor = WorkerFailureMonitor(path)
    assert not monitor.poll()
    with path.open("a") as stream:
        stream.write(
            "Scheduler hit an exception: Traceback\nfatal error: curand.h: No such file or directory\n"
        )
    assert not monitor.poll()
    with path.open("a") as stream:
        stream.write("SIGQUIT received. It usually means one child failed.\n")
    assert monitor.poll()


def test_transient_rollout_rejections_do_not_terminate_training(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text("Connection refused; retrying\nWorkflow execution failed\n")
    assert not WorkerFailureMonitor(path).poll()


def test_fatal_markers_survive_chunk_boundary_and_output_order(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text(
        "x" * (65536 - 12)
        + "SIGQUIT received.\nScheduler hit an exception: Traceback\n"
    )
    monitor = WorkerFailureMonitor(path)
    assert not monitor.poll()
    assert monitor.poll()


def test_score_centering_contract_failure_stops_without_child_crash(tmp_path):
    path = tmp_path / "worker.log"
    path.write_text(
        "x" * (65536 - 22) + "NIBBLE_SCORE_CENTERING=1 requires sampling "
        "with full-vocabulary top-k; this trajectory will be rejected.\n"
    )
    monitor = WorkerFailureMonitor(path)
    assert not monitor.poll()
    assert monitor.poll()
    assert (
        monitor.error == "Unsupported score-centering sampling contract; see worker.log"
    )
    assert not monitor.child_quit
