"""Cheap checks for allocation gates and the comparison budget."""

from dataclasses import replace
from pathlib import Path
import pytest
from nibble.spec import TrainSpec
from nibble.native import render_config
from nibble.runtime import Runtime


def test_slurm_spec_rejects_wrong_gpu_count_and_loss_composition():
    with pytest.raises(ValueError, match="requires four GPUs"):
        TrainSpec(model="x", target="slurm", max_gpus=2)
    with pytest.raises(ValueError, match="supported policy objective"):
        TrainSpec(model="x", score_centering=True)
    with pytest.raises(ValueError, match="zero staleness"):
        TrainSpec(
            model="x",
            target="slurm",
            max_gpus=4,
            execution="standalone_sync",
            max_staleness=1,
        )


def test_comparison_caps_fit_within_allocation():
    # 3 training arms, 4 evaluations, 90 seconds for API start and cleanup.
    from nibble.campaign import ARM_TIMEOUT_SECONDS, EVALUATION_TIMEOUT_SECONDS

    script = (Path(__file__).resolve().parents[1] / "jobs/compare.sbatch").read_text()
    import re

    hours, minutes, seconds = map(
        int, re.search(r"--time=(\d+):(\d+):(\d+)", script).groups()
    )
    assert (
        3 * ARM_TIMEOUT_SECONDS + 4 * EVALUATION_TIMEOUT_SECONDS + 90
        < hours * 3600 + minutes * 60 + seconds
    )


def test_pinned_configs_preserve_shared_objective(tmp_path):
    pytest.importorskip("areal")
    import areal

    source = Path(areal.__file__).resolve().parents[1]
    if not (source / "NIBBLE_PATCHED").exists():
        pytest.skip("requires pinned Nibble-patched AReaL")
    spec = TrainSpec(
        model="test",
        target="slurm",
        max_gpus=4,
        max_steps=15,
        objective="group_pg_tis",
        max_staleness=2,
    )
    runtime = Runtime(
        tmp_path,
        source,
        tmp_path,
        Path("python"),
        {"test": str(tmp_path)},
        {},
        {},
        {},
        ["0", "1", "2", "3"],
        "x" * 32,
    )
    for sc in (False, True):
        run = tmp_path / str(sc)
        run.mkdir()
        arm = replace(
            spec,
            score_centering=sc,
            fit_mappers=sc,
            cache_mode="map" if sc else "recompute",
        )
        cfg = render_config(arm, runtime, run)
        assert cfg.actor.reward_norm.mean_level == "group"
        assert cfg.actor.reward_norm.std_level is None
        assert cfg.actor.adv_norm is None
        assert cfg.gconfig.top_k == -1
        assert cfg.actor.prox_logp_method == "reuse_train_logp"
        assert cfg.rollout.backend == "sglang:d2p1t1"
        assert cfg.saver.freq_steps == 1
