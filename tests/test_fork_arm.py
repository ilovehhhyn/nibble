from pathlib import Path
from types import SimpleNamespace

import pytest

from nibble.fork_arm import (
    ARM_STEPS,
    COMPLETIONS_PER_PROMPT,
    PROMPTS_PER_STEP,
    arm_overrides,
)


@pytest.fixture
def arm_context(tmp_path):
    spec = SimpleNamespace(model="qwen", algorithm="ppo", cache_mode="recompute")
    runtime = SimpleNamespace(models={"qwen": "/models/qwen"})
    return spec, runtime, tmp_path / "trial-123"


def test_exact_arm_uses_selected_job_settings_without_mapper_flags(arm_context):
    spec, runtime, run_dir = arm_context
    overrides = arm_overrides(spec, runtime, run_dir)

    assert (ARM_STEPS, PROMPTS_PER_STEP, COMPLETIONS_PER_PROMPT) == (15, 256, 4)
    assert overrides["total_train_steps"] == 15
    assert overrides["train_dataset.batch_size"] == 256
    assert overrides["gconfig.n_samples"] == 4
    assert overrides["cluster.n_gpus_per_node"] == 3
    assert overrides["actor.backend"] == "fsdp:d1p1t1"
    assert overrides["rollout.backend"] == "sglang:d2p1t1"
    assert not any("kv_" in key or "migration" in key for key in overrides)


def test_map_arm_adds_original_migration_settings(arm_context):
    _, runtime, run_dir = arm_context
    spec = SimpleNamespace(model="qwen", algorithm="ppo", cache_mode="map")
    overrides = arm_overrides(spec, runtime, run_dir)

    assert overrides["sglang.kv_retain_on_abort"] is True
    assert overrides["sglang.kv_migration_mode"] == "map"
    assert overrides["sglang.kv_max_migration_depth"] == -1
    assert (
        overrides["sglang.kv_mapper_module"] == "areal.experimental.kvmap.sglang_bridge"
    )
    assert overrides["sglang.kv_mapper_registry"] == str(run_dir / "registry")
    assert overrides["sglang.kv_event_log"] == str(run_dir / "kv_events")
    assert overrides["rollout.kv_mapper_wait_s"] == 0


def test_rendered_config_preserves_original_ppo_defaults(arm_context, tmp_path):
    omega = pytest.importorskip("omegaconf")
    pytest.importorskip("areal.api.cli_args")
    areal_root = Path("/scratch/gpfs/ARORA/hh9077/areal-kv/AReaL-arm")
    original_path = areal_root / "examples/math/gsm8k_ppo.yaml"
    if not original_path.is_file():
        pytest.skip("original AReaL-arm PPO config is not mounted")

    from nibble.fork_arm import render_config

    spec, runtime, _ = arm_context
    runtime.areal_root = areal_root
    original = omega.OmegaConf.load(original_path)
    rendered = render_config(spec, runtime, tmp_path / "rendered")

    assert rendered.actor.optimizer.lr == original.actor.optimizer.lr
    assert rendered.actor.optimizer.type == original.actor.optimizer.type
    assert rendered.actor.reward_scaling == original.actor.reward_scaling
    assert rendered.actor.reward_bias == original.actor.reward_bias
    assert rendered.actor.adv_norm.mean_level == original.actor.adv_norm.mean_level
    assert rendered.actor.adv_norm.std_level == original.actor.adv_norm.std_level
    assert rendered.actor.use_decoupled_loss == original.actor.use_decoupled_loss
    assert rendered.critic.is_critic == original.critic.is_critic
    assert rendered.critic.optimizer.lr == rendered.actor.optimizer.lr


def test_grpo_config_retains_group_standardization(arm_context, tmp_path):
    pytest.importorskip("omegaconf")
    pytest.importorskip("areal.api.cli_args")
    from nibble.fork_arm import render_config

    spec, runtime, _ = arm_context
    spec.algorithm = "grpo"
    runtime.areal_root = Path("/scratch/gpfs/ARORA/hh9077/areal-kv/AReaL-arm")
    if not runtime.areal_root.is_dir():
        pytest.skip("original fork is not mounted")
    rendered = render_config(spec, runtime, tmp_path / "grpo-rendered")
    assert rendered.critic is None
    assert rendered.actor.optimizer.lr == 6e-6
    assert rendered.actor.reward_norm.group_size == 4
    assert rendered.actor.reward_norm.mean_level == "group"
    assert rendered.actor.reward_norm.std_level == "group"
    assert rendered.actor.rejection_sampling.action == "mask"
    assert rendered.actor.rejection_sampling.upper == 5
    assert rendered.actor.ppo_n_minibatches == 1
    assert rendered.actor.use_decoupled_loss


def test_final_evaluation_selects_actor_not_critic(tmp_path, monkeypatch):
    from nibble.fork_campaign import actor_checkpoint

    monkeypatch.setenv("USER", "hh9077")
    run = tmp_path / "trial"
    checkpoint_root = run / "areal/checkpoints/hh9077/nibble/trial"
    for role in ("default", "critic"):
        path = checkpoint_root / role / "epoch0epochstep14globalstep14"
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
    selected = actor_checkpoint(run)
    assert selected.parent.name == "default"
