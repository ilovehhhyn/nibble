"""Run the original PPO/GRPO GSM8K arm; optionally center scores and map KV."""

from contextlib import contextmanager, nullcontext
import os
from pathlib import Path
import subprocess
import time

from .report import atomic_json

ARM_STEPS = 15
PROMPTS_PER_STEP = 256
COMPLETIONS_PER_PROMPT = 4


def arm_overrides(spec, runtime, run_dir):
    """Exact AReaL CLI overrides for the selected 15-step mapper arm."""
    run_dir = Path(run_dir)
    fileroot = run_dir / "areal"
    values = {
        "experiment_name": "nibble",
        "trial_name": run_dir.name,
        "scheduler.type": "local",
        "cluster.n_gpus_per_node": 3,
        "cluster.fileroot": str(fileroot),
        "cluster.name_resolve.nfs_record_root": str(fileroot / "name_resolve"),
        "rollout.backend": "sglang:d2p1t1",
        "rollout.enable_rollout_tracing": True,
        "actor.backend": "fsdp:d1p1t1",
        "actor.path": runtime.models[spec.model],
        "actor.attn_impl": "sdpa",
        "ref.attn_impl": "sdpa",
        "train_dataset.batch_size": PROMPTS_PER_STEP,
        "gconfig.n_samples": COMPLETIONS_PER_PROMPT,
        "saver.freq_steps": 1,
        "saver.freq_epochs": None,
        # The comparison uses a shared external 100-prompt evaluation.
        "evaluator.freq_steps": None,
        "evaluator.freq_epochs": None,
        "sglang.attention_backend": "flashinfer",
        "total_train_steps": ARM_STEPS,
    }
    if spec.algorithm == "ppo":
        values["critic.attn_impl"] = "sdpa"
    if spec.cache_mode == "map":
        values.update(
            {
                "sglang.kv_retain_on_abort": True,
                "sglang.kv_migration_mode": "map",
                "sglang.kv_max_migration_depth": -1,
                "sglang.kv_mapper_module": "areal.experimental.kvmap.sglang_bridge",
                "sglang.kv_mapper_registry": str(run_dir / "registry"),
                "sglang.kv_event_log": str(run_dir / "kv_events"),
                "rollout.kv_mapper_wait_s": 0,
            }
        )
    return values


def render_config(spec, runtime, run_dir):
    """Render the original PPO config plus only the selected arm overrides."""
    from omegaconf import OmegaConf
    from areal.api.cli_args import GRPOConfig, load_expr_config

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    values = arm_overrides(spec, runtime, run_dir)
    config = OmegaConf.load(
        runtime.areal_root / f"examples/math/gsm8k_{spec.algorithm}.yaml"
    )
    for key, value in values.items():
        OmegaConf.update(
            config, key, value, merge=False, force_add=(key == "total_train_steps")
        )
    if spec.cache_mode == "map":
        (run_dir / "registry").mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "areal.yaml"
    OmegaConf.save(config, config_path)
    parsed, _ = load_expr_config(["--config", str(config_path)], GRPOConfig)
    return parsed


@contextmanager
def _cado_watcher(runtime, run_dir):
    """Run the copied cado sidecar against this trial's AReaL logs and cgroup."""
    run_dir = Path(run_dir)
    fileroot = run_dir / "areal"
    log_dir = (
        fileroot / "logs" / os.environ.get("USER", "hh9077") / "nibble" / run_dir.name
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    cado_dir = run_dir / "audit"
    done_file = run_dir / "cado.done"
    done_file.unlink(missing_ok=True)
    watcher_log = (run_dir / "cado_watch.log").open("a")
    env = runtime.environment(run_dir)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(runtime.cado_src), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    watcher = Path(__file__).resolve().parents[2] / "tools" / "cado_watch.py"
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split(":", 2)[-1].lstrip("/")
    command = [
        str(runtime.python),
        str(watcher),
        "--log-dir",
        str(log_dir),
        "--extra-log",
        str(run_dir / "worker.log"),
        "--run-dir",
        str(cado_dir),
        "--run-id",
        run_dir.name,
        "--source",
        "gsm8k",
        "--cgroup-dir",
        str(cgroup),
        "--gpu",
        "nvml",
        "--action",
        "warn",
        "--done-file",
        str(done_file),
    ]
    proc = subprocess.Popen(
        command,
        cwd=runtime.areal_root,
        env=env,
        stdout=watcher_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        yield
    finally:
        done_file.touch()
        try:
            proc.wait(timeout=120)
            if proc.returncode:
                raise RuntimeError("Cado watcher failed; see cado_watch.log")
        finally:
            watcher_log.close()


def run(spec, runtime, run_dir):
    """Train on the full original GSM8K train split using AReaL PPOTrainer."""
    from areal import PPOTrainer
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer
    from .mapper import fitter_service
    from .comparison_data import prepare_training

    run_dir = Path(run_dir)
    wall_started = time.perf_counter()
    config = render_config(spec, runtime, run_dir)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = get_custom_dataset(
        split="train", dataset_config=config.train_dataset, tokenizer=tokenizer
    )
    shared_dataset, _, data_metadata = prepare_training(tokenizer)
    if list(shared_dataset) != list(train_dataset):
        raise ValueError(
            "shared comparison training set differs from the original fork"
        )
    atomic_json(run_dir / "data-manifest.json", data_metadata)
    valid_dataset = get_custom_dataset(
        split="test", dataset_config=config.valid_dataset, tokenizer=tokenizer
    )
    workflow_kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    eval_workflow_kwargs["temperature"] = 0.6

    fitter_lifecycle = (
        fitter_service(runtime, run_dir, ARM_STEPS)
        if spec.fit_mappers
        else nullcontext()
    )
    trainer_class = (
        PPOTrainer  # Original gsm8k_rl.py uses this trainer for both configs.
    )
    workflow = "areal.workflow.openai.math_agent.MathAgent"
    with _cado_watcher(runtime, run_dir):
        with fitter_lifecycle:
            with trainer_class(
                config, train_dataset=train_dataset, valid_dataset=valid_dataset
            ) as trainer:
                initialized = time.perf_counter()
                trainer.train(
                    workflow=workflow,
                    workflow_kwargs=workflow_kwargs,
                    eval_workflow=workflow,
                    eval_workflow_kwargs=eval_workflow_kwargs,
                )
                trained = time.perf_counter()
    ended = time.perf_counter()
    atomic_json(
        run_dir / "native-result.json",
        {
            "initialization_seconds": initialized - wall_started,
            "training_seconds": trained - initialized,
            "teardown_seconds": ended - trained,
            "native_total_seconds": ended - wall_started,
            "steps": ARM_STEPS,
            "requested_optimizer_steps": ARM_STEPS,
            "trainer": f"AReaL {trainer_class.__name__}",
            "objective": f"AReaL original clipped {spec.algorithm.upper()}"
            + (" + weighted score centering" if spec.score_centering else ""),
            "score_centering": spec.score_centering,
            "spec": spec.to_dict(),
            "training_dataset": "full GSM8K train split via get_custom_dataset",
            "validation_dataset": "full GSM8K test split via get_custom_dataset",
            "internal_evaluation": "disabled; common 100-prompt evaluation is external",
            "cache_mode": spec.cache_mode,
            "mapper_fitter": (
                "background service; training_seconds excludes shutdown tail"
                if spec.fit_mappers
                else "disabled"
            ),
            "checkpoints_retained": True,
        },
    )
