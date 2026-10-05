"""Adapter to Helen's AReaL fork. Does not patch the underlying trainer or scheduler."""

from pathlib import Path
import importlib
import time


def import_symbol(path):
    module, symbol = path.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def render_config(spec, runtime, run_dir):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(runtime.areal_root / "examples/math/gsm8k_grpo.yaml")
    values = {
        "experiment_name": "nibble",
        "trial_name": run_dir.name,
        "seed": spec.seed,
        "total_train_epochs": 1,
        "total_train_steps": spec.max_steps,
        "cluster.n_nodes": 1,
        "cluster.n_gpus_per_node": 1
        if spec.target == "login_pilot"
        else (3 if spec.target == "slurm" else 2),
        "cluster.fileroot": str(run_dir / "areal"),
        "cluster.name_resolve.nfs_record_root": str(run_dir / "name_resolve"),
        "scheduler.type": "local",
        "actor.backend": "fsdp:d1p1t1",
        "actor.path": runtime.models[spec.model],
        "actor.attn_impl": "sdpa",
        "actor.weight_update_mode": "disk",
        "actor.optimizer.lr": spec.learning_rate,
        "actor.mb_spec.max_tokens_per_mb": 512
        if spec.target == "login_pilot"
        else 2048,
        "actor.gradient_checkpointing": True,
        "actor.kl_ctl": 0.0,
        "ref": None,
        "valid_dataset": None,
        "rollout.backend": "sglang:d2p1t1"
        if spec.target == "slurm"
        else "sglang:d1p1t1",
        "rollout.max_head_offpolicyness": spec.max_staleness,
        "rollout.max_concurrent_rollouts": spec.max_concurrent_rollouts,
        "rollout.enable_rollout_tracing": True,
        "rollout.dump_to_file": True,
        "gconfig.n_samples": spec.rollouts_per_prompt,
        "gconfig.max_new_tokens": spec.max_new_tokens,
        "gconfig.max_tokens": spec.max_prompt_tokens + spec.max_new_tokens,
        "gconfig.temperature": 1.0,
        "gconfig.top_p": 1.0,
        "gconfig.top_k": -1,
        "train_dataset.batch_size": spec.prompts_per_step,
        "train_dataset.shuffle": False,
        "train_dataset.num_workers": 0,
        "train_dataset.pin_memory": False,
        "sglang.attention_backend": "flashinfer",
        "sglang.context_length": spec.max_prompt_tokens + spec.max_new_tokens,
        "sglang.mem_fraction_static": 0.15 if spec.target == "login_pilot" else 0.65,
        "sglang.max_running_requests": spec.max_concurrent_rollouts,
        "sglang.disable_cuda_graph": True,
        "saver.freq_epochs": None,
        "saver.freq_steps": 1 if spec.target == "slurm" else spec.max_steps,
        "saver.mode": "sync",
        "recover.mode": "disabled",
        "recover.freq_epochs": None,
        "recover.freq_secs": None,
        "evaluator.freq_epochs": None,
        "evaluator.freq_steps": None,
        "stats_logger.wandb.mode": "disabled",
    }
    if spec.objective == "group_pg_tis":
        values.update(
            {
                "actor.reward_scaling": 1.0,
                "actor.reward_bias": 0.0,
                "actor.reward_norm.std_level": None,
                "actor.adv_norm": None,
                "actor.recompute_logprob": False,
                "actor.use_decoupled_loss": True,
                "actor.prox_logp_method": "reuse_train_logp",
                "actor.rejection_sampling": None,
                "actor.optimizer.warmup_steps_proportion": 0.0,
                "actor.discount": 1.0,
                "actor.gae_lambda": 1.0,
            }
        )
    if spec.cache_mode == "map":
        values.update(
            {
                "sglang.kv_retain_on_abort": True,
                "sglang.kv_migration_mode": "map",
                "sglang.kv_mapper_module": "areal.experimental.kvmap.sglang_bridge",
                "sglang.kv_mapper_registry": str(
                    (run_dir / "registry")
                    if spec.fit_mappers
                    else Path(spec.mapper_registry).resolve()
                ),
                "sglang.kv_max_migration_depth": 1,
                "sglang.kv_event_log": str(run_dir / "kv-events"),
            }
        )
    if spec.fit_mappers:
        (run_dir / "registry").mkdir(exist_ok=True)
    for key, value in values.items():
        OmegaConf.update(cfg, key, value, merge=False)
    path = run_dir / "areal.yaml"
    OmegaConf.save(cfg, path)
    from areal.api.cli_args import GRPOConfig, load_expr_config

    parsed, _ = load_expr_config(["--config", str(path)], GRPOConfig)
    return parsed


def run(spec, runtime, run_dir):
    from areal import PPOTrainer
    from areal.utils.hf_utils import load_hf_tokenizer
    from .report import atomic_json

    config = render_config(spec, runtime, run_dir)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    data_fn = import_symbol(runtime.datasets[spec.dataset])
    dataset = data_fn(
        tokenizer=tokenizer,
        count=spec.prompts_per_step * (spec.max_steps + 2),
        max_prompt_tokens=spec.max_prompt_tokens,
    )
    if len(dataset) < spec.prompts_per_step * spec.max_steps:
        raise ValueError("dataset is too short for requested optimizer steps")
    workflow_class = runtime.environments[spec.environment]
    workflow_kwargs = {
        "reward_fn": runtime.rewards[spec.reward].replace(":", "."),
        "gconfig": config.gconfig,
        "tokenizer": config.tokenizer_path,
        "enable_thinking": False,
    }
    if workflow_class == "nibble.workflow:AuditedRLVRWorkflow":
        workflow_kwargs.update(audit_on=spec.audit_on, audit_dir=str(run_dir / "audit"))
    wall = time.perf_counter()
    trainer_class = PPOTrainer
    if spec.fit_mappers:
        from .mapper import fitting_trainer

        trainer_class = fitting_trainer(PPOTrainer, spec, runtime, run_dir)
    with trainer_class(config, train_dataset=dataset, valid_dataset=None) as trainer:
        initialized = time.perf_counter()
        trainer.train(
            workflow=workflow_class.replace(":", "."),
            workflow_kwargs=workflow_kwargs,
        )
        trained = time.perf_counter()
    ended = time.perf_counter()
    atomic_json(
        run_dir / "native-result.json",
        {
            "initialization_seconds": initialized - wall,
            "training_seconds": trained - initialized,
            "teardown_seconds": ended - trained,
            "native_total_seconds": ended - wall,
            "requested_optimizer_steps": spec.max_steps,
            "trainer": "AReaL PPOTrainer / FSDP GRPO",
            "score_centering": spec.score_centering,
            "objective": spec.objective,
        },
    )


def gsm8k_dataset(*, tokenizer, count, max_prompt_tokens):
    from datasets import Dataset, load_dataset

    rows = []
    data = load_dataset("openai/gsm8k", "main", split="train")
    for item in data:
        messages = [
            {
                "role": "user",
                "content": item["question"] + "\nPut the final answer in \\boxed{}.",
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(ids) <= max_prompt_tokens:
            rows.append({"messages": messages, "answer": item["answer"]})
        if len(rows) == count:
            break
    return Dataset.from_list(rows)


def digit_dataset(*, tokenizer, count, max_prompt_tokens):
    """Tiny synthetic preference environment; not a model-quality benchmark."""
    from datasets import Dataset

    rows = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Choose one digit from 0 through 9. Reply with only that digit.",
                }
            ],
            "answer": "9",
        }
        for _ in range(count)
    ]
    ids = tokenizer.apply_chat_template(
        rows[0]["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) > max_prompt_tokens:
        raise ValueError("digit prompt exceeds max_prompt_tokens")
    return Dataset.from_list(rows)


def digit_reward(prompt, completions, prompt_ids, completion_ids, answer, **kwargs):
    """Synthetic wiring reward: first generated digit / 9, including prose. Not a quality metric."""
    import re

    text = completions.strip()
    match = re.search(r"[0-9]", text)
    return int(match.group(0)) / 9.0 if match else 0.0


def strict_gsm8k_reward(
    prompt, completions, prompt_ids, completion_ids, answer, **kwargs
):
    """Use the original verifier, allowing infrastructure exceptions to reach Cado."""
    from areal.reward import get_math_verify_worker

    return get_math_verify_worker().verify(str(completions), str(answer))


def benchmark_dataset(*, tokenizer, count, max_prompt_tokens):
    """Use only original holdout indices for training; calibration/dev stay separate.

    Original 'holdout' refers to the mapper experiment's training-set partition,
    not the official GSM8K test set used for final quality evaluation.
    """
    import os
    import json
    from datasets import Dataset, load_dataset

    split = json.loads(Path(os.environ["NIBBLE_SPLITS"]).read_text())
    data = load_dataset("openai/gsm8k", "main", split="train")
    rows = []
    for idx in split["indices"]["holdout"]:
        item = data[idx]
        messages = [
            {
                "role": "user",
                "content": item["question"] + "\nPut the final answer in \\boxed{}.",
            }
        ]
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if len(ids) <= max_prompt_tokens:
            rows.append(
                {"messages": messages, "answer": item["answer"], "dataset_index": idx}
            )
        if len(rows) == count:
            break
    return Dataset.from_list(rows)
