"""Official Hugging Face TRL synchronous PPO and GRPO comparison adapters.

TRL owns the optimization objectives. This module supplies the benchmark's
prompt order, MathAgent-compatible reward, SGLang rollout transport, and
checkpoint publication between synchronous updates.
"""

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import hashlib
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

TRL_VERSION = "0.28.0"
TRL_COMMIT = "49ef33428c47235991acb4e185ea599b70c6dab4"
UPDATES = 15
PROMPTS_PER_UPDATE = 256
COMPLETIONS_PER_PROMPT = 4
COMPLETIONS_PER_UPDATE = PROMPTS_PER_UPDATE * COMPLETIONS_PER_PROMPT
MAX_NEW_TOKENS = 1024
MAX_PROMPT_TOKENS = 1024
GRADER_WORKERS = 4
GRADER_TIMEOUT_SECONDS = 15


def _grader_worker_init():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


class MathRewardPool:
    def __init__(self):
        self.executor = ProcessPoolExecutor(
            max_workers=GRADER_WORKERS,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_grader_worker_init,
        )
        self.inflight = threading.Semaphore(GRADER_WORKERS)

    def __call__(self, completion, answer):
        from .comparison_data import math_reward

        with self.inflight:
            future = self.executor.submit(math_reward, completion, answer)
            return float(future.result(timeout=GRADER_TIMEOUT_SECONDS))

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)


def _jsonl(path, record):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()


def _audit_run(run_dir, enabled, runtime):
    if not enabled:
        return None
    import tplane

    reader = tplane.create_nvml_gpu_reader(
        indices=tuple(int(x) for x in runtime.gpu_ids[:3])
    )
    cgroup = Path("/sys/fs/cgroup") / Path(
        "/proc/self/cgroup"
    ).read_text().strip().split(":", 2)[-1].lstrip("/")
    return tplane.Run(
        run_id=Path(run_dir).name,
        store=tplane.FileRecordStore(Path(run_dir) / "audit"),
        options=tplane.RunOptions(
            gpus_held=len(reader.total_bytes),
            policy=tplane.Policy(
                on_agent=tplane.Action.ABORT,
                on_grader=tplane.Action.ABORT,
                on_infra=tplane.Action.ABORT,
                on_timeout=tplane.Action.ABORT,
                max_attempts=1,
            ),
        ),
        read_host=tplane.cgroup_memory_reader(cgroup),
        read_gpu=reader.read,
        gpu_total_bytes=reader.total_bytes,
    )


def _save_checkpoint(model, tokenizer, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _start_rollout_servers(runtime, run_dir, model_path, spec):
    from .sync_baseline import _server_command, _wait_ready

    if len(runtime.gpu_ids) < 3:
        raise ValueError(
            "TRL synchronous comparison requires two rollout GPUs and one trainer GPU"
        )
    context_length = 32768
    ports = [30000 + (os.getpid() % 10000) * 2, 30001 + (os.getpid() % 10000) * 2]
    bases = [f"http://127.0.0.1:{port}" for port in ports]
    processes, logs = [], []
    ready = False
    try:
        for index, (gpu, port) in enumerate(
            zip(runtime.gpu_ids[:2], ports, strict=True)
        ):
            log = (Path(run_dir) / f"trl-sglang-{index}.log").open("w")
            logs.append(log)
            env = runtime.environment(run_dir)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            command = _server_command(
                model_path,
                port,
                context_length,
                spec.max_concurrent_rollouts,
                spec.seed,
            )
            command[command.index("0.65")] = "0.8"
            command.remove("--disable-cuda-graph")
            process = subprocess.Popen(
                [str(runtime.python), *command],
                cwd=runtime.areal_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
        for process, base, index in zip(
            processes, bases, range(len(bases)), strict=True
        ):
            _wait_ready(process, base, Path(run_dir) / f"trl-sglang-{index}.log")
        ready = True
        return bases, processes, logs
    finally:
        if not ready:
            from .sync_baseline import _terminate

            for process in reversed(processes):
                _terminate(process)
            for stream in logs:
                stream.close()


def _sample_one(base, tokenizer, prompt_ids, row, reward_fn, audit, step, index):
    from .sync_baseline import _gen
    import tplane

    if audit is None:
        generated = _gen(base, prompt_ids, MAX_NEW_TOKENS, tokenizer)
        reward = reward_fn(generated["text"], row["answer"])
    else:
        with audit.unit(
            uuid.uuid4().hex, kind=tplane.UnitKind.TRAJECTORY, source="trl-sync"
        ) as unit:
            with unit.stage(tplane.Stage.ROLLOUT):
                generated = _gen(base, prompt_ids, MAX_NEW_TOKENS, tokenizer)
            unit.tokens(
                tokens_in=len(prompt_ids), tokens_out=len(generated["output_ids"])
            )
            with unit.stage(tplane.Stage.GRADE):
                reward = reward_fn(generated["text"], row["answer"])
            unit.reward(float(reward))
            unit.tag("optimizer_step", str(step))
            unit.tag("policy_head", str(step - 1))
            unit.tag("policy_tail", str(step - 1))
            unit.tag("rollout_index", str(index))
    if not isinstance(reward, (int, float)):
        raise ValueError("shared GSM8K verifier returned a non-numeric score")
    return generated, float(reward)


def _formatted_prompts(dataset, selected, tokenizer):
    rows = []
    for index in selected:
        row = dataset[int(index)]
        messages = row["messages"]
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        prompt_ids = [int(token) for token in prompt_ids]
        if len(prompt_ids) > MAX_PROMPT_TOKENS:
            raise ValueError(
                "shared filtered prompt exceeded the 1024 token cap after chat templating"
            )
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        rows.append(
            {
                "messages": messages,
                "answer": row["answer"],
                "prompt_ids": prompt_ids,
                "prompt_text": prompt_text,
            }
        )
    return rows


def _dataset_and_manifest(tokenizer, run_dir):
    from .comparison_data import prepare_training

    dataset, selected, metadata = prepare_training(tokenizer)
    if len(selected) != UPDATES * PROMPTS_PER_UPDATE:
        raise ValueError(f"expected 3840 shared prompt indices, got {len(selected)}")
    rows = _formatted_prompts(dataset, selected, tokenizer)
    atomic_json = importlib.import_module("nibble.report").atomic_json
    atomic_json(Path(run_dir) / "data-manifest.json", metadata)
    return rows, metadata


def _reward_value(reward, *, scale=10.0, bias=-0.5):
    value = (float(reward) + bias) * scale
    if not (-float("inf") < value < float("inf")):
        raise ValueError("transformed reward is non-finite")
    return value


def _create_metric_recorder(run_dir):
    metrics_path = Path(run_dir) / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    return metrics_path


def _callback_class(run_dir, tokenizer, bases, reward_fn, audit, rows, algorithm, spec):
    from transformers import TrainerCallback
    from .sync_baseline import _update_weights

    by_text = {row["prompt_text"]: row for row in rows}
    metrics_path = _create_metric_recorder(run_dir)
    rollout_stats = {}

    def rollout_func(prompts, trainer):
        step = int(trainer.state.global_step) + 1
        started = time.perf_counter()

        def one(position_and_prompt):
            position, prompt = position_and_prompt
            row = by_text[prompt]
            base = bases[position % len(bases)]
            generated, reward = _sample_one(
                base,
                tokenizer,
                row["prompt_ids"],
                row,
                reward_fn,
                audit,
                step,
                position % COMPLETIONS_PER_PROMPT,
            )
            return {
                "prompt_ids": row["prompt_ids"],
                "completion_ids": generated["output_ids"],
                "logprobs": generated["behavior_logprobs"],
                "prompt_messages": row["messages"],
                "answers": row["answer"],
                "precomputed_reward": reward,
                "sample_text": generated["text"],
                "sample_truncated": generated["truncated"],
            }

        with ThreadPoolExecutor(max_workers=spec.max_concurrent_rollouts) as pool:
            outputs = list(pool.map(one, enumerate(prompts)))
        rollout_stats[step] = {
            "rollout_seconds": time.perf_counter() - started,
            "output_tokens": sum(len(output["completion_ids"]) for output in outputs),
            "rewards": [output["precomputed_reward"] for output in outputs],
            "truncations": sum(bool(output["sample_truncated"]) for output in outputs),
        }
        return {key: [output[key] for output in outputs] for key in outputs[0]}

    def reward_func(prompts, completions, precomputed_reward, **kwargs):
        return [_reward_value(score) for score in precomputed_reward]

    class SyncCallback(TrainerCallback):
        def __init__(self):
            self.trainer = None
            self.last_step_time = time.perf_counter()
            self.last_update_seconds = 0.0
            self.grad_norms = []
            self.pending_records = {}
            self.logs_by_step = {}
            self.written_steps = set()

        def _flush(self, step):
            if step not in self.pending_records or step not in self.logs_by_step:
                return
            record = self.pending_records.pop(step)
            logs = self.logs_by_step.pop(step)
            policy_loss = logs.get(
                "loss/policy_avg", logs.get("loss", logs.get("train_loss"))
            )
            value_loss = logs.get("loss/value_avg", 0.0)
            grad_norms = record.pop("_grad_norms")
            if policy_loss is None or not grad_norms:
                raise RuntimeError(
                    f"TRL did not report loss and gradient norm for step {step}"
                )
            record["train_loss"] = float(policy_loss) + (
                float(value_loss) if algorithm == "ppo" else 0.0
            )
            record["optimizer_hook_grad_norm"] = float(
                sum(grad_norms) / len(grad_norms)
            )
            record["grad_norm"] = float(
                logs.get("grad_norm", record["optimizer_hook_grad_norm"])
            )
            record["entropy"] = float(
                logs.get("objective/entropy", logs.get("policy/entropy", 0.0))
            )
            record["train_seconds"] = float(logs.get("train_runtime", 0.0))
            if not all(
                math.isfinite(float(record[key])) for key in ("train_loss", "grad_norm")
            ):
                raise RuntimeError(
                    f"TRL logged non-finite optimization metrics at step {step}"
                )
            _jsonl(metrics_path, record)
            self.written_steps.add(step)

        def on_train_begin(self, args, state, control, optimizer=None, **kwargs):
            import torch

            raw_optimizer = getattr(optimizer, "optimizer", optimizer)
            if raw_optimizer is None:
                raise RuntimeError("TRL did not provide its optimizer at train start")

            def clip_and_measure(opt, opt_args, opt_kwargs):
                parameters = [
                    parameter
                    for group in opt.param_groups
                    for parameter in group["params"]
                ]
                norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
                self.grad_norms.append(float(norm.detach().float().cpu()))

            raw_optimizer.register_step_pre_hook(clip_and_measure)
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = int(state.global_step)
            if step > 0 and step not in self.written_steps:
                self.logs_by_step[step] = dict(logs or {})
                self._flush(step)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            trainer = self.trainer
            model = trainer.accelerator.unwrap_model(trainer.model)
            if algorithm == "ppo":
                model = model.policy
            checkpoint = Path(run_dir) / "checkpoints" / f"step-{step:04d}"
            update_started = time.perf_counter()
            _save_checkpoint(model, tokenizer, checkpoint)
            for base in bases:
                _update_weights(base, checkpoint)
            self.last_update_seconds = time.perf_counter() - update_started
            stats = rollout_stats.pop(step, {})
            rewards = stats.get("rewards", [])
            if not rewards:
                raise RuntimeError(
                    f"TRL produced no shared-verifier rewards for step {step}"
                )
            if step in self.pending_records or step in self.written_steps:
                raise RuntimeError(f"TRL step {step} has duplicate metric callbacks")
            self.pending_records[step] = {
                "step": step,
                "rewards": rewards,
                "mean_reward": sum(rewards) / len(rewards),
                "rollout_seconds": float(stats.get("rollout_seconds", 0.0)),
                "update_seconds": self.last_update_seconds,
                "output_tokens": int(stats.get("output_tokens", 0)),
                "truncations": int(stats.get("truncations", 0)),
                "step_seconds": time.perf_counter() - self.last_step_time,
                "_grad_norms": list(self.grad_norms),
            }
            self.grad_norms.clear()
            self.last_step_time = time.perf_counter()
            self._flush(step)
            return control

        def verify_complete(self):
            if (
                self.pending_records
                or self.logs_by_step
                or self.written_steps != set(range(1, UPDATES + 1))
            ):
                raise RuntimeError(
                    f"TRL metric callbacks incomplete: written={sorted(self.written_steps)}"
                )

    callback = SyncCallback()
    return callback, rollout_func, reward_func, rollout_stats


def _launch_grpo(
    spec, runtime, run_dir, model_path, tokenizer, rows, metadata, bases, audit, grader
):
    importlib.import_module("setuptools")
    import torch
    from datasets import Dataset
    import trl.import_utils as trl_import_utils

    trl_import_utils.is_weave_available = lambda: False
    from trl.trainer import GRPOConfig, GRPOTrainer
    from trl.generation.vllm_generation import VLLMGeneration

    table = Dataset.from_list(
        [
            {
                "prompt": row["prompt_text"],
                "answer": row["answer"],
                "prompt_messages": row["messages"],
            }
            for row in rows
        ]
    )
    callback, rollout_func, reward_func, _ = _callback_class(
        run_dir,
        tokenizer,
        bases,
        grader,
        audit,
        rows,
        "grpo",
        spec,
    )
    original_init = VLLMGeneration._init_vllm
    original_sync = VLLMGeneration.sync_weights
    VLLMGeneration._init_vllm = lambda self: None
    VLLMGeneration.sync_weights = lambda self: None
    args = GRPOConfig(
        output_dir=str(Path(run_dir) / "trl-output"),
        run_name=Path(run_dir).name,
        seed=spec.seed,
        data_seed=0,
        max_steps=UPDATES,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=256,
        generation_batch_size=COMPLETIONS_PER_UPDATE,
        num_generations=COMPLETIONS_PER_PROMPT,
        num_iterations=1,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        learning_rate=spec.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.017,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        epsilon=0.4,
        beta=0.0,
        scale_rewards="group",
        loss_type="grpo",
        shuffle_dataset=False,
        use_vllm=True,
        vllm_mode="server",
        vllm_importance_sampling_correction=False,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
    )
    importlib.import_module("transformers").set_seed(spec.seed)
    model = importlib.import_module(
        "transformers"
    ).AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    trainer = GRPOTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        train_dataset=table,
        reward_funcs=[reward_func],
        rollout_func=rollout_func,
    )
    callback.trainer = trainer
    trainer.add_callback(callback)
    start = time.perf_counter()
    trainer.train()
    callback.verify_complete()
    training_seconds = time.perf_counter() - start
    VLLMGeneration._init_vllm = original_init
    VLLMGeneration.sync_weights = original_sync
    if trainer.state.global_step != UPDATES:
        raise RuntimeError(
            f"TRL GRPO completed {trainer.state.global_step}/{UPDATES} updates"
        )
    return (
        training_seconds,
        Path(run_dir) / "checkpoints" / f"step-{UPDATES:04d}",
        {
            "trainer": "Hugging Face TRL 0.28.0 GRPOTrainer",
            "objective": "official clipped GRPO, epsilon=0.4, group_size=4, official group reward std scaling, beta=0",
            "rollout_transport": "official GRPO rollout_func contract backed by two synchronous SGLang servers",
            "optimizer": "AdamW lr=spec.learning_rate, weight_decay=0.017, betas=(0.9,0.999), eps=1e-8, max_grad_norm=1.0",
            "upstream_commit": TRL_COMMIT,
            "train_started_perf": start,
            "train_ended_perf": start + training_seconds,
            "nonmatching_fork_features": [
                "AReaL's actor/critic-specific scheduling and its ratio rejection sampler are not part of GRPO"
            ],
        },
    )


def _launch_ppo(
    spec, runtime, run_dir, model_path, tokenizer, rows, metadata, bases, audit, grader
):
    importlib.import_module("setuptools")
    import torch
    from torch.utils.data import DataLoader
    from datasets import Dataset
    import trl.import_utils as trl_import_utils

    trl_import_utils.is_weave_available = lambda: False
    ppo_module = importlib.import_module("trl.experimental.ppo.ppo_trainer")
    from trl.experimental.ppo import PPOConfig, PPOTrainer
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification

    prompts_by_tokens = {tuple(row["prompt_ids"]): row for row in rows}
    pending_scores = {}
    audit_run = audit
    original_batch_generation = ppo_module.batch_generation
    original_selective = ppo_module.selective_log_softmax

    def external_generate(lm_backbone, queries, pad_token_id, generation_config):
        step = int(trainer_holder["trainer"].state.global_step) + 1
        call_started = time.perf_counter()
        context_length = queries.shape[1]
        prompts = [
            [int(x) for x in row.tolist() if int(x) != int(pad_token_id)]
            for row in queries
        ]

        def generate_one(index_and_prompt):
            index, prompt_ids = index_and_prompt
            row = prompts_by_tokens[tuple(prompt_ids)]
            generated, raw_reward = _sample_one(
                bases[index % len(bases)],
                tokenizer,
                prompt_ids,
                row,
                grader,
                audit_run,
                step,
                index % COMPLETIONS_PER_PROMPT,
            )
            return index, prompt_ids, generated, raw_reward

        with ThreadPoolExecutor(max_workers=spec.max_concurrent_rollouts) as pool:
            sampled = list(pool.map(generate_one, enumerate(prompts)))
        generated_items = [item[2] for item in sampled]
        stats = rollout_stats.setdefault(
            step, {"output_tokens": 0, "truncations": 0, "rewards": [], "calls": 0}
        )
        for _, prompt_ids, generated, raw_reward in sampled:
            completion = generated["output_ids"]
            key = (
                tuple(prompt_ids),
                tuple(token for token in completion if token != tokenizer.eos_token_id),
            )
            pending_scores[key] = _reward_value(raw_reward)
            stats["output_tokens"] += len(completion)
            stats["truncations"] += int(bool(generated["truncated"]))
            stats["rewards"].append(raw_reward)
        max_len = max((len(item["output_ids"]) for item in generated_items), default=0)
        response_ids = torch.full(
            (len(generated_items), max_len),
            int(pad_token_id),
            dtype=queries.dtype,
            device=queries.device,
        )
        behavior = torch.zeros(
            (len(generated_items), max_len, 1),
            dtype=torch.float32,
            device=queries.device,
        )
        for index, item in enumerate(generated_items):
            length = len(item["output_ids"])
            response_ids[index, :length] = torch.tensor(
                item["output_ids"], dtype=queries.dtype, device=queries.device
            )
            behavior[index, :length, 0] = torch.tensor(
                item["behavior_logprobs"], dtype=torch.float32, device=queries.device
            )
        reward_model.context_length = context_length
        full = torch.cat((queries, response_ids), dim=1)
        stats = rollout_stats[step]
        stats["calls"] += 1
        if "started" not in stats:
            stats["started"] = call_started
        stats["rollout_seconds"] = time.perf_counter() - stats["started"]
        return full, behavior

    def selective_log_softmax(logits, labels):
        if logits.shape[-1] == 1:
            return logits.squeeze(-1)
        return original_selective(logits, labels)

    def external_batch_generation(
        model,
        queries,
        local_rollout_forward_batch_size,
        pad_token_id,
        generation_config,
    ):
        return external_generate(model, queries, pad_token_id, generation_config)

    ppo_module.batch_generation = external_batch_generation
    ppo_module.selective_log_softmax = selective_log_softmax

    class RewardBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_ids = None
            self.attention_mask = None
            self.owner = None

        def forward(self, input_ids, attention_mask=None, **kwargs):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            hidden = torch.zeros(
                (*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device
            )
            return type("Output", (), {"hidden_states": (hidden,)})()

    class VerifierRewardModel(torch.nn.Module):
        base_model_prefix = "backbone"

        def __init__(self):
            super().__init__()
            self.backbone = RewardBackbone()
            self.backbone.owner = self
            self.context_length = 0

        def score(self, hidden_states):
            input_ids = self.backbone.input_ids
            attention_mask = self.backbone.attention_mask
            logits = torch.zeros(
                (*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device
            )
            for index in range(input_ids.shape[0]):
                ids = input_ids[index][attention_mask[index].bool()].tolist()
                matches = [
                    prompt
                    for prompt in prompts_by_tokens
                    if ids[: len(prompt)] == list(prompt)
                ]
                prompt = max(matches, key=len)
                response = tuple(
                    int(token)
                    for token in ids[len(prompt) :]
                    if int(token) != tokenizer.eos_token_id
                )
                score = pending_scores[(prompt, response)]
                response_tokens = int(
                    attention_mask[index, self.backbone.owner.context_length :]
                    .sum()
                    .item()
                )
                terminal = self.backbone.owner.context_length + response_tokens - 1
                logits[index, terminal, 0] = float(score)
            return logits

    prompt_rows = []
    for row in rows:
        prompt_rows.append({"input_ids": row["prompt_ids"]})
    repeated_rows = [
        dict(row) for row in prompt_rows for _ in range(COMPLETIONS_PER_PROMPT)
    ]
    train_data = Dataset.from_list(repeated_rows)
    importlib.import_module("transformers").set_seed(spec.seed)
    policy = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    reference = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    critic = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=1,
        torch_dtype=torch.float32,
        local_files_only=True,
        attn_implementation="sdpa",
    )
    args = PPOConfig(
        output_dir=str(Path(run_dir) / "trl-output"),
        run_name=Path(run_dir).name,
        seed=spec.seed,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=256,
        learning_rate=spec.learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.017,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        num_ppo_epochs=1,
        num_mini_batches=1,
        total_episodes=COMPLETIONS_PER_UPDATE * UPDATES,
        response_length=MAX_NEW_TOKENS,
        temperature=1.0,
        stop_token="eos",
        kl_coef=0.0,
        gamma=1.0,
        lam=1.0,
        cliprange=0.4,
        cliprange_value=0.5,
        vf_coef=1.0,
        whiten_rewards=False,
        num_sample_generations=0,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    trainer_holder = {}
    callback, _, _, shared_rollout_stats = _callback_class(
        run_dir, tokenizer, bases, grader, audit, rows, "ppo", spec
    )
    rollout_stats = shared_rollout_stats
    reward_model = VerifierRewardModel()
    trainer = PPOTrainer(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=reference,
        reward_model=reward_model,
        value_model=critic,
        train_dataset=train_data,
        callbacks=[callback],
    )
    trainer_holder["trainer"] = trainer
    callback.trainer = trainer
    trainer.dataloader = DataLoader(
        train_data,
        batch_size=COMPLETIONS_PER_UPDATE,
        shuffle=False,
        collate_fn=trainer.data_collator,
        drop_last=True,
    )
    start = time.perf_counter()
    trainer.train()
    callback.verify_complete()
    training_seconds = time.perf_counter() - start
    ppo_module.batch_generation = original_batch_generation
    ppo_module.selective_log_softmax = original_selective
    if trainer.state.global_step != UPDATES:
        raise RuntimeError(
            f"TRL PPO completed {trainer.state.global_step}/{UPDATES} updates"
        )
    return (
        training_seconds,
        Path(run_dir) / "checkpoints" / f"step-{UPDATES:04d}",
        {
            "trainer": "Hugging Face TRL 0.28.0 experimental PPOTrainer with trainable value model",
            "objective": "official clipped PPO; policy clip=0.4, value clip=0.5, 4 completions per each of 256 prompts/update",
            "rollout_transport": "official PPO loop with a narrow synchronous SGLang token/log-prob adapter; TRL owns GAE, clipped policy/value losses, critic and optimization",
            "optimizer": "AdamW lr=spec.learning_rate, weight_decay=0.017, betas=(0.9,0.999), eps=1e-8, global grad clip=1.0",
            "upstream_commit": TRL_COMMIT,
            "train_started_perf": start,
            "train_ended_perf": start + training_seconds,
            "reward": "shared MathAgent-compatible verifier transformed as 10*(r-0.5); KL=0",
            "nonmatching_fork_features": [
                "TRL uses a joint policy/value optimizer and vf_coef=1.0; the fork has separately scheduled actor and critic optimizers",
                "AReaL's ratio rejection sampler cap=5 is not exposed by TRL PPO",
                "TRL's critic head is a sequence-classification score head initialized from the same base snapshot",
            ],
        },
    )


def _inner(spec_data, run_dir):
    from .runtime import Runtime
    from .report import atomic_json
    from .sync_baseline import _terminate
    from transformers import AutoTokenizer

    spec = importlib.import_module("nibble.spec").TrainSpec.from_dict(spec_data)
    runtime = Runtime.load(require_token=False)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model_path = runtime.models[spec.model]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows, metadata = _dataset_and_manifest(tokenizer, run_dir)
    audit = _audit_run(run_dir, spec.audit_on, runtime)
    bases, servers, logs = _start_rollout_servers(runtime, run_dir, model_path, spec)
    grader = MathRewardPool()
    try:
        grader(r"\boxed{2}", "2")
        if spec.algorithm == "grpo":
            training_seconds, checkpoint, details = _launch_grpo(
                spec,
                runtime,
                run_dir,
                model_path,
                tokenizer,
                rows,
                metadata,
                bases,
                audit,
                grader,
            )
        elif spec.algorithm == "ppo":
            training_seconds, checkpoint, details = _launch_ppo(
                spec,
                runtime,
                run_dir,
                model_path,
                tokenizer,
                rows,
                metadata,
                bases,
                audit,
                grader,
            )
        else:
            raise ValueError("TRL adapter supports only PPO and GRPO")
    finally:
        for process in reversed(servers):
            _terminate(process)
        for stream in logs:
            stream.close()
        if audit is not None:
            audit.finish()
        grader.close()
    ended = time.perf_counter()
    adapter_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(
        run_dir / "native-result.json",
        {
            "status": "succeeded",
            "trainer": details["trainer"],
            "objective": details["objective"],
            "execution": "standalone_sync",
            "score_centering": False,
            "requested_optimizer_steps": UPDATES,
            "completed_steps": UPDATES,
            "prompts_per_update": PROMPTS_PER_UPDATE,
            "completions_per_prompt": COMPLETIONS_PER_PROMPT,
            "completions_per_update": COMPLETIONS_PER_UPDATE,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "initialization_seconds": details.pop("train_started_perf") - started,
            "training_seconds": training_seconds,
            "teardown_seconds": ended - details.pop("train_ended_perf"),
            "native_total_seconds": ended - started,
            "final_checkpoint": str(checkpoint),
            "training_dataset": "full filtered openai/gsm8k main train split; epoch-zero DistributedSampler(seed=0) first 3840 prompt indices",
            "data_manifest": str(run_dir / "data-manifest.json"),
            "data_membership_sha256": metadata["membership_sha256"],
            "internal_evaluation": "disabled; common 100-prompt evaluation is external",
            "cado_records": bool(spec.audit_on),
            "trl_version": TRL_VERSION,
            "trl_upstream_commit": TRL_COMMIT,
            "adapter_sha256": adapter_hash,
            "source_fidelity_notes": details,
            "comparison_limitations": details["nonmatching_fork_features"],
        },
    )


def run(spec, runtime, run_dir):
    """Worker entry point: hand execution to the isolated TRL interpreter."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    env_python = (
        runtime.root / "work" / "venvs" / f"trl-{TRL_VERSION}" / "bin" / "python"
    )
    if not env_python.is_file():
        raise RuntimeError(
            f"isolated TRL runtime is missing; prepare it with tools/prepare_trl_runtime.py: {env_python}"
        )
    spec_path = run_dir / "trl-spec.json"
    spec_path.write_text(json.dumps(spec.to_dict(), sort_keys=True))
    env = runtime.environment(run_dir)
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.gpu_ids[2])
    env["TRL_EXPERIMENTAL_SILENCE"] = "1"
    subprocess.run(
        [
            str(env_python),
            "-m",
            "nibble.trl_sync",
            "--inner",
            str(spec_path),
            str(run_dir),
        ],
        cwd=runtime.root,
        env=env,
        check=True,
        timeout=spec.timeout_seconds,
    )
    return json.loads((run_dir / "native-result.json").read_text())


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "--inner":
        raise ValueError("usage: python -m nibble.trl_sync --inner SPEC_JSON RUN_DIR")
    spec_data = json.loads(Path(sys.argv[2]).read_text())
    _inner(spec_data, sys.argv[3])


if __name__ == "__main__":
    main()
