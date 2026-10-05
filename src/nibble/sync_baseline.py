"""Standalone synchronous group-policy baseline for Nibble comparisons.

This deliberately has no AReaL trainer/scheduler dependency. Rollouts are completed
against a fixed SGLang policy, then one HF AdamW update is applied and published to
both rollout servers before the next step.
"""

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid


def import_symbol(path):
    module, symbol = path.rsplit(":", 1)
    return getattr(importlib.import_module(module), symbol)


def center_group_rewards(rewards):
    """Mean-center each contiguous prompt group; never divide by group std."""
    centered = []
    for group in rewards:
        if not group:
            raise ValueError("rollout groups must be non-empty")
        mean = sum(float(x) for x in group) / len(group)
        centered.extend(float(x) - mean for x in group)
    return centered


def tis_weights(current_logprobs, behavior_logprobs, *, enabled=True, clip=2.0):
    """Detached per-token exp(current - behavior), capped above at `clip`."""
    import torch

    if current_logprobs.shape != behavior_logprobs.shape:
        raise ValueError("current and behavior token log-probabilities must align")
    if not enabled:
        return torch.ones_like(current_logprobs).detach()
    return (
        torch.exp(current_logprobs.detach() - behavior_logprobs)
        .clamp(max=clip)
        .detach()
    )


def _request(url, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw)


def _server_command(model_path, port, context_length, max_running_requests, seed):
    return [
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--skip-tokenizer-init",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "flashinfer",
        "--mem-fraction-static",
        "0.65",
        "--random-seed",
        str(seed),
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--context-length",
        str(context_length),
        "--max-running-requests",
        str(max_running_requests),
    ]


def _wait_ready(proc, base_url, log_path, deadline_seconds=300):
    until = time.monotonic() + deadline_seconds
    last_error = None
    while time.monotonic() < until:
        if proc.poll() is not None:
            tail = Path(log_path).read_text(errors="replace")[-4000:]
            raise RuntimeError(f"SGLang server exited with {proc.returncode}: {tail}")
        try:
            req = urllib.request.Request(base_url + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as response:
                if 200 <= response.status < 300:
                    return
                last_error = RuntimeError(f"health status {response.status}")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
        time.sleep(1)
    raise TimeoutError(
        f"SGLang at {base_url} was not ready in {deadline_seconds}s: {last_error}"
    )


def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _gen(base_url, input_ids, max_new_tokens, tokenizer):
    result = _request(
        base_url + "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": True,
        },
        timeout=max(60, max_new_tokens * 2),
    )
    if not isinstance(result, dict) or not isinstance(result.get("output_ids"), list):
        raise RuntimeError(f"SGLang returned a malformed generation: {result!r}")
    meta = result.get("meta_info") or {}
    raw_logprobs = meta.get("output_token_logprobs")
    if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(
        result["output_ids"]
    ):
        raise RuntimeError(
            "SGLang did not return aligned output token log-probabilities"
        )
    try:
        behavior_logprobs = [float(item[0]) for item in raw_logprobs]
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError(
            "SGLang output token log-probabilities are malformed"
        ) from exc
    output_ids = [int(x) for x in result["output_ids"]]
    return {
        "text": tokenizer.decode(output_ids, skip_special_tokens=True),
        "output_ids": output_ids,
        "behavior_logprobs": behavior_logprobs,
        "truncated": len(result["output_ids"]) >= max_new_tokens,
    }


def _prompt_ids(tokenizer, messages, max_prompt_tokens):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if len(ids) > max_prompt_tokens:
        raise ValueError(
            f"dataset prompt has {len(ids)} tokens, cap is {max_prompt_tokens}"
        )
    return [int(token) for token in ids]


def _recompute_logprobs(model, prompt_ids, output_ids, device):
    import torch

    all_ids = prompt_ids + output_ids
    if not output_ids:
        empty = torch.empty((0,), device=device, dtype=torch.float32)
        return empty, empty
    input_ids = torch.tensor([all_ids], device=device, dtype=torch.long)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    start = len(prompt_ids) - 1
    positions = torch.arange(start, start + len(output_ids), device=device)
    targets = torch.tensor(output_ids, device=device, dtype=torch.long)
    token_logps = torch.log_softmax(logits[positions].float(), dim=-1)
    selected = token_logps.gather(-1, targets[:, None]).squeeze(-1)
    entropy = -(token_logps.exp() * token_logps).sum(dim=-1)
    return selected, entropy


def _save_hf(model, tokenizer, path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def _make_cado_run(run_dir, enabled):
    if not enabled:
        return None
    import tplane

    reader = tplane.no_gpu_reader()
    return tplane.Run(
        run_id=Path(run_dir).name,
        store=tplane.FileRecordStore(Path(run_dir) / "audit"),
        options=tplane.RunOptions(
            policy=tplane.Policy(
                on_agent=tplane.Action.ABORT,
                on_grader=tplane.Action.ABORT,
                on_infra=tplane.Action.ABORT,
                on_timeout=tplane.Action.ABORT,
                max_attempts=1,
            )
        ),
        read_host=tplane.proc_status_reader(os.getpid()),
        read_gpu=reader.read,
        gpu_total_bytes=reader.total_bytes,
    )


def _rollout_one(
    base_url,
    tokenizer,
    row,
    prompt_ids,
    max_new_tokens,
    reward_fn,
    audit_run,
    step,
    rollout_index,
):
    if audit_run is None:
        generated = _gen(base_url, prompt_ids, max_new_tokens, tokenizer)
        reward = reward_fn(
            prompt=row["messages"],
            completions=generated["text"],
            prompt_ids=prompt_ids,
            completion_ids=generated["output_ids"],
            answer=row["answer"],
        )
    else:
        import tplane

        with audit_run.unit(
            uuid.uuid4().hex, kind=tplane.UnitKind.TRAJECTORY, source="sync-baseline"
        ) as unit:
            with unit.stage(tplane.Stage.ROLLOUT):
                generated = _gen(base_url, prompt_ids, max_new_tokens, tokenizer)
            unit.tokens(
                tokens_in=len(prompt_ids), tokens_out=len(generated["output_ids"])
            )
            with unit.stage(tplane.Stage.GRADE):
                reward = reward_fn(
                    prompt=row["messages"],
                    completions=generated["text"],
                    prompt_ids=prompt_ids,
                    completion_ids=generated["output_ids"],
                    answer=row["answer"],
                )
            if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
                raise RuntimeError(
                    "registered reward returned a non-finite or non-numeric value"
                )
            unit.reward(float(reward))
            unit.tag("optimizer_step", str(step))
            unit.tag("policy_head", str(step - 1))
            unit.tag("policy_tail", str(step - 1))
            unit.tag("rollout_index", str(rollout_index))
    if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise RuntimeError(
            "registered reward returned a non-finite or non-numeric value"
        )
    generated["reward"] = float(reward)
    return generated


def run(spec, runtime, run_dir):
    """Run the synchronous standalone baseline; contract matches `native.run`."""
    wall_start = time.perf_counter()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .native import import_symbol as native_import_symbol
    from .report import atomic_json

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if len(runtime.gpu_ids) < 4:
        raise ValueError(
            "sync baseline requires four visible GPUs: rollout 0/1, trainer 2, reserve 3"
        )
    if getattr(spec, "objective", "legacy_grpo") != "group_pg_tis":
        raise ValueError("sync baseline requires objective='group_pg_tis'")

    torch.manual_seed(spec.seed)
    random.seed(spec.seed)
    torch.cuda.set_device(2)
    device = torch.device("cuda:2")
    model_path = runtime.models[spec.model]
    context_length = spec.max_prompt_tokens + spec.max_new_tokens
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    data_fn = native_import_symbol(runtime.datasets[spec.dataset])
    dataset = data_fn(
        tokenizer=tokenizer,
        count=spec.prompts_per_step * (spec.max_steps + 2),
        max_prompt_tokens=spec.max_prompt_tokens,
    )
    needed = spec.prompts_per_step * spec.max_steps
    if len(dataset) < needed:
        raise ValueError("dataset is too short for requested optimizer steps")
    reward_fn = native_import_symbol(runtime.rewards[spec.reward])

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        local_files_only=True,
        device_map={"": 2},
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    for name in (
        "dropout",
        "attention_dropout",
        "hidden_dropout",
        "activation_dropout",
    ):
        if hasattr(model.config, name):
            setattr(model.config, name, 0.0)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model.gradient_checkpointing_enable()
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=spec.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.017,
    )

    ports = [30000 + (os.getpid() % 10000) * 2, 30001 + (os.getpid() % 10000) * 2]
    servers, logs = [], []
    bases = [f"http://127.0.0.1:{port}" for port in ports]
    audit_run = _make_cado_run(run_dir, spec.audit_on)
    initialized = None
    trained = None
    ended = None
    try:
        for index, (gpu, port) in enumerate(zip(runtime.gpu_ids[:2], ports)):
            log = (run_dir / f"sglang-{index}.log").open("w")
            logs.append(log)
            env = runtime.environment(run_dir)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            proc = subprocess.Popen(
                [
                    str(runtime.python),
                    *_server_command(
                        model_path,
                        port,
                        context_length,
                        spec.max_concurrent_rollouts,
                        spec.seed,
                    ),
                ],
                cwd=runtime.areal_root,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            servers.append(proc)
        for proc, base, log in zip(
            servers, bases, [run_dir / "sglang-0.log", run_dir / "sglang-1.log"]
        ):
            _wait_ready(proc, base, log)
        initialized = time.perf_counter()

        with (run_dir / "metrics.jsonl").open("w") as metrics_file:
            for step in range(spec.max_steps):
                step_start = time.perf_counter()
                rows = [
                    dataset[step * spec.prompts_per_step + i]
                    for i in range(spec.prompts_per_step)
                ]
                groups = []
                rollout_start = time.perf_counter()
                jobs = []
                with ThreadPoolExecutor(
                    max_workers=spec.max_concurrent_rollouts
                ) as pool:
                    for prompt_index, row in enumerate(rows):
                        prompt_ids = _prompt_ids(
                            tokenizer, row["messages"], spec.max_prompt_tokens
                        )
                        for rollout_index in range(spec.rollouts_per_prompt):
                            server_index = (
                                prompt_index * spec.rollouts_per_prompt + rollout_index
                            ) % 2
                            future = pool.submit(
                                _rollout_one,
                                bases[server_index],
                                tokenizer,
                                row,
                                prompt_ids,
                                spec.max_new_tokens,
                                reward_fn,
                                audit_run,
                                step + 1,
                                rollout_index,
                            )
                            jobs.append(
                                (prompt_index, rollout_index, prompt_ids, row, future)
                            )
                    generated = {}
                    for prompt_index, rollout_index, prompt_ids, row, future in jobs:
                        item = future.result()
                        item.update(
                            {
                                "prompt_ids": prompt_ids,
                                "row": row,
                                "group": prompt_index,
                            }
                        )
                        generated[(prompt_index, rollout_index)] = item
                rollout_seconds = time.perf_counter() - rollout_start

                trajectories = []
                for prompt_index, row in enumerate(rows):
                    group = []
                    for rollout_index in range(spec.rollouts_per_prompt):
                        item = generated[(prompt_index, rollout_index)]
                        group.append(item)
                        trajectories.append(item)
                    groups.append([x["reward"] for x in group])
                advantages = center_group_rewards(groups)
                total_output_tokens = sum(
                    len(item["output_ids"]) for item in trajectories
                )
                if total_output_tokens == 0:
                    raise RuntimeError(
                        "all rollouts produced empty completions; no optimizer update is possible"
                    )

                train_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                entropy_sum, loss_sum = 0.0, 0.0
                for advantage, item in zip(advantages, trajectories):
                    current, entropies = _recompute_logprobs(
                        model, item["prompt_ids"], item["output_ids"], device
                    )
                    entropy_sum += float(entropies.detach().sum().cpu())
                    behavior = torch.tensor(
                        item["behavior_logprobs"], device=device, dtype=current.dtype
                    )
                    weights = tis_weights(current, behavior, enabled=True, clip=2.0)
                    loss = (
                        -(float(advantage) * weights * current).sum()
                        / total_output_tokens
                    )
                    loss.backward()
                    loss_sum += float(loss.detach().cpu())
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_seconds = time.perf_counter() - train_start
                update_start = time.perf_counter()
                ckpt = run_dir / "checkpoints" / f"step-{step + 1:04d}"
                _save_hf(model, tokenizer, ckpt)
                for base in bases:
                    update = _request(
                        base + "/update_weights_from_disk",
                        {"model_path": str(ckpt), "load_format": "hf"},
                        timeout=600,
                    )
                    if isinstance(update, dict) and update.get("success") is False:
                        raise RuntimeError(
                            f"SGLang weight update failed at {base}: {update}"
                        )
                update_seconds = time.perf_counter() - update_start
                elapsed = time.perf_counter() - step_start
                rec = {
                    "step": step + 1,
                    "rewards": [item["reward"] for item in trajectories],
                    "mean_reward": sum(item["reward"] for item in trajectories)
                    / len(trajectories),
                    "train_loss": loss_sum,
                    "entropy": entropy_sum / total_output_tokens,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "train_seconds": train_seconds,
                    "output_tokens": total_output_tokens,
                    "truncations": sum(
                        bool(item["truncated"]) for item in trajectories
                    ),
                    "step_seconds": elapsed,
                }
                metrics_file.write(json.dumps(rec, allow_nan=False) + "\n")
                metrics_file.flush()
        trained = time.perf_counter()
    finally:
        for proc in reversed(servers):
            _terminate(proc)
        for log in logs:
            log.close()
        if audit_run is not None:
            audit_run.finish()
        ended = time.perf_counter()

    atomic_json(
        run_dir / "native-result.json",
        {
            "trainer": "standalone HF torch AdamW synchronous group-PG + TIS",
            "requested_optimizer_steps": spec.max_steps,
            "rollout_servers": 2,
            "trainer_gpu_index": 2,
            "reserved_gpu_index": 3,
            "objective": spec.objective,
            "score_centering": False,
            "initialization_seconds": initialized - wall_start,
            "training_seconds": trained - initialized,
            "teardown_seconds": ended - trained,
            "native_total_seconds": ended - wall_start,
        },
    )
