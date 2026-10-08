# Nibble

Experimental Python SDK and authenticated local API for bounded RL jobs on researchers' GPUs. Training uses Helen's private AReaL KV-cache fork, with Cado trajectory records.

The initial one-GPU interface pilot completed actual updates, checkpoints and audits. It showed **no speedup**: [initial results](docs/pilot-results.md). The current eight-run PPO/GRPO comparison is described in [the fork reproduction protocol](docs/fork-reproduction.md), with [official TRL baselines](docs/trl-baseline.md). Queued work is not a measured result.

```python
from nibble import Client

job = Client().train(
    model="qwen25-1.5b", dataset="gsm8k", reward="gsm8k",
    algorithm="ppo", execution="fork_arm", objective="fork_ppo",
    target="slurm", max_gpus=4, max_steps=15,
    prompts_per_step=256, rollouts_per_prompt=4,
    max_new_tokens=1024, max_prompt_tokens=1024,
    max_concurrent_rollouts=256, max_staleness=2,
    learning_rate=1.7e-5, timeout_seconds=20000,
    cache_mode="map", fit_mappers=True, score_centering=True, audit_on=True,
)
print(job.id)
print(job.status())
# job.wait(timeout=20030)
# job.report()
# job.cancel()
```

`target="slurm"` requires the runner to be **inside an existing allocation**. The campaign batch script obtains the allocation; the ordinary HTTP endpoint does not submit arbitrary Slurm requests. Models, datasets, rewards and environments must be registered in the trusted runner configuration. This MVP accepts named plugins, not untrusted remote code uploads.

## Loss module

[`score_centering.py`](src/nibble/score_centering.py) implements the paper's modular policy-gradient correction with optional TIS/MIS. Supply sampler top-k probabilities, trainer logprobs and advantages from any reward function. It is a loss component rather than a reward decorator. CPU tests compare its gradients with dense references, including packed sequences and padding.

The original paper objective remains available as `group_pg_tis`. The comparison presets `fork_ppo` and `fork_grpo` preserve the fork's clipped loss and optionally add its matching weighted score correction. GRPO selects the original GRPO YAML and learning_rate=6e-6. Upper-five rejection masking, sampler head transport and the single-update contract are documented in the protocol. The actual probability transport and FSDP integration are a small pinned [AReaL patch](patches/areal-score-centering.patch); stock AReaL is rejected for this objective.

## Della comparison

All work stays under /scratch/gpfs/ARORA/hh9077/nibble. Four arms per algorithm: standalone official TRL sync, direct AReaL async, Nibble background KV mapping, and Nibble mapping plus score centering. Every run trains15 updates of256 prompts with4 completions, then evaluates the same100 official GSM8K test questions.

jobs/matrix.sbatch consumes frozen code plus NIBBLE_ARM and NIBBLE_ALGORITHM; submission explicitly requests3 GPUs for sync/direct and4 for mapping, with serial afterok dependencies. See work/matrix-submission.json for actual IDs and revision. The API endpoint requires an existing allocation; it does not silently provision GPUs.

The mapper service runs concurrently on its own GPU with the fork's wait=0 configuration. Cado records real outcomes and allocated resources. Reports separate training, worker, fitting and evaluation costs and record map versus rebuild events. Score-centering overhead is measured within the existing SC allocations, without extra training jobs. The subsequent transfer replay reuses saved PPO checkpoints.

Pinned AReaL patches are verified with tools/apply_areal_patch.py; tools/prepare_trl_runtime.py installs an isolated pinned TRL runtime without changing shared packages. Tiny CPU checks do not verify GPU behavior or performance.

## Interface

`POST /v1/plan` validates placement; `POST /v1/jobs` returns immediately. Status, events, cancellation and reports are under `/v1/jobs/{id}`. A SQLite queue persists status; timeouts terminate owned subprocess trees. Interrupted jobs are marked interrupted on restart, not silently replayed. Tokens come from `NIBBLE_TOKEN`; bind the server to loopback and use SSH forwarding.

The one-GPU `login_pilot` retains tiny limits and synchronous exact recomputation. The two-GPU `local` profile and four-GPU `slurm` profile separate rollout/trainer placement. Arbitrary clusters, provisioning, optimizer resume and multi-tenant isolation are outside this MVP. Cado records execution outcomes and resource observations; it does not establish causal interpretability.

## Checks

Run CPU checks using the configured AReaL environment with Nibble `src`, the patched source tree, and Cado on `PYTHONPATH`, and `CUDA_VISIBLE_DEVICES=""`:

```bash
python -m pytest tests -q
python -m ruff check src tests tools/apply_areal_patch.py
```

GPU jobs are integration experiments, not part of this test suite.
