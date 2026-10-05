# Nibble

Experimental Python SDK and authenticated local API for bounded RL jobs on researchers' GPUs. Training uses Helen's private AReaL KV-cache fork, with Cado trajectory records.

The initial one-GPU interface pilot completed actual updates, checkpoints and audits. It showed **no speedup**: [initial results](docs/pilot-results.md). The new four-GPU comparison is described in [the experiment protocol](docs/async-comparison.md); queued work is not a measured result.

```python
from nibble import Client

job = Client().train(
    model="qwen25-1.5b", dataset="gsm8k-benchmark", reward="gsm8k-strict",
    target="slurm", max_gpus=4, max_steps=15,
    prompts_per_step=8, rollouts_per_prompt=4,
    max_new_tokens=1024, max_prompt_tokens=1024,
    max_concurrent_rollouts=64, max_staleness=2,
    timeout_seconds=900, objective="group_pg_tis",
    cache_mode="map", fit_mappers=True, score_centering=True, audit_on=True,
)
print(job.id)
print(job.status())
# job.wait(timeout=930)
# job.report()
# job.cancel()
```

`target="slurm"` requires the runner to be **inside an existing allocation**. The campaign batch script obtains the allocation; the ordinary HTTP endpoint does not submit arbitrary Slurm requests. Models, datasets, rewards and environments must be registered in the trusted runner configuration. This MVP accepts named plugins, not untrusted remote code uploads.

## Loss module

[`score_centering.py`](src/nibble/score_centering.py) implements the paper's modular policy-gradient correction with optional TIS/MIS. Supply sampler top-k probabilities, trainer logprobs and advantages from any reward function. It is a loss component rather than a reward decorator. CPU tests compare its gradients with dense references, including packed sequences and padding.

The supported centered objective is `group_pg_tis`: group-mean advantages + REINFORCE + token TIS, following the paper's common objective. It is distinct from the legacy clipped GRPO path. The actual probability transport and FSDP integration are a small pinned [AReaL patch](patches/areal-score-centering.patch); stock AReaL is rejected for this objective.

## Della campaign

All work lives under `/scratch/gpfs/ARORA/hh9077/nibble`. `jobs/compare.sbatch` runs three matched arms sequentially on four allocated GPUs: standalone synchronous HF training, direct asynchronous AReaL, then Nibble with mapping and centering. It uses the cached 1.5B model, 15 steps per arm, a 60-minute allocation cap and no automatic retries. See the protocol for exact settings, held-out evaluation, calibration, fairness limits and source revisions.

```bash
# From the configured Della checkout:
NIBBLE_PROJECT_ROOT="$PWD" sbatch jobs/compare.sbatch
# After the job writes results:
python -m nibble.campaign state/campaign-JOBID --summarize
```

The batch script depends on the scratch `work/env.sh`, cached models/data, the existing Python/CUDA runtime, and the private AReaL snapshot. Copy a clean checkout at the pinned revision, then use `python tools/apply_areal_patch.py PATH` to install the adapter; `--verify` checks the consumed file hashes. The submitted campaign uses a frozen Nibble source snapshot.

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
