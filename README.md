# Nibble

Experimental Python SDK and local API for real AReaL GRPO jobs on your GPU host,
with Cado trajectory records. This is a feasibility MVP.

## Capabilities

- CPU-only client: plan, submit, status, events, cancel, wait and report.
- Authenticated loopback API, durable job status and one runner.
- Real AReaL PPOTrainer/FSDP and forked SGLang with a GPU ceiling.
- Registered dataset, environment and reward entry points.
- Cado outcome records, native checkpoints, phase timings and explicit failures.

See [pilot results](docs/pilot-results.md) for executed checks and
[research notes](docs/research.md) for sources and deferred optimizations.

## Della quick start

Everything lives under /scratch/gpfs/ARORA/hh9077/nibble. Existing AReaL and Cado
installations are dependencies and have not been modified.

On Della:

    cd /scratch/gpfs/ARORA/hh9077/nibble
    source work/env.sh
    "$NIBBLE_PYTHON" -m nibble.cli serve

In another Della shell:

    cd /scratch/gpfs/ARORA/hh9077/nibble
    source work/env.sh
    "$NIBBLE_PYTHON" examples/train.py

The environment script loads the private token from .env.local (mode 0600).
Neither file is committed. Do not paste the token into a notebook or publish it.

For a laptop client, install this package and forward the endpoint:

    ssh -N -L 8765:127.0.0.1:8765 della-pli

Supply the same token securely via NIBBLE_TOKEN. The laptop needs neither CUDA
nor AReaL. SSH establishes access; it does not allocate cluster GPUs.

## SDK

    from nibble import Client

    client = Client()
    options = dict(
        model="qwen3-0.6b",
        dataset="digits",
        environment="math",
        reward="digit_preference",
        max_gpus=1,
        rollouts_per_prompt=2,
        prompts_per_step=2,
        max_steps=1,
        max_new_tokens=16,
        audit_on=True,
    )

    print(client.plan(**options))  # No GPU process launched
    job = client.train(**options)  # Returns immediately
    print(job.id)
    print(job.wait(timeout=240))
    print(job.report())

The synthetic digit task rewards the first generated digit divided by nine, even in prose. It tests the environment,
reward, optimizer and audit wiring. It is not GSM8K accuracy or useful model training.
A registered gsm8k dataset/reward pair is also available, but short completions
may all receive zero reward.

max_steps means optimizer steps. rollouts_per_prompt means GRPO group size.
max_concurrent_rollouts controls concurrency, not the total number of completions.
AReaL may generate extra prefetched rollouts; audited rollouts need not equal
only the samples consumed by optimizer steps.

The pilot caps requests at 2 steps, 2 prompts per step and 64 output tokens.
The wall-time budget includes startup. A client wait timeout does not cancel a job;
use job.cancel() explicitly.

## Bring your own environment

Edit the administrator-owned JSON identified by NIBBLE_RUNNER_CONFIG.
Register model snapshot paths, dataset factories (module:function), reward
functions and compatible AReaL workflow classes. HTTP accepts registered names,
not arbitrary import paths, shell commands, credentials or package uploads.

A dataset factory receives tokenizer, count and max_prompt_tokens, returning
a Hugging Face Dataset. The built-in workflow expects messages and reward fields.
A reward receives AReaL's prompt/completions/token IDs and task fields, returning
one finite scalar. Multi-turn environments require their own registered AReaL
workflow. This is a trusted single-user runner, not a sandbox for hostile code.

For another host, use [the runner template](examples/runner.example.json), point it
at an existing compatible AReaL Python environment, and set a random NIBBLE_TOKEN
of at least 24 characters. Install optional server dependencies:

    pip install -e '.[server]'

Cado and the AReaL/SGLang forks are separate pinned dependencies.

## Execution profiles

login_pilot shares one physical GPU between a small actor and rollout server,
uses disk weight updates and synchronous RL. The GPU process check immediately
before launch is a courtesy check; it cannot reserve a shared login GPU.

local plans two physical GPUs (one actor, one rollout), provided the runner has
two configured devices and max_gpus >= 2. Set max_staleness > 0 for bounded
asynchronous RL. This profile is implemented but not validated by the one-GPU pilot.
Slurm submission and cloud GPU provisioning are not implemented.

API background jobs and overlapping RL computation are different capabilities.

## Cache translation and score centering

The optional two-GPU map profile passes through the existing fork's retention and
mapper options. It requires a pre-fitted registry inside the project root.
Missing mappers fall back to recomputation in the fork. This MVP does not fit
mappers, validate arbitrary checkpoint-pair identities, or claim map mode is faster.
Inspect actual KV events before counting translated requests.

score_centering=True raises a validation error. The paper needs the actual
sampler's top-k distribution; the current batch transports only selected-token
logprobs. Implementing the loss alone would not be a correct integration.

## Recovery and artifacts

Jobs write spec, placement plan, AReaL config, worker log, events, report and native
artifacts beneath state/runs/JOB_ID. Direct launches use state/direct/JOB_ID.

Failure and cancellation terminate the owned worker process tree. Descendants are
tracked by process identity, including children with separate sessions.
Cleanup never kills by username or a shared process name.

A restart marks in-flight jobs interrupted and does not replay optimizer steps.
Inspect artifacts and surviving GPU processes before submitting again.
Full optimizer resume is not implemented. Registering saved weights starts a new run.

Cado records successful/failed trajectories. Any episode failure aborts; it is never
silently assigned zero reward. The short pilot does not enable live integrity
alarms or claim mechanistic interpretability.

## Minimal verification

    source work/env.sh
    "$NIBBLE_PYTHON" -m pytest -q tests

Tests cover invalid budgets, auth, unknown plugins, cancellation and restart
handling without GPU use. GPU runs are explicit:

    "$NIBBLE_PYTHON" -m nibble.cli direct work/pilot.json

Compare an identical spec through direct launch and SDK to measure interface
overhead. A meaningful async speed comparison also requires two allocated GPUs,
matched work, multiple steps and held-out quality.
