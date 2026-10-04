# Della pilot — 2026-10-04

**The SDK/API works with the real AReaL trainer. This pilot did not show a training speedup.**

All project code and run artifacts: /scratch/gpfs/ARORA/hh9077/nibble.
Private repository: https://github.com/ilovehhhyn/nibble

## Actual runs

Qwen3-0.6B, one H100 PCIe, GRPO, two prompts with two samples each, at most
16 output tokens per sample, one optimizer step, seed 1. Actor and sampler
share a physical GPU; weight publication uses disk. No GPUs were rented.

| Measurement | Direct launch | SDK/API launch |
|---|---:|---:|
| Launch to exit, including startup | 137.86 s | 142.34 s |
| Native initialization | 79.41 s | 83.87 s |
| Training phase | 18.17 s | 19.78 s |
| Teardown | 8.10 s | 8.56 s |
| Completed optimizer steps | 1 | 1 |
| Gradient norm | [7.1353] | [7.1353] |
| Successful Cado trajectories | 4 | 4 |
| Output tokens | 64 | 64 |
| Saved native checkpoint | Yes | Yes |

The SDK returned its job ID in **17.6 ms**.
That is submission latency, not training latency or a measured bound on all API overhead.

The specs, algorithm, model, seed, reward and audit setting matched. The kernel cache
root moved under Nibble scratch for the SDK run, and startup/OS cache state differed.
A single sequential pair cannot isolate API overhead from system noise. In particular,
the slower SDK training phase should not be attributed to HTTP without further evidence.

## What was verified

The SDK submits a background job over HTTP, polls status and obtains a report.
The worker runs AReaL PPOTrainer/FSDP with the forked SGLang server, executes
generation, reward computation, a nonzero optimizer update, disk weight publication,
Cado recording and checkpoint save. GPU memory returned to 0 MiB after completion.

Six small CPU tests passed: budgets/algorithm validation, authentication and registered
plugins, durable cancellation/restart state, virtualenv interpreter preservation,
owned-child process cleanup across sessions, and the reward fixture.

Three one-step GPU runs were executed, with 12 total audited
trajectories. Their combined launch-to-exit wall time was
391.9 seconds; this includes CPU startup and teardown
and is not a measured GPU-active duration. An earlier interpreter-path failure
used no GPU.

## Why there was an initial run

The strict digit-only reward returned zero for all four truncated completions, and
the trainer reported zero gradient. The synthetic fixture was changed to reward
the first generated digit / 9, including digits inside prose. It was checked against
the saved responses on CPU, then used in both timing runs. It is intentionally an
interface fixture, not a meaningful quality benchmark or a claim of RL improvement.

## Limits

This is synchronous RL on one shared GPU, despite asynchronous API submission.
The two-GPU asynchronous profile and existing-mapper passthrough are implemented
but unvalidated here. Cache translation and score centering were not enabled.
Score centering is explicitly rejected because sampled top-k probabilities are
not yet transported into the AReaL loss.

Slurm submission, managed GPU provisioning, full optimizer resume and multi-tenant
isolation remain outside this feasibility MVP. No larger or longer GPU run was started.

## Saved evidence

- Direct job: state/direct/17f14d21ef244b819ea471770b7000dd
- SDK job: state/runs/3b56ba9c50924ab2bf9d59392441300a
- Initial zero-gradient run: state/direct/028145a972324178b06d8984129d4755
- Structured results: [pilot-results.json](pilot-results.json)
- Dependencies and source hashes: [source-manifest.json](source-manifest.json)

Each job directory contains its specification, generated config, worker log, native
results, report, audit records and checkpoint. The CPU API was stopped after the pilot;
the README contains the command to start it again.
