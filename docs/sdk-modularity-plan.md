# Nibble SDK modularity and terminal UX plan

Status: design proposal, updated 2026-10-09. No runtime implementation or benchmark change in this document. This document does not change comparison jobs, dependencies, resources or immutable source archives. For the current held queue, incomplete memory instrumentation and ownership transfer, read [the comprehensive agent handoff](agent-handoff.md) and the durable manifest.

## Product contract

Expose one authenticated endpoint and a typed SDK. Put reward first in examples and configuration discovery, followed by environment/data, model, objective, sampling, optimizer and execution. Both API and SDK use the same schema, validation and resolved plan. Preserve an immutable comparison profile; create a separate configurable research profile. A modular refactor must retain original default behavior. User-selected changes to temperature, rewards, losses or batching can change training quality and performance; modularity is not evidence of performance neutrality.

Users can choose existing implementations from a catalog with stable IDs, version, source path, entrypoint, content hash, parameter schema, supported objectives and resource hints. The API references registered artifacts; it does not interpret an arbitrary client-local file path as a server file. Project-owned custom files can be explicitly registered as versioned artifacts on the user's runner. Multi-tenant code upload/isolation is a later feature. Python callbacks must not be silently serialized into HTTP JSON.

## Research readout

The [GRPO blog](https://nphard.io/2025/05/27/grpo.html) makes configuration, generation, reward and loss code easy to identify and change. Borrow those component boundaries. Keep the private AReaL fork's asynchronous lifecycle instead of adopting a synchronous generate/train barrier. Its example defaults and objective conventions are not our defaults.

[AReaL's paper](https://arxiv.org/abs/2505.24298) describes decoupled rollout and training with explicit staleness control. An SDK call being asynchronous is different from asynchronous RL scheduling. Our abstraction must retain policy versions, behavior logprobs, proximal recomputation, trajectory ordering and the existing scheduler.

Read Claude conversation “Grievance status check” at https://claude.ai/chat/8de33cf8-bf43-8d56-bcb1-8a2b6930388e through the user's desktop UI. Relevant discussion distinguishes request settings, changes that cascade into graph/buffer allocation, and initialization settings. Its claims about which internal fields are freely mutable are hypotheses, with conflicting earlier/later characterizations; they are not verified backend capabilities. Do not reproduce the claim that every inference setting requires redeployment.

The [vLLM paper](https://arxiv.org/abs/2309.06180) concerns KV memory management and serving throughput; it does not establish a universal runtime configuration API. [SamplingParams](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/) provides per-request generation settings. [Weight transfer](https://docs.vllm.ai/en/stable/training/weight_transfer/) documents a synchronization lifecycle for updated model weights. Neither is an optimizer learning-rate API. Learning rate is changed by the training optimizer, with an explicit scheduler interaction. These current vLLM APIs are reference designs, not proof that our pinned SGLang/AReaL backend supports equivalent changes. No backend migration is proposed.

## Tinker fields and division of responsibility

Reviewed public documentation generated from Tinker 0.30.4; API versions may evolve.

| Surface | Public fields / methods |
| --- | --- |
| SamplingParams | temperature, top_p, top_k, max_tokens, seed, stop |
| SamplingClient.sample | prompt, num_samples, sampling_params, include_prompt_logprobs, topk_prompt_logprobs, topk_sample_logprobs, target_prompt_logprobs |
| Training | forward / forward_backward: data, loss_fn, loss_fn_config; forward_backward_custom supplies a callable over returned logprobs |
| AdamParams | learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm; passed with optimizer steps |
| Model creation | base_model, LoRA rank, seed, train_mlp, train_attn, train_unembed, user_metadata, optimizer |

Sources: [sampling parameters](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/samplingparams/), [sampling client](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/), [training client](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/trainingclient/), [Adam parameters](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/types/adamparams/), [service client](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/serviceclient/).

Tinker's [losses](https://tinker-docs.thinkingmachines.ai/tinker/losses/) include cross_entropy, importance_sampling, ppo, cispo, dro and a custom path. RL inputs include target_tokens, rollout logprobs and advantages. The user's loop/cookbook produces rewards and advantages; reward/environment are not fields of SamplingParams or AdamParams. Batch size is represented partly by the supplied data batch, and num_samples controls completions. Save/load and sampling-weight publication are separate lifecycle operations. Tinker's “ppo” name alone does not imply our fork's full actor/critic trainer semantics.

## Configuration inventory

This is a practical inventory of training dependencies, not a claim that all combinations are currently supported.

| Group | Fields to expose |
| --- | --- |
| Reward FIRST | registered reward ID/version, parameters, component weights, parser/verifier, required metadata, scalar/token scope, timeout, invalid-output policy, normalization/clipping/transform |
| Environment | ID/version, reset/step protocol, tools, dependencies, turn limit, termination, observation/action format, sandbox resources, concurrency, timeouts |
| Dataset | artifact/revision, splits, prompt/answer/metadata columns, filtering, sampling/shuffling, epoch count, sampler seed, curriculum/mixing, holdout isolation |
| Model/input | model/tokenizer revisions, chat template, system prompt, prompt truncation, context limit, trainable parameters, full tuning/LoRA, actor/critic/reference setup |
| Sampling | temperature, top_p/top_k/min_p where supported, completions per prompt, max/min output tokens, EOS/stop rules, repetition penalties, seed, constrained outputs, required logprob/head transport |
| Batch geometry | distinct prompts per optimizer update, completions per prompt, microbatch token budget, gradient accumulation, maximum concurrent rollouts, prefetch/queue capacity |
| Objective | registered loss ID/version, clip bounds, KL/reference policy, entropy term, advantage estimator/group normalization, discount/GAE for applicable tasks, loss reduction/masks, rejection/importance weighting, SC parameters |
| Optimizer | type, learning rate, schedule/warmup, betas/epsilon, weight decay, gradient clipping, optimizer state precision, optimizer epochs/minibatches |
| Async execution | max staleness, policy-publication cadence, behavior/proximal/current logprob semantics, overflow/backpressure, abort/retain/resume policy, incomplete-group handling |
| Cache/fitting | map/recompute mode, mapper artifact/version, calibration recipe, fitting cadence/placement, readiness/fallback policy, actual-use telemetry |
| Numerics/training memory | compute/parameter precision, attention backend, packing and sequence isolation, activation checkpointing, compilation, deterministic settings |
| Distributed/resources | training/rollout/fitter/evaluator placement, GPU model/count/memory, CPU/RAM caps, data/tensor/pipeline parallel strategy, sharding/offload, interconnect, scheduler account/QOS/time limit |
| Evaluation/checkpoints | fixed test split and generation config, metric/reward distinction, evaluation cadence, save/retention policy, optimizer/RNG/data-cursor resume, publication/artifact outputs |
| Operations/reproducibility | audit/logging cadence, resource sampling, cancellation, bounded failure/retry policy, walltime/token/GPU-hour budgets, source/dependency hashes, config revisions, event retention |

Do not overload “batch_size” or “rollout_size.” Prefer prompts_per_update, completions_per_prompt, microbatch_tokens, gradient_accumulation and max_concurrent_rollouts. Derived completion count = prompts_per_update × completions_per_prompt. Show both prompt and completion counts. Validate normalization, group completeness and divisibility against actual trainer geometry.

## Reward module comes first

Proposed reward input: immutable prompt/trajectory, completion tokens/text, group ID, environment outcome, dataset metadata, behavior policy version and sampling-config version. Return a finite scalar score plus named components, metrics and an explicit error/timeout outcome. Token rewards require a distinct declared contract. Declare expected ranges and higher-is-better convention, and validate with real example records before a run.

Keep raw task reward separate from transformed optimization reward and held-out evaluation. The original GSM8K preset calls the original MathAgent parser/verifier and preserves (r-.5)*10. Reward definition, transform and advantage normalization are separate modules. Changing reward must not silently replace a verifier, change denominators, or enable a different loss. Record plugin hash and resolved parameters on every trajectory.

Compatibility checks must cover PPO critic requirements, GRPO group normalization/zero-variance groups, shaped multi-turn rewards, timeout behavior and downstream masks. Regrading existing samples can be offered as an explicit offline operation, never as a silent consequence of changing a reward in flight.

## When changes take effect

Proposed capabilities metadata for every field: accepted range, default, required inputs, supported backend/profile, mutability and effects on queued work.

| Scope | Examples | Proposed application rule |
| --- | --- | --- |
| Next rollout group | supported sampling settings | Pin revision when group is dispatched; all group members share it. Existing requests finish using their original settings. |
| Next optimizer update | learning rate, selected scalar optimizer settings | Apply atomically after the current update; preserve optimizer state; specify override vs schedule change. Acknowledge the effective step. |
| Controlled phase transition | reward, loss, group size, dataset mix, environment semantics | Finish or explicitly discard old work, checkpoint where needed, validate new contracts and start a versioned phase. Do not silently mix phases in an update. |
| New allocation/restart | model topology, GPU count, sharding, memory/graph geometry | Plan a restart/resume or new run; report interruption and resource implications. |

These are product semantics to implement, not claims of existing hot-update support. Only advertise fields proven for the pinned backend. The current SC adapter accepts a narrow sampling contract (including temperature1 and unfiltered support); bounded top-k/top-p or arbitrary temperature must be rejected when incompatible until the math and transport are extended and verified. Returning a superficially accepted field that the fork ignores is a correctness bug.

Use config revisions with optimistic concurrency, atomic validation and idempotent update requests. Update result reports requested/applied/rejected status, effective group/update, pending old-revision work and any phase transition. A requested change must not imply immediate adoption by already running rollouts. Keep default execution barrier-free; transition mechanics run only on requested supported changes.

## Proposed SDK UX (not runnable today)

```python
job = client.train(
    reward="gsm8k",             # first visible choice
    environment="math",
    dataset="gsm8k",
    model="qwen25-1.5b",
    profile="research",
    loss="fork_grpo",
    sampling={"temperature": 1.0, "max_new_tokens": 1024},
    batch={"prompts_per_update": 256, "completions_per_prompt": 4},
    optimizer={"learning_rate": 6e-6},
)
job.watch()                     # stream queue, training, resources, failures
job.update(optimizer={"learning_rate": 3e-6}, apply_at="next_update")
```

Catalog discovery precedes this: list reward/environment/loss IDs with source links, examples and accepted parameters; selecting a reward narrows compatible choices and supplies required data fields. SDK repr/terminal output starts with reward, task and resolved plan. Advanced options remain available by namespace. Unknown fields fail with exact field path and supported alternatives.

Proposed HTTP extensions: a versioned catalog/schema, resolved planning output, config-update endpoint and cursor-based streaming events. Keep existing routes compatible. Do not require users to manually drive backward or weight publication for the common managed-run API; offer lower-level primitives only after defining ownership of optimizer state, policy publication and queues.

## Continuous terminal and planning UX

Preflight returns requested vs resolved resources, per-role placement, estimates/ranges and evidence confidence. GPU need depends on model/critic/reference states, optimizer, precision, sequence lengths, packing, microbatches, activation checkpointing, KV concurrency, environment processes and fitter/evaluation placement. Environment size alone cannot determine memory needs. Use analytical estimates plus available prior telemetry; label unmeasured configurations. No speculative diagnostic GPU allocations in this phase.

Stream monotonic events through resumable SSE or an equivalent cursor protocol; SDK watch renders a compact terminal view. Separate scheduler queued, allocation initializing, rollout/training, evaluation/checkpoint/teardown and final outcomes. Show scheduler reason, estimated start and age of estimate; show queue counts only where permission and scheduler data permit. Fair-share/backfill does not provide a stable FIFO queue position. Unknown ETA must be displayed as unknown.

Resource output distinguishes requested allocation, observed utilization, current and peak GPU VRAM, host RAM, CPU use and GPU-hours. Include worker liveness, last completed optimizer update, accepted/error trajectory counts, token lengths, policy lag, reward and held-out metrics separately. Resource observations can lag and cannot prove OOM absence.

OOM/failure output identifies first failing role, phase, memory request/free bytes when recorded, last completed update, preserved artifacts and recovery options. Do not silently resize, change precision/batch/objective or repeat optimization. Hard-killed runs with missing Cado completion remain incomplete. Charge whole allocation once; never sum overlapping Cado inner/outer process units as independent GPU costs.

## Backend component boundaries

Keep Dataset/Sampler, Environment/MathAgent, Reward, AdvantageEstimator, ActorObjective, CriticObjective, OptimizerPolicy, RolloutBackend, WeightPublisher, Cache/Fitter, Scheduler and EventSink as explicit interfaces. Resolve plugins and immutable configuration outside the token/gradient hot path. Components should delegate to original fork implementations where available, with the original YAML as the default reference.

The synchronous-looking blog loop describes dependencies. In Nibble these execute through the existing scheduler: rollout workers continuously sample, reward/advantages prepare complete groups, trainers consume ready batches within staleness bounds, and publication follows the original lifecycle. Avoid a new per-batch weight-publication barrier, global synchronization or a new zero_grad/optimizer ordering. Preserve original packed-SDPA sequence isolation, clipped PPO/GRPO, critic only for PPO, rejection mask and NCCL publication. SC remains an analytic composition with documented restrictions.

## Implementation phases

The phase numbering below supersedes the earlier overlapping labels. These are proposed milestones, not already implemented capabilities.

### Phase 1: modular managed SDK and observable defaults

Define reward-first contracts/catalog schemas, explicit prompt/completion/microbatch naming, immutable comparison versus configurable research profiles, and resolved configuration. Extract adapters that delegate to the original private AReaL fork without changing its scheduler, optimizer ordering or default mathematics. Add read-only cursor-based telemetry and terminal watch output. Verify default reward outputs against saved trajectories, rendered defaults, input traces and original objectives/gradients. CPU parity does not establish GPU speed or accuracy neutrality.

### Phase 2: distributed training preferences

Define distributed training preferences using existing PyTorch/AReaL facilities: DDP versus FSDP/FSDP2, topology, wrapping, sharding precision, activation checkpointing/offload and distributed checkpoint/resume. PyTorch supplies primitives; correct placement, state ownership and a validated per-device memory plan are still required. Do not independently rewrite sharding. See [FSDP2 documentation](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html).

### Phase 3: change training configuration mid-run through the API

**Phase 3 will aim to allow training configuration to be changed mid-run through the API.** Provide a versioned config-update endpoint and typed SDK operation, with atomic validation, optimistic concurrency, idempotency and acknowledgments of the effective group, optimizer update or phase. Only advertise capabilities proven for the pinned backend; accepting a field while ignoring it is an error.

Supported sampling settings apply to newly dispatched complete rollout groups; existing requests retain their original revision. Supported learning-rate and scalar optimizer changes apply after the current optimizer update, preserving optimizer state and specifying scheduler override semantics. Reward, loss, group size, data mixture and environment semantic changes require controlled versioned phases, with old work completed or explicitly discarded and checkpoints where needed; never silently mix revisions within an update. Model topology, GPU count, sharding and allocation/graph geometry may require planned checkpoint/restart rather than a hot update. Report these consequences explicitly.

The current SC contract accepts narrow temperature-1/unfiltered sampling support; incompatible requested settings must be rejected until the mathematics and transport are extended and verified. Boundary application must retain original asynchronous scheduling during unchanged execution and add transition coordination only when required by a supported requested change. This phase is not implemented today.

### Later extensions and acceptance

Add versioned custom code packaging/environment integrations, broader backend capability support and explicit failure isolation after the above contracts are established. Acceptance requires unchanged default fork execution, correct revision attribution, incompatible-field rejection, honest resource estimates and recoverable event streams. Performance/accuracy neutrality requires measured parity evidence. This document launches no additional GPU jobs and does not modify the comparison profile.

## Current implementation gaps

Inspected src/nibble/spec.py, sdk.py, server.py, runtime.py and README.md. TrainSpec is frozen and comparison execution enforces fixed parameters. Temperature is not currently a public field. SDK provides plan/train/status/events/cancel/report/wait; events are fetched as a list, not a stream. Runtime holds named trusted registries but no full plugin schema/source catalog. /v1/capabilities exposes only limited lists and should eventually be generated from the same registry/schema to avoid omissions (e.g. fork_grpo). Public Slurm execution requires an existing allocation; it does not submit arbitrary jobs. There is no hot-config endpoint, general optimizer resume or managed provisioning. The example above describes future UX, not functionality already available.
