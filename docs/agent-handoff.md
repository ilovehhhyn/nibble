# Nibble: comprehensive agent handoff

Snapshot: October 9, 2026. This is an operational handoff, not a claim that the campaign is finished. Read the current durable manifest and Slurm before acting; IDs and sources in older documents are historical. The user is transferring ownership to another agent. Coordinate one writer for queue, Git and automation changes.

## Immediate state and first actions

**Two of eight training arms have completed. Six training requests and one replay request remain pending. PPO SC root 15287234 is intentionally held (`JobHeldUser`) for mandatory memory instrumentation review. Do not blindly release it.** The other six pending requests have serial dependencies. No GPU diagnostic allocation was submitted for this instrumentation work.

1. Connect through `ssh -o BatchMode=yes -o ConnectTimeout=10 della-pli`; read `work/matrix-submission.json`, this handoff, `docs/fork-reproduction.md`, `docs/ppo-training-audit.md` and `docs/sdk-modularity-plan.md` under `/scratch/gpfs/ARORA/hh9077/nibble`.
2. Read `git status --short` and the actual runtime diff. The memory changes are uncommitted WIP; the pending jobs point at older immutable archives and will not execute these changes merely because the working tree was edited.
3. Independently review and verify the WIP memory accounting, Cado failure plumbing and actor OOM observer. The implementation agent reported 33 focused CPU tests, lint, shell syntax and diff checks passing. This is not parent review, GPU-fit proof or proof of live telemetry operation.
4. Before any allocation starts, finish the documented memory preflight and safely deploy the approved observation changes through immutable sources for every relevant pending training/evaluation/replay request. Preserve completed arms, scientific geometry, resources, dependencies and replay reference. If replacing pending requests, verify zero allocation, hold the chain, journal accepted IDs, atomically update manifest/dependencies before release, and prevent any duplicate allocation.
5. Monitor the final memory attempt. If the same PPO shared-trainer OOM recurs, **stop: no fourth memory fix or rerun**. Request the specific scientific/resource change needed from the user and pause the monitor.

## Project and original backend

Nibble is an experimental managed Python SDK and authenticated API for bounded RL on researchers' own GPUs through trusted runners. The user supplies dataset/environment, reward, RL technique, rollout/step limits, GPU budget and audit preference; Nibble manages rollout scheduling, batching, placement, weight publication and Cado evidence. Later hosted GPUs are a product direction, not existing provisioning.

The **base code is Helen's private AReaL KV-cache fork**, not stock AReaL and not a replacement trainer written for this API. Original checkout: `/scratch/gpfs/ARORA/hh9077/areal-kv/AReaL-arm`; companion serving fork: `/scratch/gpfs/ARORA/hh9077/areal-kv/sglang-fork-src`. Nibble code/state/results live under `/scratch/gpfs/ARORA/hh9077/nibble`. Private remote: `https://github.com/ilovehhhyn/nibble.git`.

Completed arms use immutable `work/deps/areal-040cbfd5-nibble-v5`; pending arms use V6. Preserve these copies and `work/releases/<full-sha>`. Hash/patch evidence is in `docs/source-manifest.json`, `docs/fork-source-hashes.json` and `tools/apply_areal_patch.py`. The original fitter tools are copied byte-identically. Do not silently replace the original MathAgent, objectives, optimizer, scheduler or serving fork.

The recorded runtime includes Torch 2.9.1+cu129, Transformers 5.3.0 and SGLang 0.5.10.post1. Source `work/env.sh`; the configured `NIBBLE_PYTHON` points to the original fork's `.venv/bin/python`. Runtime Python paths include Nibble `src`, the correct pinned fork copy and `/scratch/gpfs/ARORA/hh9077/cado/tplane/src`. Do not print authentication tokens or environment secrets. TRL has its own pinned runtime; do not alter shared packages.

The original product idea is faster research iteration through async RL, learned retained-KV translation across policy updates, and score centering for sampler/trainer mismatch. A 4x speedup or six-month-to-three-week development claim is an aspiration, not a measured result. The initial one-GPU interface pilot completed real work but showed no speedup (`docs/pilot-results.md`). Dedalus/Tiki routing was exploratory discussion, not an implemented integration. Async HTTP calls and asynchronous RL scheduling are different properties.

## The eight comparison tests

These are eight bounded scientific training arms, not eight CPU unit tests. Each algorithm has four arms:

| Algorithm | Arm / manifest name | Optimization / scheduling | Cache | SC | GPUs |
|---|---|---|---|---|---:|
| PPO | direct / `areal` | original fork PPOTrainer, asynchronous | recompute | off | 3 |
| PPO | Nibble / `api-map` | same fork via API/SDK | original background map | off | 4 |
| PPO | Nibble / `api-map-sc` | same fork via API/SDK | original background map | on | 4 |
| PPO | official / `sync` | official TRL synchronous PPO | recompute | off | 3 |
| GRPO | direct / `areal` | original fork PPOTrainer with GRPO YAML | recompute | off | 3 |
| GRPO | Nibble / `api-map` | same fork via API/SDK | original background map | off | 4 |
| GRPO | Nibble / `api-map-sc` | same fork via API/SDK | original background map | on | 4 |
| GRPO | official / `sync` | official TRL synchronous GRPO | recompute | off | 3 |

GPU 0/1 are SGLang rollout servers; GPU 2 is trainer. Fork PPO actor and critic are separate co-resident processes on GPU 2. Map arms add background fitter GPU 3. GRPO has no critic. The replay is a subsequent one-GPU job with no training. These GPU counts preserve the current role layout; they are not proof of the mathematical minimum under a different architecture. Keep at most one allocation from this chain running.

### Common scientific acceptance

Model: Qwen2.5-1.5B-Instruct, model seed 1. Training membership is the same full filtered GSM8K dataset; order follows the original fork seed-0 epoch-zero sampler, with the first 3,840 requested prompts across 15 updates. Verify actual membership and prompt traces, not just identical-looking config.

Each arm must complete **15 optimizer updates, 256 distinct prompts/update, four completions/prompt**: 1,024 completions/update and 15,360 completions in the requested trained workload. Async prefetch can produce additional accepted trajectories; this does not establish extra optimizer steps. Max completion is 1,024; requested prompt cap is 1,024 and total sequence cap 2,048. Distinguish that workload cap from the serving context setting of 32,768.

Each final actor receives exactly the same first 100 official GSM8K test questions through common greedy native HF generation, max 1,024 new tokens, unchanged original `math_verify` parser/verifier. Accuracy is correct answers / 100, separate from sampled training reward. Evaluation uses no KV mapping or SC. One seed and 15 updates do not establish convergence.

### Fork fidelity

Retain original YAML, PPOTrainer for **both PPO and GRPO**, original MathAgent, clipped actor objective, PPO critic only, original actor optimizer, pre-update proximal logprob recomputation, token rejection `MASK` upper 5, original NCCL weight publication and reward transform **`(r - .5) * 10`**. PPO LR is 1.7e-5; GRPO LR is 6e-6 with original group mean/std normalization. Preserve the packed-SDPA sequence-isolation repair consistently.

PPO critic activation checkpointing is now explicitly true for all fork PPO arms after the memory repair; original default was false. It adds backward recomputation and is a disclosed deviation. Resulting timing is not unmodified original-default timing. GRPO remains critic-free.

### Mapping fidelity and evidence

Preserve background fourth-GPU fitting with original lifecycle: poll 10 seconds, wait 0, retain-on-abort, depth -1, original calibration, max tokens 2,048, batch 16, ridge .01, stride 4, same_layer. No artificial prefix limit, fabricated initial mapper or added training barrier. Missing artifacts may cause original rebuild fallback.

Inspect real retained/resumed requests, mapped translations, fallback rebuilds, readiness, fit work and shutdown tail. Fitting overlaps training; do not add overlapping fit durations to training latency. Charge the fourth GPU. A fallback-only run cannot establish cache benefit.

### Score-centering fidelity

SC adds the modular weighted top-128 head/tail correction to the original clipped actor objective; it does not replace that objective or change critic/reward math. Real probabilities travel through SGLang, ModelResponse and original MathAgent tensors, then shift exactly once at actor staging. Effective rejection weight is `f(r)=r*1[r<=5]`; tail masses floor at 1e-6, `rho=q_tail/p_tail`, and `alpha=1[rho>=.2]`. Use original response mask and pre-rejection denominator.

Verify real sampler transport, current/proximal logprob gap <=1e-5, finite nonzero gradients, q/p head masses, correction magnitude, tail ratio/flooring, masks and unsupported-contract failures. Exact importance weighting can already have zero expected score drift; never fabricate benefit. The paper tested REINFORCE; this PPO/GRPO clipped-objective composition is analytically/CPU tested, not a paper-validated PPO result.

Each SC allocation includes paired off/on forward-backward timings without optimizer steps, on small real prefixes with same-model heads and artificial advantages. Report those limitations and distinguish isolated correction compute from real SGLang transport and whole training/fourth-GPU cost.

### Official TRL baseline

Pinned TRL 0.28.0, commit `49ef33428c47235991acb4e185ea599b70c6dab4`, isolated under `work/venvs/trl-0.28.0`. Official PPO/GRPO own optimization, while two SGLang GPUs generate synchronously and a third trains. All samples for an update finish before training/publication and the next update.

See `docs/trl-baseline.md`: microbatch 4 and accumulation 256 preserve 1,024 completions/update; FP32 trainable parameters/Adam with BF16 compute and a BF16 reference. Official PPO has joint policy/value optimizer/GAE; GRPO response normalization differs from fork details. Fork upper-five rejection is not available in the same form. These are independent-framework baselines, **not a pure scheduling ablation**. Do not compare raw loss values across different definitions.

## Authoritative queue snapshot

Full per-row manifest sources are authoritative; the top-level `source_revision` cannot describe a mixed-source chain. Frozen roots are `work/releases/<full SHA>`. Current account `griffith`, partition `gpu`, QOS `gpu-short`, constraint `gpu80&a100&pcie` (A100 80GB PCIe). Pending jobs use V6.

| Job | Purpose | GPUs | CPU | RAM | Limit | Source | State / predecessor |
|---|---|---:|---:|---:|---|---|---|
| 15211873 | PPO direct | 3 | 16 | 200G | 6h | ac904ef | completed, preserve |
| 15211874 | PPO map SC-off | 4 | 16 | 200G | 6h | ac904ef | completed, preserve |
| 15287234 | PPO map SC-on | 4 | 16 | 128G | 3h | 17ae5e3 | pending HELD, no current dependency |
| 15227533 | PPO official TRL sync | 3 | 16 | 128G | 4h | 35b10cd | afterok 15287234 |
| 15227534 | GRPO direct | 3 | 16 | 128G | 3h | 35b10cd | afterok 15227533 |
| 15227535 | GRPO map SC-off | 4 | 16 | 128G | 3h | 35b10cd | afterok 15227534 |
| 15287235 | GRPO map SC-on | 4 | 16 | 128G | 3h | 17ae5e3 | afterok 15227535 |
| 15227537 | GRPO official TRL sync | 3 | 16 | 128G | 4h | 35b10cd | afterok 15287235 |
| 15227538 | transfer-only replay | 1 | 8 | 96G | 4h | 35b10cd | afterok 15227537; reference 15211873 |

Full revisions: completed `ac904ef02191a0d5fbd404052c5678f4daeef16a`; other pending `35b10cdb66db12344d029de12af7b645bd0e0e1f`; pending SC `17ae5e30688c3e7d683b5deea0661eef39887b7d`. The root replacement lacks a dependency on the aged completed Slurm record; it was released only after prior completion was verified and is now re-held for this audit.

The October 9 14:00 Eastern / 18:00 UTC PLI H100 fallback was conditional on the first remaining training request **never starting**. Job 15227532 actually started October 8 at 23:33:50 Eastern, so the condition is false. Do not migrate this chain under that expired conditional authorization. Preserve hardware timing comparability and current repair resources.

Never revive old failed/cancelled or replaced chains, including 15207804–12, 15210056–64, 15210962–70, failed 15211875, replaced old pending 15211876–81, 15226484/85, 15227532/36, 15256914/15 and 15258749/50. Preserve all history and use current manifest rows.

## Verified completed results and unresolved quality

Direct PPO 15211873: 15 finite actor gradients, 15 actor/critic checkpoint pairs, 17,408 successful Cado trajectories, final **2/100**, training 4,191.18 seconds, Slurm 4,870 seconds. Map SC-off PPO 15211874: same verified 15 updates / 17,408 trajectories, **34/100**, training 3,562.54 seconds, Slurm 4,221 seconds. Membership/requested order matched. Original-verifier CPU regrading agrees with every saved score in both evaluations (`work/evaluation-regrade-15211873-15211874.json`).

Direct PPO had 99/100 truncated repetitive outputs versus map 17/100. Direct sampled reward declined .66992→.44336 and behavior/current logprob gap increased .019452→.27912. Map reward declined .69531→.56738, gap .020309→.19246. Both reached staleness 2. These are real updates with degraded quality; the cause remains unresolved. Do not infer cache/scheduling caused the accuracy difference or describe training as healthy. Report arrays omit some losses/drift; inspect actual worker tables.

The map arm produced 14 real fit artifacts, cumulative 631.003877 seconds overlapping training, readiness about 55–66 seconds, but no actual map/resume event files were found. The report parser's zero fitting-work value is wrong; CPU postprocess actual fit summaries before publishing, preserving original reports. Absence of cache events may mean no interrupted retained workload, not a proved launch defect. No cache benefit claim. Completed evaluations lack Cado telemetry; disclose rather than fabricate it retroactively.

## Failure chronology and retry limits

Exact failure records are `work/failure-<JOB>.json`, with original `state/comparison-<JOB>` and `work/slurm-<JOB>.out`. Preserve partial checkpoints/audits and first exceptions; shutdown/RPC errors can be secondary.

| Job | First blocker / evidence | Allocation cost | Evidenced repair |
|---|---|---|---|
| 15207804 | preflight iterated unconnected original RDataset; 0 updates | 37 seconds | inspect concrete worker loader rows without connecting/consuming proxy; six CPU checks |
| 15210056 | FlashInfer JIT omitted `curand.h`; paired SGLang scheduler crashes, 0 accepted/updates | 1,687s ×3 =5,061 GPU-s | existing NVIDIA wheel include via FLASHINFER_EXTRA_CUDAFLAGS; caches under work/cache; fatal paired scheduler/SIGQUIT failfast |
| 15210962 | second critic OOM:176MiB requested,173MiB free,79.27GiB GPU; actor40.73/critic37.88GiB; one full step,4,096 Cado successes | 664s ×3 =1,992 GPU-s | memory fix 1: existing critic activation checkpointing |
| 15211875 | SC guard demanded top_k=-1; original converter omitted field→1e8, actual vocab151,936 makes it unfiltered;0 accepted | cancelled by123 after6,517s ×4=26,068 GPU-s; do not guess motive | cached model vocab guard accepts -1 or verified k>=vocab; V6 narrow fix |
| 15227532 | first actor stats tracker rejected detached float64 q-head-mass telemetry;2,276 successes,0 full steps |465s ×4=1,860 GPU-s|float32 detached statistics only;16 focused CPU checks; GPU boundary passed next allocation |
| 15256914 | second actor backward2.90GiB request/1.72 free;actor43.20/critic33.86;one full step pair,4,096 successes,grad.36151/proxgap0 |655s ×4=2,620 GPU-s|memory fix 2: one129-column sampled-label+top128 gather, removes one dense scatter;19 focused CPU checks |
| 15258749 | second actor backward2.90GiB request/1.34 free;actor45.51/critic31.92 on79.27;Torch37.65 allocated/5.20 reserved-unallocated;one full step pair,4,080 successes,grad.32887/proxgap0 |680s ×4=2,720 GPU-s|memory fix 3 FINAL: SC-only expandable allocator at actual matrix launch |

Job 15258749 ran October 9 01:55:31–02:06:51 Eastern. Its 1,022,807 output tokens and partial checkpoint pair are preserved. Fragmentation is a **candidate**, not proved by reserved-unallocated memory. Source `08f98f75cf29781dd6f713a868e85357c2944c8f` held the single-gather fix; `609a1431ddd25641e11cc7ba33f1f17dd0daf78b` held telemetry fix.

Final source 17ae5e3 sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` only for `api-map-sc`, before any Python process in `jobs/matrix.sbatch`. Eight startup checks and installed Torch native parser/syntax checks passed without CUDA initialization. **GPU driver support/fit remain unverified.** Journal `work/sc-allocator-submission-20261009T194038Z.json`; backup `work/matrix-submission-before-allocator-20261009T194038Z.json`.

`ppo_memory_fix_attempts_total=3` counts original critic checkpointing, SC gather reduction and allocator configuration. Do not reset this by renaming actor vs critic or fragmentation symptoms. If it fails again, the doubtful assumption is that the original 10,240-token SC actor and critic can fit together on the shared 80GB trainer GPU. Stop and ask for the specific scientific or resource change; no silent microbatch reduction, GPU increase or fourth repair allocation.

## Mandatory memory discipline and active WIP

The user explicitly requires documented per-device accounting **before every training, evaluation, replay or replacement allocation**, from actual model/precision/optimizer/token and microbatch geometry, KV cache, temporary full-vocabulary buffers and all co-resident processes. Account for optimizer-state creation after the first step and changing sequence lengths. Unmeasured activation/workspace/fragmentation costs remain unknown. CPU tests and a successful first update do not establish fit.

Probe actual device-wide NVML and process-local PyTorch allocated/reserved/peaks at relevant boundaries inside the authorized allocation; preserve pressure and exact first OOM, allocator snapshot where supported. Cado typed failures/resource peaks must connect to prompt monitoring: recorded failures are not delivered alerts. Global standing instruction saved at `/Users/helenhui/.codex/AGENTS.md` and adopted by the user.

**Uncommitted implementation at handoff:** modified `src/nibble/audit.py`, `src/nibble/areal_loss.py`, `jobs/matrix.sbatch`, `docs/fork-reproduction.md`; new `src/nibble/memory_audit.py`, `tests/test_memory_audit.py`, remote `AGENTS.md`. Do not accidentally include runtime WIP in a docs-only commit or assume frozen archives contain it. The implementation agent reported 33 CPU tests plus lint/shell/diff checks, with no freeze/submission/release. Root remains held.

Seven `work/memory-preflight-<pendingJOB>.json` budgets and `work/memory-preflight-evaluation100.json` exist, all `fit_proven=false`. Actual config/meta module counts: tied actor 1,543,714,304 parameters; critic 1,543,715,841 including classifier bias. Saved actor tensors total1,777,088,000 with embedding and LM-head entries, but serialized duplication does not prove untied resident weights; the budget records a candidate allowance and uncertainty. One bf1610,240×151,936 vocabulary buffer is2.898GiB. Account for weights/grads/Adam/master candidates, KV per-request and shared roles. Unknowns include activations/recompute/kernel workspaces, CUDA/NCCL, optimizer actual precision/residency, reference residency, KV admission/prefixes, fitter workspaces and fragmentation. TRL actual batch4/accum256 is recorded; fork10,240 terms cannot be mislabeled as measured TRL geometry. Evaluation/replay exclude optimizer/critic; replay flags possible next-checkpoint co-residency.

WIP Cado/NVML observer records `state/allocation-<JOB>/memory-live.json` (the Cado audit-directory parent) with physical GPU IDs, PID, free/used/peaks; regular file writes <=1/sec, immediate threshold crossing writes to `memory-alerts.jsonl`. A4GiB pressure threshold is informational, not a new scheduler abort or barrier. `failure-live.json` connects original typed failure records to bounded actual worker/Slurm OOM evidence, or unknown when absent. Nonzero child exit cannot write successful audit-complete. Verify exact paths in implementation before relying on them.

SC first-loss files are written under `NIBBLE_MEMORY_DIR`, currently `state/allocation-<JOB>/memory`, separately from allocation-parent Cado files. The probe records logical CUDA device/visibility/PID, Torch allocated/reserved/peaks/free and installs native Torch2.9.1 OOM observer before backward once. Observer records requested/total/free arguments, memory summary and best-effort allocator snapshot while preserving the original exception. Startup/forward failures before first SC loss are outside this Torch hook; allocation NVML/Cado cover more broadly. Snapshot may lack historical traces. Mock tests do not prove GPU callback operation. Changes are observational, **not a fourth memory fix**.

Primary references: [PyTorch2.9 CUDA memory notes](https://docs.pytorch.org/docs/2.9/notes/cuda.html), [memory snapshots](https://docs.pytorch.org/docs/2.9/torch_cuda_memory.html). Expandable segments are not increased capacity; Torch snapshots miss non-Torch allocations such as NCCL. The pinned native source was checked for OOM observer arguments and allocator unlock before callbacks.

## Monitoring, Cado and final acceptance

Automation ID `check-nibble-comparison`, original chat `01a104db-d3bc-7eb0-bb64-0ebadc1a7503`. Original20-minute schedule was changed to **two-minute checks** for prompt failure detection while unfinished. Remain quiet for ordinary queued/running/start notices and SSH gaps. Notify once for verified final results or unresolved actionable failures. The saved prompt includes the held-root review block. Coordinate ownership before changing this automation; it is still attached to the original chat.

Each scheduled check uses a bounded subagent for status/diagnosis/results, collects response and independently verifies material claims. Do not use Ballmer or leave an agent sleeping between checks. Short noninteractive SSH, total bound about25seconds; no repeated authentication retries within one scheduled check. Record gap as unknown in durable work files when accessible. Request login only when necessary action is blocked and actionable. Login to `della-pli` is sufficient for shared GPFS/Slurm checks; a separate `della` login is not inherently needed.

Read `state/comparison-<JOB>/campaign.json`, arm-result and evaluation-100 JSON, `work/slurm-<JOB>.out`; follow actual run_dir to native-result/report/metrics/worker/mapper logs and audits. Also inspect `state/allocation-<JOB>`, `evaluation-100-cado/audit` and deployed memory files. Slurm RUNNING alone is not productive progress, COMPLETED alone is not scientific acceptance, and stale campaign running markers survive failures.

Verify15 full updates, finite objectives/gradient norms, PPO actor/critic pairs, real trajectory success/error/reward/token/policy-version records, actual prompt traces/order and SC transport/prox-gap. Outer Cado records are tagged `optimizer_step=false`, `granularity=process`: never count them as optimizer updates or sum overlapping outer/inner GPU costs. Audit-open without audit-complete after hard kill is incomplete. The earlier invalid `Action.WARN` was replaced with supported `Action.ABORT` before TRL execution. Preserve original training and new evaluation audit distinctions.

Replay must validate every one of15 transition summaries with64training-holdout prompts each, alternating sequential/concurrent, warmups, abort .5seconds, max completion1024, fit/update costs, actual retained/translated cache use vs rebuilds, TTFT/continuation latency, replay correctness and re-prefill noise floor. It reuses all15 directPPO actor checkpoints from15211873 and performs no training. A zero replay exit can hide transition failures. Synthetic long-prefix ignore-EOS sweeps are latency-only and are not official test100 accuracy.

Use `tools/summarize_matrix.py` only on validated completed reports; independently inspect per-arm evidence. Final documents must include correct/100, training versus initialization/checkpoint/teardown/worker/fullSlurm timing, fourth-fitter and evaluation charges, failed-allocation costs separately, SC compute/transport overhead, fitting readiness/shutdown, token lengths/staleness, source fidelity, actual maps versus fallbacks and quality/implementation/hardware limitations. Charge whole allocation once. When all eight and replay validate, publish/push accurate documents to the private repo, notify once and pause the monitor. This milestone has not been reached.

## SSH persistence now installed

On Helen's Mac, native OpenSSH connection sharing is configured with ControlMaster auto, ControlPersist yes and existing serveralive checks. Existing authenticated gateway/Della masters were left intact. `/Users/helenhui/.ssh/della-keepalive.sh` reuses authenticated connections every5minutes via `/Users/helenhui/Library/LaunchAgents/com.hh9077.della-keepalive.plist`, label `com.hh9077.della-keepalive`. It never initiates background MFA when no master exists. Health log `/Users/helenhui/.ssh/della-connection-health.log`; error log alongside. Both connections verified healthy2026-10-09T19:51:03UTC. Backup config `/Users/helenhui/.ssh/config.before-nibble-persistence-20261009T154641`.

No extra autossh package was needed. Mac sleep/network loss/server expiry can still require authentication; this is not an MFA bypass or permanent-connectivity guarantee. [OpenSSH reference](https://man.openbsd.org/ssh_config).

## SDK design versus Tinker and phase 3

Read the full [SDK modularity and terminal UX plan](sdk-modularity-plan.md), which preserves the detailed conversation/research. It is a proposal, not deployed runtime functionality. Current SDK has plan/train/status/events/cancel/report/wait; events are fetched lists, TrainSpec is frozen, and there is no hot-config endpoint, general optimizer resume, arbitrary provisioning or multi-tenant isolation. `target=slurm` requires an existing allocation. Current trusted registries are not a complete versioned plugin catalog.

The product should expose one authenticated endpoint and typed SDK using the same validated schema/resolved plan. **Reward comes first**, followed by environment/data/model/loss/sampling/optimizer/execution. Keep immutable comparison presets separate from a configurable research profile. Catalogs use stable registered artifact IDs/version/hash/source/entrypoint/schema/compatibility/resource hints. A client's local filename is not a server artifact, and Python callbacks cannot silently become HTTP JSON. Custom code registration/packaging and isolation are later work.

Tinker comparison was based on public0.30.4 docs: SamplingParams includes temperature/top_p/top_k/max_tokens/seed/stop; sampling exposes num_samples/logprob options; training forward/backward takes data/loss_fn/config and custom loss; Adam parameters supplied per optimizer step include LR/betas/epsilon/decay/gradclip; model/LoRA/trainable-module selection is creation-time. Reward/environment logic belongs to the user's loop/cookbook, not SamplingParams or AdamParams. Tinker's `ppo` name does not imply the full fork actor/critic lifecycle, and asynchronous RPC is not proof of asynchronous RL scheduling.

Borrow clear component boundaries from the referenced GRPO blog without adopting its synchronous loop or defaults. Preserve Dataset/Sampler, Environment/MathAgent, Reward, AdvantageEstimator, Actor/CriticObjective, OptimizerPolicy, RolloutBackend, WeightPublisher, Cache/Fitter, Scheduler and EventSink interfaces delegating to original fork components. No new global publication barrier, optimizer-order change or implicit scientific setting change. Configuration modularity does not prove quality/performance neutrality.

The Claude “Grievance status check” discussion (`https://claude.ai/chat/8de33cf8-bf43-8d56-bcb1-8a2b6930388e`) distinguishes request settings, buffer/graph changes and initialization. Its conflicting mutability claims are hypotheses, not verified backend support. vLLM SamplingParams and weight transfer are reference designs, not an optimizer LR API or proof of our pinned SGLang capabilities; no backend migration was proposed.

**Phase1:** modular managed SDK, reward contract/catalog, explicit prompt/completion/microbatch names, profile split/resolved configuration, original-default adapters and read-only resumable telemetry. **Phase2:** distributed-training preferences using existing PyTorch/AReaL facilities (DDP/FSDP2/topology/sharding/checkpoint-resume), not a new sharding implementation. **Phase3:** allow supported **training configuration to change mid-run through the API** with versioned validation and atomic effective boundaries.

Phase3 semantics: supported sampling changes pin at next rollout group, same revision for all group members; LR/selected optimizer scalars apply after current update with explicit scheduler interaction and preserved optimizer state; reward/loss/group/dataset/environment semantic changes use controlled versioned phase transitions, never silently mix revisions within an update; model topology/GPU count/sharding/memory-graph geometry may need explicit checkpoint/restart plans. Use optimistic concurrency, idempotency, accepted/rejected/applied responses and effective group/step acknowledgments. Existing in-flight work retains its original revision. Only advertise fields proven for the pinned backend. Current SC supports narrow temperature1/unfiltered sampling; incompatible changes must be rejected rather than ignored.

Terminal/watch design should show requested/resolved roles, queued/init/training/eval/teardown, worker liveness, completed updates, accepted/error rollouts, raw/transformed reward versus held-out accuracy, memory current/peak and cost, with cursor-based recoverable events. Queue start ETA can be unknown and fairness/backfill is not FIFO. Failure output must identify first role/phase/OOM request/free and preserved evidence; never silently resize/retry. No promises of existing streaming/hot-update/provisioning capabilities.

## Ownership and remaining work

This handoff intentionally leaves the root held and the memory runtime changes reviewable rather than misrepresenting them as deployed. The next owner must finish memory review/deployment before release, then validate actual GPU execution under the final attempt3 limit. Documentation does not restart training. Completed arms and frozen archives remain intact. Resolve queue/automation ownership before parallel mutation; monitoring may read while one owner writes. SDK implementation is a separate future workstream and must not alter this immutable scientific comparison.
