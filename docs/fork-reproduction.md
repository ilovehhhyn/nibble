# Eight-arm source-faithful comparison

This protocol replaces the unstarted five-job chain documented in [the archived plan](fork-reproduction-825fd016.md). Four PPO runs and four GRPO runs share the same model, filtered GSM8K training membership and epoch-zero prompt order, original mathematical verifier, 15 updates, 256 prompts/update, four completions/prompt, max completion1024 and total sequence cap2048. Model seed is1; the original DistributedSampler uses shuffle seed0.

| Arm in each algorithm | Trainer | KV | Score centering | GPUs |
|---|---|---|---|---:|
| sync | official HF TRL0.28.0 | recompute | off | 3 |
| areal | original AReaL PPOTrainer + original YAML | recompute | off | 3 |
| api-map | same fork via HTTP/SDK | original background map | off | 4 |
| api-map-sc | same fork via HTTP/SDK | original background map | on | 4 |

The fork uses PPOTrainer for both YAMLs. PPO has a critic; GRPO has original group reward mean/std normalization and no critic. Actor optimizer, batch advantage normalization, clipping, pre-update logprob recomputation, token importance rejection and NCCL publication stay original. PPO LR is1.7e-5; GRPO LR is6e-6. All fork arms use unchanged MathAgent and math_verify.verify(parse(completion),parse(answer)). Reward transform is (reward-.5)*10, not 10*reward-.5.

## Background mapping fidelity

Copied fitter_service.py and fit_mapper_hf.py are byte-identical to original tools. GPU0/1 serve rollouts, GPU2 trains, GPU3 fits alongside training. Fitter polls adjacent saved checkpoints every10sec, using original calibration, max_tokens2048, batch16, ridge.01, stride4, same_layer. No artificial prefix limit or fabricated v0-to-v1 mapper. Mapper wait=0, retain-on-abort=true, mode=map, depth=-1 match arm.sbatch. Missing artifacts use the fork's rebuild fallback. No API fitting barrier.

Training time, full worker time, fitting work/readiness and shutdown tail are separate. The mapper allocation is charged; concurrent work is not added again to training time. Reports inspect real map/rebuild events and cannot infer cache benefit from fallback-only runs.

## Score-centering contract

[Paper Appendix A.2–A.3](https://arxiv.org/html/2609.20807v1) defines weighted head/tail centering. Original rejection defaults to token action=mask, no lower ratio bound, upper5. Effective weight is f(r)=r*1[r<=5], not cap-two TIS or lower-half MIS.

Additional actor loss: +A sum_H stopgrad(q*f(p/q)-alpha*p) log(p).
rho=q_tail/p_tail; alpha=rho*f(1/rho)=1[rho>=.2]; both tail masses floored at1e-6.

Original clipped PPO/GRPO loss stays in place; critic loss and reward stay unchanged. Correction uses original response mask, including rejected sampled-token outcomes, and original pre-rejection denominator. Requires one optimizer minibatch and the same pre-update model. Current/proximal logprob gap is audited and fails above1e-5; no alternative loss is silently substituted.

Real sampler top128 probabilities pass through SGLang, ModelResponse and original MathAgent interaction tensors. Actor staging shifts heads once. Correction gathers only heads and reuses sampled logprobs for normalization, avoiding another full-vocabulary fp32 buffer. Tests enumerate weighted expectations including rejected outcomes.

Paper experiments use REINFORCE; this is a tested analytic composition with the fork's clipped objective, not a paper-validated PPO experiment. Exact importance weighting already has zero expected weighted score, so correction can be small/zero when upper-five masking rarely binds.

Within each SC allocation, three alternating paired forward/backward timings compare correction off/on on four small real prefixes. No optimizer steps or extra GPU job. Artificial group advantages and same-model HF heads are disclosed. Actual SGLang serialization/transport cost is captured by fourth-versus-third full training.

## Baselines and evaluation

TRL supplies independent synchronous optimization, without AReaL trainer/scheduler. Transport collects one whole update across two SGLang servers, waits for completion, trains and publishes weights before the next batch. Server flags match the fork. Baselines use constant LR, original reward transform, groups4, seeded initialization, FP32 trainable weights/Adam states, BF16 compute and a smaller padded microbatch to fit one trainer GPU.

TRL retains official clipped objectives. PPO's joint policy/value optimizer and GRPO's response normalization differ from AReaL; saved source-fidelity notes document them. These are independent-framework baselines, not a pure scheduling ablation. Raw losses across different definitions are not comparable.

Every final actor is evaluated on first100 official GSM8K test questions, same prompt, greedy native HF generation, max_new_tokens1024 and original verifier. Evaluation uses no KV mapping or score correction. Accuracy is correct final answers/100, separate from sampled training reward. One15-update seed does not establish convergence.

## Subsequent transfer

After eight comparisons, the previous one-GPU transfer request reuses direct PPO fork checkpoints; no extra training. Original replay covers15 transitions,64 training-holdout prompts/transition, alternating concurrent/sequential, abort.5sec, max_new1024, re-prefill noise floor, warm-ups and synthetic long-prefix sweeps. Holdout replay accuracy differs from official test100; ignore-EOS sweeps are latency-only.

Retain all15 actor checkpoints for replay; original launcher deleted older ones. Keep the proven packed SDPA sequence isolation repair consistently in fork arms for Transformers5.3. Cado observes real Slurm cgroup/allocated GPUs. These differences are explicit.

## Verification and queue

CPU checks: eight HTTP specs; fork config defaults; dataset membership/order; real clipped-loss plus correction gradient; proxy heads; packed attention; official TRL imports/configs/tiny updates/callback ordering; patch/copied-tool hashes; shell syntax. GPU performance remains unverified until execution.

Request3 GPUs without fitting,4 with fitting, matched A100-80GB PCIe hardware. Serial afterok dependencies permit one allocation at a time. Queue IDs/frozen source are recorded in work/matrix-submission.json after submission.

## Submitted matrix — October 7, 2026

Frozen training source: `64b45bc429c0aec5b536893fa116f5afee2eb328`. Account `griffith`; matched A100-80GB PCIe nodes. The eight runs are serial and currently pending. Their full durable manifest is `work/matrix-submission.json`.

| Arm | PPO job | GRPO job | GPUs |
|---|---:|---:|---:|
| sync | 15207807 | 15207811 | 3 |
| areal | 15207804 | 15207808 | 3 |
| api-map | 15207805 | 15207809 | 4 |
| api-map-sc | 15207806 | 15207810 | 4 |

Transfer-only replay job `15207812` follows all eight training jobs and reuses checkpoints from `15207804`. It adds no training run.

Superseded pending jobs 15205429–15205433 were cancelled with elapsed 00:00:00. No comparison GPUs were consumed before submission. The first new job has no scheduler start estimate yet; queue status is not a training result.

Validation: 45 CPU tests passed against the actual pinned V5 adapter; lint, eight source hashes, copied fitter equality and shell syntax passed. GPU accuracy and speed are pending execution.

## Startup repair — October 8, 2026

The first direct PPO attempt, 15207804, failed after 37 allocation seconds before any optimizer update, checkpoint or Cado trajectory. Its source attempted to iterate the original unconnected RDataset proxy while checking training membership. PPOTrainer normally connects that proxy to its DataController during initialization.

The repair compares shared rows with the original concrete data-worker loader before startup, leaving the original remote proxy, trainer and scheduler unchanged. A CPU regression uses the real cached full dataset and verifies that validation neither connects nor consumes the proxy. The failed logs/reports are preserved. Remaining jobs had not started; their requests will be replaced with the corrected immutable source. This is one failed startup retry, not a repeat of completed training.

### Corrected queued requests

Corrected frozen source: `a5c652dc8d13d08f9fdff863b10ffc04da52dd8c`. Only 15207804 had started and failed; old 15207805–15207812 were still pending and were replaced before spending GPUs. The new requests keep the same serial resource limits and all comparison settings.

| Arm | PPO job | GRPO job | GPUs |
|---|---:|---:|---:|
| sync | 15210059 | 15210063 | 3 |
| areal | 15210056 | 15210060 | 3 |
| api-map | 15210057 | 15210061 | 4 |
| api-map-sc | 15210058 | 15210062 | 4 |

Transfer-only replay `15210064` reuses direct PPO `15210056`. Six focused CPU tests passed, including real full-data membership and unconnected-proxy regression; the fix is independently reviewed. Replacement GPU execution/results are pending. Failure evidence is in `work/failure-15207804.json`; the original submission remains in `work/matrix-submission-before-rdataset-15207804.json`.

## Sampling compiler repair — October 8, 2026

Replacement 15210056 passed the dataset preflight and initialized training workers, but the first FlashInfer sampling compile failed because Della's CUDA12.9 compiler include path omitted cuRAND headers. Both SGLang schedulers exited; the original controller continued retrying dead workers with zero accepted trajectories. The nonproductive allocation was stopped at1,687seconds (5,061 reserved GPU-seconds). No optimizer update/checkpoint was completed. Logs remain under state/comparison-15210056.

The existing PyTorch NVIDIA wheel already contains the required headers. The repair exposes that exact directory through FlashInfer's supported extra compiler flags, preserving its sampling backend and all training settings. FlashInfer/Triton/Torch/driver caches now stay under Nibble work/cache. The actual sampling header's full dependency chain passed nvcc preprocessing on CPU, without GPU execution. The API runner also stops immediately when a fatal SGLang scheduler exception and child SIGQUIT appear, including across log read boundaries; ordinary rollout rejection does not trigger it.

Compiler initialization/caching costs remain distinct from steady-state trainer latency. Failed allocations are reported separately rather than hidden from total experiment costs. All remaining requests had never started and will use the corrected archive under the same resource limits.

### Compiler-corrected requests

Frozen source: `90a4e53c6a96d7774a059dc6c61985676b7535ac`. All replacement requests retain the same resources/settings; previous unstarted requests 15210057–15210064 were cancelled with zero elapsed allocation.

| Arm | PPO job | GRPO job | GPUs |
|---|---:|---:|---:|
| sync | 15210965 | 15210969 | 3 |
| areal | 15210962 | 15210966 | 3 |
| api-map | 15210963 | 15210967 | 4 |
| api-map-sc | 15210964 | 15210968 | 4 |

Transfer-only replay `15210970` follows the eight runs and uses `15210962` checkpoints. Compiler and log-monitor fixes passed focused CPU tests and independent review; actual corrected GPU training remains pending. Durable history: `work/matrix-submission-before-curand-15210056.json`, `work/failure-15210056.json`.


## PPO critic memory repair — October 8, 2026

Job 15210962 passed the previous dataset and sampling compiler repairs and generated real rollouts. Its second critic update failed in Qwen2MLP.up_proj: a 176 MiB CUDA allocation had only 173 MiB free on the 79.27 GiB trainer GPU. Actor and critic share that GPU; the actor process retained 40.73 GiB while the critic used 37.88 GiB. The original YAML enables actor activation checkpointing but leaves critic checkpointing disabled. This is the original fork setting, not a configuration-copy mismatch. One complete step-0 actor/critic checkpoint and 4,096 successful Cado trajectories survive. The second actor update occurred before the critic exception, so completed-step count is not a count of all actor optimizer executions. No final held-out evaluation exists.

Use the engine's existing `critic.gradient_checkpointing=true` setting for all three PPO fork arms. It enables non-reentrant activation recomputation and retains the original clipped critic loss, actor loss, optimizer, reward, packed attention, 10,240-token microbatch budget, training membership/order and three-GPU training geometry. GRPO has no critic and is unchanged. This is an explicit memory-related deviation from the original fork default: it adds backward recomputation, so resulting PPO timing includes that cost and cannot be presented as unmodified original-default timing. Background fitter lifecycle remains unchanged.

A focused CPU regression compares the actual Qwen2 token-classification critic class with checkpointing off/on: packed sequence values, original clipped critic loss, every parameter gradient, finite/nonzero gradients, and reduced saved-tensor bytes. This does not establish that the full GPU run fits. The replacement first PPO allocation must validate that assumption. No additional GPU resources or shortened workloads are used.

The failed allocation consumed 664 seconds on three GPUs (1,992 reserved GPU-seconds, 0.5533 GPU-hours); report separately from successful comparison costs. Logs, report, checkpoints and audit records remain under `state/comparison-15210962`; source archive 90a4e53c6a96d7774a059dc6c61985676b7535ac remains immutable. The campaign marker still says running after the exception, while Slurm and the actual arm report identify failure; a marker alone is not success evidence. Pending dependent requests have never run and will be replaced with a corrected immutable source.


### Critic-memory-corrected requests

Frozen source: `ac904ef02191a0d5fbd404052c5678f4daeef16a`. Failed direct PPO is retried from the same seed/base model; no completed 15-update arm is repeated. Unstarted 15210963–15210970 were cancelled and replaced under the same serial resource budgets.

| Arm | PPO job | GRPO job | GPUs |
|---|---:|---:|---:|
| sync | 15211876 | 15211880 | 3 |
| areal | 15211873 | 15211877 | 3 |
| api-map | 15211874 | 15211878 | 4 |
| api-map-sc | 15211875 | 15211879 | 4 |

Transfer-only replay `15211881` reuses direct PPO `15211873`. Three focused CPU checks and independent review passed; full GPU memory fit and final accuracy/latency remain unverified until these allocations execute. Failure/frozen manifest history is retained in `work/failure-15210962.json` and `work/matrix-submission-before-critic-memory-15210962.json`.


## Score-centering transport repair — October 8, 2026

Two PPO arms completed all 15 optimizer updates and the shared 100-question evaluation before the score-centering failure. Direct AReaL 15211873 scored 2/100 with 4,191.18 seconds training and 4,870 seconds Slurm elapsed; Nibble map-off-SC 15211874 scored 34/100 with 3,562.54 seconds training and 4,221 seconds allocation elapsed. Each has 15 finite gradient norms, actor/critic checkpoint pairs and 17,408 successful Cado trajectories. Training membership and requested order match. These are partial descriptive results: direct evaluation truncated 99/100 answers versus 17/100 in Nibble, and completion lengths differ substantially. No final eight-arm comparison or causal speedup claim exists yet.

The background mapper fitted 14 real adjacent checkpoint pairs, with 631.003877 seconds of fitting work overlapping training. No actual retained-cache map/resume events were found in this arm, so fitted artifacts and lower total latency do not demonstrate a cache benefit. The current report parser incorrectly reports fitting work as zero; actual fitting summaries must be inspected for costs. Preserve original reports and disclose this limitation. The subsequent transfer replay is still needed to test actual cache use.

PPO SC job 15211875 rejected the very first rollout at 06:45:26 EDT because the original MathAgent/OpenAI client supplies top_k=100000000 when omitted, while our score-centering guard accepted only -1. The actual model vocabulary is 151936: both values leave the full vocabulary unfiltered. Temperature, top_p, frequency penalty and minimum output length were otherwise supported. The controller retried the invalid contract and never accepted a trajectory or completed an optimizer update; later worker connection failures are secondary. Slurm ultimately records cancellation by UID123 after 6,517 allocation seconds (26,068 reserved GPU-seconds, 7.2411 GPU-hours), not a successful training run. The cancellation initiator/reason beyond recorded termination is not inferred. Original logs and partial state stay under state/comparison-15211875.

Repair the guard to accept -1 or top_k >= the registered model's verified AutoConfig vocabulary size. The SC worker reads that exact cached config and supplies its vocabulary to the guard. Request payload and original MathAgent remain unchanged; truly filtered or unverified distributions still fail. The actual FlashInfer kernel treats k>=vocabulary as unfiltered. Add immediate supervisor failure on this specific unsupported-SC-contract marker, preserving ordinary rollout rejection behavior. This first fix for the SC transport contract uses a new immutable V6 dependency copy; V5 and completed runs remain unchanged.

Fifteen focused CPU checks passed, including the real original MathAgent → ArealOpenAI omitted-top-k conversion → SGLang request builder, cached actual model vocabulary, unchanged off/on sampling payload, rejection of bounded/unverified k, aligned top128 transport and failure-marker detection across log chunks. These checks do not prove GPU training/SC gradients pass. Only the failed PPO SC request and the unstarted GRPO SC request require corrected launches; keep both completed PPO arms, other pending jobs and the transfer checkpoint reference.


### Minimal score-centering replacement chain

Only PPO SC `15226484` (failed `15211875` replacement) and GRPO SC `15226485` (unstarted `15211879` replacement) use immutable repair source `56a865ef5c7677b8b8a54a38be2db93bc2a5446b` with V6. Completed `15211873` and `15211874` remain final comparison arms. Other existing requests retain critic-repaired source `ac904ef02191a0d5fbd404052c5678f4daeef16a` with V5: PPO sync `15211876`, GRPO direct `15211877`, map `15211878`, sync `15211880`, transfer `15211881`. Original V5 files still match their frozen hashes. No completed training is rerun.

PPO sync now depends on corrected PPO SC; GRPO sync now depends on corrected GRPO SC, which follows original GRPO map. Old unstarted GRPO SC was cancelled. The new PPO SC first request was held until these dependencies and the durable mixed-source manifest were committed to disk, then released. A completed old Slurm job no longer accepted a new dependency reference; the rejected initial submission accepted no job, and completion was verified before releasing the independent replacement. At most one allocation in the live chain can run. Transfer keeps the original completed direct PPO checkpoints, with no additional training.

## October 8 PPO quality audit and future Cado boundaries

The direct PPO arm genuinely updated actor and critic 15 times, but its final 2/100 accuracy is a degraded result, not evidence of healthy training. Original-verifier CPU regrading agrees with all saved scores; 99 answers truncated. See `ppo-training-audit.md` for drift trends, source deviations and remaining uncertainty. Existing completed arms are preserved.

All future batch templates now wrap the actual allocation process in the user Cado tplane implementation, retaining full child logs and NVML/cgroup telemetry. Evaluation adds real generation/grade batch EVAL units; replay adds per-transition process records plus completeness validation. Process records are explicitly tagged `optimizer_step=false` and must not be counted as optimizer steps or summed with overlapping inner costs. Failed processes abort without retries. Hard-killed allocations can leave an open marker without a final record; Slurm remains authoritative. Existing completed evaluations have no Cado telemetry and that gap is not retroactively fabricated. Six focused CPU regressions passed; GPU execution of the additions remains to be checked.


## Score-centering telemetry repair — October 9, 2026

PPO SC job `15227532` accepted 2,276 successful Cado trajectories but failed during its first actor update before any complete optimizer update or checkpoint. The original sampling-contract repair passed. The first blocking exception was the pinned AReaL stats tracker rejecting `nibble_sc_q_head_mass` with dtype `torch.float64`; its tensor-stat interface accepts only float32. The exception came from detached telemetry, not the score-centering correction mathematics. Later worker/RPC shutdown errors are secondary. Slurm elapsed 465 seconds on four GPUs: 1,860 reserved GPU-seconds (0.516667 GPU-hours), reported as failed-experiment cost. Two cache retain events and no resume events survive; no cache benefit can be claimed. Original report/log/audits remain preserved.

This first fix for the SC telemetry-dtype issue casts detached correction/head-mass/tail-ratio/gap statistics to float32 at the tracker boundary. Returned correction, float64 head/tail arithmetic, optimizer, rewards, scheduling, mapper and pinned V6 sources are unchanged. Sixteen focused CPU checks passed, including the real pinned DistributedStatsTracker with float32/float64 advantages, identical loss/gradients against disabled telemetry, finite nonzero gradients and masked-row isolation. Lint passed. GPU execution remains unverified. This issue is distinct from the earlier top-k sampling-contract repair. Only failed PPO SC and the unstarted GRPO SC request share this evidenced cause; completed arms must stay preserved. Evidence: `work/failure-15227532.json`.

### First SC actor-backward memory repair (15256914)

PPO SC job 15256914 passed the telemetry repair and completed one actor/critic step and checkpoint pair, then failed during the second actor backward with a 2.90 GiB allocation request and 1.72 GiB free on the 79.27 GiB trainer GPU. Actor and critic processes used 43.20 and 33.86 GiB. Its 4096 successful Cado trajectories and zero failed trajectories, partial checkpoints and logs are preserved; allocation cost was 655 seconds, 2620 reserved GPU-seconds (0.72778 GPU-hours). This is the first SC actor-backward memory fix; the earlier critic-memory failure and activation-checkpointing deviation remain in the history.

The correction now gathers the sampled label and top128 heads together, then splits the 129 gathered columns. This preserves the sampled-logprob engine branch, differentiable normalization, upper-five masking, correction precision, objective, 10240-token microbatch and resource budget. A single gather replaces two dense backward scatter buffers. CPU profiling shows three to two total gather scatters including the unchanged engine logprob branch, removing one full vocabulary buffer; the production bf16 buffer geometry is 10240 by 151936, or 2.898 GiB. This is an operation-level memory reduction, not a measured whole-training peak or guaranteed GPU fit. Real tiny Qwen2 packed sequence-isolated SDPA actor tests compare original clipped loss plus the original two-gather correction against the repair, including overlapping sampled/head IDs and masked rows: identical forward losses and bit-identical all-parameter gradients in the tested CPU float32 and bf16 fixtures (measured maximum absolute and norm-relative drift both zero; GPU accumulation order is not tested). No optimizer step or GPU allocation was added for these CPU checks.
