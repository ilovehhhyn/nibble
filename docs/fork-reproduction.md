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
