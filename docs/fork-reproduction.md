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
