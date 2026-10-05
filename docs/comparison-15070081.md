# Verified four-GPU API comparison

Nibble completed end to end through the HTTP API and SDK on four NVIDIA A100-SXM4-80GB GPUs in Della job **15070081**, griffith account, 2026-10-05 16:47:51–17:03:47 Eastern. Slurm COMPLETED, exit 0:0, elapsed 15m56s. Frozen implementation: `4dcd0d4c43dded6ee3ccfddff33689fc7c639cb2`. All three arms completed three optimizer updates, saved final checkpoints, and completed the same 32-example official GSM8K test evaluation. Initial checkpoint: 20/32 (62.5%).

## Latency and held-out accuracy

Seconds; training includes rollout generation, optimizer updates, checkpoint/weight transfers and mapper fitting where enabled. Worker time includes initialization and teardown plus Python/import/data overhead. Evaluation is separate; queue time is excluded.

| Arm | Initialization | Training | Teardown | Worker total | Evaluation | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| standalone-sync | 55.00 | 88.97 | 1.93 | 148.58 | 75.53 | 22/32 (68.750%) |
| areal-async | 69.89 | 85.99 | 10.21 | 185.17 | 78.85 | 16/32 (50.000%) |
| nibble-map-sc | 72.46 | 162.59 | 8.73 | 262.08 | 79.20 | 21/32 (65.625%) |

Direct AReaL training was 3.35% shorter than synchronous training (85.99 vs 88.97 seconds), but its whole worker was longer (185.17 vs 148.58 seconds). Nibble API training was 1.89x direct AReaL training time and 1.83x synchronous training time. The API submission itself took 0.00743 seconds; two mapper-fit barriers cost 72.08 seconds, already included in its 162.59-second training time. Do not subtract fitting to claim a cache speedup. This pilot demonstrates functioning API integration; it does **not** demonstrate a net latency improvement.

## Numerical and audit acceptance

Packed SDPA now uses an explicit per-sequence causal mask. The first-step train/sampler absolute log-probability gaps are 0.021109 nats (direct) and 0.018972 (API), versus approximately 4.49/4.98 in invalid historical run 15068076. Tiny CPU Qwen2 packed-vs-separate logits and gradients passed before submission. All three optimizer gradient norms are finite and nonzero: sync [0.529663, 0.377186, 0.521482], direct [0.43681, 0.51119, 0.69781], API [0.35397, 0.42401, 0.48357]. Native update-successful flags are [1,1,1] in both asynchronous arms.

Score-centering token fraction is [1,1,1] for API and [0,0,0] for direct AReaL. API centered losses [-0.0075683, 0.0043612, -0.013135] and its uncentered TIS diagnostics [-0.0075607, 0.0043741, -0.013073] are recorded as diagnostics; different loss definitions or different sampled trajectories cannot rank training quality. Score-centering adjusts the policy-gradient score estimator, not the scalar GSM8K correctness reward.

Cado records: sync 96 successful/0 failed, direct 123/0, API 121/0. Each arm consumes 96 training trajectories; asynchronous counts include prefetched work. Audit records retain reward, token cost and policy-version tags. Their extra output tokens must not be interpreted as consumed-training throughput. Raw records and tag counts are preserved in the campaign's verified-summary.json and each run's audit/records directory. This verifies audit logging, not live Cado change detection or model interpretability.

The API logged **23 actual mapped KV resumes**, **5 rebuild fallbacks** (missing_mapper), and two translation batches: 9 entries/2,385 tokens/93.29ms and 14 entries/3,734 tokens/90.51ms. These are actual cache-map events, not fallback-only execution; they still do not establish a net cache benefit. Two mapper artifacts v0→v1 and v1→v2 used 25 calibration pairs and 13,071 sampled rows each. Calibration fit R² is high but is not a held-out cache-quality guarantee. Whole mapper barriers cost 72.08s including process overhead. Phase timings are in registry/*/fit_report.json; extraction is already included in fit_total_including_extraction and must not be added twice.

## Workload and limits

Qwen2.5-1.5B-Instruct; seed 1; 8 prompts/update, 4 completions/prompt; 3 updates; completion cap 1,024 tokens; matched sampling temperature=1, top_p=1, top_k=-1, repetition_penalty=1. AdamW learning rate 6e-6. Objective is group-relative REINFORCE with token-level importance sampling capped at 2, without PPO clipping; the public algorithm field currently says grpo. Score-centering uses the modular top-128 approximation. Synchronous arm uses two SGLang rollout GPUs and one HF trainer GPU; asynchronous arms use two rollout GPUs and one FSDP trainer GPU. Fourth GPU is reserved for mapper fitting/evaluation. All GPU work executes inside Slurm.

Only three steps, one seed and 32 held-out examples: observed accuracy differences do not establish convergence, a reliable algorithm ranking, or improvement from score-centering alone. Nibble changes both cache behavior and score-centering, so this combined arm cannot isolate their individual effects. Shared-node arm order, differing rollout trajectories/prefetch and startup costs limit causal latency inference. The original 15-step run was shortened for conservative integration validation. No additional GPU run is scheduled: this pilot already establishes interface feasibility, while repeated mapper startup dominates the current implementation and merits CPU/code optimization before spending on a longer run.

## Evidence and history

All artifacts are under `/scratch/gpfs/ARORA/hh9077/nibble/state/campaign-15070081`: campaign.json, hardware.txt, initial-evaluation.json, three final *-evaluation.json files, verified-summary.json, each run's report.json, worker.log, native-result.json, Cado audit/records, and API registry/ and kv-events directories. Submission metadata is work/submission-15070081.json. The API run ID is 62bc03ba745b4503828aee812cbf04bf.

Historical attempts are preserved: 15011566 tokenizer return type failure; 15067116 invalid SGLang load_format; 15068076 completed but invalid packed SDPA attention. The corrected release fixes those causes and metric parsing. Eleven affected CPU tests, lint, clean-base patch application and seven pinned adapter hashes passed. No original source tree was overwritten.
