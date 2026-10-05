# Four-GPU async comparison

Status: code and CPU preflight completed; GPU results remain pending until the Slurm campaign finishes.

## Sources inspected

The saved Claude session **KV-cache translation for AReaL policy updates** (session 3bb6e63f-2300-45a8-8fae-95d86b01c05b) describes two experiment families: 15 real policy updates followed by replay, and live four-GPU arms with two rollout GPUs, one trainer GPU, and one fitter GPU. We read its experiment setup and result/failure discussions, the original `transfer15.sbatch`, `arm.sbatch`, fitter, calibration split tooling, and cache bridge before implementing this campaign.

The actual private GitHub branch `ilovehhhyn/AReaL:kv-cache-translation` was read and downloaded at `040cbfd50a0f4ef58cce844132a9c87c95b7e323`. Its SGLang companion is `ilovehhhyn/sglang:kv-retain` at `bc51acaba6f7f40b9b18d5d78db79d7d93b839e3`. The adapter patch is isolated under Nibble; no source tree used by older queued jobs was changed. `patches/manifest.json` pins original and patched file hashes.

Score-centering follows [arXiv v1](https://arxiv.org/html/2609.20807v1), Sections 4.1 and Appendix A/B, and the authors' [reference loss implementation](https://github.com/martin-marek/score-centering/blob/7c56e9ee2972aa57f446cf564de1a1658d14b321/losses.py). Cado uses the installed tplane Run/Unit API with strict infrastructure failure propagation.

## Matched arms

| Arm | Trainer/orchestration | Sampling | Cache | Loss |
|---|---|---|---|---|
| standalone-sync | Explicit PyTorch/HF loop; no AReaL trainer or scheduler | Two SGLang replicas; finish batch before update | Recompute | Group-centered REINFORCE + TIS |
| areal-async | Direct AReaL FSDP trainer | Two replicas; maximum policy staleness 2 | Recompute | Same objective |
| nibble-map-sc | SDK → authenticated HTTP → AReaL | Same bounded asynchronous scheduling | Adjacent-policy fitted maps, exact fallback | Same objective plus score-centering |

All arms run sequentially in one four-GPU allocation on the same node. Two GPUs generate and one trains; the fourth is reserved for fitting in arm 3 and the common held-out evaluation. All four allocated GPUs are charged to every arm, including idle time. The allocation is capped at **60 minutes (at most 4 allocated GPU-hours)**, with 15-minute per-arm and 3-minute per-fit and evaluation limits. Failure stops remaining arms without retries. This is a bounded pilot, not the paper's full experiment campaign.

Qwen2.5-1.5B-Instruct snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; 15 updates; 8 prompts/update × 4 completions; maximum 1024 prompt and 1024 generated tokens; one seed; temperature 1, top-p 1, no top-k filtering. All arms use AdamW with fp32 master parameters, bf16 computation, LR 6e-6, betas .9/.999, epsilon 1e-8, weight decay .017, gradient clipping 1, no warmup, no KL, and one update per batch. The smaller batch reduces cost relative to the old default batch of 256 prompts.

Training prompts come from the original mapper split's `holdout` partition of GSM8K **train**, excluding calibration and dev. Evaluation uses the same first 32 official GSM8K **test** prompts with deterministic HF generation, initial and after each arm. Calibration reuses 200 recorded prefixes from the original 1.5B calibration file, ridge .01, stride 4, same-layer maps, max 2048 tokens. Calibration and training indices are disjoint.

## Score-centering contract

`src/nibble/score_centering.py` accepts any reward function's detached group-relative advantages. It changes the policy-gradient loss; it does not transform a scalar reward or approximate the sampler distribution using trainer-only probabilities.

For a sampled token y and sampler top-128 head H:

```
w(v) = min(p(v)/q(v), 2)                     # TIS
rho = max(1 - sum_H q, 1e-6) / max(1 - sum_H p, 1e-6)
alpha = rho * min(1/rho, 2)
L = -A * [stopgrad(w(y)) * log p(y)
          - sum_H stopgrad(q(v)*w(v) - alpha*p(v)) * log p(v)]
A = reward - mean_reward_for_same_prompt
```

The sampler logs actual top-128 probabilities per generated token, preserved across interruptions and policy versions. The adapter shifts them to predictor positions and preserves the feature axis during padding/packing. Training gathers p on those same IDs. Missing or misaligned distributions abort the run. Text-only v1 RemoteInfEngine, TP=SP=1 are supported; v2, multimodal, filtered sampling, and tensor-parallel head gathering are unsupported.

To reproduce the paper's common objective faithfully, this is group-relative REINFORCE with TIS, **not original clipped-PPO GRPO**. The paper also uses this shared objective rather than the full algorithms behind each correction. We retain the original experiment's AdamW settings; this is not a full reproduction of the paper's SGD/Qwen3/Countdown experiments. Legacy Nibble GRPO remains available separately and rejects score-centering.

## Mapping and measurement

Before rollout resume, arm 3 fits the real adjacent policy transition on GPU 4 and atomically publishes the mapper. The full loading/hashing/fitting/waiting cost remains in training wall time. One-hop migration is allowed; unsupported or missing maps fall back to re-prefill. No unchanged-weight replay is presented as a training result. Cado tags token policy versions, and KV event logs count actual mapped resumes and fallbacks. Zero mapped resumes means the optimization was not exercised.

The Claude session's previous 2–3.4× result was resume latency on unchanged-weight replay, not training speed. It also recorded continuation changes and correctness flips on 4B. These maps remain approximate; score-centering cannot prove cache fidelity or eliminate all distribution bias.

Each arm writes `report.json`, `report.md`, `worker.log`, Cado `audit/records`, checkpoints, and native timing. The mapped arm also writes mapper logs, fit reports, registries and `kv-events.*.jsonl`. The campaign produces `comparison.json` and `comparison.md`, preserving failures and incomplete arms.

Compare held-out reward and the common uncentered-TIS diagnostic, not the absolute centered-loss value against another loss. Report initialization, training, teardown, total time, and API submission latency separately. A one-seed 15-step pilot cannot establish convergence, statistical quality equivalence, or time to a target score. Arm 3 bundles API, mapping and centering, so its difference cannot identify the individual contribution of each. Separate ablations would require additional authorized compute.

## Corrected end-to-end validation

The first rerun after job 15011566 uses `NIBBLE_MAX_STEPS=3` to complete all three arms before spending on the 15-step comparison. The model, seed, prompts per update (8), responses per prompt (4), 1024-token generation limit, official 32-prompt held-out evaluation, Cado auditing, score-centering objective and 200-prefix mapper fit stay fixed. This run measures implementation feasibility and short-run accuracy/latency; it cannot establish convergence. The default remains 15 steps. The known-bad duplicate 15059125 is held while the corrected immutable release is prepared.
