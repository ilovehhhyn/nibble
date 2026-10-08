# Fork-faithful queued reproduction

User-selected protocol: areal-kv/jobs/arm.sbatch, 15 updates. Config is the fork's examples/math/gsm8k_ppo.yaml, byte-identical to the pinned dependency copy. Full official GSM8K train split, shuffled as in the fork, 256 prompts per update, four completions per prompt, max_new_tokens=1024. Original clipped PPO, critic, reward scaling/normalization, AdamW lr=1.7e-5, microbatch token cap=10240, NCCL weight publication, SGLang flashinfer/context_length=32768/memory fraction=.8/CUDA graphs defaults remain intact. It is not the previous group-relative REINFORCE/TIS pilot.

Three serial training arms: direct fork exact-cache (3 GPUs), authenticated HTTP/SDK fork exact-cache (3 GPUs), authenticated HTTP/SDK fork mapping (4 GPUs). Each runs 15 updates and receives a common deterministic HF native-cache evaluation on the first **100 official GSM8K test questions**. One initial checkpoint evaluation also uses 100 questions. Mapping uses the original background checkpoint-polling fitter on GPU 3, ten-second polling, mapper wait=0, migration depth=-1, original calibration-v0.jsonl, batch16/max_tokens2048/lambda.01/stride4/same_layer with no artificial limit. No initial v0-v1 mapper is fabricated. Missing/not-yet-ready artifacts use the fork's existing rebuild fallback. Fitter CPU/GPU work, readiness, overlap, shutdown tail and reserved resources are reported, never treated as zero cost.

The fitter_service.py and fit_mapper_hf.py tools are byte-identical copies of the original source. fork_arm.py calls original PPOTrainer directly with original get_custom_dataset and MathAgent; there is no trainer subclass or blocking fit override. This is an experiment preset: the API validates its actual fixed workload instead of accepting a smaller count and silently training 15 updates.

## Explicit differences from the original launcher

- Proven packed-SDPA sequence isolation repair retained consistently in all fork arms for Transformers5.3. Original mask=None dictionaries allow cross-example attention; this numerical fix changes neither PPO nor scheduling.
- Common external 100-question test replaces original internal evaluation. It measures answer correctness with the original math verifier; no cache mapping is used during evaluation.
- All 15 checkpoints are retained because the subsequent transfer replay needs every transition. The original arm launcher periodically deleted all but three; deleting here would make the requested transfer workload impossible. Checkpoint write cadence is unchanged.
- Cado observes the real Slurm cgroup and allocated physical GPU IDs. Reward and training logic are untouched. Audit reports separate trajectory records from train-step units.
- Exact arms request only three GPUs rather than reserving an idle fourth. Map reserves four. Therefore GPU-second comparisons must charge actual allocation sizes; training latency is comparable hardware-role placement, not equal total reserved capacity.

## Separate score-centering overhead

A one-GPU 15-minute capped job measures existing modular group-relative REINFORCE/TIS score-centering off/on on the same four real calibration sequences and checkpoint. One warm-up per mode and three alternating paired forward/backward timings; sampler logsoftmax/top128 timing is separate. Artificial deterministic group advantages are explicitly disclosed. No optimizer updates or inference accuracy are claimed from this benchmark. It measures HF forward/loss/backward computation only, not SGLang top-k serialization/transport or whole RL training overhead. Score-centering is not inserted into original clipped PPO or DPO under an invented formula.

## Subsequent transfer workload

After all three fork arms and the score-centering cost job, one GPU replays all **15 policy transitions** using the direct fork's preserved checkpoints. Source tools/login/login_replay.sh and server_replay.py are copied; only paths, interpreter, dependency tree and Slurm-visible GPU selection are adapted. Original replay workload: 64 training-holdout prompts per transition, alternating concurrent/sequential sends, abort_after=.5s, max1024 tokens, transfer vs re-prefill plus repeated re-prefill noise floor; original warm-ups and synthetic long-prefix sweep at 2/5/10/17 seconds retained. Transfer-holdout accuracy is distinct from the common official-test accuracy100; synthetic ignore-EOS sweeps are latency-only. Fitting readiness and all fitting phase costs remain separately reported. There is no second training job for transfer.

## Modular boundaries

Environment/reward plugins, training objectives and score correction are separate. The source-faithful PPO route executes the fork's loss unchanged; the existing GRPO/group-PG and score_centering.py adapters remain separately available. DPO requires chosen/rejected preference pairs and reference log-probabilities, not a reward-function substitution (see https://arxiv.org/html/2305.18290v3). DPO training and DPO score-centering are not advertised as implemented. The reproduction prioritizes verified original behavior rather than a fake universal objective switch.

CPU preflight: full existing suite, original-default config comparisons, packed logits/gradient equivalence, exact fitter copies, source hashes, patch verification, CLI compilation and shell syntax. Queue IDs and frozen revision are appended after submission. GPU results are pending and no speed/accuracy improvement is claimed.
