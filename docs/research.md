# Sources read before implementation

AReaL: Helen's KV-cache fork at local revision 040cbfd5 and Della AReaL-arm,
including RLVRWorkflow, PPOTrainer (rl_trainer.py), local scheduler, config,
KV mapper application/bridge and retained-prefix design. SGLang companion branch:
kv-retain, local revision bc51aca. Della source trees are synchronized working
copies, not Git repositories; work/source-sha256.json records exact consumed files.

Cado: tplane Run, Unit, Policy, FileRecordStore, WatchOptions and outcome schema.
The adapter records successful and failed trajectories inline. All failures abort;
there is no infrastructure-failure-to-zero conversion. It does not infer causal
interpretability, and does not enable a statistically meaningful live change monitor
from one or two steps. Costs on overlapping trajectory records are not summed.

Score centering:
https://arxiv.org/html/2609.20807 (sections 4.1, 5 and Appendix A)
https://github.com/martin-marek/score-centering
Reviewed losses.py at 7c56e9ee2972aa57f446cf564de1a1658d14b321.

The correction needs the sampler's distribution, approximated by top-k IDs and
log probabilities plus a trainer-modeled tail, with detached correction weights.
The existing RLVR workflow retains only selected-token logprobs. Enabling a flag
without transporting the extra information would be wrong. Therefore this MVP
rejects score_centering=True. It does not substitute reward centering for score
centering, and does not advertise that the method fixes approximate cache contents.

# Measurement scope

API asynchrony means train() returns a job handle. Actual asynchronous RL requires
bounded policy staleness and overlapping rollout/training work. login_pilot keeps
max_staleness=0, shares one physical GPU through AReaL's existing local scheduler,
and uses disk weight updates (NCCL cannot connect both roles on the same GPU).
It is a feasibility profile, not a representative distributed performance result.

The matched direct/API pair uses the same real AReaL trainer, workflow, model,
dataset, reward, audit setting, and optimizer budget. It estimates interface
overhead only. Startup and filesystem/cache order confound a single pair.
No time-to-target-quality claim is possible from the synthetic digit environment.

The local two-GPU profile exposes max_staleness and optional existing KV mapping.
Map mode never fits a mapper automatically. Missing artifacts must fall back to
re-prefill in the fork, and usage must be verified in kv event logs.
Map mode, genuine asynchronous overlap, Slurm submission, managed GPU provisioning,
optimizer resume, score centering, and multi-tenant isolation are not validated
by the login-node pilot.
