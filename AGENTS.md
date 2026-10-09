# Experiment memory and failure rules

Before every future training, evaluation, or replay GPU request, calculate and save a scoped memory budget from the actual model configuration, per-device roles, and verified token/batch geometry. Include weights, gradients/optimizer state only when trained, logits/backward buffers, KV/calibration/fitter work, co-resident processes, and explicit unknown activations/workspaces. A component estimate never proves GPU fit; unverified TRL geometry or serialized tied-head residency must remain unknown.

Probe actual allocation and process memory through Cado/NVML and PyTorch telemetry. Inspect low-headroom events and typed Cado failure records promptly; preserve first-cause evidence, partial trajectories/checkpoints and full reserved GPU costs. Do not declare success from scheduler state or a zero wrapper exit.

PPO memory repair attempt3 is the final authorized attempt, following critic checkpointing and the SC single-gather repair. Observability adds no fix attempt. If the same memory blocker recurs, stop and report the doubtful shared-GPU fit assumption; do not relabel the failure or silently submit attempt4. Preserve immutable sources, pinned dependencies, completed arms, the serial one-allocation limit and authorized resource budgets.
