# Comparison job 15011566: blocked before training

Frozen revision: f33eb8dd95fcc9d08cf11c635eecc11c9e96b2ea. Account: griffith. Slurm FAILED, exit 1:0. Started 2026-10-05 14:13:31 Eastern; ended 14:16:44; allocation elapsed 193 seconds.

The first blocker is src/nibble/sync_baseline.py:182: _prompt_ids() iterates a mapping returned by tokenizer.apply_chat_template(), attempting int(input_ids). The fix is to extract input_ids from the mapping before converting token IDs. This diagnosis has not been patched into the frozen release or GPU-tested.

Initial official GSM8K held-out evaluation: 20/32 correct (62.5%), 81.37 seconds. Standalone worker elapsed 81.77 seconds, with zero completed training steps. Neither AReaL nor Nibble API arm launched. No training gradients, audit trajectories, training checkpoints, KV events or score-centering measurements were produced. Therefore no latency speedup, training improvement, mapper-fitting cost comparison or convergence conclusion can be drawn.

Original evidence is preserved under state/campaign-15011566 and work/slurm-15011566.out; monitor diagnosis is work/monitor-failure-15011566.json.

The duplicate job 15059125 (arora) remained pending when checked. It uses the same frozen release and is likely to encounter the same blocker. It was retained under the explicit instruction not to cancel the sole surviving run. Monitoring is paused after reporting this terminal blocker; no further jobs were submitted or training retried.
