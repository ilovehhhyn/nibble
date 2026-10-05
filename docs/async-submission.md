# Queued comparison

Submitted Slurm job **15011566**, 2026-10-04. Last verified state: **PENDING**.

The job contains all three training arms, sequentially on the same four GPUs: standalone synchronous, direct AReaL asynchronous, and Nibble with real adjacent-policy cache mapping and score-centering. Each arm requests 15 optimizer steps. Total allocation cap is 60 minutes. Della assigned the gputest partition and gpu-test QoS under its submission policy.

Frozen source revision: `f33eb8dd95fcc9d08cf11c635eecc11c9e96b2ea`.

- Batch log: `/scratch/gpfs/ARORA/hh9077/nibble/work/slurm-15011566.out`
- Comparison: `/scratch/gpfs/ARORA/hh9077/nibble/state/campaign-15011566/comparison.md`
- Machine-readable results: same directory, `comparison.json`.
- Each arm's report points to its Cado records, checkpoints, loss metrics and KV events.

There are no latency or quality measurements from this queued campaign yet. [Protocol](async-comparison.md) and [CPU validation](async-validation.md) describe what is implemented and what remains unverified. The frozen snapshot is used even if later documentation changes on main.
