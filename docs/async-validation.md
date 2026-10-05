# Pre-submission validation

2026-10-04, on Della with CUDA_VISIBLE_DEVICES empty:

- 30 CPU tests passed, including dense gradient references, a packed-sequence loss integration check, probability transport/padding tests, API contracts and campaign configuration/budget checks.
- Ruff and Python compilation passed for Nibble source; patch/source hash verification passed for all seven modified AReaL files.
- Both native AReaL configurations parsed using the exact pinned source and cached model. Offline data loading verified 136 disjoint training prompts and availability of the official 1319-example test set.
- An actual Cado CPU smoke recorded a successful trajectory with reward 1 and a verifier failure with reward null. Infrastructure failure was not converted into reward zero.
- `sbatch --test-only` accepted the four-GPU, 60-minute request after removing an explicit partition directive as required by Della policy.

No GPU training was executed on the login node during this implementation. These checks do not establish successful multi-GPU execution, learning quality, cache fidelity or a speedup. The batch job is the first complete integration experiment, and stops at its first failing arm.

The source snapshot is frozen when submitted. The patched dependency is hash-checked before execution; original AReaL trees and older queued jobs are untouched. Third-party library deprecation warnings remain.
