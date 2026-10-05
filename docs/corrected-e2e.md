# Corrected end-to-end run

Job 15067116, griffith account, 4 GPUs, 60-minute cap. Frozen code ed5430d3ce7268c6dea10faabd6322c80572fb4b. Three matched optimizer updates per arm; all other campaign workload and evaluation settings unchanged.

The Transformers 5.3 tokenizer mapping return is normalized by explicitly requesting return_dict=False in baseline and dataset builders. The actual Qwen2.5 tokenizer and 136 benchmark prompts passed CPU preflight. 32 CPU tests, lint, compilation and pinned adapter verification passed.

Results are pending GPU execution. Job 15059125 with the faulty release is held and must not be released. Results will be saved under state/campaign-15067116.

## GPU attempt 15067116

The tokenizer fix passed real rollout generation. Job ran for 193 seconds, then the synchronous baseline failed at its first disk weight update: the SGLang scheduler rejects load_format=hf (ValueError: hf is not a valid LoadFormat). A first optimizer update and checkpoint step-0001 were produced, but the step metric was not committed because reloading failed; async/API arms were not launched. Logs remain under state/campaign-15067116.

The next release uses the actual supported auto load format. It also explicitly disables model-generation defaults for top-k and repetition penalty: top_k=-1 and repetition_penalty=1.0 in both baseline and group_pg_tis AReaL requests. Previous baseline generation inherited top_k=20 and all arms otherwise inherited repetition_penalty=1.1; that mismatch would invalidate a matched sampling comparison. The new AReaL adapter is a separate source copy under work/deps/areal-040cbfd5-nibble-v2, preserving the prior source and reports.

## Current validation job

15068076 (griffith), four GPUs, one-hour cap, three matched updates per arm. Frozen revision 61e1d0c26ad6ce8cafc921a83483ae9ae8bed19e. The actual installed SGLang loader validates auto in the regression. Eight baseline checks and ten affected adapter/config/loss checks passed; patch application and seven source hashes verified. The obsolete duplicate 15059125 was cancelled when 15067116 started. Results pending under state/campaign-15068076.

## Numerical validity repair after 15068076

All three arms completed, including API submission, score-centering, mapper fitting and real mapped resumes. However, the initial native train/sampler log-probability gap was approximately 4.5 nats. CPU inspection confirmed that packed SDPA inputs with null attention masks allow cross-example attention. A tiny Qwen2 reference showed maximum logit error 0.0769967 with the original packing and 0.0 with an explicit per-sequence causal mask. The previous results (sync 23/32, direct 21/32, API 20/32; training 86.95/77.63/163.59 seconds) are provisional and cannot validate the intended RL comparison. The corrected adapter preserves SDPA but supplies a block causal mask; logits and gradients are checked against separately evaluated sequences. Report parsing is also corrected for /avg metrics and baseline gradients. Historical v2 source and all reports are preserved.
