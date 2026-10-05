# Corrected end-to-end run

Job 15067116, griffith account, 4 GPUs, 60-minute cap. Frozen code ed5430d3ce7268c6dea10faabd6322c80572fb4b. Three matched optimizer updates per arm; all other campaign workload and evaluation settings unchanged.

The Transformers 5.3 tokenizer mapping return is normalized by explicitly requesting return_dict=False in baseline and dataset builders. The actual Qwen2.5 tokenizer and 136 benchmark prompts passed CPU preflight. 32 CPU tests, lint, compilation and pinned adapter verification passed.

Results are pending GPU execution. Job 15059125 with the faulty release is held and must not be released. Results will be saved under state/campaign-15067116.

## GPU attempt 15067116

The tokenizer fix passed real rollout generation. Job ran for 193 seconds, then the synchronous baseline failed at its first disk weight update: the SGLang scheduler rejects load_format=hf (ValueError: hf is not a valid LoadFormat). A first optimizer update and checkpoint step-0001 were produced, but the step metric was not committed because reloading failed; async/API arms were not launched. Logs remain under state/campaign-15067116.

The next release uses the actual supported auto load format. It also explicitly disables model-generation defaults for top-k and repetition penalty: top_k=-1 and repetition_penalty=1.0 in both baseline and group_pg_tis AReaL requests. Previous baseline generation inherited top_k=20 and all arms otherwise inherited repetition_penalty=1.1; that mismatch would invalidate a matched sampling comparison. The new AReaL adapter is a separate source copy under work/deps/areal-040cbfd5-nibble-v2, preserving the prior source and reports.
