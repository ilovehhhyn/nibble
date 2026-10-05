# Corrected end-to-end run

Job 15067116, griffith account, 4 GPUs, 60-minute cap. Frozen code ed5430d3ce7268c6dea10faabd6322c80572fb4b. Three matched optimizer updates per arm; all other campaign workload and evaluation settings unchanged.

The Transformers 5.3 tokenizer mapping return is normalized by explicitly requesting return_dict=False in baseline and dataset builders. The actual Qwen2.5 tokenizer and 136 benchmark prompts passed CPU preflight. 32 CPU tests, lint, compilation and pinned adapter verification passed.

Results are pending GPU execution. Job 15059125 with the faulty release is held and must not be released. Results will be saved under state/campaign-15067116.
