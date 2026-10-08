# Standalone synchronous HF TRL baseline

The comparison uses official Hugging Face TRL v0.28.0 at commit
49ef33428c47235991acb4e185ea599b70c6dab4. It provides PPO at
trl.experimental.ppo and GRPO's custom rollout_func. The adapter imports
official PPOTrainer, GRPOTrainer, and config classes; it does not use AReaL
trainers or scheduling.

Primary upstream references:
- [TRL v0.28.0](https://github.com/huggingface/trl/tree/v0.28.0)
- [PPO trainer](https://github.com/huggingface/trl/blob/v0.28.0/trl/experimental/ppo/ppo_trainer.py)
- [GRPO trainer](https://github.com/huggingface/trl/blob/v0.28.0/trl/trainer/grpo_trainer.py)
- [Rollout callback](https://github.com/huggingface/trl/blob/v0.28.0/trl/generation/vllm_generation.py)
- [Package constraints](https://github.com/huggingface/trl/blob/v0.28.0/pyproject.toml)

## Runtime setup

Run python tools/prepare_trl_runtime.py on Della's login node. It creates an
isolated work/venvs/trl-0.28.0 environment and installs only pinned TRL there.
A .pth file shares existing AReaL dependencies; Torch, Transformers, and other
shared packages are not changed. CPU import validation reports Torch 2.9.1,
Transformers 5.3.0, and TRL 0.28.0. Setup launches no GPU work.

## Training contract

Both algorithms use the shared filtered full GSM8K train split and the seed-0
epoch-zero sampler's first 3,840 prompt indices. Each update consumes 256
distinct prompts and four samples per prompt (1,024 completions). Prompts are
chat-templated with the shared suffix and checked against the 1,024-token cap;
generation is capped at 1,024 tokens. Rewards use the shared MathAgent
math_verify verifier and original transform (reward - 0.5) * 10. The common
100-prompt test runs outside this adapter.

TRL owns PPO/GRPO losses, value model, GAE, clipping, and optimizer updates.
Rollouts run synchronously on two SGLang GPUs; TRL trains on the third. PPO
patches TRL's full-batch batch_generation hook and dispatches all 1,024
rollouts concurrently. It supplies sampled-token log probabilities in a
narrow tensor, avoiding dense vocabulary logits while leaving PPO loss, GAE,
critic, and optimizer updates inside TRL. GRPO uses TRL's rollout_func and
official group reward standard-deviation scaling. Weights publish to both
rollout servers after each update.

PPO uses 15 updates, policy clip 0.4, value clip 0.5, KL coefficient zero,
constant learning rate, and FP32 trainable policy/value parameters with
bfloat16 autocast so Adam moments use FP32. The reference policy uses
bfloat16. TRL uses one joint policy/value optimizer and a sequence-classifier
value head initialized from the base checkpoint; AReaL has separate actor/
critic optimizer scheduling. AReaL's ratio rejection sampler cap of 5 is not
available in TRL. GRPO uses epsilon 0.4, four-sample groups, official group
reward scaling, and KL coefficient zero. Its sequence-level normalization
differs from the fork's token-level advantage details.

Both configs use microbatches of 4 and accumulation of 256, preserving the
1,024-completion optimizer update while limiting activation memory. SGLang
uses the fork's 32,768 context, 0.8 static memory fraction, FlashInfer, and
enabled CUDA graphs. Login-node validation covers setup/import/config only;
actual training needs a three-GPU allocation and was not run here.

Successful runs write data-manifest.json, 15 metrics.jsonl rows, a checkpoint
after each update, and native-result.json with source/version, objective,
optimizer, timing, dataset membership, and limitations. Missing loss or
gradient-norm metrics fail the run rather than becoming zero.

Reward parsing runs in a four-process pool created with the spawn start method,
so it is safe after CUDA initialization and avoids math_verify's main-thread
signal handler restriction. Each result has a 15-second wait bound; the shared
math_reward function and parser/verifier defaults remain unchanged. Cado reads
NVML usage for the three allocated GPUs and host memory from the trial cgroup.
