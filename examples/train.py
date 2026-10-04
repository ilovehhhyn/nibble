"""Run through an SSH tunnel; this client needs no CUDA or AReaL installation."""

from nibble import Client

client = Client()
job = client.train(
    model="qwen3-0.6b",
    dataset="digits",
    environment="math",
    reward="digit_preference",
    max_gpus=1,
    rollouts_per_prompt=2,
    prompts_per_step=2,
    max_steps=1,
    max_new_tokens=16,
    audit_on=True,
)
print("job:", job.id)
print(job.wait())
print(job.report())
