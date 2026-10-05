"""Paper group-relative REINFORCE + TIS, optionally with score centering.

Pinned FSDP calls this with shifted, packed [tokens,...] trajectory fields.
This is distinct from legacy clipped PPO. AReaL computes group-mean advantages.
"""

import os
import torch
from .score_centering import score_centering_loss


def loss_from_logits(logits, data, *, temperature=1.0):
    """Token-mean loss; only active responses need sampler heads. TP=SP=1."""
    if temperature != 1.0:
        raise ValueError("the reproduction fixes temperature=1 to match sampler q")
    mask = data["loss_mask"].reshape(-1).bool()
    ids = data["input_ids"].reshape(-1).long()
    flat = logits.reshape(-1, logits.shape[-1])[: len(mask)]
    # Sequence endpoints are masked, so rolled labels cannot cross examples.
    labels = ids.roll(-1)[mask]
    active_logits = flat[mask].float()
    normalizer = torch.logsumexp(active_logits, dim=-1)
    sampled = active_logits.gather(-1, labels[:, None]).squeeze(-1) - normalizer
    behavior = data["logprobs"].reshape(-1)[mask].detach()
    advantage = data["advantages"].reshape(-1)[mask].detach()
    weights = torch.exp(sampled.detach() - behavior).clamp(max=2.0)
    plain = -advantage * weights * sampled
    centered = os.environ.get("NIBBLE_SCORE_CENTERING") == "1"
    if centered:
        head_ids = data["sampler_topk_ids"].reshape(len(mask), -1)[mask].long()
        q_head = data["sampler_topk_logprobs"].reshape(len(mask), -1)[mask]
        p_head = active_logits.gather(-1, head_ids) - normalizer[:, None]
        per_token = score_centering_loss(
            None,
            head_ids,
            q_head,
            sampled,
            advantage,
            train_head_logprobs=p_head,
            sampled_sampler_logprob=behavior,
            importance="tis",
            reduction="none",
        )
    else:
        per_token = plain
    from areal.utils import stats_tracker

    stats_tracker.denominator(
        n_nibble_tokens=torch.ones_like(sampled, dtype=torch.bool)
    )
    stats_tracker.stat(
        nibble_policy_loss=per_token.detach(),
        nibble_uncentered_tis_loss=plain.detach(),
        nibble_abs_logp_gap=(sampled.detach() - behavior).abs(),
        nibble_sc_tokens=torch.full_like(sampled, float(centered)),
        denominator="n_nibble_tokens",
    )
    return per_token.mean()
