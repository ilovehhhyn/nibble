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


def clipped_ppo_score_centering_correction(
    logits, data, config, *, sampled_logprobs=None
):
    """Correction-only score-centering term for fork-clipped PPO.

    The original PPO loss remains unchanged. This adds
    A * sum_H sg[q*f(p/q)-alpha*p] log(p), using token rejection mask
    f(r)=r*1[r<=5]. Its reduction uses the unmodified response mask count.
    """
    if config.is_critic:
        raise ValueError("score centering is actor-only")
    if config.ppo_n_minibatches != 1:
        raise ValueError("score centering requires one PPO optimizer minibatch")
    if config.prox_logp_method != "recompute":
        raise ValueError("score centering requires recomputed proximal logprobs")
    rs = config.rejection_sampling
    if rs is None:
        raise ValueError("score centering requires token rejection sampling")

    def value(v):
        return getattr(v, "value", v)

    if str(value(rs.level)).lower() not in ("token", "level.token"):
        raise ValueError("score centering requires token-level rejection sampling")
    if str(value(rs.action)).lower() not in ("mask", "action.mask"):
        raise ValueError("score centering requires rejection action=mask")
    if (
        getattr(rs, "metric", "ratio") != "ratio"
        or getattr(rs, "lower", None) is not None
    ):
        raise ValueError("score centering requires ratio masking without a lower bound")
    if getattr(config, "importance_sampling_level", "token") != "token":
        raise ValueError("score centering requires token-level PPO importance ratios")
    if float(rs.upper) != 5.0:
        raise ValueError("score centering requires rejection upper=5")
    if not config.use_decoupled_loss:
        raise ValueError("score centering requires decoupled PPO loss")
    if config.temperature != 1.0:
        raise ValueError("score centering requires temperature=1")
    if "prox_logp" not in data:
        raise ValueError("score centering requires recomputed prox_logp for auditing")

    mask = data["loss_mask"].reshape(-1).bool()
    ids = data["input_ids"].reshape(-1).long()
    flat_logits = logits.reshape(-1, logits.shape[-1])[: len(mask)]
    if not bool(mask.any()):
        return flat_logits.sum() * 0.0
    labels = ids.roll(-1)
    if sampled_logprobs is None:
        # Small CPU tests may omit the engine's already-computed logprobs.
        selected = flat_logits.gather(-1, labels[:, None]).squeeze(-1).float()
        sampled_all = selected - torch.logsumexp(flat_logits.float(), dim=-1)
    else:
        sampled_all = sampled_logprobs.reshape(-1)[: len(mask)]
    sampled = sampled_all[mask].float()
    prox = data["prox_logp"].reshape(-1)[mask].detach().float()
    prox_gap = (sampled.detach() - prox).abs()
    if bool((prox_gap > 1e-5).any()):
        raise ValueError(
            f"score centering requires prox/current logprob gap <= 1e-5; got {prox_gap.max().item():.3g}"
        )
    sampled_logits = (
        flat_logits.gather(-1, ids.roll(-1)[:, None]).squeeze(-1)[mask].float()
    )
    log_normalizer = sampled_logits - sampled
    all_head_ids = data["sampler_topk_ids"].reshape(len(mask), -1).long()
    head_ids = all_head_ids[mask]
    head_q = data["sampler_topk_logprobs"].reshape(len(mask), -1)[mask]
    behavior = data["logprobs"].reshape(-1)[mask].detach()
    advantage = data["advantages"].reshape(-1)[mask].detach()
    from .score_centering import score_centering_loss

    head_logits = flat_logits.gather(-1, all_head_ids)
    head_logits = head_logits[mask].float()
    train_head_logprobs = head_logits - log_normalizer[:, None]
    correction = score_centering_loss(
        None,
        head_ids,
        head_q,
        torch.zeros_like(sampled),
        advantage,
        train_head_logprobs=train_head_logprobs,
        sampled_sampler_logprob=behavior,
        importance="ppo_mask",
        reduction="none",
    )
    from areal.utils import stats_tracker

    q_coverage = head_q.double().exp().sum(-1).clamp(max=1.0)
    p_coverage = train_head_logprobs.detach().double().exp().sum(-1).clamp(max=1.0)
    tail_floor = 1e-6
    q_tail = (1.0 - q_coverage).clamp(min=tail_floor)
    p_tail = (1.0 - p_coverage).clamp(min=tail_floor)
    tail_rho = q_tail / p_tail
    stats_tracker.denominator(n_nibble_sc_tokens=torch.ones_like(mask[mask]))
    stats_tracker.stat(
        nibble_sc_correction=correction.detach(),
        nibble_sc_q_head_mass=q_coverage.detach(),
        nibble_sc_p_head_mass=p_coverage.detach(),
        nibble_sc_tail_rho=tail_rho.detach(),
        nibble_sc_abs_logp_gap=(sampled.detach() - behavior).abs(),
        nibble_sc_prox_logp_gap=prox_gap.detach(),
        nibble_sc_prox_gap_within_tolerance=(prox_gap <= 1e-5).float(),
        denominator="n_nibble_sc_tokens",
    )
    return correction.mean()
