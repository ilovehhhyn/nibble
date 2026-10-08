"""PyTorch REINFORCE losses with paper-faithful score centering.

For each token, score_centering_loss implements
    L = -A * (sg[w_y] log p_y - sum_H sg[q_v w_v - alpha p_v] log p_v),
    rho = q(T) / p(T),  alpha = rho * f(1 / rho),
where H is the sampler's recorded top-k. The unobserved sampler tail is modeled
as q_v = rho * p_v. importance=None uses f=1; 'tis uses f(r)=min(r, 2);
'mis' uses r in [0.5, 5] and zero outside, matching Appendix A.

Inputs have arbitrary token-prefix dimensions. Full trainer logprobs have shape
[..., V]. Alternatively, callers may provide gathered trainer head logprobs
[..., K] plus optional trainer tail log-mass [...]. The tail mass is recovered
from normalized trainer logprobs as 1 minus the head mass if omitted. Sampler values, advantages, importance
weights, and centering coefficients are detached from autograd.
"""

from typing import Literal

import torch
from torch import Tensor

ImportanceMode = Literal["tis", "mis", "ppo_mask"] | None


def _validate_logprobs(logprobs: Tensor, name: str) -> None:
    if not logprobs.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if bool((torch.isnan(logprobs) | torch.isposinf(logprobs) | (logprobs > 0)).any()):
        raise ValueError(f"{name} must contain log probabilities in [-inf, 0]")


def _log1mexp(log_mass: Tensor) -> Tensor:
    """Return log(1-exp(x)) accurately for x <= 0; x=0 maps to -inf."""
    # A normalized full-vocabulary log-softmax can sum to 1+roundoff.
    log_mass = torch.minimum(log_mass, torch.zeros_like(log_mass))
    split = -0.6931471805599453
    return torch.where(
        log_mass < split,
        torch.log1p(-torch.exp(log_mass)),
        torch.log(-torch.expm1(log_mass)),
    )


_TAIL_MASS_FLOOR = 1e-6  # Paper Appendix A.3 uses eps=1e-6 for both tails.


def _tail_log_mass(head_logprobs: Tensor) -> Tensor:
    """Compute log(1 - head mass) in float64, floored as in Appendix A.3."""
    log_mass = _log1mexp(torch.logsumexp(head_logprobs.to(torch.float64), dim=-1))
    return torch.maximum(log_mass, log_mass.new_tensor(_TAIL_MASS_FLOOR).log())


def _importance_weight(log_ratio: Tensor, mode: ImportanceMode) -> Tensor:
    """Evaluate supported paper weights from log(p/q), avoiding ratio overflow."""
    if mode is None:
        return torch.ones_like(log_ratio)
    if mode == "tis":
        return torch.exp(torch.minimum(log_ratio, log_ratio.new_tensor(2.0).log()))
    if mode == "mis":
        lower, upper = log_ratio.new_tensor(0.5).log(), log_ratio.new_tensor(5.0).log()
        return torch.where(
            (log_ratio >= lower) & (log_ratio <= upper),
            torch.exp(log_ratio),
            torch.zeros_like(log_ratio),
        )
    if mode == "ppo_mask":
        upper = log_ratio.new_tensor(5.0).log()
        return torch.where(
            log_ratio <= upper, torch.exp(log_ratio), torch.zeros_like(log_ratio)
        )
    raise ValueError("importance must be None, 'tis', 'mis', or 'ppo_mask'")


def _head_weighted_sampler_mass(
    train_head_logprobs: Tensor,
    sampler_head_logprobs: Tensor,
    mode: ImportanceMode,
) -> Tensor:
    """Compute q*f(p/q) per head token without materializing unstable ratios."""
    if mode is None:
        return torch.exp(sampler_head_logprobs.to(torch.float64))
    log_p = train_head_logprobs.to(torch.float64)
    log_q = sampler_head_logprobs.to(torch.float64)
    log_ratio = log_p - log_q
    if mode == "tis":
        return torch.exp(torch.minimum(log_p, log_q + log_ratio.new_tensor(2.0).log()))
    if mode == "mis":
        low, high = log_ratio.new_tensor(0.5).log(), log_ratio.new_tensor(5.0).log()
        return torch.where(
            (log_ratio >= low) & (log_ratio <= high),
            torch.exp(log_p),
            torch.zeros_like(log_p),
        )
    if mode == "ppo_mask":
        log_ratio = log_p - log_q
        return torch.where(
            log_ratio <= log_ratio.new_tensor(5.0).log(),
            torch.exp(log_p),
            torch.zeros_like(log_p),
        )
    raise ValueError("importance must be None, 'tis', 'mis', or 'ppo_mask'")


def _tail_alpha(log_rho: Tensor, mode: ImportanceMode) -> Tensor:
    """Return alpha=rho*f(1/rho), including exact zero-tail limits."""
    if mode is None:
        return torch.exp(log_rho)
    if mode == "tis":
        # rho * min(1/rho, 2) = min(1, 2*rho).
        return torch.exp(
            torch.minimum(
                log_rho + log_rho.new_tensor(2.0).log(), torch.zeros_like(log_rho)
            )
        )
    if mode == "mis":
        low, high = log_rho.new_tensor(0.2).log(), log_rho.new_tensor(2.0).log()
        return torch.where(
            (log_rho >= low) & (log_rho <= high),
            torch.ones_like(log_rho),
            torch.zeros_like(log_rho),
        )
    if mode == "ppo_mask":
        return (log_rho >= log_rho.new_tensor(0.2).log()).to(log_rho.dtype)
    raise ValueError("importance must be None, 'tis', 'mis', or 'ppo_mask'")


def grpo_policy_gradient_loss(
    sampled_train_logprobs: Tensor,
    sampled_sampler_logprob: Tensor,
    advantage: Tensor,
    *,
    importance: ImportanceMode = "tis",
    mask: Tensor | None = None,
    reduction: Literal["none", "sum", "mean"] = "mean",
) -> Tensor:
    """Group-relative REINFORCE loss with detached sampled-token weights.

    The caller supplies group-centered advantages, normally reward minus the
    prompt-group mean. This is the shared REINFORCE objective with TIS
    (f(r)=min(r,2)) by default; it does not add a PPO clipped surrogate.
    Score centering composes through score_centering_loss and its Appendix A.2
    head/tail correction.
    """
    if sampled_train_logprobs.shape != sampled_sampler_logprob.shape:
        raise ValueError(
            "sampled trainer and sampler logprobs must have matching shapes"
        )
    if advantage.shape != sampled_train_logprobs.shape:
        raise ValueError("advantage must match sampled logprob shape")
    if importance not in (None, "tis", "mis", "ppo_mask"):
        raise ValueError("importance must be None, 'tis', 'mis', or 'ppo_mask'")
    _validate_logprobs(sampled_train_logprobs, "sampled_train_logprobs")
    _validate_logprobs(sampled_sampler_logprob, "sampled_sampler_logprob")
    weights = (
        _importance_weight(
            sampled_train_logprobs.detach().to(torch.float64)
            - sampled_sampler_logprob.detach().to(torch.float64),
            importance,
        )
        .to(sampled_train_logprobs.dtype)
        .detach()
    )
    losses = -advantage.detach() * weights * sampled_train_logprobs
    if mask is not None:
        if mask.shape != losses.shape:
            raise ValueError("mask must match sampled logprob shape")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be boolean")
        losses = torch.where(mask, losses, torch.zeros_like(losses))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    denominator = losses.new_tensor(losses.numel()) if mask is None else mask.sum()
    if bool((denominator == 0).item()):
        raise ValueError("mean reduction requires at least one valid token")
    return losses.sum() / denominator


def score_centering_loss(
    train_logprobs: Tensor | None,
    sampler_topk_ids: Tensor,
    sampler_topk_logprobs: Tensor,
    sampled_train_logprobs: Tensor,
    advantage: Tensor,
    *,
    train_head_logprobs: Tensor | None = None,
    train_tail_log_mass: Tensor | None = None,
    sampled_sampler_logprob: Tensor | None = None,
    importance: ImportanceMode = None,
    mask: Tensor | None = None,
    reduction: Literal["none", "sum", "mean"] = "none",
) -> Tensor:
    """Return masked per-token score-centering loss.

    Pass exactly one trainer representation: full logprobs [..., V], or gathered
    trainer heads [..., K] with optional tail log-mass log(p(T)) [...]. When
    omitted in gathered mode it is reconstructed from normalized logprobs as
    1 minus the head mass. Sampler
    top-k logprobs are actual q probabilities, not logits or head-renormalized
    values. Importance composition needs sampled_sampler_logprob for the
    sampled token even when it lies outside top-k.

    The output retains gradients through sampled and gathered trainer logprobs.
    'mean' divides by the valid-token count, matching token-average GRPO.
    As in paper Appendix A.3, inferred tails are floored at 1e-6.
    """
    if (train_logprobs is None) == (train_head_logprobs is None):
        raise ValueError("provide exactly one of train_logprobs or train_head_logprobs")
    if reduction not in ("none", "sum", "mean"):
        raise ValueError("reduction must be 'none', 'sum', or 'mean'")
    if importance not in (None, "tis", "mis", "ppo_mask"):
        raise ValueError("importance must be None, 'tis', 'mis', or 'ppo_mask'")
    if sampler_topk_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("sampler_topk_ids must contain integer token IDs")
    if sampler_topk_ids.ndim < 1 or sampler_topk_ids.shape[-1] == 0:
        raise ValueError("sampler top-k must contain at least one token")
    if sampler_topk_logprobs.shape != sampler_topk_ids.shape:
        raise ValueError("sampler_topk_logprobs must match sampler_topk_ids shape")
    if sampled_train_logprobs.shape != sampler_topk_ids.shape[:-1]:
        raise ValueError("sampled_train_logprobs must match token-prefix shape")
    if advantage.shape != sampled_train_logprobs.shape:
        raise ValueError("advantage must match token-prefix shape")
    if mask is not None:
        if mask.shape != sampled_train_logprobs.shape:
            raise ValueError("mask must match token-prefix shape")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be boolean")
    _validate_logprobs(sampler_topk_logprobs, "sampler_topk_logprobs")
    _validate_logprobs(sampled_train_logprobs, "sampled_train_logprobs")

    if train_logprobs is not None:
        if train_logprobs.ndim != sampler_topk_ids.ndim:
            raise ValueError(
                "train_logprobs must have one vocabulary axis after token axes"
            )
        if train_logprobs.shape[:-1] != sampled_train_logprobs.shape:
            raise ValueError("train_logprobs must match token-prefix shape")
        if sampler_topk_ids.device != train_logprobs.device:
            raise ValueError("trainer and sampler tensors must be on the same device")
        _validate_logprobs(train_logprobs, "train_logprobs")
        valid_ids = (sampler_topk_ids >= 0) & (
            sampler_topk_ids < train_logprobs.shape[-1]
        )
        active = (
            torch.ones_like(sampled_train_logprobs, dtype=torch.bool)
            if mask is None
            else mask
        )
        if bool((~valid_ids & active.unsqueeze(-1)).any()):
            raise ValueError("sampler_topk_ids are outside the trainer vocabulary")
        safe_ids = torch.where(active.unsqueeze(-1), sampler_topk_ids, 0).to(torch.long)
        train_head = torch.gather(train_logprobs, -1, safe_ids)
        if mask is not None:
            train_head = torch.where(
                mask.unsqueeze(-1), train_head, torch.full_like(train_head, -torch.inf)
            )
        train_tail_log_mass = _tail_log_mass(train_head)
    else:
        if train_head_logprobs is None:
            raise ValueError("gathered trainer heads require train_head_logprobs")
        if train_head_logprobs.shape != sampler_topk_ids.shape:
            raise ValueError("train_head_logprobs must match sampler_topk_ids shape")
        if (
            train_tail_log_mass is not None
            and train_tail_log_mass.shape != sampled_train_logprobs.shape
        ):
            raise ValueError("train_tail_log_mass must match token-prefix shape")
        if sampler_topk_ids.device != train_head_logprobs.device:
            raise ValueError("trainer and sampler tensors must be on the same device")
        train_head = train_head_logprobs
        _validate_logprobs(train_head, "train_head_logprobs")
        if mask is not None:
            train_head = torch.where(
                mask.unsqueeze(-1), train_head, torch.full_like(train_head, -torch.inf)
            )
        if train_tail_log_mass is None:
            train_tail_log_mass = _tail_log_mass(train_head)
        else:
            _validate_logprobs(train_tail_log_mass, "train_tail_log_mass")

    if (
        train_head.device != sampler_topk_logprobs.device
        or sampled_train_logprobs.device != train_head.device
        or advantage.device != train_head.device
        or sampler_topk_ids.device != train_head.device
    ):
        raise ValueError("all loss tensors must be on the same device")
    assert train_tail_log_mass is not None
    effective_sampler_head = sampler_topk_logprobs
    if mask is not None:
        effective_sampler_head = torch.where(
            mask.unsqueeze(-1),
            sampler_topk_logprobs,
            torch.full_like(sampler_topk_logprobs, -torch.inf),
        )
    sampler_tail_log_mass = _tail_log_mass(effective_sampler_head)
    train_tail_log_mass = train_tail_log_mass.to(torch.float64)
    train_tail_log_mass = torch.maximum(
        train_tail_log_mass,
        train_tail_log_mass.new_tensor(_TAIL_MASS_FLOOR).log(),
    )
    if bool((train_tail_log_mass > 0).any()):
        raise ValueError("trainer tail log-mass must be at most zero")

    # Appendix A.3 floors both tails at 1e-6, so their ratio is defined.
    log_rho = sampler_tail_log_mass - train_tail_log_mass
    alpha = _tail_alpha(log_rho, importance)

    train_head64 = train_head.to(torch.float64)
    sampler_head64 = effective_sampler_head.to(torch.float64)
    head_mass = _head_weighted_sampler_mass(train_head64, sampler_head64, importance)
    coefficients = head_mass - alpha.unsqueeze(-1) * torch.exp(train_head64)
    coefficients = coefficients.detach()
    correction = (
        (coefficients * train_head64).sum(dim=-1).to(sampled_train_logprobs.dtype)
    )

    if importance is None:
        sampled_weight = torch.ones_like(sampled_train_logprobs)
    else:
        if sampled_sampler_logprob is None:
            raise ValueError("importance composition requires sampled_sampler_logprob")
        if sampled_sampler_logprob.shape != sampled_train_logprobs.shape:
            raise ValueError("sampled_sampler_logprob must match token-prefix shape")
        if sampled_sampler_logprob.device != train_head.device:
            raise ValueError("all loss tensors must be on the same device")
        _validate_logprobs(sampled_sampler_logprob, "sampled_sampler_logprob")
        sampled_weight = _importance_weight(
            sampled_train_logprobs.detach().to(torch.float64)
            - sampled_sampler_logprob.detach().to(torch.float64),
            importance,
        ).to(sampled_train_logprobs.dtype)
    token_loss = -advantage.detach() * (
        sampled_weight.detach() * sampled_train_logprobs - correction
    )

    if mask is not None:
        if mask.shape != token_loss.shape:
            raise ValueError("mask must match token-prefix shape")
        if mask.dtype != torch.bool:
            raise TypeError("mask must be boolean")
        token_loss = torch.where(mask, token_loss, torch.zeros_like(token_loss))
    if reduction == "none":
        return token_loss
    if reduction == "sum":
        return token_loss.sum()
    denominator = (
        token_loss.new_tensor(token_loss.numel()) if mask is None else mask.sum()
    )
    if bool((denominator == 0).item()):
        raise ValueError("mean reduction requires at least one valid token")
    return token_loss.sum() / denominator
