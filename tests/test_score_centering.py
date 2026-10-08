"""CPU reference checks for score centering values, gradients, and contracts."""

import pytest
import torch

from nibble.score_centering import score_centering_loss


def _dense_reference(logits, sampler_probs, sampled_ids, advantage, mask=None):
    """Full-vocabulary Eq. 13 objective."""
    logp = torch.log_softmax(logits, dim=-1)
    center = (sampler_probs * logp).sum(-1)
    sampled = logp.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
    losses = -advantage * (sampled - center)
    if mask is not None:
        losses = torch.where(mask, losses, torch.zeros_like(losses))
    return losses


def test_topk_matches_dense_reference_value_and_gradient():
    torch.manual_seed(8)
    logits = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    q_logits = torch.randn(2, 3, 7, dtype=torch.float64)
    q = q_logits.softmax(-1)
    sampled_ids = torch.randint(7, (2, 3))
    advantage = torch.randn(2, 3, dtype=torch.float64)
    logp = logits.log_softmax(-1)
    q_head, ids = q.topk(3, dim=-1)
    sampled_train = logp.gather(-1, sampled_ids.unsqueeze(-1)).squeeze(-1)
    got = score_centering_loss(logp, ids, q_head.log(), sampled_train, advantage)
    p_head = logp.gather(-1, ids).exp()
    rho = (1.0 - q_head.sum(-1)) / (1.0 - p_head.sum(-1))
    in_head = torch.zeros_like(q, dtype=torch.bool).scatter(-1, ids, True)
    q_hat = torch.where(in_head, q, rho.detach().unsqueeze(-1) * logp.detach().exp())
    expected = _dense_reference(logits, q_hat, sampled_ids, advantage)
    got_grad = torch.autograd.grad(got.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    assert torch.allclose(got_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_matching_policies_make_centering_a_noop():
    logits = torch.tensor(
        [[0.3, -0.8, 0.1, 1.0]], dtype=torch.float64, requires_grad=True
    )
    logp = logits.log_softmax(-1)
    q = logp.detach().exp()
    ids = torch.tensor([[3, 0]])
    sampled = torch.tensor([1])
    advantage = torch.tensor([2.0], dtype=torch.float64)
    loss = score_centering_loss(
        logp,
        ids,
        q.gather(-1, ids).log(),
        logp.gather(-1, sampled[:, None]).squeeze(-1),
        advantage,
    )
    vanilla = -advantage * logp.gather(-1, sampled[:, None]).squeeze(-1)
    assert torch.allclose(loss, vanilla, atol=1e-12, rtol=1e-12)
    loss_grad = torch.autograd.grad(loss.sum(), logits, retain_graph=True)[0]
    vanilla_grad = torch.autograd.grad(vanilla.sum(), logits)[0]
    assert torch.allclose(loss_grad, vanilla_grad, atol=1e-12)


def test_constant_reward_expected_gradient_is_zero_under_mismatch():
    trainer_logits = torch.tensor(
        [[0.1, 0.8, -0.6, 0.2]], dtype=torch.float64, requires_grad=True
    )
    sampler_logits = torch.tensor([[-0.7, 0.1, 0.9, -0.2]], dtype=torch.float64)
    logp = trainer_logits.log_softmax(-1)
    q = sampler_logits.softmax(-1)
    ids = torch.arange(4).view(1, 1, 4).expand(1, 4, 4)
    trainer_vocab = logp[:, None, :].expand(1, 4, 4)
    sampler_vocab = q.log()[:, None, :].expand(1, 4, 4)
    sampled_train = logp.expand(4, 4).diagonal().unsqueeze(0)
    losses = score_centering_loss(
        trainer_vocab,
        ids,
        sampler_vocab,
        sampled_train,
        torch.ones(1, 4, dtype=torch.float64),
    )
    dense_expected = _dense_reference(
        trainer_vocab,
        q[:, None, :].expand(1, 4, 4),
        torch.arange(4).view(1, 4),
        torch.ones(1, 4, dtype=torch.float64),
    )
    expected_grad = torch.autograd.grad(
        (losses * q).sum(), trainer_logits, retain_graph=True
    )[0]
    dense_grad = torch.autograd.grad(
        (dense_expected * q).sum(), trainer_logits, retain_graph=True
    )[0]
    assert torch.allclose(expected_grad, dense_grad, atol=1e-12)
    assert torch.allclose(expected_grad, torch.zeros_like(expected_grad), atol=1e-12)
    uncentered = -(q * logp).sum(-1).sum()
    drift = torch.autograd.grad(uncentered, trainer_logits)[0]
    assert drift.abs().max() > 1e-3


def test_mask_and_mean_reduce_only_valid_tokens():
    logits = torch.tensor(
        [
            [[0.0, 1.0, -1.0], [1.0, -0.5, 0.2], [0.4, -0.1, 0.3]],
            [[-0.2, 0.3, 0.8], [1.2, -0.4, 0.1], [0.0, 0.5, -0.5]],
        ],
        dtype=torch.float64,
    )
    logp = logits.log_softmax(-1)
    q = torch.tensor(
        [
            [[0.2, 0.5, 0.3], [0.1, 0.2, 0.7], [0.3, 0.3, 0.4]],
            [[0.6, 0.1, 0.3], [0.2, 0.3, 0.5], [0.1, 0.7, 0.2]],
        ],
        dtype=torch.float64,
    )
    ids = torch.arange(3).expand(2, 3, 3).clone()
    ids[~torch.tensor([[True, False, True], [False, True, False]])] = 0
    sampled = torch.tensor([[0, 1, 2], [2, 1, 0]])
    sample_lp = logp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    advantage = torch.ones(2, 3, dtype=torch.float64)
    mask = torch.tensor([[True, False, True], [False, True, False]])
    loss = score_centering_loss(logp, ids, q.log(), sample_lp, advantage, mask=mask)
    assert torch.equal(loss[~mask], torch.zeros_like(loss[~mask]))
    mean = score_centering_loss(
        logp, ids, q.log(), sample_lp, advantage, mask=mask, reduction="mean"
    )
    assert torch.allclose(mean, loss.sum() / mask.sum())
    with pytest.raises(ValueError, match="at least one valid token"):
        score_centering_loss(
            logp,
            ids,
            q.log(),
            sample_lp,
            advantage,
            mask=torch.zeros_like(mask),
            reduction="mean",
        )


def test_gathered_head_infers_tail_mass_from_complete_head():
    logits = torch.tensor(
        [[0.2, -0.3, 0.5, 1.2]], dtype=torch.float64, requires_grad=True
    )
    sampler_logits = torch.tensor([[0.1, -0.1, 1.0, -0.5]], dtype=torch.float64)
    logp, q = logits.log_softmax(-1), sampler_logits.softmax(-1)
    ids = torch.tensor([[3, 1]])
    train_head = logp.gather(-1, ids)
    q_head_logp = q.gather(-1, ids).log()
    sample_id = torch.tensor([0])
    sample_logp = logp.gather(-1, sample_id[:, None]).squeeze(-1)
    tail = (1.0 - train_head.exp().sum(-1)).log()
    result = score_centering_loss(
        None,
        ids,
        q_head_logp,
        sample_logp,
        torch.ones(1, dtype=torch.float64),
        train_head_logprobs=train_head,
        train_tail_log_mass=tail,
    )
    p_head = train_head.exp()
    rho = (1.0 - q_head_logp.exp().sum(-1)) / (1.0 - p_head.sum(-1))
    in_head = torch.zeros_like(q, dtype=torch.bool).scatter(-1, ids, True)
    q_hat = torch.where(in_head, q, rho.detach().unsqueeze(-1) * logp.detach().exp())
    expected = _dense_reference(
        logits, q_hat, sample_id, torch.ones(1, dtype=torch.float64)
    )
    result_grad = torch.autograd.grad(result.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits, retain_graph=True)[0]
    assert torch.allclose(result_grad, expected_grad, atol=1e-12, rtol=1e-12)
    inferred = score_centering_loss(
        None,
        ids,
        q_head_logp,
        sample_logp,
        torch.ones(1, dtype=torch.float64),
        train_head_logprobs=train_head,
    )
    inferred_grad = torch.autograd.grad(inferred.sum(), logits)[0]
    assert torch.allclose(inferred_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_tis_composition_matches_dense_reference_for_head_and_tail():
    logits = torch.tensor(
        [[0.1, -0.4, 0.7, -0.2]], dtype=torch.float64, requires_grad=True
    )
    q_logits = torch.tensor([[0.2, 0.1, -0.8, 0.5]], dtype=torch.float64)
    logp, q = logits.log_softmax(-1), q_logits.softmax(-1)
    ids = torch.tensor([[3, 1]])
    q_head = q.gather(-1, ids)
    p_head = logp.gather(-1, ids).exp()
    rho = (1.0 - q_head.sum(-1)) / (1.0 - p_head.sum(-1))
    sample_id = torch.tensor([2])
    sample_p = logp.gather(-1, sample_id[:, None]).squeeze(-1)
    sample_q = q.gather(-1, sample_id[:, None]).squeeze(-1).log()
    advantage = torch.tensor([1.7], dtype=torch.float64)
    head_weights = (p_head / q_head).clamp(max=2.0)
    alpha = rho * torch.minimum(1.0 / rho, torch.tensor(2.0, dtype=torch.float64))
    coeff = q_head * head_weights - alpha[:, None] * p_head
    sampled_weight = (sample_p - sample_q).exp().clamp(max=2.0)
    expected = -advantage * (
        sampled_weight * sample_p - (coeff * logp.gather(-1, ids)).sum(-1)
    )
    got = score_centering_loss(
        logp,
        ids,
        q_head.log(),
        sample_p,
        advantage,
        sampled_sampler_logprob=sample_q,
        importance="tis",
    )
    assert torch.allclose(got, expected, atol=1e-12, rtol=1e-12)


def test_bad_shapes_and_inconsistent_tail_mass_fail_clearly():
    lp = torch.log_softmax(torch.tensor([[0.1, 0.2, -0.1]]), -1)
    ids = torch.tensor([[0, 1]])
    with pytest.raises(ValueError, match="match sampler_topk_ids"):
        score_centering_loss(
            lp, ids, torch.zeros((1, 1)), torch.zeros(1), torch.ones(1)
        )
    with pytest.raises(ValueError, match="in \\[-inf, 0\\]"):
        score_centering_loss(
            None,
            ids,
            torch.tensor([[-0.2, -0.3]]),
            torch.zeros(1),
            torch.ones(1),
            train_head_logprobs=torch.tensor([[-0.2, -0.3]]),
            train_tail_log_mass=torch.tensor([0.1]),
        )


def test_zero_sampler_tail_is_defined_without_epsilon():
    logp = torch.log_softmax(torch.tensor([[0.1, 0.2, -0.3]], dtype=torch.float64), -1)
    ids = torch.tensor([[0, 1, 2]])
    q_logp = logp.detach()
    sample = logp[:, 0]
    result = score_centering_loss(logp, ids, q_logp, sample, torch.ones_like(sample))
    assert torch.allclose(result, -sample, atol=1e-12)


def test_shared_grpo_helper_uses_detached_tis_weight_and_group_advantage():
    from nibble.score_centering import grpo_policy_gradient_loss

    train = torch.tensor([[-0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    sampler = torch.tensor([[-0.5, -0.4]], dtype=torch.float64)
    advantage = torch.tensor([[1.0, -1.0]], dtype=torch.float64)
    result = grpo_policy_gradient_loss(train, sampler, advantage, reduction="none")
    expected_weight = (train.detach() - sampler).exp().clamp(max=2.0)
    expected = -advantage * expected_weight * train
    assert torch.allclose(result, expected)
    assert torch.allclose(
        torch.autograd.grad(result.sum(), train)[0], -advantage * expected_weight
    )


def test_mis_tail_alpha_and_gradient_match_dense_reference_at_rho_not_one():
    logits = torch.tensor(
        [[0.3, 0.0, -0.4, -1.0]], dtype=torch.float64, requires_grad=True
    )
    sampler_logits = torch.tensor([[-0.6, -0.6, -0.2, -0.2]], dtype=torch.float64)
    logp, q = logits.log_softmax(-1), sampler_logits.softmax(-1)
    ids = torch.tensor([[0, 1]])
    q_head = q.gather(-1, ids)
    p_head = logp.gather(-1, ids).exp()
    rho = (1.0 - q_head.sum(-1)) / (1.0 - p_head.sum(-1))
    assert not torch.allclose(rho, torch.ones_like(rho))
    assert 0.2 < rho.item() < 2.0
    sample_id = torch.tensor([0])
    sample_p = logp.gather(-1, sample_id[:, None]).squeeze(-1)
    sample_q = q.gather(-1, sample_id[:, None]).squeeze(-1).log().detach()
    advantage = torch.tensor([1.3], dtype=torch.float64)

    in_head = torch.zeros_like(q, dtype=torch.bool).scatter(-1, ids, True)
    q_hat = torch.where(in_head, q, rho.detach().unsqueeze(-1) * logp.detach().exp())
    dense_ratio = logp.detach().exp() / q_hat.detach()
    dense_weight = torch.where(
        (dense_ratio >= 0.5) & (dense_ratio <= 5.0),
        dense_ratio,
        torch.zeros_like(dense_ratio),
    )
    sampled_weight = (sample_p.detach() - sample_q).exp().clamp(0.5, 5.0)
    expected = -advantage * (
        sampled_weight * sample_p
        - (q_hat.detach() * dense_weight.detach() * logp).sum(-1)
    )
    got = score_centering_loss(
        logp,
        ids,
        q_head.log(),
        sample_p,
        advantage,
        sampled_sampler_logprob=sample_q,
        importance="mis",
    )
    got_grad = torch.autograd.grad(got.sum(), logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), logits)[0]
    assert torch.allclose(got_grad, expected_grad, atol=1e-12, rtol=1e-12)


def test_ppo_mask_mode_matches_exact_dense_weighted_gradient():
    from nibble.score_centering import score_centering_loss

    logits = torch.tensor(
        [[0.2, -0.1, 0.4, -0.7]], dtype=torch.float64, requires_grad=True
    )
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    logp = logits.log_softmax(-1)
    ids = torch.tensor([[0, 1]])
    q_head = q.gather(-1, ids)
    p_head = logp.detach().exp().gather(-1, ids)
    rho = ((1 - q_head.sum(-1)) / (1 - p_head.sum(-1))).clamp(min=1e-6)
    head_ratio = p_head / q_head
    head_weighted_q = torch.where(head_ratio <= 5.0, p_head, 0.0)
    alpha = (rho >= 0.2).to(logp.dtype)
    head_coeff = head_weighted_q - alpha[:, None] * p_head
    advantage = torch.tensor([1.7], dtype=torch.float64)
    sampled_id = torch.tensor([3])
    sample_p = logp.gather(-1, sampled_id[:, None]).squeeze(-1)
    sample_q = q.gather(-1, sampled_id[:, None]).squeeze(-1).log()
    sample_ratio = (sample_p.detach() - sample_q).exp()
    sample_weight = torch.where(sample_ratio <= 5.0, sample_ratio, 0.0)
    base_clipped_ppo = -advantage * sample_weight * sample_p
    exact_sc_addon = advantage * (head_coeff.detach() * logp.gather(-1, ids)).sum(-1)
    expected = base_clipped_ppo + exact_sc_addon
    got = score_centering_loss(
        logp,
        ids,
        q_head.log(),
        sample_p,
        advantage,
        sampled_sampler_logprob=sample_q,
        importance="ppo_mask",
    )
    assert torch.allclose(
        torch.autograd.grad(got.sum(), logits, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), logits)[0],
        atol=1e-12,
        rtol=1e-12,
    )
