"""CPU integration reference for packed AReaL score-centering loss inputs."""

from types import SimpleNamespace

import torch

from nibble.areal_loss import loss_from_logits


def test_packed_multisequence_head_loss_matches_dense_tis_gradient(monkeypatch):
    """Shifted masks prevent cross-sequence labels; padded rows contribute nothing."""
    from areal import utils

    monkeypatch.setenv("NIBBLE_SCORE_CENTERING", "1")
    monkeypatch.setattr(
        utils,
        "stats_tracker",
        SimpleNamespace(
            denominator=lambda **kwargs: None,
            stat=lambda *args, **kwargs: None,
        ),
    )

    torch.manual_seed(31)
    vocab_size = 6
    token_count = 9
    # First two sequences have lengths 4 and 3. The final two entries model
    # transport padding and have zero loss mask.
    input_ids = torch.tensor([1, 2, 3, 4, 2, 5, 1, 0, 0])
    loss_mask = torch.tensor([0, 1, 1, 0, 1, 1, 0, 0, 0], dtype=torch.bool)
    logits = torch.randn(
        token_count + 2, vocab_size, dtype=torch.float64, requires_grad=True
    )
    sampler_logits = torch.randn(token_count, vocab_size, dtype=torch.float64)
    sampler_probs = sampler_logits.softmax(-1)
    top_q, top_ids = sampler_probs.topk(2, dim=-1)
    # Input uses already-shifted actor fields. Inactive prompt/end/pad rows hold
    # zero IDs and log-probabilities, as the patched rollout workflow does.
    sampler_topk_ids = torch.zeros_like(top_ids)
    sampler_topk_logprobs = torch.zeros_like(top_q)
    sampler_topk_ids[loss_mask] = top_ids[loss_mask]
    sampler_topk_logprobs[loss_mask] = top_q[loss_mask].log()

    behavior_logprobs = torch.zeros(token_count, dtype=torch.float64)
    all_labels = input_ids.roll(-1)
    behavior_logprobs[loss_mask] = (
        sampler_probs[loss_mask]
        .gather(-1, all_labels[loss_mask, None])
        .squeeze(-1)
        .log()
    )
    advantages = torch.zeros(token_count, dtype=torch.float64)
    advantages[1:3] = 1.0  # first completion in a 2-sample prompt group
    advantages[4:6] = -1.0  # second completion in that group

    data = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "logprobs": behavior_logprobs,
        "advantages": advantages,
        "sampler_topk_ids": sampler_topk_ids,
        "sampler_topk_logprobs": sampler_topk_logprobs,
    }
    monkeypatch.setenv("NIBBLE_SCORE_CENTERING", "1")
    got = loss_from_logits(logits, data)

    active_ids = input_ids[loss_mask]
    labels = all_labels[loss_mask]
    active_logits = logits[:token_count][loss_mask].float()
    train_logp = active_logits.log_softmax(-1)
    sampler_active = sampler_probs[loss_mask]
    head_ids = top_ids[loss_mask]
    q_head = sampler_active.gather(-1, head_ids)
    p_head = train_logp.gather(-1, head_ids).exp()
    rho = (1.0 - q_head.sum(-1)) / (1.0 - p_head.sum(-1))
    in_head = torch.zeros_like(sampler_active, dtype=torch.bool).scatter(
        -1, head_ids, True
    )
    q_hat = torch.where(
        in_head, sampler_active, rho[:, None] * train_logp.detach().exp()
    )
    dense_ratio = train_logp.detach().exp() / q_hat.detach()
    dense_tis = torch.minimum(dense_ratio, torch.tensor(2.0))
    sample_train = train_logp.gather(-1, labels[:, None]).squeeze(-1)
    sample_behavior = behavior_logprobs[loss_mask]
    sample_tis = (sample_train.detach() - sample_behavior).exp().clamp(max=2.0)
    active_advantages = advantages[loss_mask]
    dense_losses = -active_advantages * (
        sample_tis * sample_train
        - (q_hat.detach() * dense_tis.detach() * train_logp).sum(-1)
    )
    expected = dense_losses.mean()
    got_grad = torch.autograd.grad(got, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]

    assert torch.allclose(got_grad, expected_grad, atol=2e-7, rtol=2e-7)
    assert torch.equal(got_grad[-2:], torch.zeros_like(got_grad[-2:]))
    assert active_ids.numel() == 4
