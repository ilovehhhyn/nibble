"""CPU integration reference for packed AReaL score-centering loss inputs."""

from types import SimpleNamespace

import pytest
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


def test_clipped_ppo_score_centering_correction_is_additive_and_audited(monkeypatch):
    from types import SimpleNamespace

    from areal import utils
    from nibble.areal_loss import clipped_ppo_score_centering_correction

    monkeypatch.setattr(
        utils,
        "stats_tracker",
        SimpleNamespace(
            denominator=lambda **kwargs: None, stat=lambda *args, **kwargs: None
        ),
    )
    logits = torch.tensor(
        [[0.3, -0.1, 0.2, -0.5], [0.1, 0.2, -0.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    input_ids = torch.tensor([0, 1])
    mask = torch.tensor([True, False])
    labels = input_ids.roll(-1)
    sampled = logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    q = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    top_q, top_ids = q.topk(2, dim=-1)
    data = {
        "loss_mask": mask,
        "input_ids": input_ids,
        "advantages": torch.tensor([1.2, 0.0]),
        "logprobs": torch.tensor([sampled[0].item(), 0.0]),
        "prox_logp": torch.tensor([sampled[0].item(), 0.0]),
        "sampler_topk_ids": torch.cat([top_ids, torch.zeros((1, 2), dtype=torch.long)]),
        "sampler_topk_logprobs": torch.cat(
            [top_q.log(), torch.zeros((1, 2), dtype=torch.float64)]
        ),
    }
    cfg = SimpleNamespace(
        is_critic=False,
        ppo_n_minibatches=1,
        prox_logp_method="recompute",
        rejection_sampling=SimpleNamespace(level="token", action="mask", upper=5.0),
        use_decoupled_loss=True,
        temperature=1.0,
    )
    correction = clipped_ppo_score_centering_correction(
        logits, data, cfg, sampled_logprobs=sampled
    )
    assert correction.isfinite()
    assert torch.autograd.grad(correction, logits)[0][1].abs().sum() == 0
    data["prox_logp"][0] += 1e-3
    with pytest.raises(ValueError, match="prox/current logprob gap"):
        clipped_ppo_score_centering_correction(
            logits, data, cfg, sampled_logprobs=sampled
        )


def test_actual_fork_clipped_loss_plus_centering_cancels_constant_reward_drift(
    monkeypatch,
):
    """Enumerate all sampled labels, including upper-five rejected outcomes."""
    from types import SimpleNamespace
    from areal import utils
    from areal.api.cli_args import RejectionSamplingConfig
    from areal.utils.functional import ppo_actor_loss_fn
    from nibble.areal_loss import clipped_ppo_score_centering_correction

    monkeypatch.setattr(
        utils,
        "stats_tracker",
        SimpleNamespace(
            denominator=lambda **kwargs: None, stat=lambda *args, **kwargs: None
        ),
    )
    logits = torch.tensor([0.8, 0.15, 0.05], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([0.01, 0.89, 0.10], dtype=torch.float64)
    rows = logits[None, :].expand(3, -1)
    ids = torch.tensor([2, 0, 1])
    labels = ids.roll(-1)
    sampled = rows.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
    mask = torch.ones(3, dtype=torch.bool)
    rs = RejectionSamplingConfig(metric="ratio", upper=5)
    # Multiplying A by N*q simulates expectation over the categorical sampler.
    advantages = 3 * q
    data = dict(
        input_ids=ids,
        loss_mask=mask,
        logprobs=q.log(),
        prox_logp=sampled.detach(),
        advantages=advantages,
        sampler_topk_ids=torch.arange(3)[None, :].expand(3, -1),
        sampler_topk_logprobs=q.log()[None, :].expand(3, -1),
    )
    cfg = SimpleNamespace(
        is_critic=False,
        ppo_n_minibatches=1,
        prox_logp_method="recompute",
        rejection_sampling=rs,
        use_decoupled_loss=True,
        temperature=1.0,
    )
    base, stat = ppo_actor_loss_fn(
        sampled,
        sampled.detach(),
        q.log(),
        advantages,
        eps_clip=0.4,
        loss_mask=mask,
        rejection_sampling=rs,
    )
    assert not stat["behave_mask"][0]  # r=80 is rejected; do not reject C.
    base_grad = torch.autograd.grad(base, logits, retain_graph=True)[0]
    correction = clipped_ppo_score_centering_correction(
        rows, data, cfg, sampled_logprobs=sampled
    )
    centered_grad = torch.autograd.grad(base + correction, logits)[0]
    assert base_grad.abs().max() > 0.01
    assert centered_grad.abs().max() < 2e-7


@pytest.mark.parametrize("advantage_dtype", [torch.float32, torch.float64])
def test_clipped_correction_actual_tracker_preserves_loss_and_gradient(
    monkeypatch, advantage_dtype
):
    # Real tracker rejects double stats; telemetry must not change loss math.
    from areal import utils
    from areal.utils.stats_tracker import DistributedStatsTracker
    from nibble.areal_loss import clipped_ppo_score_centering_correction

    logits = torch.tensor(
        [[0.3, -0.1, 0.2, -0.5], [0.1, 0.2, -0.2, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    ids = torch.tensor([0, 1])
    sampled = logits.log_softmax(-1).gather(-1, ids.roll(-1)[:, None]).squeeze(-1)
    q = torch.tensor([[0.001, 0.2, 0.3, 0.499]], dtype=torch.float64)
    head_q, head_ids = q.topk(4, dim=-1)
    data = dict(
        loss_mask=torch.tensor([True, False]),
        input_ids=ids,
        advantages=torch.tensor([1.2, 0.0], dtype=advantage_dtype),
        logprobs=sampled.detach(),
        prox_logp=sampled.detach(),
        sampler_topk_ids=torch.cat([head_ids, torch.zeros((1, 4), dtype=torch.long)]),
        sampler_topk_logprobs=torch.cat([head_q.log(), torch.zeros((1, 4))]),
    )
    cfg = SimpleNamespace(
        is_critic=False,
        ppo_n_minibatches=1,
        prox_logp_method="recompute",
        rejection_sampling=SimpleNamespace(level="token", action="mask", upper=5.0),
        use_decoupled_loss=True,
        temperature=1.0,
    )
    monkeypatch.setattr(
        utils,
        "stats_tracker",
        SimpleNamespace(
            denominator=lambda **kwargs: None,
            stat=lambda **kwargs: None,
        ),
    )
    reference = clipped_ppo_score_centering_correction(
        logits, data, cfg, sampled_logprobs=sampled
    )
    reference_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    tracker = DistributedStatsTracker()
    monkeypatch.setattr(utils, "stats_tracker", tracker)
    got = clipped_ppo_score_centering_correction(
        logits, data, cfg, sampled_logprobs=sampled
    )
    got_grad = torch.autograd.grad(got, logits)[0]
    assert torch.equal(got, reference)
    assert torch.equal(got_grad, reference_grad)
    assert got.isfinite() and got_grad.isfinite().all()
    assert got_grad.abs().sum() > 0
    assert torch.equal(got_grad[1], torch.zeros_like(got_grad[1]))
    for key in [
        "nibble_sc_correction",
        "nibble_sc_q_head_mass",
        "nibble_sc_p_head_mass",
        "nibble_sc_tail_rho",
    ]:
        value = tracker.stats[key][0]
        assert value.dtype == torch.float32
        assert value.isfinite().all() and not value.requires_grad
    assert torch.equal(
        tracker.stats["nibble_sc_correction"][0], got.detach().float()[None]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_single_gather_packed_qwen_clipped_loss_parameter_gradient_parity(
    monkeypatch, dtype
):
    # Real packed actor, original clipped loss plus correction, overlapping IDs.
    from areal import utils
    from areal.utils.functional import ppo_actor_loss_fn
    from areal.api.cli_args import RejectionSamplingConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from nibble.areal_loss import clipped_ppo_score_centering_correction
    from nibble.score_centering import score_centering_loss

    monkeypatch.setattr(
        utils,
        "stats_tracker",
        SimpleNamespace(denominator=lambda **kwargs: None, stat=lambda **kwargs: None),
    )
    torch.manual_seed(71)
    config = Qwen2Config(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    actor = Qwen2ForCausalLM(config).to(dtype).eval()
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    mask = torch.tensor([False, True, True, False, False, True, True, False])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]])
    allowed = torch.zeros(8, 8, dtype=torch.bool)
    allowed[:4, :4] = torch.ones(4, 4, dtype=torch.bool).tril()
    allowed[4:, 4:] = torch.ones(4, 4, dtype=torch.bool).tril()
    attention = torch.zeros(1, 1, 8, 8, dtype=dtype).masked_fill(
        ~allowed[None, None], torch.finfo(dtype).min
    )
    labels = ids.roll(-1)
    heads = torch.stack((labels, (labels + 1) % 37, (labels + 2) % 37), -1)
    qlog = torch.tensor([0.001, 0.20, 0.25]).log().expand(8, -1)
    cfg = SimpleNamespace(
        is_critic=False,
        ppo_n_minibatches=1,
        prox_logp_method="recompute",
        rejection_sampling=RejectionSamplingConfig(metric="ratio", upper=5),
        use_decoupled_loss=True,
        temperature=1.0,
    )
    outputs = []
    for single in [False, True]:
        actor.zero_grad(set_to_none=True)
        logits = actor(
            input_ids=ids[None],
            position_ids=positions,
            attention_mask=attention,
            use_cache=False,
        ).logits[0]
        logits.retain_grad()
        sampled = logits.float().log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)
        data = dict(
            loss_mask=mask,
            input_ids=ids,
            advantages=torch.tensor([0.0, 1.0, -2.0, 0.0, 0.0, 1.5, -0.5, 0.0]),
            logprobs=sampled.detach() - 0.02,
            prox_logp=sampled.detach(),
            sampler_topk_ids=heads,
            sampler_topk_logprobs=qlog,
        )
        base, _ = ppo_actor_loss_fn(
            sampled,
            sampled.detach(),
            data["logprobs"],
            data["advantages"],
            eps_clip=0.4,
            loss_mask=mask,
            rejection_sampling=cfg.rejection_sampling,
        )
        if single:
            correction = clipped_ppo_score_centering_correction(
                logits, data, cfg, sampled_logprobs=sampled
            )
        else:
            selected = logits.gather(-1, labels[:, None]).squeeze(-1)[mask].float()
            h = logits.gather(-1, heads)[mask].float()
            head_logp = h - (selected - sampled[mask])[:, None]
            correction = score_centering_loss(
                None,
                heads[mask],
                qlog[mask],
                torch.zeros_like(sampled[mask]),
                data["advantages"][mask],
                train_head_logprobs=head_logp,
                sampled_sampler_logprob=data["logprobs"][mask],
                importance="ppo_mask",
                reduction="none",
            ).mean()
        loss = base + correction
        loss.backward()
        assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
        outputs.append(
            (loss.detach(), {n: p.grad.clone() for n, p in actor.named_parameters()})
        )
    assert torch.equal(outputs[0][0], outputs[1][0])
    for name, before in outputs[0][1].items():
        after = outputs[1][1][name]
        assert torch.isfinite(after).all(), name
        assert torch.equal(after, before), name
    assert sum(g.abs().sum() for g in outputs[1][1].values()) > 0


def test_single_gather_removes_one_dense_backward_allocation():
    # CPU profiler scopes gather change; this is not whole GPU peak proof.
    from torch.profiler import ProfilerActivity, profile

    x0 = torch.randn(32, 257)
    labels = torch.arange(32)[:, None]
    heads = torch.cat((labels, (labels + 1) % 257), -1)
    results = []
    for single in [False, True]:
        x = x0.clone().requires_grad_()
        sampled = x.log_softmax(-1).gather(-1, labels).squeeze(-1)
        if single:
            both = x.gather(-1, torch.cat((labels, heads), -1))
            selected, h = both[:, 0], both[:, 1:]
        else:
            selected = x.gather(-1, labels).squeeze(-1)
            h = x.gather(-1, heads)
        loss = (h - (selected - sampled)[:, None]).square().sum()
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
        ) as prof:
            loss.backward()
        scatters = [e for e in prof.events() if e.name == "aten::scatter_add_"]
        gathers = [e for e in prof.events() if e.name == "aten::gather_backward"]
        results.append((loss.detach(), x.grad, scatters, gathers))
    assert torch.equal(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
    assert [len(r[2]) for r in results] == [3, 2]
    assert all(e.input_shapes[0] == [32, 257] for r in results for e in r[2])
    assert (
        sum(e.cpu_memory_usage for e in results[0][3])
        - sum(e.cpu_memory_usage for e in results[1][3])
        == x0.numel() * x0.element_size()
    )
