"""Critic recomputation must preserve packed SDPA values and clipped gradients."""

import copy

import torch
from transformers import AutoModelForTokenClassification, Qwen2Config

from nibble.attention import packed_causal_mask


def test_critic_checkpointing_preserves_original_loss_and_gradients():
    from areal.utils.functional import ppo_critic_loss_fn

    torch.manual_seed(13)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0.0,
        num_labels=1,
    )
    config._attn_implementation = "sdpa"
    original = AutoModelForTokenClassification.from_config(config).train()
    for module in original.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0  # Same disable_dropout behavior as the original engine.
    checkpointed = copy.deepcopy(original)
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    assert checkpointed.is_gradient_checkpointing
    ids = torch.tensor([[1, 3, 5, 7, 2, 4, 6]])
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    mask = packed_causal_mask(positions)
    loss_mask = torch.tensor([[False, True, True, True, False, True, True]])
    old_values = torch.tensor([[0.0, 0.7, -0.8, 0.05, 0.0, -0.1, 0.2]])
    returns = torch.tensor([[0.0, -0.3, 0.4, 0.5, 0.0, -0.6, 0.3]])

    def update(model):
        saved_bytes = 0

        def pack(tensor):
            nonlocal saved_bytes
            saved_bytes += tensor.numel() * tensor.element_size()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            value = model(
                ids,
                position_ids=positions,
                attention_mask={"full_attention": mask, "sliding_attention": mask},
                use_cache=False,
            ).logits.squeeze(-1)
            loss, stat = ppo_critic_loss_fn(
                value=value,
                old_value=old_values,
                target_value=returns,
                value_eps_clip=0.5,
                loss_mask=loss_mask,
            )
        loss.backward()
        grads = {name: p.grad for name, p in model.named_parameters()}
        assert all(g is not None and torch.isfinite(g).all() for g in grads.values())
        assert any(g.count_nonzero() for g in grads.values())
        assert stat["clip_mask"].any()  # Exercise the original clipped branch.
        return value.detach(), loss.detach(), grads, saved_bytes

    expected_value, expected_loss, expected_grads, original_bytes = update(original)
    actual_value, actual_loss, actual_grads, checkpointed_bytes = update(checkpointed)
    torch.testing.assert_close(actual_value, expected_value, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual_loss, expected_loss, atol=1e-6, rtol=1e-6)
    for name, gradient in actual_grads.items():
        torch.testing.assert_close(gradient, expected_grads[name], atol=1e-6, rtol=1e-5)
    assert checkpointed_bytes < original_bytes
