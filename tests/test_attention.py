"""Packed Qwen2 SDPA must match separate sequences, including gradients."""

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from nibble.attention import packed_causal_mask


def test_packed_sdpa_preserves_sequence_logits_and_gradients():
    torch.manual_seed(10)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        attention_dropout=0.0,
    )
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    sequences = [torch.tensor([[1, 3, 5, 7]]), torch.tensor([[2, 4, 6]])]
    ids = torch.cat(sequences, dim=-1)
    positions = torch.tensor([[0, 1, 2, 3, 0, 1, 2]])
    reference = torch.cat([model(x, use_cache=False).logits for x in sequences], dim=1)
    mask = packed_causal_mask(positions)
    packed = model(
        ids,
        position_ids=positions,
        attention_mask={"full_attention": mask, "sliding_attention": mask},
        use_cache=False,
    ).logits
    torch.testing.assert_close(packed, reference, atol=1e-6, rtol=1e-6)
    parameter = model.model.layers[0].self_attn.q_proj.weight
    expected = torch.autograd.grad(reference.square().sum(), parameter)[0]
    actual = torch.autograd.grad(packed.square().sum(), parameter)[0]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert not mask[0, 0, 4, :4].any()
    assert not mask[0, 0, 1, 2:].any()
