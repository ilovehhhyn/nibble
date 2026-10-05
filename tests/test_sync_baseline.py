import pytest
import torch

from nibble.sync_baseline import center_group_rewards, tis_weights


def test_group_rewards_are_centered_by_mean_without_variance_scaling():
    assert center_group_rewards([[1.0, 3.0], [4.0, 4.0, 10.0]]) == [
        -1.0,
        1.0,
        -2.0,
        -2.0,
        4.0,
    ]


def test_tis_weights_are_detached_and_clipped_at_two():
    current = torch.tensor([0.0, 3.0], requires_grad=True)
    behavior = torch.tensor([0.0, 0.0])
    weights = tis_weights(current, behavior)
    assert torch.allclose(weights, torch.tensor([1.0, 2.0]))
    assert not weights.requires_grad


def test_tis_can_be_disabled_and_requires_aligned_tokens():
    current = torch.tensor([2.0])
    behavior = torch.tensor([-4.0])
    assert torch.equal(tis_weights(current, behavior, enabled=False), torch.ones(1))
    with pytest.raises(ValueError, match="align"):
        tis_weights(torch.zeros(1), torch.zeros(2))


def test_empty_group_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        center_group_rewards([[]])
