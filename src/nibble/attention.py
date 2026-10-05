"""Sequence-isolated causal attention for a packed text microbatch."""

import torch


def packed_causal_mask(position_ids: torch.Tensor) -> torch.Tensor:
    """Return boolean SDPA mask [1,1,S,S]; True allows attention.

    AReaL packs independent sequences into one batch row and resets positions
    to zero at each boundary. A token can see only earlier tokens in its own
    sequence. Transport padding starts new inactive sequences and cannot leak
    into the preceding example. This adapter supports text with TP=SP=1.
    """
    if position_ids.ndim != 2 or position_ids.shape[0] != 1:
        raise ValueError("packed text positions must have shape [1, S]")
    positions = position_ids[0]
    sequence_ids = (positions == 0).cumsum(dim=0)
    index = torch.arange(positions.numel(), device=positions.device)
    same_sequence = sequence_ids[:, None] == sequence_ids[None, :]
    causal = index[:, None] >= index[None, :]
    return (same_sequence & causal)[None, None]
