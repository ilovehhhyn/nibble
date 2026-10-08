import pytest
import torch

pytest.importorskip("areal")

from areal.api.cli_args import GenerationHyperparameters, MicroBatchSpec
from areal.api.io_struct import HttpGenerationResult, ModelRequest
from areal.engine.sglang_remote import SGLangBackend
from areal.utils.data import (
    concat_batch,
    pack_tensor_dict,
    pad_packed_tensor_dict,
    split_batch,
    split_padded_tensor_dict_into_mb_list,
)


def _top_logprob_rows(num_tokens: int, k: int = 128):
    return [
        [[-float(rank + 1), rank, None] for rank in range(k)] for _ in range(num_tokens)
    ]


def _generation_response(num_tokens: int = 2):
    return {
        "meta_info": {
            "finish_reason": {"type": "stop"},
            "output_token_logprobs": [(-0.5, i + 100, None) for i in range(num_tokens)],
            "output_top_logprobs": _top_logprob_rows(num_tokens),
        }
    }


def test_score_centering_request_requests_top128_and_requires_supported_sampling(
    monkeypatch,
):
    monkeypatch.setenv("NIBBLE_SCORE_CENTERING", "1")
    monkeypatch.setenv("NIBBLE_OBJECTIVE", "group_pg_tis")
    backend = SGLangBackend()
    request = ModelRequest(
        input_ids=[1, 2],
        gconfig=GenerationHyperparameters(top_k=-1),
    )

    http_request = backend.build_generation_request(request, with_lora=False, version=0)

    assert http_request.payload["top_logprobs_num"] == 128
    assert http_request.payload["sampling_params"]["temperature"] == 1.0
    assert http_request.payload["sampling_params"]["top_p"] == 1.0
    assert http_request.payload["sampling_params"]["top_k"] == -1
    assert http_request.payload["sampling_params"]["repetition_penalty"] == 1.0

    request.gconfig.top_p = 0.9
    with pytest.raises(ValueError, match="NIBBLE_SCORE_CENTERING=1 requires sampling"):
        backend.build_generation_request(request, with_lora=False, version=0)


def test_score_centering_response_parses_aligned_top_logprobs(monkeypatch):
    monkeypatch.setenv("NIBBLE_SCORE_CENTERING", "1")

    result = SGLangBackend().parse_generation_response(_generation_response())

    assert result.output_tokens == [100, 101]
    assert result.sampler_topk_ids is not None
    assert result.sampler_topk_logprobs is not None
    assert result.sampler_topk_ids[1] == list(range(128))
    assert result.sampler_topk_logprobs[0] == [-float(i + 1) for i in range(128)]


def test_score_centering_missing_or_short_head_fails_loudly(monkeypatch):
    monkeypatch.setenv("NIBBLE_SCORE_CENTERING", "1")
    response = _generation_response()
    del response["meta_info"]["output_top_logprobs"]
    with pytest.raises(ValueError, match="did not return token-aligned"):
        SGLangBackend().parse_generation_response(response)

    response = _generation_response()
    response["meta_info"]["output_top_logprobs"][0].pop()
    with pytest.raises(ValueError, match="exactly 128"):
        SGLangBackend().parse_generation_response(response)


def test_score_centering_fields_survive_batch_concat_and_split():
    short_len, long_len, k = 3, 5, 128
    trajectories = []
    for seq_len in (short_len, long_len):
        trajectories.append(
            {
                "input_ids": torch.arange(seq_len).unsqueeze(0),
                "attention_mask": torch.ones((1, seq_len), dtype=torch.bool),
                "sampler_topk_ids": torch.full((1, seq_len, k), 17, dtype=torch.int64),
                "sampler_topk_logprobs": torch.full(
                    (1, seq_len, k), -2.0, dtype=torch.float32
                ),
            }
        )

    batched, meta = concat_batch(trajectories)
    split = split_batch(batched, meta)

    assert batched["sampler_topk_ids"].shape == (2, long_len, k)
    assert [item["sampler_topk_ids"].shape for item in split] == [
        (1, short_len, k),
        (1, long_len, k),
    ]
    assert torch.equal(
        split[0]["sampler_topk_ids"], trajectories[0]["sampler_topk_ids"]
    )
    assert torch.equal(
        split[0]["sampler_topk_logprobs"], trajectories[0]["sampler_topk_logprobs"]
    )


def test_http_generation_result_rejects_missing_rows():
    with pytest.raises(ValueError, match="rows must match"):
        HttpGenerationResult(
            output_tokens=[7],
            output_logprobs=[-0.1],
            stop_reason="stop",
            sampler_topk_ids=[],
            sampler_topk_logprobs=[],
        )


def test_score_centering_fields_follow_microbatch_packing_and_padding():
    data = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
        "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        "sampler_topk_ids": torch.arange(2 * 3 * 128).reshape(2, 3, 128),
        "sampler_topk_logprobs": torch.zeros((2, 3, 128), dtype=torch.float32),
    }

    mb_list = split_padded_tensor_dict_into_mb_list(
        data, MicroBatchSpec(n_mbs=2), sync_mbs=False
    )
    assert len(mb_list.mbs) == 2
    for microbatch in mb_list.mbs:
        assert microbatch["sampler_topk_ids"].ndim == 3
        packed = pack_tensor_dict(microbatch)
        assert packed["sampler_topk_ids"].shape == (3, 128)
        padded, pad_length, _, _ = pad_packed_tensor_dict(packed, pad_to_length=256)
        assert pad_length == 253
        assert padded["sampler_topk_ids"].shape == (256, 128)
        assert padded["sampler_topk_logprobs"].shape == (256, 128)


def test_interaction_exports_sampler_heads_and_zero_prompt_rows():
    from areal.api.io_struct import ModelResponse
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    response = ModelResponse(
        input_tokens=[10, 11],
        output_tokens=[12, 13],
        output_logprobs=[-0.3, -0.4],
        sampler_topk_ids=[[2, 3], [4, 5]],
        sampler_topk_logprobs=[[-0.2, -1.6], [-0.5, -1.1]],
    )
    interaction = InteractionWithTokenLogpReward(model_response=response)
    data = interaction.to_tensor_dict()
    assert data["sampler_topk_ids"].tolist() == [[[0, 0], [0, 0], [2, 3], [4, 5]]]
    assert torch.allclose(
        data["sampler_topk_logprobs"],
        torch.tensor([[[0.0, 0.0], [0.0, 0.0], [-0.2, -1.6], [-0.5, -1.1]]]),
    )
