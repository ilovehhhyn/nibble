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


def test_cached_tokenizer_prompt_ids_and_benchmark_dataset_cpu(monkeypatch):
    from pathlib import Path

    from transformers import AutoTokenizer
    from nibble.native import benchmark_dataset
    from nibble.sync_baseline import _prompt_ids

    repo_root = Path(__file__).resolve().parents[1]
    split_path = repo_root / "work" / "data" / "splits.json"
    cache_root = repo_root.parent / "hf_cache"
    model_path = (
        cache_root
        / "hub"
        / "models--Qwen--Qwen2.5-1.5B-Instruct"
        / "snapshots"
        / "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    )
    if not split_path.is_file() or not model_path.is_dir():
        pytest.skip("cached Qwen2.5-1.5B tokenizer or benchmark split is unavailable")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    messages = [{"role": "user", "content": "hello"}]
    actual = _prompt_ids(tokenizer, messages, max_prompt_tokens=256)
    expected = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    assert isinstance(actual, list)
    assert actual == expected

    monkeypatch.setenv("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    monkeypatch.setenv("NIBBLE_SPLITS", str(split_path))
    dataset = benchmark_dataset(tokenizer=tokenizer, count=136, max_prompt_tokens=1024)
    assert len(dataset) == 136
    lengths = [len(_prompt_ids(tokenizer, row["messages"], 1024)) for row in dataset]
    assert max(lengths) <= 1024


def test_prompt_ids_requests_a_list_from_tokenizer():
    from nibble.sync_baseline import _prompt_ids

    class MappingByDefaultTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            return [42, 43, 44]

    assert _prompt_ids(MappingByDefaultTokenizer(), [], 3) == [42, 43, 44]
