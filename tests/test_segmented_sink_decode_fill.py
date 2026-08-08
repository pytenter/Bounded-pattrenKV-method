from __future__ import annotations

import torch

from models.segmented_cache import append_decode, build_cache_from_prefill, cache_segment_stats, reconstruct_full_k, validate_cache


def make_kv(tokens: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(offset, offset + 2 * tokens * 16, dtype=torch.float16).reshape(1, 2, tokens, 16)
    return base / 37, (base + 5) / 41


def test_multi_token_append_splits_between_sink_and_recent() -> None:
    key, value = make_kv(3)
    append_k, append_v = make_kv(4, offset=1000)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    append_decode(cache, append_k, append_v)

    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 5
    assert stats["recent_tokens"] == 2
    validate_cache(cache)

    expected = torch.cat([key, append_k], dim=2)
    assert reconstruct_full_k(cache).shape == expected.shape
    assert torch.equal(cache.sink_k, expected[:, :, :5, :])
    assert torch.equal(cache.recent_k, expected[:, :, 5:, :])


def test_sink_fill_precedes_pending_and_packing() -> None:
    key, value = make_kv(3)
    append_k, append_v = make_kv(6, offset=2000)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=2, group_size=4, k_bits=2, v_bits=2)
    append_decode(cache, append_k, append_v)

    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 5
    assert stats["recent_tokens"] == 2
    assert stats["packed_history_tokens"] == 0
    assert stats["pending_history_tokens"] == 2
    validate_cache(cache)


def test_exactly_filled_sink_serializes_round_trip() -> None:
    key, value = make_kv(3)
    append_k, append_v = make_kv(2, offset=3000)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    append_decode(cache, append_k, append_v)
    validate_cache(cache)
    assert cache_segment_stats(cache)["sink_tokens"] == 5
