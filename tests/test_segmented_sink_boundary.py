from __future__ import annotations

import torch

from models.segmented_cache import (
    append_decode,
    build_cache_from_prefill,
    cache_segment_stats,
    reconstruct_full_k,
    serialize_cache,
    deserialize_cache,
    validate_cache,
)


def make_kv(tokens: int, *, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(offset, offset + 2 * tokens * 16, dtype=torch.float16).reshape(1, 2, tokens, 16)
    return base / 101, (base + 3) / 103


def test_decode_fills_sink_one_token_at_a_time() -> None:
    key, value = make_kv(3)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    assert cache_segment_stats(cache)["sink_tokens"] == 3
    validate_cache(cache)

    for expected_sink, expected_recent in [(4, 0), (5, 0), (5, 1)]:
        next_k, next_v = make_kv(1, offset=1000 + expected_sink)
        append_decode(cache, next_k, next_v)
        stats = cache_segment_stats(cache)
        assert stats["sink_tokens"] == expected_sink
        assert stats["recent_tokens"] == expected_recent
        validate_cache(cache)


def test_decode_after_exactly_filled_sink_goes_to_recent() -> None:
    key, value = make_kv(5)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    next_k, next_v = make_kv(1, offset=2000)
    append_decode(cache, next_k, next_v)
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 5
    assert stats["recent_tokens"] == 1
    validate_cache(cache)


def test_prefill_longer_than_sink_keeps_existing_behavior() -> None:
    key, value = make_kv(8)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    original_sink = cache.sink_k.clone()
    next_k, next_v = make_kv(1, offset=3000)
    append_decode(cache, next_k, next_v)
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 5
    assert stats["recent_tokens"] == 4
    assert torch.equal(cache.sink_k, original_sink)
    validate_cache(cache)


def test_partially_filled_sink_serializes_round_trip() -> None:
    key, value = make_kv(3)
    cache = build_cache_from_prefill(key, value, sink_length=5, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    restored = deserialize_cache(serialize_cache(cache))
    assert cache_segment_stats(restored) == cache_segment_stats(cache)
    assert torch.equal(reconstruct_full_k(restored), reconstruct_full_k(cache))
    validate_cache(restored)
