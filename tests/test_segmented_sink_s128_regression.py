from __future__ import annotations

from models.segmented_cache import append_decode, build_cache_from_prefill, cache_segment_stats, deserialize_cache, serialize_cache, validate_cache

from tests.test_segmented_cache_semantics import make_kv


def test_s128_prefill_117_fills_sink_before_recent() -> None:
    key, value = make_kv(117)
    cache = build_cache_from_prefill(key, value, sink_length=128, recent_length=128, group_size=16, k_bits=2, v_bits=2)
    assert cache_segment_stats(cache)["sink_tokens"] == 117

    for step in range(11):
        next_k, next_v = make_kv(1)
        append_decode(cache, next_k + step, next_v + step)
        stats = cache_segment_stats(cache)
        assert stats["sink_tokens"] == 118 + step
        assert stats["recent_tokens"] == 0
        validate_cache(cache)

    next_k, next_v = make_kv(1)
    append_decode(cache, next_k + 12, next_v + 12)
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 128
    assert stats["recent_tokens"] == 1
    validate_cache(cache)


def test_s128_sink_plus_recent_serializes_round_trip() -> None:
    key, value = make_kv(117)
    cache = build_cache_from_prefill(key, value, sink_length=128, recent_length=128, group_size=16, k_bits=2, v_bits=2)
    append_k, append_v = make_kv(12)
    append_decode(cache, append_k, append_v)
    restored = deserialize_cache(serialize_cache(cache))
    assert cache_segment_stats(restored) == cache_segment_stats(cache)
    validate_cache(restored)
