from __future__ import annotations

from models.segmented_cache import CHUNKED_CACHE_MODE, ROLLING_CACHE_MODE, normalize_cache_mode, segment_lengths


def test_rolling_and_chunked_cache_config_separation() -> None:
    rolling = segment_lengths(total_tokens=320, sink_length=64, recent_length=256)
    chunked = segment_lengths(total_tokens=320, sink_length=0, recent_length=0)
    assert rolling["sink_tokens"] == 64
    assert rolling["recent_tokens"] == 256
    assert chunked["sink_tokens"] == 0
    assert chunked["recent_tokens"] == 0
    assert normalize_cache_mode("chunked") == CHUNKED_CACHE_MODE
    assert normalize_cache_mode("rolling") == ROLLING_CACHE_MODE
