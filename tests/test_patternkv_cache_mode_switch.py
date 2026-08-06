from __future__ import annotations

from types import SimpleNamespace

import pytest

from bench.aime24_int2_wave1 import stable_hash
from models.llama_patternkv import patternkv_cache_mode


def test_default_cache_mode_is_segmented_rolling(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_MODE", raising=False)
    monkeypatch.delenv("PATTERNKV_CACHE_PATH", raising=False)
    assert patternkv_cache_mode(SimpleNamespace()) == "segmented_rolling"


def test_explicit_cache_modes(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_MODE", raising=False)
    assert patternkv_cache_mode(SimpleNamespace(patternkv_cache_mode="legacy_tuple_chunked")) == "legacy_tuple_chunked"
    assert patternkv_cache_mode(SimpleNamespace(patternkv_cache_mode="segmented_chunked")) == "segmented_chunked"
    assert patternkv_cache_mode(SimpleNamespace(patternkv_cache_mode="segmented_rolling")) == "segmented_rolling"


def test_cache_mode_env_overrides_config(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_CACHE_MODE", "segmented_chunked")
    assert patternkv_cache_mode(SimpleNamespace(patternkv_cache_mode="segmented_rolling")) == "segmented_chunked"


def test_legacy_path_alias_maps_to_legacy_chunked(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_MODE", raising=False)
    monkeypatch.setenv("PATTERNKV_CACHE_PATH", "legacy")
    assert patternkv_cache_mode(SimpleNamespace()) == "legacy_tuple_chunked"


def test_invalid_cache_mode_rejected(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_MODE", raising=False)
    with pytest.raises(ValueError):
        patternkv_cache_mode(SimpleNamespace(patternkv_cache_mode="segmented_recentish"))


def test_cache_mode_enters_config_hash() -> None:
    chunked = {"method": "patternkv", "patternkv_cache_mode": "segmented_chunked", "seed": 42}
    rolling = {"method": "patternkv", "patternkv_cache_mode": "segmented_rolling", "seed": 42}
    assert stable_hash(chunked) != stable_hash(rolling)
