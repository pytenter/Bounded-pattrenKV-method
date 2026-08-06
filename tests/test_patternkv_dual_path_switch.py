from __future__ import annotations

from types import SimpleNamespace

from bench.aime24_int2_wave1 import stable_hash
from models.llama_patternkv import patternkv_cache_path


def test_default_patternkv_cache_path_is_segmented(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_PATH", raising=False)
    assert patternkv_cache_path(SimpleNamespace()) == "segmented"


def test_explicit_patternkv_cache_path_switch(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_CACHE_PATH", raising=False)
    assert patternkv_cache_path(SimpleNamespace(patternkv_cache_path="legacy")) == "legacy"
    assert patternkv_cache_path(SimpleNamespace(patternkv_cache_path="segmented")) == "segmented"


def test_env_patternkv_cache_path_overrides_config(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_CACHE_PATH", "legacy")
    assert patternkv_cache_path(SimpleNamespace(patternkv_cache_path="segmented")) == "legacy"


def test_cache_path_enters_config_hash() -> None:
    base = {"method": "patternkv", "patternkv_cache_path": "segmented", "seed": 42}
    legacy = {"method": "patternkv", "patternkv_cache_path": "legacy", "seed": 42}
    assert stable_hash(base) != stable_hash(legacy)

