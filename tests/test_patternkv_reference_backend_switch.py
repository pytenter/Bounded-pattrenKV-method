from __future__ import annotations

from types import SimpleNamespace

from models.llama_patternkv import patternkv_cache_mode, patternkv_equivalence_backend


def test_backend_switch_does_not_change_cache_mode(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_EQUIVALENCE_BACKEND", "reference")
    config = SimpleNamespace(patternkv_cache_mode="segmented_chunked", patternkv_cache_path="segmented")
    assert patternkv_equivalence_backend(config) == "reference"
    assert patternkv_cache_mode(config) == "segmented_chunked"
