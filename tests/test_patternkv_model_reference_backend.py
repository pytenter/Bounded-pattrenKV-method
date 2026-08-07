from __future__ import annotations

from types import SimpleNamespace

from models.llama_patternkv import patternkv_equivalence_backend


def test_patternkv_equivalence_backend_defaults_to_production(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_EQUIVALENCE_BACKEND", raising=False)
    monkeypatch.delenv("PATTERNKV_FORCE_REFERENCE_ATTENTION", raising=False)
    assert patternkv_equivalence_backend(SimpleNamespace()) == "production"


def test_patternkv_equivalence_backend_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_EQUIVALENCE_BACKEND", "reference")
    assert patternkv_equivalence_backend(SimpleNamespace()) == "reference"
