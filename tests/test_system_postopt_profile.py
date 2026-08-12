import os

import torch

from bench.postopt_system_reprofile import GROUPS, component_rows_from_snapshot
from models.llama_patternkv import patternkv_mixed_v_backend
from models.segmented_cache import normalize_cache_backend
from quant.matmul import patternkv_gqa_v_backend, patternkv_page_v_reader_backend
from quant.patternkv_profile import profile_range, profile_snapshot, record_counter, reset_profile


def test_system_profile_disabled_by_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_SYSTEM_PROFILE", raising=False)
    monkeypatch.delenv("PATTERNKV_PROFILE", raising=False)
    reset_profile()
    with profile_range("disabled_by_default"):
        _ = sum(range(8))
    assert profile_snapshot(reset=True) == {}


def test_system_profile_does_not_change_output(monkeypatch):
    x = torch.arange(16, dtype=torch.float32)
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "0")
    reset_profile()
    with profile_range("simple"):
        y0 = x.sin() * 2.0
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    with profile_range("simple"):
        y1 = x.sin() * 2.0
    torch.testing.assert_close(y0, y1)


def test_failed_gqa_backend_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
    monkeypatch.delenv("PATTERNKV_MIXED_V_BACKEND", raising=False)
    assert patternkv_mixed_v_backend() == "fused"


def test_paged_cache_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_BACKEND", raising=False)
    assert normalize_cache_backend(None) == "contiguous"


def test_page_native_reader_not_default(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_profile_component_counters_reset(monkeypatch):
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    record_counter("selector_total", calls=2, tokens=128)
    assert profile_snapshot(reset=False)["selector_total"]["calls"] == 2
    reset_profile()
    assert profile_snapshot(reset=False) == {}


def test_profile_component_counters_increment(monkeypatch):
    monkeypatch.setenv("PATTERNKV_SYSTEM_PROFILE", "1")
    reset_profile()
    record_counter("mixed_v_fused_attention", calls=3, tokens=384)
    snap = profile_snapshot(reset=True)
    assert snap["mixed_v_fused_attention"]["calls"] == 3
    assert snap["mixed_v_fused_attention"]["tokens"] == 384


def test_no_double_count_in_component_report():
    assert "decode_decoder_model_forward" not in {item for values in GROUPS.values() for item in values}
    assert "mixed_v_v2_compute" not in GROUPS["mixed_v"]
    assert "mixed_v_v4_compute" not in GROUPS["mixed_v"]
    result = {
        "context_tokens": 8192,
        "decode_tokens": 2,
        "decode_total_ms": 1.0,
        "profile_snapshot": {
            "decode_decoder_model_forward": {"total_us": 1000.0, "calls": 2.0, "tokens": 0.0},
            "mixed_v_fused_attention": {"total_us": 200.0, "calls": 2.0, "tokens": 1024.0},
            "mixed_v_v2_compute": {"total_us": 120.0, "calls": 2.0, "tokens": 768.0},
            "mixed_v_v4_compute": {"total_us": 80.0, "calls": 2.0, "tokens": 256.0},
        },
    }
    rows = component_rows_from_snapshot(result)
    mixed = next(row for row in rows if row["component"] == "mixed_v")
    assert mixed["profile_total_us"] == 200.0
    assert mixed["share_percent"] == 20.0
