from __future__ import annotations

import torch

from models.llama_patternkv import patternkv_use_bi_prefill_kproj
from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_kproj_available,
    batch_invariant_kproj_counters,
    flag_enabled,
    record_bi_prefill_kproj,
    record_normal_decode_kproj_call,
    reset_batch_invariant_kproj_counters,
    selected_backend,
)


def test_bi_kproj_prefill_dispatch_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(None) is True


def test_bi_kproj_decode_dispatch_disabled(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(("patternkv_segmented_cache_v1",)) is False


def test_bi_kproj_flag_disabled_baseline(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_BATCH_INVARIANT_KPROJ", raising=False)
    assert flag_enabled() is False
    assert patternkv_use_bi_prefill_kproj(None) is False


def test_prefill_detection_initial_cache(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(None) is True


def test_prefill_detection_decode_cache(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(("patternkv_segmented_cache_v1", None)) is False


def test_prefill_dispatch_no_serial_loop() -> None:
    reset_batch_invariant_kproj_counters()
    record_bi_prefill_kproj(rows=1024, kernel_launches=1)
    counters = batch_invariant_kproj_counters()
    assert counters["bi_prefill_serial_dispatches"] == 0
    assert counters["bi_prefill_fallback_calls"] == 0


def test_prefill_dispatch_v2_backend(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_KPROJ_BACKEND", "v2")
    assert selected_backend() == "v2"


def test_prefill_runtime_counter_semantics() -> None:
    reset_batch_invariant_kproj_counters()
    record_bi_prefill_kproj(rows=4)
    record_normal_decode_kproj_call()
    counters = batch_invariant_kproj_counters()
    assert counters["bi_prefill_kproj_calls"] == 1
    assert counters["bi_prefill_kproj_rows"] == 4
    assert counters["bi_decode_kproj_calls"] == 0
    assert counters["normal_decode_kproj_calls"] == 1


def test_prefill_runtime_kernel_counter_if_available() -> None:
    if not batch_invariant_kproj_available():
        return
    x = torch.randn((2, 3, 16), device="cuda", dtype=torch.float16)
    w = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    reset_batch_invariant_kproj_counters()
    batch_invariant_k_projection(x, w, backend="v2")
    counters = batch_invariant_kproj_counters()
    assert counters["bi_kproj_v2_calls"] == 1
    assert counters["bi_kproj_serial_request_dispatches"] == 0
    assert counters["bi_kproj_fallback_calls"] == 0
