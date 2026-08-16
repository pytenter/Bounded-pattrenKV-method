from __future__ import annotations

import torch

from models.llama_patternkv import patternkv_use_bi_kproj, patternkv_use_bi_prefill_kproj, patternkv_use_bi_prefill_vproj, patternkv_use_bi_vproj
from bench.run_bi_vproj_cost_benefit import classify_cost_benefit, drift_reduction_ratio, prefill_overhead_percent
from quant.batch_invariant_kproj import (
    BI_KV_PREFILL_PROJ_MODE,
    BI_K_PREFILL_PROJ_MODE,
    NORMAL_PREFILL_PROJ_MODE,
    batch_invariant_k_projection,
    batch_invariant_kproj_available,
    batch_invariant_kproj_counters,
    flag_enabled,
    patternkv_mode_aware_equivalence_policy,
    patternkv_prefill_projection_mode,
    patternkv_prefill_projection_mode_description,
    patternkv_prefill_projection_mode_policy,
    prefill_proj_mode,
    record_bi_prefill_kproj,
    record_bi_prefill_vproj,
    record_normal_decode_kproj_call,
    record_normal_decode_vproj_call,
    record_normal_prefill_kproj_call,
    record_normal_prefill_vproj_call,
    recommended_patternkv_serving_prefill_proj_mode,
    reset_batch_invariant_kproj_counters,
    selected_backend,
    strict_patternkv_prefill_proj_mode,
)


def test_bi_kproj_prefill_dispatch_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(None) is True


def test_bi_kproj_decode_dispatch_enabled(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_kproj(("patternkv_segmented_cache_v1",)) is True


def test_bi_kproj_flag_disabled_baseline(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_PREFILL_PROJ_MODE", raising=False)
    monkeypatch.delenv("PATTERNKV_BATCH_INVARIANT_KPROJ", raising=False)
    assert flag_enabled() is False
    assert patternkv_use_bi_prefill_kproj(None) is False


def test_prefill_detection_initial_cache(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(None) is True


def test_prefill_detection_decode_cache_uses_bi_kproj(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_use_bi_prefill_kproj(("patternkv_segmented_cache_v1", None)) is False
    assert patternkv_use_bi_kproj(("patternkv_segmented_cache_v1", None)) is True


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


def test_prefill_proj_mode_normal(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "normal")
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert prefill_proj_mode() == "normal"
    assert patternkv_use_bi_prefill_kproj(None) is False
    assert patternkv_use_bi_prefill_vproj(None) is False


def test_prefill_proj_mode_bi_k(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_k")
    assert prefill_proj_mode() == "bi_k"
    assert patternkv_use_bi_prefill_kproj(None) is True
    assert patternkv_use_bi_prefill_vproj(None) is False


def test_prefill_proj_mode_bi_kv(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_kv")
    assert prefill_proj_mode() == "bi_kv"
    assert patternkv_use_bi_prefill_kproj(None) is True
    assert patternkv_use_bi_prefill_vproj(None) is True


def test_prefill_proj_mode_legacy_bi_k(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_PREFILL_PROJ_MODE", raising=False)
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert prefill_proj_mode() == "bi_k"


def test_prefill_proj_mode_default_normal(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_PREFILL_PROJ_MODE", raising=False)
    monkeypatch.delenv("PATTERNKV_BATCH_INVARIANT_KPROJ", raising=False)
    assert patternkv_prefill_projection_mode() == NORMAL_PREFILL_PROJ_MODE


def test_prefill_proj_mode_legacy_flag_maps_to_bi_k(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_PREFILL_PROJ_MODE", raising=False)
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_prefill_projection_mode() == BI_K_PREFILL_PROJ_MODE


def test_prefill_proj_mode_explicit_normal_overrides_legacy(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "normal")
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_prefill_projection_mode() == NORMAL_PREFILL_PROJ_MODE


def test_prefill_proj_mode_explicit_bi_k(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_k")
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "0")
    assert patternkv_prefill_projection_mode() == BI_K_PREFILL_PROJ_MODE


def test_prefill_proj_mode_explicit_bi_kv(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_kv")
    monkeypatch.setenv("PATTERNKV_BATCH_INVARIANT_KPROJ", "1")
    assert patternkv_prefill_projection_mode() == BI_KV_PREFILL_PROJ_MODE


def test_prefill_proj_mode_invalid_rejected(monkeypatch) -> None:
    for value in ("foo", "strict", "deterministic", "1", "true"):
        monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", value)
        try:
            patternkv_prefill_projection_mode()
        except ValueError as exc:
            assert "normal" in str(exc)
            assert "bi_k" in str(exc)
            assert "bi_kv" in str(exc)
        else:
            raise AssertionError(f"invalid mode was accepted: {value}")


def test_recommended_serving_mode_is_bi_k() -> None:
    assert recommended_patternkv_serving_prefill_proj_mode() == BI_K_PREFILL_PROJ_MODE


def test_strict_mode_is_bi_kv() -> None:
    assert strict_patternkv_prefill_proj_mode() == BI_KV_PREFILL_PROJ_MODE


def test_normal_mode_dispatch_contract() -> None:
    desc = patternkv_prefill_projection_mode_description("normal")
    assert desc["prefill_k"] == "normal"
    assert desc["prefill_v"] == "normal"
    assert desc["decode_k"] == "normal"
    assert desc["decode_v"] == "normal"


def test_bi_k_mode_dispatch_contract() -> None:
    desc = patternkv_prefill_projection_mode_description("bi_k")
    assert desc["prefill_k"] == "bi_v2"
    assert desc["prefill_v"] == "normal"
    assert desc["decode_k"] == "bi_v2"
    assert desc["decode_v"] == "normal"


def test_bi_kv_mode_dispatch_contract() -> None:
    desc = patternkv_prefill_projection_mode_description("bi_kv")
    assert desc["prefill_k"] == "bi_v2"
    assert desc["prefill_v"] == "bi_v2"
    assert desc["decode_k"] == "bi_v2"
    assert desc["decode_v"] == "bi_v2"


def test_bi_kv_decode_uses_bi_kv(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_kv")
    past = ("patternkv_segmented_cache_v1", None)
    assert patternkv_use_bi_prefill_kproj(past) is False
    assert patternkv_use_bi_kproj(past) is True
    assert patternkv_use_bi_prefill_vproj(past) is False
    assert patternkv_use_bi_vproj(past) is True


def test_mode_aware_equivalence_policy() -> None:
    policy = patternkv_mode_aware_equivalence_policy()
    assert policy["normal"]["batch_state_equivalence"] == "not_promised"
    assert policy["bi_k"]["v_centroid"] == "numerical_metric"
    assert policy["bi_kv"]["compressed_kv_state"] == "request_local_exact"
    mode_policy = patternkv_prefill_projection_mode_policy()
    assert mode_policy["historical_default"] == "normal"
    assert mode_policy["recommended_serving_mode"] == "bi_k"
    assert mode_policy["strict_mode"] == "bi_kv"


def test_bi_vproj_prefill_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_kv")
    assert patternkv_use_bi_prefill_vproj(None) is True


def test_bi_vproj_decode_not_used(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_k")
    assert patternkv_use_bi_prefill_kproj(("patternkv_segmented_cache_v1",)) is False
    assert patternkv_use_bi_kproj(("patternkv_segmented_cache_v1",)) is True
    assert patternkv_use_bi_prefill_vproj(("patternkv_segmented_cache_v1",)) is False
    assert patternkv_use_bi_vproj(("patternkv_segmented_cache_v1",)) is False


def test_projection_mode_counters() -> None:
    reset_batch_invariant_kproj_counters()
    record_normal_prefill_kproj_call()
    record_normal_prefill_vproj_call()
    record_bi_prefill_kproj(rows=8)
    record_bi_prefill_vproj(rows=8)
    record_normal_decode_kproj_call()
    record_normal_decode_vproj_call()
    counters = batch_invariant_kproj_counters()
    assert counters["normal_prefill_kproj_calls"] == 1
    assert counters["normal_prefill_vproj_calls"] == 1
    assert counters["bi_prefill_kproj_calls"] == 1
    assert counters["bi_prefill_vproj_calls"] == 1
    assert counters["bi_prefill_vproj_rows"] == 8
    assert counters["normal_decode_kproj_calls"] == 1
    assert counters["normal_decode_vproj_calls"] == 1
    assert counters["bi_decode_vproj_calls"] == 0


def test_cost_benefit_classifier() -> None:
    classification, next_task = classify_cost_benefit(
        overhead_percent=4.0,
        logit_drift_reduction_ratio=2.5,
    )
    assert classification == "BI_VPROJ_COST_BENEFIT_SUPPORTED"
    assert next_task == "INTEGRATE_BATCH_INVARIANT_KVPROJ_PREFILL_RUNTIME"


def test_drift_reduction_metric() -> None:
    assert drift_reduction_ratio(0.4, 0.2) == 2.0
    assert drift_reduction_ratio(0.4, 0.0) == float("inf")


def test_prefill_overhead_metric() -> None:
    assert abs(prefill_overhead_percent(100.0, 105.0) - 5.0) < 1e-9


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
