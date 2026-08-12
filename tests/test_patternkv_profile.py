import csv
import io
import os

import pytest
import torch

from quant.patternkv_profile import (
    cache_mutation_snapshot,
    merge_profile_rows,
    profile_range,
    profile_snapshot,
    record_cache_mutation,
    record_counter,
    record_temp_allocation,
    reset_profile,
    temp_allocation_snapshot,
)


def test_profile_disabled_does_not_change_output(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "0")
    reset_profile()
    x = torch.arange(8, dtype=torch.float32)
    with profile_range("disabled_component", tokens=8):
        y = x * 2.0 + 1.0
    assert torch.equal(y, x * 2.0 + 1.0)
    assert profile_snapshot(reset=True) == {}


def test_profile_counters_reset(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    record_counter("selector_total", tokens=128, bytes_copied=4)
    snap = profile_snapshot(reset=False)
    assert snap["selector_total"]["calls"] == 1
    assert snap["selector_total"]["tokens"] == 128
    reset_profile()
    assert profile_snapshot(reset=False) == {}


def test_profile_timers_aggregate(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    with profile_range("cpu_component", tokens=4):
        _ = sum(range(32))
    rows = merge_profile_rows(profile_snapshot(reset=True), decode_tokens=2, decode_total_us=1000.0)
    row = next(row for row in rows if row["component"] == "cpu_component")
    assert row["calls"] == 1
    assert row["calls_per_generated_token"] == 0.5
    assert row["total_us"] >= 0.0
    assert set(row) >= {"component", "calls", "total_us", "mean_us", "percent_decode_time"}


def test_profile_csv_schema_valid(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    record_counter("cache_cat_events", calls=3, bytes_copied=1024)
    rows = merge_profile_rows(profile_snapshot(reset=True), decode_tokens=4, decode_total_us=2000.0)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    parsed = list(csv.DictReader(io.StringIO(buf.getvalue())))
    assert parsed[0]["component"] == "cache_cat_events"
    assert int(parsed[0]["calls"]) == 3


def test_cache_mutation_counter_classification_and_bytes(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    old = torch.zeros(1, 2, 3, dtype=torch.float16)
    app = torch.zeros(1, 2, 1, dtype=torch.float16)
    result = torch.zeros(1, 2, 4, dtype=torch.float16)
    record_cache_mutation("packed_v2_payload", old, app, result)
    rows = cache_mutation_snapshot()
    row = next(row for row in rows if row["category"] == "packed_v2_payload")
    assert row["calls"] == 1
    assert row["old_bytes"] == old.numel() * old.element_size()
    assert row["append_bytes"] == app.numel() * app.element_size()
    assert row["estimated_copy_bytes"] == (old.numel() + app.numel()) * old.element_size()
    assert row["largest_result_bytes"] == result.numel() * result.element_size()


def test_temp_allocation_snapshot_schema(monkeypatch):
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    value = torch.zeros(2, 3, dtype=torch.float32)
    record_temp_allocation("mixed_v_attn2_compact", value)
    rows = temp_allocation_snapshot(decode_tokens=2)
    assert rows[0]["tensor"] == "mixed_v_attn2_compact"
    assert rows[0]["shape"] == "2x3"
    assert rows[0]["dtype"] == "torch.float32"
    assert rows[0]["bytes_per_decode_token"] == value.numel() * value.element_size() / 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for mixed V backend profile smoke")
def test_fused_and_reference_backend_both_profile(monkeypatch):
    from bench.bench_mixed_v_kernel_perf import build_case
    from quant.matmul import cuda_attn_v_mixed_fused_with_base, get_patternkv_mixed_v_counters, reset_patternkv_mixed_v_counters

    data = build_case("mixed25", 128, seed=321)
    p2 = data["p2"]
    p4 = data["p4"]
    monkeypatch.setenv("PATTERNKV_PROFILE", "1")
    reset_profile()
    reset_patternkv_mixed_v_counters()
    fused = cuda_attn_v_mixed_fused_with_base(
        128,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        p4[0],
        p4[1],
        p4[2],
        data["precision"],
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        32,
        8,
    )
    assert torch.isfinite(fused).all()
    snap = profile_snapshot(reset=True)
    counters = get_patternkv_mixed_v_counters()
    assert snap["mixed_v_fused_attention"]["calls"] == 1
    assert snap["mixed_v_kernel_launches"]["calls"] == 2
    assert counters["mixed_v_fused_calls"] == 1

    reset_profile()
    with profile_range("mixed_v_reference_attention", tokens=128):
        ref_like = fused + 0
    assert torch.equal(ref_like, fused)
    snap = profile_snapshot(reset=True)
    assert snap["mixed_v_reference_attention"]["calls"] == 1
