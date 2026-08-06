from __future__ import annotations

import torch

from bench.aime24_int2_wave1 import BitwidthConfig, effective_bitwidth, stable_hash
from bench.paper_config import cache_storage_summary
from models.segmented_cache import build_cache_from_prefill, serialize_cache


def test_effective_bitwidth_reports_components() -> None:
    stats = effective_bitwidth(BitwidthConfig(method="pattern_k2v2_s64_r256", total_tokens=4096, sink_length=64, recent_length=256, k_bits=2, v_bits=2))
    assert stats["payload_bits_per_scalar"] > 2.0
    assert stats["metadata_bits_per_scalar"] > 0.0
    assert stats["pattern_centroid_overhead_bits_per_scalar"] > 0.0
    assert stats["total_effective_bits_per_scalar"] >= stats["payload_bits_per_scalar"]


def test_config_hash_changes_with_sink_recent_and_mask() -> None:
    base = {"config": "pattern", "sink_length": 0, "recent_length": 128, "mask": ""}
    sink = {"config": "pattern", "sink_length": 64, "recent_length": 256, "mask": ""}
    mask = {"config": "pattern", "sink_length": 0, "recent_length": 128, "mask": "abc"}
    assert stable_hash(base) != stable_hash(sink)
    assert stable_hash(base) != stable_hash(mask)


def test_cache_storage_summary_reports_pattern_dynamic_components() -> None:
    key = torch.arange(1 * 2 * 8 * 16, dtype=torch.float16).reshape(1, 2, 8, 16) / 17
    value = key + 0.5
    cache = build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=4,
        group_size=4,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=torch.zeros(2, 3, 16, dtype=torch.float16),
        v_centroids=torch.zeros(2, 3, 16, dtype=torch.float16),
    )
    stats = cache_storage_summary("patternkv_paper", [serialize_cache(cache)], total_cached_tokens=8, residual_length=4)
    assert stats["k_centroid_bytes"] > 0
    assert stats["v_centroid_bytes"] > 0
    assert stats["k_assignment_theoretical_compact_bits"] > 0
    assert stats["v_assignment_theoretical_compact_bits"] > 0
    assert stats["v_gate_theoretical_compact_bits"] == cache.v_pattern_mask.numel()
    assert stats["assignment_actual_python_tensor_bits"] > stats["k_assignment_theoretical_compact_bits"]
