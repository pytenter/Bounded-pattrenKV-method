from __future__ import annotations

from bench.aime24_int2_wave1 import BitwidthConfig, effective_bitwidth, stable_hash


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
