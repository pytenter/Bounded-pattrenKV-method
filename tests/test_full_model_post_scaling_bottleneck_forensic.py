from __future__ import annotations

import torch

from scripts import full_model_post_scaling_bottleneck_forensic as forensic


def test_memory_snapshot_schema_cpu_safe(monkeypatch) -> None:
    calls = []

    class _Cuda:
        @staticmethod
        def synchronize(device: object) -> None:
            calls.append(("sync", device))

        @staticmethod
        def mem_get_info(device: object) -> tuple[int, int]:
            return 10, 20

        @staticmethod
        def memory_allocated(device: object) -> int:
            return 1

        @staticmethod
        def memory_reserved(device: object) -> int:
            return 2

        @staticmethod
        def max_memory_allocated(device: object) -> int:
            return 3

        @staticmethod
        def max_memory_reserved(device: object) -> int:
            return 4

    monkeypatch.setattr(torch, "cuda", _Cuda)

    row = forensic.memory_snapshot("phase", "cuda:0")

    assert row == {
        "phase": "phase",
        "allocated_bytes": 1,
        "reserved_bytes": 2,
        "max_allocated_bytes": 3,
        "max_reserved_bytes": 4,
        "mem_get_info_free_bytes": 10,
        "mem_get_info_total_bytes": 20,
    }
    assert calls == [("sync", "cuda:0")]


def test_tensor_ownership_summary_categorizes_cache_tensors() -> None:
    cache = {
        "packed_k": torch.empty((2, 4), dtype=torch.uint8),
        "packed_v4_scale": torch.empty((1,), dtype=torch.float16),
        "v_assignment_idx": torch.empty((2, 3), dtype=torch.int64),
        "centroid_state_pool": {"k_centroids": torch.empty((1, 2, 3), dtype=torch.float16)},
    }

    rows = forensic.collect_tensors(cache)
    summary = {row["category"]: row["physical_allocated_bytes"] for row in forensic.summarize_tensor_rows(rows)}

    assert summary["compressed_k_payload"] == 8
    assert summary["quant_scale"] == 2
    assert summary["precision_assignment_pattern_metadata"] == 48
    assert summary["centroid_state"] == 12


def test_oom_payload_flattens_without_crashing() -> None:
    payloads = [
        {
            "point": {"phase": "memory", "method": "FP16_FULL_MODEL", "context_length": 4096, "batch_size": 4, "decode_tokens": 8},
            "status": "OOM",
            "oom_phase": "initial_prefill",
            "oom_error": "CUDA out of memory",
            "memory_lifecycle": [{"phase": "after_model_load", "allocated_bytes": 1}],
            "tensor_summary": [],
        }
    ]

    lifecycle, breakdown, oom = forensic.flatten_memory(payloads)

    assert lifecycle[0]["phase"] == "after_model_load"
    assert breakdown == []
    assert oom[0]["oom_phase"] == "initial_prefill"


def test_profile_schema_handles_missing_optional_fields() -> None:
    payloads = [
        {
            "point": {"phase": "profile", "method": "CAUSAL_V4_25_FULL_MODEL", "context_length": 2048, "batch_size": 1, "decode_tokens": 8},
            "status": "PASS",
            "run_result": {"decode_only_wall_ms": 100.0, "output_tokens": 8},
            "module_profile": [{"component": "attention_total", "calls": 32, "total_ms": 20.0, "mean_ms": 0.625}],
            "profile_rows": [{"component": "cache_append", "calls": 32, "total_us": 1000.0}],
        }
    ]

    components, allocations, mutations = forensic.flatten_profile(payloads)

    assert len(components) == 2
    assert allocations == []
    assert mutations == []
    assert {row["source"] for row in components} == {"module_cuda_events", "patternkv_profile_range"}


def test_protocol_invariant_preservation_is_carried_from_repaired_gate() -> None:
    assert forensic.REPAIRED_DIR.name == "full_model_scaling_decode_only_protocol_repair_v1"
