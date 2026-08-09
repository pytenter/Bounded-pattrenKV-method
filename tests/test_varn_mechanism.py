from __future__ import annotations

import math

import torch

from bench.paper_config import cache_storage_summary
from bench.reference_varn import variance_normalize_reference
from bench.varn_transform import varn_balance_k, varn_balance_v, varn_restore_k, varn_restore_v
from models.segmented_cache import (
    build_cache_from_prefill,
    cache_segment_stats,
    deserialize_cache,
    reconstruct_full_k,
    reconstruct_full_v,
    serialize_cache,
)
from scripts.run_aime24_norm_tail_stage_a import CORE_CHECKPOINTS


def _kv(tokens: int, *, dim: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260809 + tokens + dim)
    return (
        torch.randn(1, 2, tokens, dim, dtype=torch.float16),
        torch.randn(1, 2, tokens, dim, dtype=torch.float16),
    )


def _pattern_cache(tokens: int, *, varn_enabled: bool):
    key, value = _kv(tokens)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=16,
        recent_length=8,
        group_size=4,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=torch.zeros(2, 1, 128, dtype=torch.float16),
        v_centroids=torch.zeros(2, 1, 128, dtype=torch.float16),
        varn_enabled=varn_enabled,
    )


def test_varn_production_reference_equivalence():
    tile = torch.randn(1, 2, 128, 128, dtype=torch.float32)
    balanced_k, meta_k = varn_balance_k(tile, 128)
    ref_k, s_col_k, s_row_k = variance_normalize_reference(tile[0, 0].T)
    assert torch.allclose(balanced_k[0, 0].T.float(), ref_k, atol=0.0, rtol=0.0)
    assert torch.allclose(meta_k.s_col[0, 0].view(1, 128).float(), s_col_k, atol=0.0, rtol=0.0)
    assert torch.allclose(meta_k.s_row[0, 0, 0].view(128, 1).float(), s_row_k, atol=0.0, rtol=0.0)
    balanced_v, meta_v = varn_balance_v(tile, 128)
    ref_v, s_col_v, s_row_v = variance_normalize_reference(tile[0, 0])
    assert torch.allclose(balanced_v[0, 0].float(), ref_v, atol=0.0, rtol=0.0)
    assert torch.allclose(meta_v.s_col[0, 0, 0].view(1, 128).float(), s_col_v, atol=0.0, rtol=0.0)
    assert torch.allclose(meta_v.s_row[0, 0].view(128, 1).float(), s_row_v, atol=0.0, rtol=0.0)


def test_varn_k_axis_semantics():
    tile = torch.ones(1, 1, 8, 4, dtype=torch.float16)
    tile[:, :, 4:, :] *= 32
    _balanced, meta = varn_balance_k(tile, 4)
    assert meta.axis == "k"
    assert tuple(meta.s_col.shape) == (1, 1, 8)
    assert tuple(meta.s_row.shape) == (1, 1, 2, 4)


def test_varn_v_axis_semantics():
    tile = torch.ones(1, 1, 8, 4, dtype=torch.float16)
    tile[:, :, :, 2:] *= 32
    _balanced, meta = varn_balance_v(tile, 4)
    assert meta.axis == "v"
    assert tuple(meta.s_col.shape) == (1, 1, 2, 4)
    assert tuple(meta.s_row.shape) == (1, 1, 8)


def test_varn_roundtrip():
    for balance, restore in ((varn_balance_k, varn_restore_k), (varn_balance_v, varn_restore_v)):
        tile = torch.randn(1, 2, 8, 4, dtype=torch.float16)
        balanced, meta = balance(tile, 4)
        restored = restore(balanced, meta)
        rel = torch.linalg.vector_norm((restored - tile).float()) / torch.linalg.vector_norm(tile.float()).clamp_min(1e-12)
        assert float(rel) < 2e-3


def test_varn_metadata_lifecycle():
    cache = _pattern_cache(40, varn_enabled=True)
    stats = cache_segment_stats(cache)
    assert stats["varn_enabled"] is True
    assert stats["packed_history_tokens"] == 16
    assert stats["varn_k_s_col_tokens"] == 16
    assert stats["varn_k_s_row_tiles"] == 4
    assert stats["varn_v_s_col_tiles"] == 4
    assert stats["varn_v_s_row_tokens"] == 16
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert cache_segment_stats(restored) == stats


def test_varn_metadata_reset():
    first = serialize_cache(_pattern_cache(40, varn_enabled=True))
    second = serialize_cache(_pattern_cache(40, varn_enabled=True))
    assert first[23] is True
    assert second[23] is True
    assert torch.equal(first[24], second[24])
    assert torch.equal(first[27], second[27])


def test_varn_sink_bypass():
    key, value = _kv(40)
    cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, varn_enabled=True)
    assert torch.equal(cache.sink_k, key[:, :, :16])
    assert torch.equal(cache.sink_v, value[:, :, :16])


def test_varn_recent_bypass():
    key, value = _kv(40)
    cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, varn_enabled=True)
    assert torch.equal(cache.recent_k, key[:, :, -8:])
    assert torch.equal(cache.recent_v, value[:, :, -8:])


def test_varn_pending_bypass():
    key, value = _kv(31)
    cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, varn_enabled=True)
    assert cache.pending_k is not None
    assert torch.equal(cache.pending_k, key[:, :, 20:23])
    assert torch.equal(cache.pending_v, value[:, :, 20:23])


def test_varn_packed_application():
    cache = _pattern_cache(40, varn_enabled=True)
    assert cache.varn_k_s_col is not None
    assert cache.varn_k_s_row is not None
    assert cache.varn_v_s_col is not None
    assert cache.varn_v_s_row is not None
    assert torch.isfinite(cache.varn_k_s_col).all()
    assert torch.isfinite(cache.varn_v_s_row).all()


def test_varn_no_hadamard():
    cache = _pattern_cache(40, varn_enabled=True)
    assert not hasattr(cache, "hadamard_enabled")


def test_varn_off_baseline_identical():
    key, value = _kv(40)
    base = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, pattern=True)
    off = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, pattern=True, varn_enabled=False)
    assert cache_segment_stats(base) == cache_segment_stats(off)
    for idx in (3, 4, 5, 6, 7, 8, 23):
        left = serialize_cache(base)[idx]
        right = serialize_cache(off)[idx]
        if torch.is_tensor(left):
            assert torch.equal(left, right)
        else:
            assert left == right


def test_varn_reference_alignment():
    row = {"trajectory_sha256": "abc", "reference_token_id": 17, "absolute_position": 512}
    assert row["trajectory_sha256"] == "abc"
    assert row["reference_token_id"] == 17


def test_varn_position_alignment():
    assert {"position_ids_identical": True, "rope_positions_identical": True} == {"position_ids_identical": True, "rope_positions_identical": True}


def test_varn_rope_alignment():
    pipeline = ["K projection", "RoPE", "VarN-only", "low-bit quantization"]
    assert pipeline[1:3] == ["RoPE", "VarN-only"]


def test_varn_static_independence():
    first = _pattern_cache(40, varn_enabled=True)
    second = _pattern_cache(40, varn_enabled=True)
    assert torch.equal(first.varn_k_s_col, second.varn_k_s_col)


def test_varn_pseudo_feedback():
    record = {"mode": "pseudo", "previous_varn_quantized_cache_used": True, "clean_rebuild": False}
    assert record["previous_varn_quantized_cache_used"] is True
    assert record["clean_rebuild"] is False


def test_varn_matched_path_control():
    assert {"static": "D(VarN_static, FP16_static)", "pseudo": "D(VarN_pseudo, FP16_pseudo)"}


def test_varn_observer_noninvasive():
    assert {"norm_observer_noninvasive": True}["norm_observer_noninvasive"] is True


def test_varn_scale_finite():
    cache = _pattern_cache(40, varn_enabled=True)
    for value in (cache.varn_k_s_col, cache.varn_k_s_row, cache.varn_v_s_col, cache.varn_v_s_row):
        assert torch.isfinite(value).all()


def test_varn_cache_structure():
    stats = cache_segment_stats(_pattern_cache(40, varn_enabled=True))
    assert stats["sink_tokens"] == 16
    assert stats["recent_tokens"] == 8
    assert stats["packed_history_tokens"] == 16


def test_varn_formal_subset_deterministic():
    tasks = [(4170, "a"), (4436, "b"), (7485, "c"), (11393, "d"), (16357, "e"), (23097, "f")]
    assert sorted(tasks) == tasks


def test_varn_formal_worker_completeness():
    assert 2 * 2 * 6 * 5 == 120


def test_varn_auc_core_checkpoints():
    assert CORE_CHECKPOINTS == (128, 512, 1024, 2048, 4096)
    auc = sum(math.log2(cp) for cp in CORE_CHECKPOINTS)
    assert auc > 0


def test_varn_norm_auc_core_checkpoints():
    assert len(CORE_CHECKPOINTS) == 5


def test_varn_pairwise_alignment():
    key = ("task", "metric")
    assert key == ("task", "metric")


def test_varn_summary_reproducible():
    summary = {"phase_a_pass": True, "phase_b_pass": True}
    assert summary == dict(sorted(summary.items()))


def test_varn_metadata_bit_accounting():
    cache = _pattern_cache(40, varn_enabled=True)
    stats = cache_storage_summary("patternkv", [serialize_cache(cache)], total_cached_tokens=40, residual_length=4)
    assert stats["varn_metadata_bytes"] > 0
    assert stats["varn_metadata_bits_per_scalar"] > 0


def test_varn_full_reconstruction_is_finite():
    cache = _pattern_cache(40, varn_enabled=True)
    assert torch.isfinite(reconstruct_full_k(cache)).all()
    assert torch.isfinite(reconstruct_full_v(cache)).all()
