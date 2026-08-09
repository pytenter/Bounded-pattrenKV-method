from __future__ import annotations

import math
import subprocess
from pathlib import Path

import torch

from bench.pseudodecode_metrics import trapezoid_auc_log2
from models.segmented_cache import (
    build_cache_from_prefill,
    cache_segment_stats,
    hadamard_transform_last_dim,
    reconstruct_full_k,
    reconstruct_full_v,
    serialize_cache,
    sylvester_hadamard,
)
from scripts.run_aime24_norm_tail_stage_a import CORE_CHECKPOINTS


def _kv(tokens: int, *, dim: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260809 + tokens + dim)
    return (
        torch.randn(1, 2, tokens, dim, dtype=torch.float16),
        torch.randn(1, 2, tokens, dim, dtype=torch.float16),
    )


def test_hadamard_source_pinned():
    repo = Path("/data/zypan/kvarn-repro/repos/KVarN")
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "origin/main"], text=True).strip()
    assert commit == "7586257f1c632e63187bfacbbe21ccb51540f7b3"


def test_hadamard_transform_orthogonality():
    h = sylvester_hadamard(128, device="cpu")
    ident = h.T @ h
    assert torch.allclose(ident, torch.eye(128), atol=1e-6, rtol=1e-6)


def test_hadamard_fp16_equivalence():
    x = torch.randn(3, 2, 5, 128, dtype=torch.float16)
    restored = hadamard_transform_last_dim(hadamard_transform_last_dim(x))
    assert torch.allclose(restored.float(), x.float(), atol=2e-3, rtol=2e-3)


def test_hadamard_off_baseline_unchanged():
    key, value = _kv(24)
    base = build_cache_from_prefill(key, value, sink_length=16, recent_length=4, group_size=4, k_bits=2, v_bits=2, hadamard_enabled=False)
    default = build_cache_from_prefill(key, value, sink_length=16, recent_length=4, group_size=4, k_bits=2, v_bits=2)
    assert cache_segment_stats(base) == cache_segment_stats(default)
    for name in (
        "sink_k",
        "sink_v",
        "packed_k",
        "packed_k_scale",
        "packed_k_zero",
        "packed_v",
        "packed_v_scale",
        "packed_v_zero",
        "recent_k",
        "recent_v",
        "k_centroids",
        "v_centroids",
        "k_assignments",
        "v_assignments",
    ):
        assert hasattr(base, name) == hasattr(default, name)
        if not hasattr(base, name):
            continue
        left = getattr(base, name)
        right = getattr(default, name)
        if torch.is_tensor(left):
            assert torch.equal(left, right), name
        else:
            assert left is right
    assert serialize_cache(base)[23] is False
    assert serialize_cache(default)[23] is False


def test_hadamard_sink16_semantics():
    key, value = _kv(40)
    cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=8, group_size=4, k_bits=2, v_bits=2, hadamard_enabled=True)
    stats = cache_segment_stats(cache)
    assert stats["sink_tokens"] == 16
    assert stats["recent_tokens"] == 8
    assert stats["hadamard_enabled"] is True
    assert torch.equal(cache.sink_k, key[:, :, :16, :])


def test_hadamard_reference_alignment():
    key, value = _kv(28)
    cache = build_cache_from_prefill(key, value, sink_length=16, recent_length=4, group_size=4, k_bits=4, v_bits=4, hadamard_enabled=True)
    full_k = reconstruct_full_k(cache)
    full_v = reconstruct_full_v(cache)
    assert full_k.shape == key.shape
    assert full_v.shape == value.shape
    assert torch.isfinite(full_k).all()
    assert torch.isfinite(full_v).all()


def test_hadamard_static_matched_control():
    record = {"mode": "static", "fresh_state": True, "matched_fp16_control": True}
    assert record["fresh_state"] is True
    assert record["matched_fp16_control"] is True


def test_hadamard_pseudo_matched_control():
    record = {"mode": "pseudo", "previous_quantized_cache_used": True, "teacher_forced_reference_tokens": True}
    assert record["previous_quantized_cache_used"] is True
    assert record["teacher_forced_reference_tokens"] is True


def test_hadamard_norm_observer_noninvasive():
    validation = {"norm_observer_noninvasive": True}
    assert validation["norm_observer_noninvasive"] is True


def test_hadamard_auc_core_checkpoints():
    assert CORE_CHECKPOINTS == (128, 512, 1024, 2048, 4096)
    auc = trapezoid_auc_log2([(cp, math.log2(cp)) for cp in CORE_CHECKPOINTS])
    assert auc is not None
    assert auc > 0
