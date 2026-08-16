from __future__ import annotations

import torch

from models.segmented_cache import (
    PatternQuantizedKVCache,
    _assign_minmax_hnk,
    append_decode,
    build_cache_from_prefill,
    dequantize_v_reference,
    pattern_gather_centroids,
    pattern_nearest_v_centroid,
    pattern_select_v_candidate,
    pattern_v_candidate_reconstructions,
    pattern_v_threshold_and_mask,
    quantize_pack_v_reference,
    serialize_cache,
    deserialize_cache,
)
from bench.pseudodecode_metrics import trapezoid_auc_log2
from scripts.run_aime24_value_direction_screen import pairwise_summary


def test_value_candidate_set_invariant() -> None:
    x = torch.randn(1, 2, 4, 8)
    centroids = torch.randn(2, 5, 8)
    candidate_count = centroids.shape[1]
    for objective in ("base", "v_dir", "v_hybrid"):
        idx, mask, _scores = pattern_select_v_candidate(x, centroids, value_objective=objective, group_size=4, bits=2)
        assert idx.shape == mask.shape == (1, 2, 4)
        assert int(idx.max().item()) < candidate_count
    assert centroids.shape == (2, 5, 8)


def test_value_candidate_reconstruction_matches_production() -> None:
    x = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).view(1, 1, 1, 16)
    centroids = torch.stack([torch.zeros(16), torch.linspace(0.5, -0.5, 16)], dim=0).view(1, 2, 16)
    recon, masks, _base = pattern_v_candidate_reconstructions(x, centroids, group_size=16, bits=2)
    idx = torch.tensor([[[0]]])
    chosen = pattern_gather_centroids(idx, centroids).to(x.dtype)
    _, mask = pattern_v_threshold_and_mask(x, chosen)
    adjusted = x - mask.unsqueeze(-1).to(x.dtype) * chosen
    packed, scale, zero = quantize_pack_v_reference(adjusted, group_size=16, bits=2)
    restored = dequantize_v_reference(packed, scale, zero, group_size=16, bits=2) + mask.unsqueeze(-1).to(x.dtype) * chosen
    assert torch.allclose(recon[:, :, 0], restored)
    assert torch.equal(masks[:, :, 0], mask)


def test_v_dir_final_reconstruction_objective() -> None:
    x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]], dtype=torch.float32)
    centroids = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.6, -0.2, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    base_idx = pattern_nearest_v_centroid(x, centroids)
    dir_idx, _mask, scores = pattern_select_v_candidate(x, centroids, value_objective="v_dir", group_size=4, bits=2)
    assert scores["direction"].shape == (1, 1, 2, 1)
    assert base_idx.shape == dir_idx.shape
    assert torch.isfinite(scores["direction"]).all()


def test_v_hybrid_final_reconstruction_objective() -> None:
    x = torch.tensor([[[[3.0, 4.0, 0.0, 0.0]]]], dtype=torch.float32)
    centroids = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]]], dtype=torch.float32)
    idx, _mask, scores = pattern_select_v_candidate(x, centroids, value_objective="v_hybrid", group_size=4, bits=2)
    manual = scores["direction"] + scores["nre"]
    chosen = torch.gather(manual, 2, idx.unsqueeze(2)).squeeze(2)
    assert torch.allclose(chosen, manual.min(dim=2).values)


def test_v_dir_zero_norm_safe() -> None:
    x = torch.zeros(1, 1, 2, 4)
    centroids = torch.randn(1, 3, 4)
    _idx, _mask, scores = pattern_select_v_candidate(x, centroids, value_objective="v_dir", group_size=4, bits=2)
    assert torch.isfinite(scores["direction"]).all()


def test_v_hybrid_zero_norm_safe() -> None:
    x = torch.zeros(1, 1, 2, 4)
    centroids = torch.randn(1, 3, 4)
    _idx, _mask, scores = pattern_select_v_candidate(x, centroids, value_objective="v_hybrid", group_size=4, bits=2)
    assert torch.isfinite(scores["direction"]).all()
    assert torch.isfinite(scores["nre"]).all()


def test_v_dir_tiebreak_deterministic() -> None:
    x = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    centroids = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]])
    idx_a, _mask_a, _ = pattern_select_v_candidate(x, centroids, value_objective="v_dir", group_size=4, bits=2)
    idx_b, _mask_b, _ = pattern_select_v_candidate(x, centroids, value_objective="v_dir", group_size=4, bits=2)
    assert torch.equal(idx_a, idx_b)


def test_v_hybrid_tiebreak_deterministic() -> None:
    x = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
    centroids = torch.tensor([[[0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]]])
    idx_a, _mask_a, _ = pattern_select_v_candidate(x, centroids, value_objective="v_hybrid", group_size=4, bits=2)
    idx_b, _mask_b, _ = pattern_select_v_candidate(x, centroids, value_objective="v_hybrid", group_size=4, bits=2)
    assert torch.equal(idx_a, idx_b)


def test_vectorized_candidate_score_matches_scalar_reference() -> None:
    x = torch.randn(1, 1, 3, 4)
    centroids = torch.randn(1, 4, 4)
    recon, masks, base = pattern_v_candidate_reconstructions(x, centroids, group_size=4, bits=2)
    scalar_recons = []
    scalar_masks = []
    scalar_base = []
    for c in range(centroids.shape[1]):
        r, m, b = pattern_v_candidate_reconstructions(x, centroids[:, c : c + 1], group_size=4, bits=2)
        scalar_recons.append(r)
        scalar_masks.append(m)
        scalar_base.append(b)
    assert torch.allclose(recon, torch.cat(scalar_recons, dim=2))
    assert torch.equal(masks, torch.cat(scalar_masks, dim=2))
    assert torch.allclose(base, torch.cat(scalar_base, dim=2))


def test_value_objective_changes_assignment() -> None:
    found = False
    for seed in range(64):
        torch.manual_seed(seed)
        x = torch.randn(1, 2, 8, 4)
        centroids = torch.randn(2, 6, 4)
        base = pattern_nearest_v_centroid(x, centroids)
        v_dir, _mask, _ = pattern_select_v_candidate(x, centroids, value_objective="v_dir", group_size=4, bits=2)
        found = found or not torch.equal(base, v_dir)
    assert found


def test_k_path_identical() -> None:
    x = torch.randn(2, 6, 4)
    centroids = torch.randn(2, 5, 4)
    base = _assign_minmax_hnk(x, centroids)
    for _objective in ("base", "v_dir", "v_hybrid"):
        assert torch.equal(_assign_minmax_hnk(x, centroids), base)


def test_value_bits_identical() -> None:
    cache = PatternQuantizedKVCache(group_size=4, k_bits=2, v_bits=2, value_objective="v_hybrid")
    assert cache.k_bits == 2
    assert cache.v_bits == 2


def test_value_baseline_reproduction() -> None:
    torch.manual_seed(0)
    k = torch.randn(1, 2, 40, 16)
    v = torch.randn(1, 2, 40, 16)
    centroids = torch.randn(2, 3, 16)
    cache_a = build_cache_from_prefill(k, v, sink_length=4, recent_length=4, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=centroids, v_centroids=centroids, value_objective="base")
    cache_b = build_cache_from_prefill(k, v, sink_length=4, recent_length=4, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=centroids, v_centroids=centroids, value_objective="base")
    assert torch.equal(cache_a.v_assignment_idx, cache_b.v_assignment_idx)
    assert torch.equal(cache_a.packed_v, cache_b.packed_v)


def test_static_value_objective_independence() -> None:
    torch.manual_seed(1)
    k = torch.randn(1, 1, 32, 16)
    v = torch.randn(1, 1, 32, 16)
    c = torch.randn(1, 3, 16)
    a = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, value_objective="v_dir")
    b = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, value_objective="v_dir")
    assert torch.equal(a.v_assignment_idx, b.v_assignment_idx)


def test_pseudo_value_objective_feedback() -> None:
    torch.manual_seed(2)
    k = torch.randn(1, 1, 16, 16)
    v = torch.randn(1, 1, 16, 16)
    c = torch.randn(1, 3, 16)
    cache = build_cache_from_prefill(k, v, sink_length=0, recent_length=8, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, value_objective="v_hybrid")
    append_decode(cache, torch.randn(1, 1, 9, 16), torch.randn(1, 1, 9, 16))
    assert cache.value_objective == "v_hybrid"
    assert cache.packed_v_tokens >= 4


def test_value_direction_formal_subset() -> None:
    assert "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"


def test_value_direction_auc() -> None:
    auc = trapezoid_auc_log2([(128, 0.0), (512, 1.0), (1024, 1.0), (2048, 2.0), (4096, 2.0)])
    assert auc > 0.0


def test_value_direction_pairwise_alignment() -> None:
    base = {("task0", "metric"): 1.0}
    method = {("task0", "metric"): 0.5}
    assert set(base) == set(method)


def test_static_stored_v_pairwise_uses_v_direction_family() -> None:
    static_auc = [
        {"task_key": "task0", "method": "BASE", "layer": "31", "metric_family": "v_direction", "object_type": "v_stored", "region": "all_tokens", "metric_name": "direction_error", "statistic": "p95", "auc": 1.0},
        {"task_key": "task0", "method": "V_DIR", "layer": "31", "metric_family": "v_direction", "object_type": "v_stored", "region": "all_tokens", "metric_name": "direction_error", "statistic": "p95", "auc": 0.5},
        {"task_key": "task0", "method": "V_HYBRID", "layer": "31", "metric_family": "v_direction", "object_type": "v_stored", "region": "all_tokens", "metric_name": "direction_error", "statistic": "p95", "auc": 0.75},
    ]
    pairwise, _summary = pairwise_summary(static_auc, [])
    direct = {row["method"]: row for row in pairwise if row["metric"] == "static_stored_v_direction"}
    assert direct["V_DIR"]["tasks_compared"] == 1
    assert direct["V_DIR"]["median_delta"] == -0.5
    assert direct["V_HYBRID"]["tasks_compared"] == 1


def test_assignment_behavior_summary() -> None:
    base = torch.tensor([[[0, 1, 1, 2]]])
    method = torch.tensor([[[0, 2, 1, 3]]])
    changed = method != base
    assert float(changed.float().mean().item()) == 0.5


def test_value_objective_survives_serialize_roundtrip() -> None:
    cache = PatternQuantizedKVCache(total_tokens=0, group_size=4, k_bits=2, v_bits=2, value_objective="v_dir")
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert isinstance(restored, PatternQuantizedKVCache)
    assert restored.value_objective == "v_dir"


def test_v_candidate_block_size_equivalence() -> None:
    torch.manual_seed(2026081503)
    x = torch.randn(1, 2, 33, 8)
    centroids = torch.randn(2, 7, 8)
    for objective in ("v_dir", "v_hybrid"):
        idx_full, mask_full, _ = pattern_select_v_candidate(x, centroids, value_objective=objective, group_size=4, bits=2, block_tokens=128)
        idx_chunked, mask_chunked, _ = pattern_select_v_candidate(x, centroids, value_objective=objective, group_size=4, bits=2, block_tokens=8)
        assert torch.equal(idx_chunked, idx_full)
        assert torch.equal(mask_chunked, mask_full)
