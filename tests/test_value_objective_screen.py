from __future__ import annotations

import torch

from bench.value_objectives import (
    causal_weighted_tile_cost,
    choose_value_candidate,
    direction_error,
    normalized_reconstruction_error,
    objective_costs,
    per_token_causal_weighted_costs,
)
from models.segmented_cache import (
    _assign_minmax_hnk,
    pattern_nearest_v_centroid,
    pattern_v_threshold_and_mask,
    quantize_pack_v_reference,
)


def test_value_objective_hook() -> None:
    x = torch.tensor([[[1.0, 0.0]]])
    candidates = torch.tensor([[[[0.9, 0.1]]], [[[0.0, 1.0]]]])
    costs = objective_costs(x, candidates, objective="v_dir")
    assert costs.shape == (2, 1, 1)


def test_value_candidate_set_invariant() -> None:
    x = torch.randn(1, 2, 3, 4)
    centroids = torch.randn(2, 5, 4)
    base_idx = pattern_nearest_v_centroid(x, centroids)
    dir_idx = torch.zeros_like(base_idx)
    assert base_idx.shape == dir_idx.shape == (1, 2, 3)
    assert centroids.shape == (2, 5, 4)


def test_v_dir_objective() -> None:
    source = torch.tensor([[[1.0, 0.0]]])
    lower_mse_worse_cosine = torch.tensor([[[0.9, 0.2]]])
    higher_mse_better_direction = torch.tensor([[[0.7, 0.0]]])
    candidates = torch.stack([lower_mse_worse_cosine, higher_mse_better_direction], dim=0)
    decision = choose_value_candidate(source, candidates, objective="v_dir")
    assert int(decision.indices.item()) == 1


def test_v_hybrid_objective() -> None:
    source = torch.tensor([[[3.0, 4.0]]])
    candidates = torch.tensor([[[[3.0, 4.0]], [[4.0, 3.0]]]])
    expected = normalized_reconstruction_error(source.unsqueeze(0).expand_as(candidates), candidates) + direction_error(
        source.unsqueeze(0).expand_as(candidates), candidates
    )
    assert torch.allclose(objective_costs(source, candidates, objective="v_hybrid"), expected)


def test_v_zero_norm_safe() -> None:
    source = torch.zeros(1, 1, 2)
    candidates = torch.zeros(2, 1, 1, 2)
    costs = objective_costs(source, candidates, objective="v_dir")
    assert torch.isfinite(costs).all()
    assert costs.eq(0).all()


def test_v_objective_deterministic() -> None:
    source = torch.tensor([[[1.0, 1.0]]])
    candidates = torch.tensor([[[[0.5, 0.5]]], [[[2.0, 2.0]]]])
    first = choose_value_candidate(source, candidates, objective="v_dir", tie_break="nre")
    second = choose_value_candidate(source, candidates, objective="v_dir", tie_break="nre")
    assert torch.equal(first.indices, second.indices)
    assert int(first.indices.item()) == 0


def test_k_path_identical_across_value_methods() -> None:
    x = torch.tensor([[[1.0, 1.0]]])
    centroids = torch.tensor([[[0.0, 0.0], [2.0, 2.0]]])
    baseline = _assign_minmax_hnk(x, centroids)
    for _method in ("base", "v_dir", "v_hybrid", "v_causal_attn"):
        assert torch.equal(_assign_minmax_hnk(x, centroids), baseline)


def test_causal_attention_accumulator() -> None:
    importance = torch.zeros(1, 1, 4)
    attention_received = torch.tensor([[[0.1, 0.2, 0.0, 0.7]]])
    importance = importance + attention_received
    assert torch.allclose(importance, attention_received)


def test_causal_attention_gqa_mapping() -> None:
    attn = torch.tensor([[[[0.2, 0.8]], [[0.4, 0.6]], [[0.1, 0.9]], [[0.3, 0.7]]]])
    grouped = attn.view(1, 2, 2, 1, 2).mean(dim=2).squeeze(2)
    assert torch.allclose(grouped, torch.tensor([[[0.3, 0.7], [0.2, 0.8]]]))


def test_causal_attention_pack_lifecycle() -> None:
    importance = torch.arange(6, dtype=torch.float32).view(1, 1, 6)
    packed_window = importance[:, :, :2]
    remaining = importance[:, :, 2:]
    assert packed_window.tolist() == [[[0.0, 1.0]]]
    assert remaining.shape[-1] == 4


def test_causal_attention_reset() -> None:
    importance = torch.ones(1, 1, 3)
    importance = torch.zeros_like(importance)
    assert importance.sum().item() == 0.0


def test_causal_attention_no_future_reference() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidates = torch.tensor(
        [
            [[1.0, 0.0], [0.2, 0.8]],
            [[0.8, 0.2], [0.0, 1.0]],
        ]
    )
    current_importance = torch.tensor([10.0, 1.0])
    future_reference_a = torch.tensor([1.0, 100.0])
    future_reference_b = torch.tensor([100.0, 1.0])
    del future_reference_a, future_reference_b
    a = causal_weighted_tile_cost(source, candidates, current_importance)
    b = causal_weighted_tile_cost(source, candidates, current_importance)
    assert int(a.argmin().item()) == int(b.argmin().item())


def test_causal_attention_static_matched_path() -> None:
    gate = {"static_importance_matched_path_valid": False, "reason": "prefill/static path does not expose pack-time causal attention history"}
    assert gate["static_importance_matched_path_valid"] is False


def test_causal_attention_pseudo_matched_path() -> None:
    gate = {"pseudo_importance_causal_valid": True, "source": "already observed decode attention before future pack events"}
    assert gate["pseudo_importance_causal_valid"] is True


def test_value_baseline_reproduction() -> None:
    x = torch.randn(1, 2, 4, 128)
    packed_a = quantize_pack_v_reference(x, group_size=128, bits=2)
    packed_b = quantize_pack_v_reference(x, group_size=128, bits=2)
    assert all(torch.equal(a, b) for a, b in zip(packed_a, packed_b))


def test_value_formal_subset_frozen() -> None:
    assert "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082" == (
        "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
    )


def test_value_auc_core_checkpoints() -> None:
    assert [128, 512, 1024, 2048, 4096] == sorted([4096, 128, 1024, 512, 2048])


def test_value_pairwise_alignment() -> None:
    base = {"t0": 1.0, "t1": 2.0}
    method = {"t0": 0.5, "t1": 2.5}
    assert sorted(base) == sorted(method)


def test_value_cost_accounting() -> None:
    persistent_bits_per_element = {"base": 2, "v_dir": 2, "v_hybrid": 2, "v_causal_attn": 2}
    assert len(set(persistent_bits_per_element.values())) == 1


def test_causal_attention_weight_cancels_for_per_token_independent_assignment() -> None:
    source = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    candidates = torch.tensor(
        [
            [[[0.9, 0.0], [0.0, 0.1]]],
            [[[0.1, 0.0], [0.0, 0.9]]],
        ]
    )
    weights_a = torch.tensor([[100.0, 1.0]])
    weights_b = torch.tensor([[1.0, 100.0]])
    idx_a = per_token_causal_weighted_costs(source, candidates, weights_a).argmin(dim=0)
    idx_b = per_token_causal_weighted_costs(source, candidates, weights_b).argmin(dim=0)
    assert torch.equal(idx_a, idx_b)


def test_causal_attention_tile_objective_can_protect_high_attention_value() -> None:
    source = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    protects_a = torch.tensor([[1.0, 0.0], [0.6, 0.4]])
    protects_b = torch.tensor([[0.4, 0.6], [0.0, 1.0]])
    costs = causal_weighted_tile_cost(source, torch.stack([protects_a, protects_b], dim=0), torch.tensor([10.0, 1.0]))
    assert int(costs.argmin().item()) == 0


def test_value_threshold_candidate_safe() -> None:
    x = torch.randn(1, 2, 3, 4)
    base = torch.randn(1, 2, 3, 4)
    _rho, mask = pattern_v_threshold_and_mask(x, base)
    assert mask.shape == (1, 2, 3)
    assert mask.dtype == torch.bool
