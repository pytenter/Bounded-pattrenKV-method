from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import (
    ReferencePatternKVView,
    pattern_gather_centroids,
    reference_attention,
    reference_dequant_k,
    reference_dequant_v,
    reference_patternkv_attention,
    reference_reconstruct_k,
    reference_reconstruct_v,
    repeat_kv_for_gqa,
)
from models.segmented_cache import quantize_pack_k_reference, quantize_pack_v_reference


def _make_view(mask: torch.Tensor) -> ReferencePatternKVView:
    packed_k_input = torch.arange(256, dtype=torch.float16).reshape(1, 1, 16, 16)
    packed_v_input = torch.arange(256, 512, dtype=torch.float16).reshape(1, 1, 16, 16)
    chunk_k = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 512
    chunk_v = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 528
    packed_k, packed_k_scale, packed_k_zero = quantize_pack_k_reference(packed_k_input, group_size=16, bits=2)
    packed_v, packed_v_scale, packed_v_zero = quantize_pack_v_reference(packed_v_input, group_size=16, bits=2)
    return ReferencePatternKVView(
        packed_k=packed_k,
        packed_k_scale=packed_k_scale,
        packed_k_zero=packed_k_zero,
        packed_v=packed_v,
        packed_v_scale=packed_v_scale,
        packed_v_zero=packed_v_zero,
        chunk_k=chunk_k,
        chunk_v=chunk_v,
        k_centroids=torch.tensor([[[0.5] * 16, [1.5] * 16]], dtype=torch.float16),
        v_centroids=torch.tensor([[[1.0] * 16, [2.0] * 16]], dtype=torch.float16),
        k_assignments=torch.tensor([[[0] * 16]], dtype=torch.long),
        v_assignment_idx=torch.tensor([[[1] * 16]], dtype=torch.long),
        v_pattern_mask=mask,
        total_tokens=17,
        packed_k_tokens=16,
        packed_v_tokens=16,
        group_size=16,
        k_bits=2,
        v_bits=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        cache_mode="legacy_tuple_chunked",
    )


def test_reference_v_attention_gate_modes() -> None:
    query = torch.zeros(1, 4, 1, 16, dtype=torch.float16)
    gate_true_view = _make_view(torch.tensor([[[1] * 16]], dtype=torch.uint8))
    gate_false_view = _make_view(torch.tensor([[[0] * 16]], dtype=torch.uint8))

    true_output, _, _, _ = reference_patternkv_attention(query, gate_true_view)
    false_output, _, _, _ = reference_patternkv_attention(query, gate_false_view)
    reconstructed_true_v = reference_reconstruct_v(gate_true_view)
    reconstructed_false_v = reference_reconstruct_v(gate_false_view)
    assert true_output.shape == false_output.shape == (1, 4, 1, 16)
    assert torch.isfinite(true_output).all()
    assert torch.isfinite(false_output).all()
    assert not torch.allclose(reconstructed_true_v, reconstructed_false_v)
