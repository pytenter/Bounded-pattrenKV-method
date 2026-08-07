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


def test_reference_k_attention_matches_explicit_reconstruction() -> None:
    query = torch.arange(32, dtype=torch.float16).reshape(1, 2, 1, 16)
    packed_k_input = torch.arange(256, dtype=torch.float16).reshape(1, 1, 16, 16)
    packed_v_input = torch.arange(256, 512, dtype=torch.float16).reshape(1, 1, 16, 16)
    chunk_k = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 512
    chunk_v = torch.arange(16, dtype=torch.float16).reshape(1, 1, 1, 16) + 528
    packed_k, packed_k_scale, packed_k_zero = quantize_pack_k_reference(packed_k_input, group_size=16, bits=2)
    packed_v, packed_v_scale, packed_v_zero = quantize_pack_v_reference(packed_v_input, group_size=16, bits=2)
    view = ReferencePatternKVView(
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
        v_pattern_mask=torch.tensor([[[1] * 16]], dtype=torch.uint8),
        total_tokens=17,
        packed_k_tokens=16,
        packed_v_tokens=16,
        group_size=16,
        k_bits=2,
        v_bits=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        cache_mode="legacy_tuple_chunked",
    )
    output, probs, key_for_attention, value_for_attention = reference_patternkv_attention(query, view)
    manual_k = reference_reconstruct_k(view)
    manual_v = reference_reconstruct_v(view)
    manual_output = reference_attention(
        query,
        repeat_kv_for_gqa(manual_k, 2, expected_heads=2, tensor_name="manual_k"),
        repeat_kv_for_gqa(manual_v, 2, expected_heads=2, tensor_name="manual_v"),
    )
    assert torch.allclose(output, manual_output)
    assert probs.shape == (1, 2, 1, 17)
    assert key_for_attention.shape == (1, 2, 17, 16)
    assert value_for_attention.shape == (1, 2, 17, 16)
