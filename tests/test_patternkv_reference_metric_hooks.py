from __future__ import annotations

import torch

from bench.patternkv_equivalence_reference import (
    ReferencePatternKVView,
    reference_patternkv_attention,
)


def test_reference_attention_details_do_not_mutate_inputs() -> None:
    query = torch.randn(1, 2, 1, 4, dtype=torch.float16)
    chunk_k = torch.randn(1, 1, 3, 4, dtype=torch.float16)
    chunk_v = torch.randn(1, 1, 3, 4, dtype=torch.float16)
    query_before = query.clone()
    chunk_k_before = chunk_k.clone()
    chunk_v_before = chunk_v.clone()
    view = ReferencePatternKVView(
        packed_k=None,
        packed_k_scale=None,
        packed_k_zero=None,
        packed_v=None,
        packed_v_scale=None,
        packed_v_zero=None,
        chunk_k=chunk_k,
        chunk_v=chunk_v,
        k_centroids=None,
        v_centroids=None,
        k_assignments=None,
        v_assignment_idx=None,
        v_pattern_mask=None,
        total_tokens=3,
        packed_k_tokens=0,
        packed_v_tokens=0,
        group_size=4,
        k_bits=2,
        v_bits=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        cache_mode="segmented_chunked",
    )

    output, probs, _, _, details = reference_patternkv_attention(query, view, return_details=True)

    assert output.shape == (1, 2, 1, 4)
    assert probs.shape == (1, 2, 1, 3)
    assert details["attention_scores"].shape == (1, 2, 1, 3)
    assert torch.equal(query, query_before)
    assert torch.equal(chunk_k, chunk_k_before)
    assert torch.equal(chunk_v, chunk_v_before)
