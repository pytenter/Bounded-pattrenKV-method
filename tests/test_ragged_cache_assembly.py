from __future__ import annotations

import math

import torch

from models.segmented_cache import (
    PatternKVCentroidStatePool,
    PatternQuantizedKVCache,
    advance_request_total_tokens,
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_decode_position_ids,
    get_total_tokens_per_request,
    serialize_cache,
    set_request_total_tokens,
)
from quant.page_batch import PatternKVBatchMetadata, PatternKVOperatorReadyPagePools


def _page_pools(tokens: int, *, centroids: torch.Tensor) -> PatternKVOperatorReadyPagePools:
    page_size = 128
    pages = math.ceil(tokens / page_size)
    device = centroids.device
    valid = [page_size] * pages
    if pages:
        valid[-1] = tokens - page_size * (pages - 1)
    metadata = PatternKVBatchMetadata(
        request_indptr=torch.tensor([0, pages], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([tokens], dtype=torch.int32, device=device),
        num_pages=torch.tensor([pages], dtype=torch.int32, device=device),
        v2_page_table=torch.arange(pages, dtype=torch.int32, device=device).view(1, pages),
        v4_page_table=torch.full((1, pages), -1, dtype=torch.int32, device=device),
        metadata_page_table=torch.arange(pages, dtype=torch.int32, device=device).view(1, pages),
        precision_bitmap=torch.zeros((pages, 4), dtype=torch.int32, device=device),
        v2_counts=torch.tensor(valid, dtype=torch.int16, device=device),
        v4_counts=torch.zeros((pages,), dtype=torch.int16, device=device),
        valid_tokens=torch.tensor(valid, dtype=torch.int16, device=device),
        v4_prefix_counts=torch.zeros((pages, page_size + 1), dtype=torch.int16, device=device),
    )
    payload_tokens = max(tokens, 1)
    return PatternKVOperatorReadyPagePools(
        v2_payload_pool=torch.zeros((1, payload_tokens, 1, 1), dtype=torch.int32, device=device),
        v4_payload_pool=torch.zeros((1, 0, 1, 1), dtype=torch.int32, device=device),
        v2_scale_pool=torch.zeros((1, payload_tokens, 1, 1), device=device),
        v2_zero_pool=torch.zeros((1, payload_tokens, 1, 1), device=device),
        v4_scale_pool=torch.zeros((1, 0, 1, 1), device=device),
        v4_zero_pool=torch.zeros((1, 0, 1, 1), device=device),
        v2_pattern_pool=torch.zeros((1, payload_tokens, 1), dtype=torch.uint8, device=device),
        v4_pattern_pool=torch.zeros((1, 0, 1), dtype=torch.uint8, device=device),
        v2_assignment_pool=torch.zeros((1, payload_tokens, 1), dtype=torch.int32, device=device),
        v4_assignment_pool=torch.zeros((1, 0, 1), dtype=torch.int32, device=device),
        v2_page_offsets=torch.arange(pages, dtype=torch.int32, device=device) * page_size,
        v4_page_offsets=torch.full((pages,), -1, dtype=torch.int32, device=device),
        metadata=metadata,
        centroids=centroids,
        group_size=1,
        nh=1,
        nh_kv=1,
        head_dim=1,
        page_size=page_size,
    )


def _cache(total: int, *, packed: int | None = None, slot_value: float = 1.0) -> PatternQuantizedKVCache:
    packed = total - 128 if packed is None else packed
    recent = total - packed
    k_static = torch.tensor([[[slot_value], [slot_value + 1.0]]])
    v_static = torch.tensor([[[slot_value + 2.0], [slot_value + 3.0]]])
    pool = PatternKVCentroidStatePool.create(k_static, v_static, max_slots=1, max_dynamic_centroids=1)
    pool.allocate(torch.tensor([0]))
    pool.k_centroid_pool[0, 0, 2, 0] = slot_value + 10.0
    pool.v_centroid_pool[0, 0, 2, 0] = slot_value + 20.0
    pool.k_counts[0] = 3
    pool.v_counts[0] = 3
    cache = PatternQuantizedKVCache(
        sink_k=torch.zeros((1, 1, 0, 1)),
        sink_v=torch.zeros((1, 1, 0, 1)),
        packed_k=torch.zeros((1, 1, 1, packed), dtype=torch.int32),
        packed_k_scale=torch.zeros((1, 1, 1, packed)),
        packed_k_zero=torch.zeros((1, 1, 1, packed)),
        packed_v=torch.zeros((1, 1, packed, 1), dtype=torch.int32),
        packed_v_scale=torch.zeros((1, 1, packed, 1)),
        packed_v_zero=torch.zeros((1, 1, packed, 1)),
        pending_k=torch.zeros((1, 1, 0, 1)),
        pending_v=torch.zeros((1, 1, 0, 1)),
        recent_k=torch.zeros((1, 1, recent, 1)),
        recent_v=torch.zeros((1, 1, recent, 1)),
        total_tokens=total,
        packed_k_tokens=packed,
        packed_v_tokens=packed,
        sink_length=0,
        recent_length=128,
        group_size=1,
        k_bits=2,
        v_bits=2,
        k_assignments=torch.zeros((1, 1, packed), dtype=torch.long),
        v_assignments=torch.zeros((1, 1, packed), dtype=torch.uint8),
        v_assignment_idx=torch.zeros((1, 1, packed), dtype=torch.long),
        v_pattern_mask=torch.zeros((1, 1, packed), dtype=torch.uint8),
        k_centroids=pool.current_k(torch.tensor([0])),
        v_centroids=pool.current_v(torch.tensor([0])),
        v_precision_mask=torch.zeros((1, packed), dtype=torch.bool),
        packed_v4=torch.zeros((1, 1, 0, 1), dtype=torch.int32),
        packed_v4_scale=torch.zeros((1, 1, 0, 1)),
        packed_v4_zero=torch.zeros((1, 1, 0, 1)),
        packed_v4_tokens=0,
        operator_ready_page_pools=_page_pools(packed, centroids=pool.current_v(torch.tensor([0]))),
        centroid_state_pool=pool,
        centroid_state_indices=torch.tensor([0]),
    )
    set_request_total_tokens(cache, [total])
    cache.request_packed_k_tokens = torch.tensor([packed])
    cache.request_packed_v_tokens = torch.tensor([packed])
    cache.request_packed_v4_tokens = torch.tensor([0])
    return cache


def test_ragged_cache_preserves_per_request_total_tokens() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    assert get_total_tokens_per_request(cache).tolist() == [384, 513]
    assert cache.total_tokens == 513


def test_ragged_cache_b2_lengths_384_513() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    assert cache.request_packed_k_tokens.tolist() == [256, 385]
    assert cache.request_packed_v_tokens.tolist() == [256, 385]


def test_ragged_cache_b4_distinct_lengths() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513), _cache(642), _cache(771)])
    assert get_total_tokens_per_request(cache).tolist() == [384, 513, 642, 771]


def test_ragged_position_ids_b2() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    assert get_decode_position_ids(cache, 1).tolist() == [[384], [513]]


def test_ragged_position_ids_b4() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513), _cache(642), _cache(771)])
    assert get_decode_position_ids(cache, 1).tolist() == [[384], [513], [642], [771]]


def test_ragged_position_ids_reorder() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(771), _cache(384)])
    assert get_decode_position_ids(cache, 2).tolist() == [[771, 772], [384, 385]]


def test_explicit_position_ids_preserved() -> None:
    explicit = torch.tensor([[7], [9]])
    assert explicit.tolist() == [[7], [9]]


def test_ragged_decode_length_increment_independent() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    advance_request_total_tokens(cache, 1)
    assert get_total_tokens_per_request(cache).tolist() == [385, 514]


def test_ragged_centroid_slots_unique_after_assembly() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513), _cache(642)])
    assert cache.centroid_state_indices.tolist() == [0, 1, 2]


def test_ragged_centroid_state_copied_not_aliased() -> None:
    left = _cache(384, slot_value=11.0)
    cache = assemble_ragged_patternkv_cache([left, _cache(513, slot_value=21.0)])
    before = cache.centroid_state_pool.k_centroid_pool[0].clone()
    left.centroid_state_pool.k_centroid_pool[0].fill_(999.0)
    assert torch.equal(cache.centroid_state_pool.k_centroid_pool[0], before)


def test_ragged_page_indptr_after_pool_assembly() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    assert cache.operator_ready_page_pools.metadata.request_indptr.tolist() == [0, 2, 6]
    assert cache.operator_ready_page_pools.metadata.num_pages.tolist() == [2, 4]


def test_ragged_page_ownership_after_assembly() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    table = cache.operator_ready_page_pools.metadata.metadata_page_table
    assert table[0, :2].tolist() == [0, 1]
    assert table[1, :4].tolist() == [2, 3, 4, 5]
    assert table[0, 2:].tolist() == [-1, -1]


def test_ragged_serialize_deserialize_preserves_lengths() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(384), _cache(513)])
    restored = deserialize_cache(serialize_cache(cache), pattern=True)
    assert get_total_tokens_per_request(restored).tolist() == [384, 513]
    assert restored.operator_ready_page_pools.metadata.request_indptr.tolist() == [0, 2, 6]


def test_equal_length_cache_backward_compatible() -> None:
    cache = assemble_ragged_patternkv_cache([_cache(512), _cache(512)])
    assert cache.total_tokens == 512
    assert get_decode_position_ids(cache, 1).tolist() == [[512], [512]]
