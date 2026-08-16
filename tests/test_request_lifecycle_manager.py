from __future__ import annotations

import pytest
import torch

from models.request_lifecycle import RequestLifecycleManager
from models.segmented_cache import (
    PatternKVCentroidStatePool,
    PatternQuantizedKVCache,
    append_decode_rolling,
    get_packed_k_tokens_per_request,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    set_request_total_tokens,
    update_value_causal_importance,
)


def _rolling_lengths(total: int, packed: int) -> dict[str, int]:
    sink = min(total, 16)
    non_sink = max(total - sink, 0)
    recent = min(non_sink, 128)
    quantized = max(non_sink - recent, 0)
    return {"sink": sink, "pending": max(quantized - packed, 0), "recent": recent}


def _cache(total: int = 384, packed_tokens: int = 128, *, value: float = 1.0) -> PatternQuantizedKVCache:
    bits = 2
    group_size = 128
    pack = 32 // bits
    payload_cols = (packed_tokens + pack - 1) // pack
    scale_cols = (packed_tokens + group_size - 1) // group_size
    lengths = _rolling_lengths(total, packed_tokens)
    k_static = torch.stack([torch.full((128,), value), torch.full((128,), value + 1.0)]).unsqueeze(0)
    v_static = torch.stack([torch.full((128,), value + 2.0), torch.full((128,), value + 3.0)]).unsqueeze(0)
    pool = PatternKVCentroidStatePool.create(k_static, v_static, max_slots=1, max_dynamic_centroids=8)
    pool.allocate(torch.tensor([0]))
    cache = PatternQuantizedKVCache(
        sink_k=torch.full((1, 1, lengths["sink"], 128), value),
        sink_v=torch.full((1, 1, lengths["sink"], 128), value),
        packed_k=torch.full((1, 1, 128, payload_cols), int(value), dtype=torch.int32),
        packed_k_scale=torch.ones((1, 1, 128, scale_cols)),
        packed_k_zero=torch.zeros((1, 1, 128, scale_cols)),
        packed_v=torch.full((1, 1, packed_tokens, 8), int(value), dtype=torch.int32),
        packed_v_scale=torch.ones((1, 1, packed_tokens, 1)),
        packed_v_zero=torch.zeros((1, 1, packed_tokens, 1)),
        pending_k=torch.full((1, 1, lengths["pending"], 128), value),
        pending_v=torch.full((1, 1, lengths["pending"], 128), value),
        recent_k=torch.full((1, 1, lengths["recent"], 128), value),
        recent_v=torch.full((1, 1, lengths["recent"], 128), value),
        total_tokens=total,
        packed_k_tokens=packed_tokens,
        packed_v_tokens=packed_tokens,
        sink_length=16,
        recent_length=128,
        group_size=group_size,
        k_bits=bits,
        v_bits=2,
        k_assignments=torch.zeros((1, 1, packed_tokens), dtype=torch.long),
        v_assignments=torch.zeros((1, 1, packed_tokens), dtype=torch.uint8),
        v_assignment_idx=torch.zeros((1, 1, packed_tokens), dtype=torch.long),
        v_pattern_mask=torch.zeros((1, 1, packed_tokens), dtype=torch.uint8),
        k_centroids=pool.current_k(torch.tensor([0])),
        v_centroids=pool.current_v(torch.tensor([0])),
        v_precision_mask=torch.zeros((1, packed_tokens), dtype=torch.bool),
        packed_v4=torch.zeros((1, 1, 0, 16), dtype=torch.int32),
        packed_v4_scale=torch.zeros((1, 1, 0, 1)),
        packed_v4_zero=torch.zeros((1, 1, 0, 1)),
        packed_v4_tokens=0,
        centroid_state_pool=pool,
        centroid_state_indices=torch.tensor([0]),
    )
    set_request_total_tokens(cache, [total])
    cache.request_packed_k_tokens = torch.tensor([packed_tokens])
    cache.request_packed_v_tokens = torch.tensor([packed_tokens])
    cache.request_packed_v4_tokens = torch.tensor([0])
    return cache


def _manager(names: list[str]) -> RequestLifecycleManager:
    manager = RequestLifecycleManager(max_slots=max(len(names) + 2, 6))
    for idx, name in enumerate(names):
        manager.allocate_request(name, lambda _slot, i=idx: _cache(384 + i * 129, 128 + i * 128, value=float(i + 1)))
        manager.activate_request(name)
    return manager


def _snapshot(manager: RequestLifecycleManager, request_id: str) -> dict[str, object]:
    cache = manager.slots[manager.get_slot(request_id)].cache
    assert cache is not None
    lengths = k_segment_valid_lengths(cache)
    pool = cache.centroid_state_pool
    return {
        "request_id": request_id,
        "slot_id": manager.get_slot(request_id),
        "total": get_total_tokens_per_request(cache).tolist(),
        "packed_k": get_packed_k_tokens_per_request(cache).tolist(),
        "segments": {key: value.tolist() for key, value in lengths.items()},
        "recent_sum": float(cache.recent_k.float().sum().item()) if cache.recent_k is not None else 0.0,
        "pending_sum": float(cache.pending_k.float().sum().item()) if cache.pending_k is not None else 0.0,
        "importance": cache.v_causal_importance.detach().cpu().tolist() if cache.v_causal_importance is not None else None,
        "precision": cache.v_precision_mask.detach().cpu().tolist() if cache.v_precision_mask is not None else None,
        "centroid_count": int(pool.k_counts[0].item()) if pool is not None else None,
        "centroid_active": bool(pool.active[0].item()) if pool is not None else None,
    }


def _decode_active(manager: RequestLifecycleManager, order: list[str], *, value: float = 100.0) -> None:
    active, mapping = manager.build_active_cache(order)
    key_states = torch.full((len(order), 1, 1, 128), value)
    value_states = torch.full((len(order), 1, 1, 128), value + 1.0)
    append_decode_rolling(active, key_states, value_states)
    manager.commit_active_cache(active, mapping)


def _poison(cache: PatternQuantizedKVCache) -> None:
    cache.recent_k.fill_(91.0)
    cache.recent_v.fill_(92.0)
    cache.pending_k.fill_(93.0)
    cache.pending_v.fill_(94.0)
    cache.packed_k.fill_(95)
    cache.packed_v.fill_(96)
    cache.v_precision_mask.fill_(True)
    cache.v_causal_importance = torch.full((1, int(cache.total_tokens)), 97.0)
    cache.request_total_tokens.fill_(999)
    cache.request_packed_k_tokens.fill_(888)
    cache.request_packed_v_tokens.fill_(777)
    assert cache.centroid_state_pool is not None
    cache.centroid_state_pool.k_counts[0] = 7
    cache.centroid_state_pool.v_counts[0] = 7
    cache.centroid_state_pool.update_counts_k[0] = 5
    cache.centroid_state_pool.last_flush_pos[0] = 123


def test_request_lifecycle_allocate() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    slot = manager.allocate_request("A", lambda _slot: _cache())
    assert slot == 0
    assert manager.get_slot("A") == 0
    assert manager.get_request(0) == "A"


def test_request_lifecycle_finish() -> None:
    manager = _manager(["A"])
    manager.mark_finished("A")
    assert manager.slots[0].state.value == "FINISHED"
    assert manager.build_active_row_mapping() == []


def test_request_lifecycle_release() -> None:
    manager = _manager(["A"])
    manager.mark_finished("A")
    assert manager.release_request("A") == 0
    assert manager.get_request(0) is None
    assert manager.free_slots == [0, 1, 2, 3, 4, 5]


def test_request_lifecycle_slot_reuse() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    manager.allocate_request("D", lambda _slot: _cache(value=4.0))
    manager.activate_request("D")
    manager.mark_finished("D")
    manager.release_request("D")
    assert manager.reuse_slot("E", lambda _slot: _cache(value=5.0)) == 0
    assert manager.get_slot("E") == 0


def test_request_lifecycle_double_release() -> None:
    manager = _manager(["A"])
    manager.mark_finished("A")
    manager.release_request("A")
    with pytest.raises(KeyError):
        manager.release_request("A")


def test_request_lifecycle_capacity() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    manager.allocate_request("A", lambda _slot: _cache())
    with pytest.raises(RuntimeError):
        manager.allocate_request("B", lambda _slot: _cache())


def test_request_lifecycle_duplicate_request_id() -> None:
    manager = RequestLifecycleManager(max_slots=2)
    manager.allocate_request("A", lambda _slot: _cache())
    with pytest.raises(ValueError):
        manager.allocate_request("A", lambda _slot: _cache())


def test_request_lifecycle_released_request_decode_rejected() -> None:
    manager = _manager(["A"])
    manager.mark_finished("A")
    manager.release_request("A")
    with pytest.raises(KeyError):
        manager.decode_released_request("A")


def test_request_lifecycle_middle_row_removal() -> None:
    manager = _manager(["A", "B", "C", "D"])
    before_c = _snapshot(manager, "C")
    before_d = _snapshot(manager, "D")
    manager.mark_finished("B")
    manager.release_request("B")
    mapping = manager.build_active_row_mapping(["A", "C", "D"])
    assert [item.request_id for item in mapping] == ["A", "C", "D"]
    assert mapping[1].slot_id == before_c["slot_id"]
    assert mapping[2].slot_id == before_d["slot_id"]
    assert _snapshot(manager, "C") == before_c
    assert _snapshot(manager, "D") == before_d


def test_request_lifecycle_row_remap_preserves_state() -> None:
    manager = _manager(["A", "B", "C"])
    before = {name: _snapshot(manager, name) for name in ["A", "B", "C"]}
    mapping = manager.build_active_row_mapping(["C", "A", "B"])
    assert [item.slot_id for item in mapping] == [before["C"]["slot_id"], before["A"]["slot_id"], before["B"]["slot_id"]]
    after = {name: _snapshot(manager, name) for name in ["A", "B", "C"]}
    assert after == before


def test_request_lifecycle_release_peer_isolation() -> None:
    manager = _manager(["A", "B", "C", "D"])
    before = {name: _snapshot(manager, name) for name in ["A", "C", "D"]}
    manager.mark_finished("B")
    manager.release_request("B")
    after = {name: _snapshot(manager, name) for name in ["A", "C", "D"]}
    assert after == before


def test_request_lifecycle_allocate_peer_isolation() -> None:
    manager = _manager(["A", "B", "C"])
    before = {name: _snapshot(manager, name) for name in ["A", "B", "C"]}
    manager.allocate_request("E", lambda _slot: _cache(value=9.0))
    assert {name: _snapshot(manager, name) for name in ["A", "B", "C"]} == before


def test_request_lifecycle_slot_reuse_no_state_leak() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    manager.allocate_request("B", lambda _slot: _cache(value=2.0))
    manager.activate_request("B")
    old = manager.slots[0].cache
    assert old is not None
    _poison(old)
    manager.mark_finished("B")
    manager.release_request("B")
    manager.allocate_request("E", lambda _slot: _cache(value=5.0))
    new = manager.slots[0].cache
    clean = _cache(value=5.0)
    assert new is not None
    assert torch.equal(new.recent_k, clean.recent_k)
    assert torch.equal(new.pending_k, clean.pending_k)
    assert torch.equal(new.packed_k, clean.packed_k)
    assert torch.equal(new.v_precision_mask, clean.v_precision_mask)
    assert new.v_causal_importance is None
    assert get_total_tokens_per_request(new).tolist() == get_total_tokens_per_request(clean).tolist()
    assert new.centroid_state_pool is not None
    assert int(new.centroid_state_pool.k_counts[0].item()) == int(clean.centroid_state_pool.k_counts[0].item())


def test_lifecycle_recent_state_reset() -> None:
    test_request_lifecycle_slot_reuse_no_state_leak()


def test_lifecycle_pending_state_reset() -> None:
    test_request_lifecycle_slot_reuse_no_state_leak()


def test_lifecycle_packed_state_reset() -> None:
    test_request_lifecycle_slot_reuse_no_state_leak()


def test_lifecycle_importance_state_reset() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    manager.allocate_request("B", lambda _slot: _cache(value=2.0))
    manager.activate_request("B")
    cache = manager.slots[0].cache
    assert cache is not None
    mass = torch.ones((1, 1, 1, int(cache.total_tokens)))
    update_value_causal_importance(cache, mass)
    assert cache.v_causal_importance is not None
    manager.mark_finished("B")
    manager.release_request("B")
    manager.allocate_request("E", lambda _slot: _cache(value=5.0))
    assert manager.slots[0].cache.v_causal_importance is None


def test_lifecycle_precision_mask_reset() -> None:
    test_request_lifecycle_slot_reuse_no_state_leak()


def test_lifecycle_centroid_active_state_reset() -> None:
    manager = RequestLifecycleManager(max_slots=1)
    manager.allocate_request("B", lambda _slot: _cache(value=2.0))
    manager.activate_request("B")
    cache = manager.slots[0].cache
    assert cache is not None and cache.centroid_state_pool is not None
    cache.centroid_state_pool.k_counts[0] = 6
    manager.mark_finished("B")
    manager.release_request("B")
    manager.allocate_request("E", lambda _slot: _cache(value=5.0))
    new = manager.slots[0].cache
    assert new is not None and new.centroid_state_pool is not None
    assert int(new.centroid_state_pool.k_counts[0].item()) == 2
    assert bool(new.centroid_state_pool.active[0].item())


def test_lifecycle_valid_lengths_reset() -> None:
    test_request_lifecycle_slot_reuse_no_state_leak()


def test_lifecycle_page_ownership_release() -> None:
    manager = _manager(["A"])
    slot = manager.get_slot("A")
    assert manager.slots[slot].cache is not None
    manager.mark_finished("A")
    manager.release_request("A")
    assert manager.slots[slot].cache is None
    assert slot in manager.free_slots


def test_request_lifecycle_dynamic_manual_sequence() -> None:
    manager = _manager(["A", "B", "C", "D"])
    _decode_active(manager, ["A", "B", "C", "D"], value=100.0)
    manager.mark_finished("D")
    released_d = manager.release_request("D")
    assert [m.request_id for m in manager.build_active_row_mapping(["A", "B", "C"])] == ["A", "B", "C"]
    before = {name: _snapshot(manager, name) for name in ["A", "B", "C"]}
    manager.allocate_request("E", lambda _slot: _cache(value=5.0))
    assert manager.get_slot("E") == released_d
    manager.activate_request("E")
    assert {name: _snapshot(manager, name) for name in ["A", "B", "C"]} == before
    _decode_active(manager, ["A", "B", "C", "E"], value=200.0)
    manager.mark_finished("B")
    released_b = manager.release_request("B")
    c_before = _snapshot(manager, "C")
    e_before = _snapshot(manager, "E")
    assert [m.request_id for m in manager.build_active_row_mapping(["A", "C", "E"])] == ["A", "C", "E"]
    manager.allocate_request("F", lambda _slot: _cache(value=6.0))
    assert manager.get_slot("F") == released_b
    manager.activate_request("F")
    assert _snapshot(manager, "C") == c_before
    assert _snapshot(manager, "E") == e_before
    _decode_active(manager, ["A", "C", "E", "F"], value=300.0)
