from __future__ import annotations

import pytest
import torch

from models.request_lifecycle import DecodeReadyAdmission, DynamicAddRemoveBatchRunner, RequestLifecycleManager
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


def _make_cache_for_request(request_id: str) -> PatternQuantizedKVCache:
    index = ord(request_id) - ord("A")
    return _cache(384 + index * 129, 128 + index * 64, value=float(index + 1))


def _manager(names: list[str], *, capacity: int = 7) -> RequestLifecycleManager:
    manager = RequestLifecycleManager(max_slots=capacity)
    for name in names:
        manager.admit_decode_ready_request(name, lambda _slot, rid=name: _make_cache_for_request(rid))
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
        "recent_tail": cache.recent_k[:, :, -1:, :].detach().cpu().tolist() if cache.recent_k is not None and cache.recent_k.numel() else None,
        "pending_sum": float(cache.pending_k.float().sum().item()) if cache.pending_k is not None else 0.0,
        "importance": cache.v_causal_importance.detach().cpu().tolist() if cache.v_causal_importance is not None else None,
        "precision": cache.v_precision_mask.detach().cpu().tolist() if cache.v_precision_mask is not None else None,
        "centroid_count": int(pool.k_counts[0].item()) if pool is not None else None,
        "centroid_active": bool(pool.active[0].item()) if pool is not None else None,
    }


def _page_ownership(manager: RequestLifecycleManager, request_ids: list[str]) -> dict[str, dict[str, object]]:
    active, mapping = manager.build_active_cache(request_ids)
    lengths = k_segment_valid_lengths(active)
    out = {}
    page_cursor = 0
    for item in mapping:
        row = item.row_idx
        packed_v = int(active.request_packed_v_tokens[row].item())
        page_count = max((packed_v + 127) // 128, 1)
        pages = list(range(page_cursor, page_cursor + page_count))
        page_cursor += page_count
        out[item.request_id] = {
            "slot_id": item.slot_id,
            "seq_len": int(get_total_tokens_per_request(active)[row].item()),
            "segments": {key: int(value[row].item()) for key, value in lengths.items()},
            "pages": pages,
        }
    return out


def _decode_value(value: float):
    def decode(active_cache: PatternQuantizedKVCache, _mapping):
        batch = len(_mapping)
        key_states = torch.full((batch, 1, 1, 128), value)
        value_states = torch.full((batch, 1, 1, 128), value + 1.0)
        append_decode_rolling(active_cache, key_states, value_states)
        return active_cache

    return decode


def _run_iteration(
    runner: DynamicAddRemoveBatchRunner,
    order: list[str],
    *,
    value: float,
    finished: list[str] | None = None,
    admit: list[str] | None = None,
):
    admissions = [
        DecodeReadyAdmission(request_id=name, cache_factory=lambda _slot, rid=name: _make_cache_for_request(rid))
        for name in (admit or [])
    ]
    return runner.run_decode_iteration(
        active_order=order,
        finished_request_ids=finished or [],
        admissions=admissions,
        decode_step=_decode_value(value),
    )


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
    cache.centroid_state_pool.update_counts_k[0] = 5
    cache.centroid_state_pool.last_flush_pos[0] = 123


def test_dynamic_single_remove_add() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    _run_iteration(runner, ["A", "B", "C", "D"], value=100.0)
    before = {name: _snapshot(manager, name) for name in ["A", "C", "D"]}
    result = _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    assert result.active_request_ids == ("A", "C", "D", "E")
    assert "B" not in manager.request_to_slot
    assert {name: before[name]["slot_id"] for name in before} == {name: _snapshot(manager, name)["slot_id"] for name in before}
    assert manager.get_request(manager.get_slot("E")) == "E"


def test_dynamic_middle_remove_add() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    _run_iteration(runner, ["A", "B", "C", "D"], value=100.0)
    before_c = _snapshot(manager, "C")
    before_d = _snapshot(manager, "D")
    result = _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    rows = {item.request_id: item.row_idx for item in result.row_mapping}
    assert rows["C"] == 1
    assert rows["D"] == 2
    assert _snapshot(manager, "C")["slot_id"] == before_c["slot_id"]
    assert _snapshot(manager, "D")["slot_id"] == before_d["slot_id"]


def test_dynamic_multi_iteration_membership() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    r0 = _run_iteration(runner, ["A", "B", "C", "D"], value=100.0)
    r1 = _run_iteration(runner, ["A", "B", "C", "D"], value=110.0)
    r2 = _run_iteration(runner, ["A", "C", "D", "E"], value=120.0, finished=["B"], admit=["E"])
    r3 = _run_iteration(runner, ["A", "D", "E", "F"], value=130.0, finished=["C"], admit=["F"])
    r4 = _run_iteration(runner, ["D", "E", "F", "G"], value=140.0, finished=["A"], admit=["G"])
    assert [result.active_request_ids for result in (r0, r1, r2, r3, r4)] == [
        ("A", "B", "C", "D"),
        ("A", "B", "C", "D"),
        ("A", "C", "D", "E"),
        ("A", "D", "E", "F"),
        ("D", "E", "F", "G"),
    ]


def test_survivor_trajectory_preserved() -> None:
    dynamic = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(dynamic)
    d_initial = dynamic.get_slot("D")
    _run_iteration(runner, ["A", "B", "C", "D"], value=100.0)
    _run_iteration(runner, ["A", "C", "D", "E"], value=110.0, finished=["B"], admit=["E"])
    _run_iteration(runner, ["A", "D", "E", "F"], value=120.0, finished=["C"], admit=["F"])
    _run_iteration(runner, ["D", "E", "F", "G"], value=130.0, finished=["A"], admit=["G"])

    reference = _manager(["A", "B", "C", "D"])
    ref_runner = DynamicAddRemoveBatchRunner(reference)
    for value in (100.0, 110.0, 120.0, 130.0):
        _run_iteration(ref_runner, ["A", "B", "C", "D"], value=value)

    dyn = _snapshot(dynamic, "D")
    ref = _snapshot(reference, "D")
    assert dynamic.get_slot("D") == d_initial
    assert dyn["total"] == ref["total"]
    assert dyn["packed_k"] == ref["packed_k"]
    assert dyn["segments"] == ref["segments"]
    assert dyn["recent_tail"] == ref["recent_tail"]


def test_dynamic_slot_reuse_poison() -> None:
    manager = RequestLifecycleManager(max_slots=4)
    for name in ["A", "B", "C", "D"]:
        manager.admit_decode_ready_request(name, lambda _slot, rid=name: _make_cache_for_request(rid))
    b_slot = manager.get_slot("B")
    b_cache = manager.slots[b_slot].cache
    assert b_cache is not None
    _poison(b_cache)
    runner = DynamicAddRemoveBatchRunner(manager)
    result = _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    assert b_slot in result.released_slot_ids
    assert manager.get_slot("E") == b_slot
    e_cache = manager.slots[manager.get_slot("E")].cache
    clean = _make_cache_for_request("E")
    assert e_cache is not None
    assert int(e_cache.v_precision_mask.sum().item()) == 0
    assert e_cache.v_causal_importance is None
    assert get_total_tokens_per_request(e_cache).tolist() == [int(get_total_tokens_per_request(clean)[0].item()) + 1]
    assert torch.equal(e_cache.recent_k[:, :, :-1, :], clean.recent_k[:, :, 1:, :])


def test_dynamic_page_ownership() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    before = _page_ownership(manager, ["A", "B", "C", "D"])
    _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    after = _page_ownership(manager, ["A", "C", "D", "E"])
    assert "B" not in after
    for name in ["A", "C", "D"]:
        assert after[name]["slot_id"] == before[name]["slot_id"]
        assert after[name]["pages"]
    all_pages = [tuple(data["pages"]) for data in after.values()]
    assert len(all_pages) == len(set(all_pages))


def test_multiple_row_remaps() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    d_slot = manager.get_slot("D")
    rows = []
    for result in [
        _run_iteration(runner, ["A", "B", "C", "D"], value=100.0),
        _run_iteration(runner, ["A", "C", "D", "E"], value=110.0, finished=["B"], admit=["E"]),
        _run_iteration(runner, ["A", "D", "E", "F"], value=120.0, finished=["C"], admit=["F"]),
        _run_iteration(runner, ["D", "E", "F", "G"], value=130.0, finished=["A"], admit=["G"]),
    ]:
        rows.append({item.request_id: item.row_idx for item in result.row_mapping}["D"])
        assert manager.get_slot("D") == d_slot
    assert rows == [3, 2, 1, 0]


def test_admission_state_clean() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    e_cache = manager.slots[manager.get_slot("E")].cache
    clean = _make_cache_for_request("E")
    assert e_cache is not None
    assert get_total_tokens_per_request(e_cache).tolist() == [get_total_tokens_per_request(clean)[0].item() + 1]
    assert e_cache.v_causal_importance is None
    assert int(e_cache.v_precision_mask.sum().item()) == 0


def test_removed_request_cannot_reappear() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    _run_iteration(runner, ["A", "C", "D", "E"], value=200.0, finished=["B"], admit=["E"])
    with pytest.raises(KeyError):
        manager.build_active_row_mapping(["A", "B", "C"])
    with pytest.raises(KeyError):
        manager.decode_released_request("B")


def test_dynamic_true_batch_invariants() -> None:
    manager = _manager(["A", "B", "C", "D"])
    runner = DynamicAddRemoveBatchRunner(manager)
    seen_batches = []

    def decode(active_cache: PatternQuantizedKVCache, mapping):
        seen_batches.append(len(mapping))
        assert len(mapping) == 4
        key_states = torch.full((4, 1, 1, 128), 123.0)
        value_states = torch.full((4, 1, 1, 128), 124.0)
        append_decode_rolling(active_cache, key_states, value_states)
        return active_cache

    runner.run_decode_iteration(
        active_order=["A", "C", "D", "E"],
        finished_request_ids=["B"],
        admissions=[DecodeReadyAdmission("E", lambda _slot: _make_cache_for_request("E"))],
        decode_step=decode,
    )
    assert seen_batches == [4]
