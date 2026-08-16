from __future__ import annotations

import pytest
import torch

from models.request_lifecycle import ContinuousBatchScheduler, DecodeReadyAdmission, RequestLifecycleManager
from models.segmented_cache import (
    PatternKVCentroidStatePool,
    PatternQuantizedKVCache,
    append_decode_rolling,
    get_packed_k_tokens_per_request,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    set_request_total_tokens,
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


def _admission(request_id: str) -> DecodeReadyAdmission:
    return DecodeReadyAdmission(request_id, lambda _slot, rid=request_id: _make_cache_for_request(rid))


def _scheduler(names: list[str], *, capacity: int = 4, max_slots: int | None = None) -> ContinuousBatchScheduler:
    scheduler = ContinuousBatchScheduler(RequestLifecycleManager(max_slots=max_slots or capacity), capacity)
    scheduler.submit_many(_admission(name) for name in names)
    return scheduler


def _decode_value(value: float, seen_batches: list[int] | None = None):
    def decode(active_cache: PatternQuantizedKVCache, mapping):
        if seen_batches is not None:
            seen_batches.append(len(mapping))
        batch = len(mapping)
        key_states = torch.full((batch, 1, 1, 128), value)
        value_states = torch.full((batch, 1, 1, 128), value + 1.0)
        append_decode_rolling(active_cache, key_states, value_states)
        return active_cache

    return decode


def _snapshot(scheduler: ContinuousBatchScheduler, request_id: str) -> dict[str, object]:
    manager = scheduler.lifecycle
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


def _page_ownership(scheduler: ContinuousBatchScheduler) -> dict[str, dict[str, object]]:
    active, mapping = scheduler.lifecycle.build_active_cache(scheduler.running_request_ids)
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


def _poison(cache: PatternQuantizedKVCache) -> None:
    cache.recent_k.fill_(91.0)
    cache.recent_v.fill_(92.0)
    cache.pending_k.fill_(93.0)
    cache.pending_v.fill_(94.0)
    cache.packed_k.fill_(95)
    cache.packed_v.fill_(96)
    assert cache.centroid_state_pool is not None
    cache.centroid_state_pool.k_counts[0] = 7
    cache.centroid_state_pool.update_counts_k[0] = 5
    cache.centroid_state_pool.last_flush_pos[0] = 123


def test_scheduler_waiting_running_finished_transitions() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    assert result.decoded_request_ids == ("A", "B", "C", "D")
    assert result.running_request_ids == ("A", "C", "D", "E")
    assert result.waiting_request_ids == ("F", "G")
    assert result.finished_archive == ("B",)


def test_automatic_finished_removal() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E"], max_slots=5)
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    assert "B" not in result.running_request_ids
    assert "B" not in [item.request_id for item in result.next_row_mapping]
    with pytest.raises(KeyError):
        scheduler.lifecycle.decode_released_request("B")


def test_automatic_waiting_admission() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E"], max_slots=5)
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    assert result.admitted_request_ids == ("A", "B", "C", "D", "E")
    assert result.running_request_ids == ("A", "C", "D", "E")
    assert scheduler.lifecycle.get_request(scheduler.lifecycle.get_slot("E")) == "E"


def test_continuous_refill() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    r0 = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    r1 = scheduler.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["C"])
    r2 = scheduler.advance_iteration(decode_step=_decode_value(120.0), finished_request_ids=["A"])
    assert r0.running_request_ids == ("A", "C", "D", "E")
    assert r1.running_request_ids == ("A", "D", "E", "F")
    assert r2.running_request_ids == ("D", "E", "F", "G")


def test_multi_finish_multi_admit() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B", "C"])
    assert result.running_request_ids == ("A", "D", "E", "F")
    assert result.waiting_request_ids == ("G",)
    assert result.finished_archive == ("B", "C")


def test_waiting_queue_order() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B", "C"])
    assert result.admitted_request_ids[-2:] == ("E", "F")
    assert result.waiting_request_ids == ("G",)


def test_late_arrival_waits_until_capacity() -> None:
    scheduler = _scheduler(["A", "B", "C", "D"], max_slots=6)
    r0 = scheduler.advance_iteration(decode_step=_decode_value(100.0))
    assert r0.running_request_ids == ("A", "B", "C", "D")
    scheduler.submit_many([_admission("E"), _admission("F")])
    assert scheduler.waiting_request_ids() == ("E", "F")
    r1 = scheduler.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["B"])
    assert r1.running_request_ids == ("A", "C", "D", "E")
    assert r1.waiting_request_ids == ("F",)


def test_scheduler_slot_reuse_poison() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E"], max_slots=4)
    scheduler.advance_iteration(decode_step=_decode_value(100.0))
    b_slot = scheduler.lifecycle.get_slot("B")
    b_cache = scheduler.lifecycle.slots[b_slot].cache
    assert b_cache is not None
    _poison(b_cache)
    result = scheduler.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["B"])
    assert b_slot in result.released_slot_ids
    assert scheduler.lifecycle.get_slot("E") in result.released_slot_ids
    e_cache = scheduler.lifecycle.slots[scheduler.lifecycle.get_slot("E")].cache
    clean = _make_cache_for_request("E")
    assert e_cache is not None
    assert int(e_cache.v_precision_mask.sum().item()) == 0
    assert e_cache.v_causal_importance is None
    assert get_total_tokens_per_request(e_cache).tolist() == get_total_tokens_per_request(clean).tolist()


def test_scheduler_page_ownership() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E"], max_slots=5)
    scheduler.advance_iteration(decode_step=_decode_value(100.0))
    before = _page_ownership(scheduler)
    result = scheduler.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["B"])
    after = _page_ownership(scheduler)
    assert "B" not in after
    assert "E" in after
    for name in ["A", "C", "D"]:
        assert after[name]["slot_id"] == before[name]["slot_id"]
    all_pages = [tuple(data["pages"]) for data in after.values()]
    assert len(all_pages) == len(set(all_pages))
    assert result.running_request_ids == ("A", "C", "D", "E")


def test_iteration_metadata_rebuild() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F"], max_slots=6)
    r0 = scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    r1 = scheduler.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["C"])
    assert [item.request_id for item in r0.next_row_mapping] == ["A", "C", "D", "E"]
    assert [item.row_idx for item in r0.next_row_mapping] == [0, 1, 2, 3]
    assert [item.request_id for item in r1.next_row_mapping] == ["A", "D", "E", "F"]
    assert [item.row_idx for item in r1.next_row_mapping] == [0, 1, 2, 3]
    assert [item.slot_id for item in r1.next_row_mapping] == [scheduler.lifecycle.get_slot(name) for name in ["A", "D", "E", "F"]]


def test_survivor_trajectory_preserved() -> None:
    dynamic = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    dynamic.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["B"])
    dynamic.advance_iteration(decode_step=_decode_value(110.0), finished_request_ids=["C"])
    dynamic.advance_iteration(decode_step=_decode_value(120.0), finished_request_ids=["A"])
    dynamic.advance_iteration(decode_step=_decode_value(130.0))

    reference = _scheduler(["A", "B", "C", "D"], max_slots=4)
    for value in (100.0, 110.0, 120.0, 130.0):
        reference.advance_iteration(decode_step=_decode_value(value))

    dyn = _snapshot(dynamic, "D")
    ref = _snapshot(reference, "D")
    assert dyn["slot_id"] == 3
    assert dyn["total"] == ref["total"]
    assert dyn["packed_k"] == ref["packed_k"]
    assert dyn["segments"] == ref["segments"]
    assert dyn["recent_tail"] == ref["recent_tail"]


def test_multiple_row_remaps() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E", "F", "G"], max_slots=7)
    rows = []
    for value, finished in [(100.0, ["B"]), (110.0, ["C"]), (120.0, ["A"]), (130.0, [])]:
        result = scheduler.advance_iteration(decode_step=_decode_value(value), finished_request_ids=finished)
        rows.append({item.request_id: item.row_idx for item in result.decoded_row_mapping}["D"])
        assert scheduler.lifecycle.get_slot("D") == 3
    assert rows == [3, 2, 1, 0]


def test_partial_batch() -> None:
    scheduler = _scheduler(["A", "B"], capacity=4, max_slots=4)
    seen_batches: list[int] = []
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0, seen_batches))
    assert result.running_request_ids == ("A", "B")
    assert seen_batches == [2]


def test_scheduler_drains_to_empty() -> None:
    scheduler = _scheduler(["A", "B"], capacity=4, max_slots=4)
    scheduler.advance_iteration(decode_step=_decode_value(100.0), finished_request_ids=["A", "B"])
    assert not scheduler.has_work()
    assert scheduler.running_request_ids == []
    assert scheduler.waiting_request_ids() == ()
    assert scheduler.finished_request_ids == ["A", "B"]
    result = scheduler.advance_iteration(decode_step=_decode_value(110.0))
    assert not result.decode_executed


def test_true_batch_production_invariants() -> None:
    scheduler = _scheduler(["A", "B", "C", "D", "E"], max_slots=5)
    seen_batches: list[int] = []
    result = scheduler.advance_iteration(decode_step=_decode_value(100.0, seen_batches), finished_request_ids=["B"])
    assert seen_batches == [4]
    assert result.running_request_ids == ("A", "C", "D", "E")
    assert {
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "serial_rmsnorm_request_dispatches": 0,
        "historical_fp16_k_materialization": 0,
        "historical_fp16_v_materialization": 0,
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
    }
