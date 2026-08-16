from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import ceil
from typing import Any, Callable, Iterable, Sequence

import torch

from models.segmented_cache import (
    PatternQuantizedKVCache,
    assemble_ragged_patternkv_cache,
    cache_batch_size,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    _slice_ragged_request_cache,
    set_request_total_tokens,
    validate_cache,
)


class RequestLifecycleState(str, Enum):
    FREE = "FREE"
    ALLOCATED = "ALLOCATED"
    ACTIVE = "ACTIVE"
    FINISHED = "FINISHED"
    RELEASED = "RELEASED"


@dataclass(frozen=True)
class ActiveRowMapping:
    row_idx: int
    request_id: str
    slot_id: int
    generation: int


@dataclass
class RequestSlot:
    slot_id: int
    state: RequestLifecycleState = RequestLifecycleState.FREE
    request_id: str | None = None
    generation: int = 0
    cache: PatternQuantizedKVCache | None = None


@dataclass(frozen=True)
class DecodeReadyAdmission:
    request_id: str
    cache_factory: Callable[[int], PatternQuantizedKVCache] | PatternQuantizedKVCache


@dataclass(frozen=True)
class DynamicBatchIterationResult:
    iteration: int
    active_request_ids: tuple[str, ...]
    row_mapping: tuple[ActiveRowMapping, ...]
    removed_request_ids: tuple[str, ...]
    admitted_request_ids: tuple[str, ...]
    released_slot_ids: tuple[int, ...]
    admitted_slot_ids: tuple[int, ...]


@dataclass(frozen=True)
class ContinuousBatchIterationResult:
    iteration: int
    decoded_request_ids: tuple[str, ...]
    decoded_row_mapping: tuple[ActiveRowMapping, ...]
    finished_request_ids: tuple[str, ...]
    released_slot_ids: tuple[int, ...]
    admitted_request_ids: tuple[str, ...]
    admitted_slot_ids: tuple[int, ...]
    running_request_ids: tuple[str, ...]
    waiting_request_ids: tuple[str, ...]
    finished_archive: tuple[str, ...]
    next_row_mapping: tuple[ActiveRowMapping, ...]
    decode_executed: bool


def _slice_batch(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    return value[row : row + 1].contiguous()


def _slice_tokens(value: torch.Tensor | None, row: int, token_dim: int, tokens: int) -> torch.Tensor | None:
    sliced = _slice_batch(value, row)
    if not torch.is_tensor(sliced):
        return None
    dim = token_dim if token_dim >= 0 else sliced.dim() + token_dim
    return sliced.narrow(dim, 0, max(int(tokens), 0)).contiguous()


def _slice_centroid_pool(cache: PatternQuantizedKVCache, row: int):
    pool = cache.centroid_state_pool
    slots = cache.centroid_state_indices
    if pool is None or not torch.is_tensor(slots):
        return None, None
    slot = slots[row : row + 1].long()
    k_static = pool.k_centroid_pool[slot, :, : int(pool.static_centroid_count), :].squeeze(0).contiguous()
    v_static_count = int(pool.static_v_centroid_count or pool.static_centroid_count)
    v_static = pool.v_centroid_pool[slot, :, :v_static_count, :].squeeze(0).contiguous()
    new_pool = type(pool).create(
        k_static,
        v_static,
        max_slots=1,
        max_dynamic_centroids=max(int(pool.k_centroid_pool.shape[2]) - int(pool.static_centroid_count), 0),
    )
    new_pool.allocate(torch.tensor([0], dtype=torch.long, device=k_static.device))
    source = int(slot[0].item())
    new_pool.k_centroid_pool[0].copy_(pool.k_centroid_pool[source])
    new_pool.v_centroid_pool[0].copy_(pool.v_centroid_pool[source])
    new_pool.k_counts[0] = pool.k_counts[source]
    new_pool.v_counts[0] = pool.v_counts[source]
    new_pool.update_counts_k[0] = pool.update_counts_k[source]
    new_pool.update_counts_v[0] = pool.update_counts_v[source]
    new_pool.last_flush_pos[0] = pool.last_flush_pos[source]
    new_pool.active[0] = pool.active[source]
    return new_pool, torch.tensor([0], dtype=torch.long, device=k_static.device)


def _request_scalar(cache: PatternQuantizedKVCache, name: str, fallback: int, row: int) -> int:
    value = getattr(cache, name, None)
    if torch.is_tensor(value):
        return int(value[row].item())
    return int(fallback)


def extract_request_cache(cache: PatternQuantizedKVCache, row: int) -> PatternQuantizedKVCache:
    if not isinstance(cache, PatternQuantizedKVCache):
        raise TypeError("request lifecycle currently requires PatternQuantizedKVCache")
    row = int(row)
    if row < 0 or row >= cache_batch_size(cache):
        raise IndexError(f"row_idx out of range: {row}")
    totals = get_total_tokens_per_request(cache)
    lengths = k_segment_valid_lengths(cache)
    row_cache = _slice_ragged_request_cache(cache, row, lengths)
    if getattr(row_cache, "operator_ready_page_pools", None) is None:
        validate_cache(row_cache)
    return row_cache


class RequestLifecycleManager:
    def __init__(self, max_slots: int) -> None:
        if int(max_slots) <= 0:
            raise ValueError("max_slots must be positive")
        self.max_slots = int(max_slots)
        self.slots = [RequestSlot(slot_id=slot_id) for slot_id in range(self.max_slots)]
        self.free_slots = list(range(self.max_slots))
        self.request_to_slot: dict[str, int] = {}
        self.active_request_ids: list[str] = []

    def allocate_request(
        self,
        request_id: str,
        cache_factory: Callable[[int], PatternQuantizedKVCache] | PatternQuantizedKVCache | None = None,
    ) -> int:
        request_id = str(request_id)
        if request_id in self.request_to_slot:
            raise ValueError(f"duplicate request_id: {request_id}")
        if not self.free_slots:
            raise RuntimeError("request slot capacity exhausted")
        slot_id = self.free_slots.pop(0)
        slot = self.slots[slot_id]
        slot.generation += 1
        slot.state = RequestLifecycleState.ALLOCATED
        slot.request_id = request_id
        slot.cache = self._build_cache(cache_factory, slot_id)
        self.request_to_slot[request_id] = slot_id
        self.validate_lifecycle_state()
        return slot_id

    def activate_request(self, request_id: str) -> None:
        slot = self._slot_for_request(request_id)
        if slot.state not in (RequestLifecycleState.ALLOCATED, RequestLifecycleState.ACTIVE):
            raise RuntimeError(f"request {request_id} cannot be activated from {slot.state.value}")
        if slot.cache is None:
            raise RuntimeError(f"request {request_id} has no allocated cache state")
        slot.state = RequestLifecycleState.ACTIVE
        if request_id not in self.active_request_ids:
            self.active_request_ids.append(request_id)
        self.validate_lifecycle_state()

    def mark_finished(self, request_id: str) -> None:
        slot = self._slot_for_request(request_id)
        if slot.state == RequestLifecycleState.FINISHED:
            raise RuntimeError(f"request {request_id} already finished")
        if slot.state != RequestLifecycleState.ACTIVE:
            raise RuntimeError(f"request {request_id} cannot finish from {slot.state.value}")
        slot.state = RequestLifecycleState.FINISHED
        self.active_request_ids = [rid for rid in self.active_request_ids if rid != request_id]
        self.validate_lifecycle_state()

    def release_request(self, request_id: str) -> int:
        slot = self._slot_for_request(request_id)
        if slot.state != RequestLifecycleState.FINISHED:
            raise RuntimeError(f"request {request_id} cannot be released from {slot.state.value}")
        slot_id = slot.slot_id
        if slot.cache is not None and slot.cache.centroid_state_pool is not None and torch.is_tensor(slot.cache.centroid_state_indices):
            slot.cache.centroid_state_pool.free(slot.cache.centroid_state_indices.long())
        slot.cache = None
        slot.request_id = None
        slot.state = RequestLifecycleState.RELEASED
        del self.request_to_slot[request_id]
        if slot_id in self.free_slots:
            raise RuntimeError(f"slot {slot_id} is already free")
        self.free_slots.insert(0, slot_id)
        slot.state = RequestLifecycleState.FREE
        self.validate_lifecycle_state()
        return slot_id

    def reuse_slot(self, request_id: str, cache_factory: Callable[[int], PatternQuantizedKVCache] | PatternQuantizedKVCache | None = None) -> int:
        return self.allocate_request(request_id, cache_factory)

    def get_slot(self, request_id: str) -> int:
        return self._slot_for_request(request_id).slot_id

    def get_request(self, slot_id: int) -> str | None:
        return self.slots[int(slot_id)].request_id

    def build_active_row_mapping(self, request_ids: Iterable[str] | None = None) -> list[ActiveRowMapping]:
        ids = list(self.active_request_ids if request_ids is None else request_ids)
        mappings = []
        seen: set[str] = set()
        for row_idx, request_id in enumerate(ids):
            if request_id in seen:
                raise ValueError(f"duplicate active request_id: {request_id}")
            seen.add(request_id)
            slot = self._slot_for_request(request_id)
            if slot.state != RequestLifecycleState.ACTIVE:
                raise RuntimeError(f"request {request_id} is not active")
            mappings.append(ActiveRowMapping(row_idx=row_idx, request_id=request_id, slot_id=slot.slot_id, generation=slot.generation))
        return mappings

    def build_active_cache(self, request_ids: Iterable[str] | None = None) -> tuple[PatternQuantizedKVCache, list[ActiveRowMapping]]:
        mappings = self.build_active_row_mapping(request_ids)
        caches = []
        for mapping in mappings:
            cache = self.slots[mapping.slot_id].cache
            if cache is None:
                raise RuntimeError(f"active request {mapping.request_id} has no cache")
            caches.append(cache)
        return assemble_ragged_patternkv_cache(caches), mappings

    def commit_active_cache(self, active_cache: PatternQuantizedKVCache, mappings: list[ActiveRowMapping]) -> None:
        if cache_batch_size(active_cache) != len(mappings):
            raise ValueError("active cache batch size does not match row mapping")
        for mapping in mappings:
            slot = self.slots[mapping.slot_id]
            if slot.request_id != mapping.request_id or slot.generation != mapping.generation or slot.state != RequestLifecycleState.ACTIVE:
                raise RuntimeError(f"stale or invalid row mapping for request {mapping.request_id}")
            slot.cache = extract_request_cache(active_cache, mapping.row_idx)
        self.validate_lifecycle_state()

    def finish_and_release_requests(self, request_ids: Iterable[str]) -> list[int]:
        released = []
        for request_id in request_ids:
            self.mark_finished(request_id)
            released.append(self.release_request(request_id))
        return released

    def admit_decode_ready_request(
        self,
        request_id: str,
        cache_factory: Callable[[int], PatternQuantizedKVCache] | PatternQuantizedKVCache,
    ) -> int:
        slot_id = self.allocate_request(request_id, cache_factory)
        self.activate_request(request_id)
        return slot_id

    def decode_released_request(self, request_id: str) -> None:
        slot = self._slot_for_request(request_id)
        if slot.state != RequestLifecycleState.ACTIVE:
            raise RuntimeError(f"request {request_id} is not active")

    def validate_lifecycle_state(self) -> None:
        if len(self.free_slots) != len(set(self.free_slots)):
            raise RuntimeError("duplicate free slot entry")
        owned_slots = set(self.request_to_slot.values())
        if len(owned_slots) != len(self.request_to_slot):
            raise RuntimeError("multiple requests own the same slot")
        for slot_id in self.free_slots:
            slot = self.slots[slot_id]
            if slot.state != RequestLifecycleState.FREE or slot.request_id is not None or slot.cache is not None:
                raise RuntimeError(f"free slot {slot_id} still has live ownership")
        for request_id, slot_id in self.request_to_slot.items():
            slot = self.slots[slot_id]
            if slot.request_id != request_id:
                raise RuntimeError(f"request_to_slot mismatch for {request_id}")
            if slot.state == RequestLifecycleState.FREE:
                raise RuntimeError(f"owned slot {slot_id} is marked free")
        active_seen: set[str] = set()
        for request_id in self.active_request_ids:
            if request_id in active_seen:
                raise RuntimeError(f"duplicate active request: {request_id}")
            active_seen.add(request_id)
            slot = self._slot_for_request(request_id)
            if slot.state != RequestLifecycleState.ACTIVE:
                raise RuntimeError(f"active mapping includes non-active request {request_id}")

    def _slot_for_request(self, request_id: str) -> RequestSlot:
        request_id = str(request_id)
        if request_id not in self.request_to_slot:
            raise KeyError(f"unknown request_id: {request_id}")
        return self.slots[self.request_to_slot[request_id]]

    def _build_cache(
        self,
        cache_factory: Callable[[int], PatternQuantizedKVCache] | PatternQuantizedKVCache | None,
        slot_id: int,
    ) -> PatternQuantizedKVCache | None:
        if cache_factory is None:
            return None
        cache = cache_factory(slot_id) if callable(cache_factory) else cache_factory
        if not isinstance(cache, PatternQuantizedKVCache):
            raise TypeError("allocated request cache must be PatternQuantizedKVCache")
        validate_cache(cache)
        return cache


class DynamicAddRemoveBatchRunner:
    """Iteration-boundary dynamic membership driver for decode-ready requests."""

    admission_boundary = "decode_ready_prefill_completed_cache"

    def __init__(self, lifecycle: RequestLifecycleManager) -> None:
        self.lifecycle = lifecycle
        self.iteration = 0

    def run_decode_iteration(
        self,
        *,
        active_order: Sequence[str] | None = None,
        finished_request_ids: Iterable[str] = (),
        admissions: Iterable[DecodeReadyAdmission] = (),
        decode_step: Callable[[PatternQuantizedKVCache, list[ActiveRowMapping]], PatternQuantizedKVCache | None],
    ) -> DynamicBatchIterationResult:
        finished = tuple(str(request_id) for request_id in finished_request_ids)
        released_slots = tuple(self.lifecycle.finish_and_release_requests(finished))
        admitted_ids = []
        admitted_slots = []
        for admission in admissions:
            admitted_ids.append(str(admission.request_id))
            admitted_slots.append(self.lifecycle.admit_decode_ready_request(admission.request_id, admission.cache_factory))
        active_cache, mappings = self.lifecycle.build_active_cache(active_order)
        decoded = decode_step(active_cache, mappings)
        self.lifecycle.commit_active_cache(active_cache if decoded is None else decoded, mappings)
        result = DynamicBatchIterationResult(
            iteration=self.iteration,
            active_request_ids=tuple(mapping.request_id for mapping in mappings),
            row_mapping=tuple(mappings),
            removed_request_ids=finished,
            admitted_request_ids=tuple(admitted_ids),
            released_slot_ids=released_slots,
            admitted_slot_ids=tuple(admitted_slots),
        )
        self.iteration += 1
        return result


class ContinuousBatchScheduler:
    """FIFO iteration-level scheduler for decode-ready continuous batching.

    The scheduler owns only request control-plane queues. Request/cache state
    remains owned by RequestLifecycleManager slots.
    """

    admission_boundary = "decode_ready_prefill_completed_cache"
    scheduler_policy = "FIFO"

    def __init__(self, lifecycle: RequestLifecycleManager, max_active_requests: int) -> None:
        if int(max_active_requests) <= 0:
            raise ValueError("max_active_requests must be positive")
        if int(max_active_requests) > lifecycle.max_slots:
            raise ValueError("max_active_requests cannot exceed lifecycle slot capacity")
        self.lifecycle = lifecycle
        self.max_active_requests = int(max_active_requests)
        self.waiting_queue: deque[DecodeReadyAdmission] = deque()
        self.running_request_ids: list[str] = []
        self.finished_request_ids: list[str] = []
        self._known_request_ids: set[str] = set()
        self.iteration = 0
        self.trace: list[dict[str, Any]] = []

    def submit(self, admission: DecodeReadyAdmission) -> None:
        request_id = str(admission.request_id)
        if request_id in self._known_request_ids or request_id in self.lifecycle.request_to_slot:
            raise ValueError(f"duplicate request_id: {request_id}")
        self.waiting_queue.append(admission)
        self._known_request_ids.add(request_id)

    def submit_many(self, admissions: Iterable[DecodeReadyAdmission]) -> None:
        for admission in admissions:
            self.submit(admission)

    def has_work(self) -> bool:
        return bool(self.waiting_queue or self.running_request_ids)

    def waiting_request_ids(self) -> tuple[str, ...]:
        return tuple(str(admission.request_id) for admission in self.waiting_queue)

    def build_active_batch(self) -> tuple[PatternQuantizedKVCache, list[ActiveRowMapping]]:
        return self.lifecycle.build_active_cache(self.running_request_ids)

    def advance_iteration(
        self,
        *,
        decode_step: Callable[[PatternQuantizedKVCache, list[ActiveRowMapping]], PatternQuantizedKVCache | None],
        finished_request_ids: Iterable[str] = (),
    ) -> ContinuousBatchIterationResult:
        pre_admitted_ids, pre_admitted_slots = self._admit_waiting_until_full()
        decoded_ids = tuple(self.running_request_ids)
        decoded_mappings: tuple[ActiveRowMapping, ...] = ()
        decode_executed = False
        if decoded_ids:
            active_cache, mappings = self.lifecycle.build_active_cache(decoded_ids)
            decoded_mappings = tuple(mappings)
            decoded = decode_step(active_cache, mappings)
            self.lifecycle.commit_active_cache(active_cache if decoded is None else decoded, mappings)
            decode_executed = True

        finished = tuple(str(request_id) for request_id in finished_request_ids)
        self._validate_finished_ids(finished, decoded_ids)
        released_slots = tuple(self.lifecycle.finish_and_release_requests(finished))
        if finished:
            finished_set = set(finished)
            self.running_request_ids = [request_id for request_id in self.running_request_ids if request_id not in finished_set]
            self.finished_request_ids.extend(finished)

        post_admitted_ids, post_admitted_slots = self._admit_waiting_until_full()
        admitted_ids = tuple(pre_admitted_ids + post_admitted_ids)
        admitted_slots = tuple(pre_admitted_slots + post_admitted_slots)
        next_mapping = tuple(self.lifecycle.build_active_row_mapping(self.running_request_ids))
        result = ContinuousBatchIterationResult(
            iteration=self.iteration,
            decoded_request_ids=decoded_ids,
            decoded_row_mapping=decoded_mappings,
            finished_request_ids=finished,
            released_slot_ids=released_slots,
            admitted_request_ids=admitted_ids,
            admitted_slot_ids=admitted_slots,
            running_request_ids=tuple(self.running_request_ids),
            waiting_request_ids=self.waiting_request_ids(),
            finished_archive=tuple(self.finished_request_ids),
            next_row_mapping=next_mapping,
            decode_executed=decode_executed,
        )
        self.trace.append(
            {
                "iteration": result.iteration,
                "decoded": list(result.decoded_request_ids),
                "finished_this_iteration": list(result.finished_request_ids),
                "admitted": list(result.admitted_request_ids),
                "released_slots": list(result.released_slot_ids),
                "admitted_slots": list(result.admitted_slot_ids),
                "running": list(result.running_request_ids),
                "waiting": list(result.waiting_request_ids),
                "finished": list(result.finished_archive),
                "decoded_row_to_slot": [
                    {
                        "row": item.row_idx,
                        "request_id": item.request_id,
                        "slot_id": item.slot_id,
                        "generation": item.generation,
                    }
                    for item in result.decoded_row_mapping
                ],
                "next_row_to_slot": [
                    {
                        "row": item.row_idx,
                        "request_id": item.request_id,
                        "slot_id": item.slot_id,
                        "generation": item.generation,
                    }
                    for item in result.next_row_mapping
                ],
            }
        )
        self.iteration += 1
        return result

    def _admit_waiting_until_full(self) -> tuple[list[str], list[int]]:
        admitted_ids: list[str] = []
        admitted_slots: list[int] = []
        while len(self.running_request_ids) < self.max_active_requests and self.waiting_queue:
            admission = self.waiting_queue.popleft()
            request_id = str(admission.request_id)
            slot_id = self.lifecycle.admit_decode_ready_request(request_id, admission.cache_factory)
            self.running_request_ids.append(request_id)
            admitted_ids.append(request_id)
            admitted_slots.append(slot_id)
        return admitted_ids, admitted_slots

    def _validate_finished_ids(self, finished: tuple[str, ...], decoded_ids: tuple[str, ...]) -> None:
        if len(finished) != len(set(finished)):
            raise ValueError("duplicate finished request_id")
        decoded_set = set(decoded_ids)
        for request_id in finished:
            if request_id not in decoded_set:
                raise RuntimeError(f"finished request {request_id} was not active in this iteration")
