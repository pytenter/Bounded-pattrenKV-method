from __future__ import annotations

import os
import hashlib
from dataclasses import dataclass
from math import lcm
from typing import Any

import torch

from quant.new_pack import pack_tensor, triton_quantize_and_pack_along_last_dim, unpack_tensor
from quant.patternkv_profile import profile_range, record_cache_mutation, record_counter, tensor_bytes


@dataclass
class QuantizedKVCache:
    sink_k: torch.Tensor | None = None
    sink_v: torch.Tensor | None = None
    packed_k: torch.Tensor | None = None
    packed_k_scale: torch.Tensor | None = None
    packed_k_zero: torch.Tensor | None = None
    packed_v: torch.Tensor | None = None
    packed_v_scale: torch.Tensor | None = None
    packed_v_zero: torch.Tensor | None = None
    pending_k: torch.Tensor | None = None
    pending_v: torch.Tensor | None = None
    recent_k: torch.Tensor | None = None
    recent_v: torch.Tensor | None = None
    total_tokens: int = 0
    packed_k_tokens: int = 0
    packed_v_tokens: int = 0
    sink_length: int = 0
    recent_length: int = 128
    group_size: int = 128
    k_bits: int = 2
    v_bits: int = 2
    pack_count_k: int = 0
    pack_count_v: int = 0
    cache_mode: str = "segmented_rolling"
    chunk_length: int = 128
    cache_backend: str = "contiguous"
    request_total_tokens: torch.Tensor | None = None
    request_packed_k_tokens: torch.Tensor | None = None
    request_packed_v_tokens: torch.Tensor | None = None
    request_packed_v4_tokens: torch.Tensor | None = None


@dataclass
class PatternQuantizedKVCache(QuantizedKVCache):
    k_assignments: torch.Tensor | None = None
    v_assignments: torch.Tensor | None = None
    v_assignment_idx: torch.Tensor | None = None
    v_pattern_mask: torch.Tensor | None = None
    k_centroids: torch.Tensor | None = None
    v_centroids: torch.Tensor | None = None
    centroid_updates_k: int = 0
    centroid_updates_v: int = 0
    value_objective: str = "base"
    v_precision_selector: str = "base_v2"
    v4_budget_fraction: float = 0.0
    random_selector_seed: int = 20260809
    v_precision_mask: torch.Tensor | None = None
    packed_v4: torch.Tensor | None = None
    packed_v4_scale: torch.Tensor | None = None
    packed_v4_zero: torch.Tensor | None = None
    packed_v4_tokens: int = 0
    v2_pattern_mask: torch.Tensor | None = None
    v2_assignment_idx: torch.Tensor | None = None
    v4_pattern_mask: torch.Tensor | None = None
    v4_assignment_idx: torch.Tensor | None = None
    capacity_backend: str = "baseline"
    capacity_buffers: Any | None = None
    v_causal_importance: torch.Tensor | None = None
    v_oracle_importance: torch.Tensor | None = None
    operator_ready_page_pools: Any | None = None
    centroid_state_pool: Any | None = None
    centroid_state_indices: torch.Tensor | None = None
    centroid_flush_mask: torch.Tensor | None = None


def cache_batch_size(cache: QuantizedKVCache) -> int:
    for value in (
        cache.sink_k,
        cache.packed_k,
        cache.pending_k,
        cache.recent_k,
        getattr(cache, "v_precision_mask", None),
        getattr(cache, "centroid_state_indices", None),
    ):
        if torch.is_tensor(value) and value.dim() > 0:
            return int(value.shape[0])
    if torch.is_tensor(getattr(cache, "request_total_tokens", None)):
        return int(cache.request_total_tokens.numel())
    return 1


def get_total_tokens_per_request(cache: QuantizedKVCache, *, device: torch.device | None = None) -> torch.Tensor:
    if torch.is_tensor(getattr(cache, "request_total_tokens", None)):
        out = cache.request_total_tokens.to(dtype=torch.long)
        return out.to(device=device) if device is not None else out
    target_device = device
    if target_device is None:
        for value in (cache.sink_k, cache.packed_k, cache.pending_k, cache.recent_k):
            if torch.is_tensor(value):
                target_device = value.device
                break
    return torch.full((cache_batch_size(cache),), int(cache.total_tokens), dtype=torch.long, device=target_device)


def all_requests_equal_length(cache: QuantizedKVCache) -> bool:
    lengths = get_total_tokens_per_request(cache)
    return bool(lengths.numel() <= 1 or torch.equal(lengths, torch.full_like(lengths, int(lengths[0].item()))))


def set_request_total_tokens(cache: QuantizedKVCache, lengths: torch.Tensor | list[int] | tuple[int, ...] | None) -> None:
    if lengths is None:
        cache.request_total_tokens = None
        return
    device = None
    for value in (cache.sink_k, cache.packed_k, cache.pending_k, cache.recent_k):
        if torch.is_tensor(value):
            device = value.device
            break
    tensor = torch.as_tensor(lengths, dtype=torch.long, device=device)
    cache.request_total_tokens = tensor.contiguous()
    if tensor.numel() == 1 or bool(torch.equal(tensor, torch.full_like(tensor, int(tensor[0].item())))):
        cache.total_tokens = int(tensor[0].item())
    else:
        cache.total_tokens = int(tensor.max().item())


def advance_request_total_tokens(cache: QuantizedKVCache, amount: int = 1) -> None:
    set_request_total_tokens(cache, get_total_tokens_per_request(cache) + int(amount))


def get_decode_position_ids(cache: QuantizedKVCache, q_len: int, *, device: torch.device | None = None) -> torch.Tensor:
    starts = get_total_tokens_per_request(cache, device=device)
    offsets = torch.arange(int(q_len), dtype=torch.long, device=starts.device).unsqueeze(0)
    return starts.unsqueeze(1) + offsets


def _request_vector(cache: QuantizedKVCache, name: str, fallback: int, *, device: torch.device | None = None) -> torch.Tensor:
    value = getattr(cache, name, None)
    if torch.is_tensor(value):
        out = value.to(dtype=torch.long)
        return out.to(device=device) if device is not None else out
    return torch.full((cache_batch_size(cache),), int(fallback), dtype=torch.long, device=device)


_RAGGED_K_COUNTERS = {
    "ragged_batch_forward_calls": 0,
    "ragged_requests_processed": 0,
    "ragged_k_path_calls": 0,
    "ragged_k_mask_calls": 0,
    "ragged_k_invalid_tail_tokens": 0,
    "serial_request_dispatches": 0,
    "historical_fp16_k_materialization": 0,
}


def reset_ragged_k_counters() -> None:
    for key in _RAGGED_K_COUNTERS:
        _RAGGED_K_COUNTERS[key] = 0


def get_ragged_k_counters() -> dict[str, int]:
    return dict(_RAGGED_K_COUNTERS)


def record_ragged_k_counter(key: str, amount: int = 1) -> None:
    if key not in _RAGGED_K_COUNTERS:
        raise KeyError(f"unknown ragged K counter: {key}")
    _RAGGED_K_COUNTERS[key] += int(amount)


def get_packed_k_tokens_per_request(cache: QuantizedKVCache, *, device: torch.device | None = None) -> torch.Tensor:
    return _request_vector(cache, "request_packed_k_tokens", int(cache.packed_k_tokens), device=device)


def k_segment_valid_lengths(cache: QuantizedKVCache, *, device: torch.device | None = None) -> dict[str, torch.Tensor]:
    totals = get_total_tokens_per_request(cache, device=device)
    packed = get_packed_k_tokens_per_request(cache, device=totals.device)
    sink_limit = torch.full_like(totals, int(cache.sink_length))
    recent_limit = torch.full_like(totals, int(cache.recent_length))
    zeros = torch.zeros_like(totals)
    sink_values = torch.minimum(totals, sink_limit)
    non_sink = torch.maximum(totals - sink_values, zeros)
    recent_values = torch.minimum(non_sink, recent_limit)
    quantized_history = torch.maximum(non_sink - recent_values, zeros)
    pending_values = torch.maximum(quantized_history - packed, zeros)
    return {
        "sink": sink_values.to(dtype=torch.long),
        "packed": packed.to(dtype=torch.long, device=totals.device),
        "pending": pending_values.to(dtype=torch.long),
        "recent": recent_values.to(dtype=torch.long),
    }


def build_k_segment_validity_mask(
    cache: QuantizedKVCache,
    value_parts: list[tuple[str, int]],
    *,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if not torch.is_tensor(getattr(cache, "request_total_tokens", None)) and not torch.is_tensor(getattr(cache, "request_packed_k_tokens", None)):
        return None
    target_device = device
    if target_device is None:
        for value in (cache.sink_k, cache.packed_k, cache.pending_k, cache.recent_k):
            if torch.is_tensor(value):
                target_device = value.device
                break
    lengths = k_segment_valid_lengths(cache, device=target_device)
    masks = []
    invalid = None
    for name, physical_length in value_parts:
        physical = int(physical_length)
        if physical <= 0:
            continue
        valid = lengths[name].clamp(min=0, max=physical)
        positions = torch.arange(physical, dtype=torch.long, device=valid.device).unsqueeze(0)
        segment_mask = positions < valid.unsqueeze(1)
        masks.append(segment_mask)
        segment_invalid = physical - valid
        invalid = segment_invalid if invalid is None else invalid + segment_invalid
    if not masks:
        return None
    mask = torch.cat(masks, dim=1)
    if invalid is not None:
        invalid_count = int(invalid.sum().item())
        if invalid_count:
            record_ragged_k_counter("ragged_k_mask_calls", 1)
            record_ragged_k_counter("ragged_k_invalid_tail_tokens", invalid_count)
    return mask


@dataclass
class PatternKVCentroidStatePool:
    k_centroid_pool: torch.Tensor
    v_centroid_pool: torch.Tensor
    k_counts: torch.Tensor
    v_counts: torch.Tensor
    update_counts_k: torch.Tensor
    update_counts_v: torch.Tensor
    last_flush_pos: torch.Tensor
    active: torch.Tensor
    static_centroid_count: int
    static_v_centroid_count: int | None = None

    @classmethod
    def create(
        cls,
        k_static: torch.Tensor,
        v_static: torch.Tensor,
        *,
        max_slots: int,
        max_dynamic_centroids: int,
        initial_slots: torch.Tensor | None = None,
    ) -> "PatternKVCentroidStatePool":
        if k_static.dim() not in (3, 4) or v_static.dim() not in (3, 4):
            raise ValueError("static centroid banks must be [H,M,D] or [B,H,M,D]")
        if k_static.dim() != v_static.dim():
            raise ValueError("K/V static centroid rank mismatch")
        k_head_dim = 0 if k_static.dim() == 3 else 1
        k_count_dim = 1 if k_static.dim() == 3 else 2
        v_head_dim = 0 if v_static.dim() == 3 else 1
        v_count_dim = 1 if v_static.dim() == 3 else 2
        if k_static.shape[k_head_dim] != v_static.shape[v_head_dim] or k_static.shape[-1] != v_static.shape[-1]:
            raise ValueError("K/V static centroid geometry mismatch")
        if k_static.dim() == 4 and k_static.shape[0] != v_static.shape[0]:
            raise ValueError("K/V request-local static centroid batch mismatch")
        slots = int(max_slots)
        k_static_count = int(k_static.shape[k_count_dim])
        v_static_count = int(v_static.shape[v_count_dim])
        heads = int(k_static.shape[k_head_dim])
        dim = int(k_static.shape[-1])
        k_total = k_static_count + int(max_dynamic_centroids)
        v_total = v_static_count + int(max_dynamic_centroids)
        k_pool = torch.empty((slots, heads, k_total, dim), dtype=k_static.dtype, device=k_static.device)
        v_pool = torch.empty((slots, heads, v_total, dim), dtype=v_static.dtype, device=v_static.device)
        k_counts = torch.full((slots,), k_static_count, dtype=torch.int32, device=k_static.device)
        v_counts = torch.full((slots,), v_static_count, dtype=torch.int32, device=v_static.device)
        if k_static.dim() == 3:
            k_pool[:, :, :k_static_count, :] = k_static.unsqueeze(0)
            v_pool[:, :, :v_static_count, :] = v_static.unsqueeze(0)
        else:
            if initial_slots is None:
                initial_slots = torch.arange(int(k_static.shape[0]), dtype=torch.long, device=k_static.device)
            initial_slots = initial_slots.to(device=k_static.device, dtype=torch.long)
            if int(initial_slots.numel()) != int(k_static.shape[0]):
                raise ValueError("request-local static centroid slots must match static centroid batch")
            k_pool[:, :, :k_static_count, :] = k_static[0].unsqueeze(0)
            v_pool[:, :, :v_static_count, :] = v_static[0].unsqueeze(0)
            k_pool[initial_slots, :, :k_static_count, :] = k_static
            v_pool[initial_slots, :, :v_static_count, :] = v_static
        return cls(
            k_centroid_pool=k_pool,
            v_centroid_pool=v_pool,
            k_counts=k_counts,
            v_counts=v_counts,
            update_counts_k=torch.zeros(slots, dtype=torch.int32, device=k_static.device),
            update_counts_v=torch.zeros(slots, dtype=torch.int32, device=v_static.device),
            last_flush_pos=torch.zeros(slots, dtype=torch.int32, device=k_static.device),
            active=torch.zeros(slots, dtype=torch.bool, device=k_static.device),
            static_centroid_count=k_static_count,
            static_v_centroid_count=v_static_count,
        )

    def allocate(self, slots: torch.Tensor) -> None:
        self.active[slots.long()] = True

    def free(self, slots: torch.Tensor) -> None:
        slots = slots.long()
        self.k_counts[slots] = int(self.static_centroid_count)
        self.v_counts[slots] = int(self.static_v_centroid_count or self.static_centroid_count)
        self.update_counts_k[slots] = 0
        self.update_counts_v[slots] = 0
        self.last_flush_pos[slots] = 0
        self.active[slots] = False

    def current_k(self, slots: torch.Tensor) -> torch.Tensor:
        return self.k_centroid_pool[slots.long()]

    def current_v(self, slots: torch.Tensor) -> torch.Tensor:
        return self.v_centroid_pool[slots.long()]


CHUNKED_CACHE_MODE = "segmented_chunked"
ROLLING_CACHE_MODE = "segmented_rolling"
CONTIGUOUS_CACHE_BACKEND = "contiguous"
FIXED_PAGE_CACHE_BACKEND = "paged"
DEFAULT_PAGE_SIZE = 128
CAPACITY_GROWTH_BASELINE = "baseline"
CAPACITY_GROWTH_FIXED = "fixed_capacity"
CAPACITY_GROWTH_CHUNKED = "chunked_capacity"


_CAPACITY_COUNTERS = {
    "historical_append_calls": 0,
    "historical_torch_cat_calls": 0,
    "historical_realloc_events": 0,
    "historical_old_bytes_copied": 0,
    "historical_new_bytes_written": 0,
    "capacity_growth_events": 0,
    "capacity_growth_old_bytes_copied": 0,
    "reserved_capacity_bytes": 0,
    "logical_valid_bytes": 0,
    "unused_capacity_bytes": 0,
    "historical_materialization_calls": 0,
    "historical_materialized_bytes": 0,
}


def reset_capacity_cache_counters() -> None:
    for key in _CAPACITY_COUNTERS:
        _CAPACITY_COUNTERS[key] = 0


def get_capacity_cache_counters() -> dict[str, int]:
    return dict(_CAPACITY_COUNTERS)


def normalize_capacity_growth_backend(value: str | None = None) -> str:
    backend = str(value or os.environ.get("PATTERNKV_CACHE_GROWTH_BACKEND", CAPACITY_GROWTH_BASELINE)).strip().lower()
    aliases = {
        "fixed": CAPACITY_GROWTH_FIXED,
        "capacity": CAPACITY_GROWTH_FIXED,
        "chunked": CAPACITY_GROWTH_CHUNKED,
        "grow_by_chunk": CAPACITY_GROWTH_CHUNKED,
    }
    backend = aliases.get(backend, backend)
    if backend not in (CAPACITY_GROWTH_BASELINE, CAPACITY_GROWTH_FIXED, CAPACITY_GROWTH_CHUNKED):
        raise ValueError("PATTERNKV_CACHE_GROWTH_BACKEND must be 'baseline', 'fixed_capacity', or 'chunked_capacity'")
    return backend


@dataclass(frozen=True)
class PageDescriptor:
    stream: str
    page_id: int
    logical_start_token: int
    valid_tokens: int
    page_size: int
    shape: tuple[int, ...]
    dtype: str
    device: str


class FixedPageBuffer:
    """Fixed-size page storage for one logical token stream.

    The token dimension can differ by stream: FP16 K/V use dim=2, packed K uses
    dim=3, compact V payload/metadata use dim=2, and precision masks use dim=1.
    """

    def __init__(self, *, stream: str, page_size: int = DEFAULT_PAGE_SIZE, token_dim: int = 2) -> None:
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")
        self.stream = str(stream)
        self.page_size = int(page_size)
        self.token_dim = int(token_dim)
        self.pages: list[torch.Tensor] = []
        self.num_tokens = 0
        self.page_allocations = 0
        self.page_writes = 0
        self.page_crossings = 0
        self.new_bytes_written = 0
        self.old_bytes_copied = 0

    def logical_length(self) -> int:
        return int(self.num_tokens)

    def page_count(self) -> int:
        return len(self.pages)

    def last_page_fill(self) -> int:
        if not self.pages:
            return 0
        rem = self.num_tokens % self.page_size
        return self.page_size if rem == 0 else rem

    def allocated_slots(self) -> int:
        return self.page_count() * self.page_size

    def valid_slots(self) -> int:
        return self.logical_length()

    def fragmentation_slots(self) -> int:
        return max(self.allocated_slots() - self.valid_slots(), 0)

    def fragmentation_bytes(self) -> int:
        if not self.pages:
            return 0
        page = self.pages[0]
        bytes_per_slot = page.numel() * page.element_size() // self.page_size
        return self.fragmentation_slots() * bytes_per_slot

    def _normalize_dim(self, dim: int, rank: int) -> int:
        return dim if dim >= 0 else rank + dim

    def _allocate_page_like(self, value: torch.Tensor) -> torch.Tensor:
        token_dim = self._normalize_dim(self.token_dim, value.dim())
        shape = list(value.shape)
        shape[token_dim] = self.page_size
        page = torch.empty(tuple(shape), dtype=value.dtype, device=value.device)
        self.pages.append(page)
        self.page_allocations += 1
        return page

    def append(self, value: torch.Tensor) -> None:
        self.append_block(value)

    def append_block(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError("FixedPageBuffer.append_block expects a tensor")
        token_dim = self._normalize_dim(self.token_dim, value.dim())
        tokens = int(value.shape[token_dim])
        if tokens == 0:
            return
        cursor = 0
        while cursor < tokens:
            page_id = self.num_tokens // self.page_size
            offset = self.num_tokens % self.page_size
            if page_id == len(self.pages):
                self._allocate_page_like(value)
            page = self.pages[page_id]
            take = min(tokens - cursor, self.page_size - offset)
            dst = page.narrow(token_dim, offset, take)
            src = value.narrow(token_dim, cursor, take).contiguous()
            dst.copy_(src)
            self.page_writes += 1
            self.new_bytes_written += tensor_bytes(src)
            self.num_tokens += take
            cursor += take
            if cursor < tokens:
                self.page_crossings += 1

    def view_page(self, page_id: int) -> torch.Tensor:
        return self.pages[int(page_id)]

    def iterate_pages(self):
        token_dim = self._normalize_dim(self.token_dim, self.pages[0].dim()) if self.pages else self.token_dim
        for page_id, page in enumerate(self.pages):
            valid = min(self.page_size, self.num_tokens - page_id * self.page_size)
            if valid <= 0:
                continue
            yield page_id, page.narrow(token_dim, 0, valid)

    def descriptors(self) -> list[PageDescriptor]:
        out = []
        for page_id, page in enumerate(self.pages):
            valid = min(self.page_size, max(self.num_tokens - page_id * self.page_size, 0))
            out.append(
                PageDescriptor(
                    stream=self.stream,
                    page_id=page_id,
                    logical_start_token=page_id * self.page_size,
                    valid_tokens=valid,
                    page_size=self.page_size,
                    shape=tuple(page.shape),
                    dtype=str(page.dtype),
                    device=str(page.device),
                )
            )
        return out

    def materialize_contiguous(self) -> torch.Tensor | None:
        parts = [part for _page_id, part in self.iterate_pages()]
        if not parts:
            return None
        token_dim = self._normalize_dim(self.token_dim, parts[0].dim())
        return torch.cat(parts, dim=token_dim).contiguous()

    def stats(self) -> dict[str, int | str]:
        return {
            "stream": self.stream,
            "page_size": self.page_size,
            "tokens": self.logical_length(),
            "page_count": self.page_count(),
            "last_page_fill": self.last_page_fill(),
            "page_allocations": self.page_allocations,
            "page_writes": self.page_writes,
            "page_crossings": self.page_crossings,
            "old_bytes_copied": self.old_bytes_copied,
            "new_bytes_written": self.new_bytes_written,
            "allocated_slots": self.allocated_slots(),
            "valid_slots": self.valid_slots(),
            "fragmentation_slots": self.fragmentation_slots(),
            "fragmentation_bytes": self.fragmentation_bytes(),
        }


class RecentRingBuffer:
    """Fixed-capacity recent window with logical oldest-to-newest materialization."""

    def __init__(self, *, capacity: int = DEFAULT_PAGE_SIZE, token_dim: int = 2, stream: str = "recent") -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = int(capacity)
        self.token_dim = int(token_dim)
        self.stream = str(stream)
        self.buffer: torch.Tensor | None = None
        self.start = 0
        self.length = 0
        self.page_writes = 0
        self.rollover_count = 0
        self.old_bytes_copied = 0
        self.new_bytes_written = 0

    def _normalize_dim(self, dim: int, rank: int) -> int:
        return dim if dim >= 0 else rank + dim

    def _allocate_like(self, value: torch.Tensor) -> None:
        token_dim = self._normalize_dim(self.token_dim, value.dim())
        shape = list(value.shape)
        shape[token_dim] = self.capacity
        self.buffer = torch.empty(tuple(shape), dtype=value.dtype, device=value.device)

    def _read_logical(self, start: int, tokens: int) -> torch.Tensor | None:
        if self.buffer is None or tokens <= 0:
            return None
        token_dim = self._normalize_dim(self.token_dim, self.buffer.dim())
        logical_start = (self.start + int(start)) % self.capacity
        first = min(tokens, self.capacity - logical_start)
        parts = [self.buffer.narrow(token_dim, logical_start, first)]
        remain = tokens - first
        if remain:
            parts.append(self.buffer.narrow(token_dim, 0, remain))
        return torch.cat(parts, dim=token_dim).contiguous() if len(parts) > 1 else parts[0].contiguous()

    def _write_at(self, offset: int, value: torch.Tensor) -> None:
        if self.buffer is None:
            self._allocate_like(value)
        assert self.buffer is not None
        token_dim = self._normalize_dim(self.token_dim, self.buffer.dim())
        tokens = int(value.shape[token_dim])
        first = min(tokens, self.capacity - offset)
        self.buffer.narrow(token_dim, offset, first).copy_(value.narrow(token_dim, 0, first).contiguous())
        if tokens > first:
            self.buffer.narrow(token_dim, 0, tokens - first).copy_(value.narrow(token_dim, first, tokens - first).contiguous())
        self.page_writes += 1
        self.new_bytes_written += tensor_bytes(value)

    def append(self, value: torch.Tensor) -> torch.Tensor | None:
        return self.append_block(value)

    def append_block(self, value: torch.Tensor) -> torch.Tensor | None:
        if not torch.is_tensor(value):
            raise TypeError("RecentRingBuffer.append_block expects a tensor")
        token_dim = self._normalize_dim(self.token_dim, value.dim())
        tokens = int(value.shape[token_dim])
        if tokens == 0:
            return None
        if self.buffer is None:
            self._allocate_like(value)
        overflow_parts = []
        if tokens >= self.capacity:
            old = self.materialize_contiguous()
            if old is not None:
                overflow_parts.append(old)
            prefix = tokens - self.capacity
            if prefix:
                overflow_parts.append(value.narrow(token_dim, 0, prefix).contiguous())
            keep = value.narrow(token_dim, prefix, self.capacity).contiguous()
            self.start = 0
            self.length = self.capacity
            self._write_at(0, keep)
            self.rollover_count += 1
        else:
            overflow = max(self.length + tokens - self.capacity, 0)
            if overflow:
                old = self._read_logical(0, overflow)
                if old is not None:
                    overflow_parts.append(old)
                    self.old_bytes_copied += tensor_bytes(old)
                self.start = (self.start + overflow) % self.capacity
                self.length -= overflow
                self.rollover_count += 1
            write_offset = (self.start + self.length) % self.capacity
            self._write_at(write_offset, value.contiguous())
            self.length += tokens
        if not overflow_parts:
            return None
        return torch.cat(overflow_parts, dim=token_dim).contiguous() if len(overflow_parts) > 1 else overflow_parts[0]

    def logical_length(self) -> int:
        return int(self.length)

    def materialize_contiguous(self) -> torch.Tensor | None:
        return self._read_logical(0, self.length)

    def stats(self) -> dict[str, int | str]:
        return {
            "stream": self.stream,
            "capacity": self.capacity,
            "tokens": self.logical_length(),
            "page_writes": self.page_writes,
            "rollover_count": self.rollover_count,
            "old_bytes_copied": self.old_bytes_copied,
            "new_bytes_written": self.new_bytes_written,
        }


class FixedPageCacheStorage:
    """Descriptor-oriented fixed-page storage for PatternKV cache streams."""

    DEFAULT_TOKEN_DIMS = {
        "packed_k": 3,
        "packed_k_scale": 3,
        "packed_k_zero": 3,
        "packed_v": 2,
        "packed_v_scale": 2,
        "packed_v_zero": 2,
        "packed_v4": 2,
        "packed_v4_scale": 2,
        "packed_v4_zero": 2,
        "v_precision_mask": 1,
        "k_assignments": 2,
        "v_pattern_mask": 2,
        "v_assignment_idx": 2,
    }

    def __init__(self, *, page_size: int = DEFAULT_PAGE_SIZE) -> None:
        self.page_size = int(page_size)
        self.buffers: dict[str, FixedPageBuffer] = {}

    def buffer(self, stream: str, *, token_dim: int | None = None) -> FixedPageBuffer:
        if stream not in self.buffers:
            dim = self.DEFAULT_TOKEN_DIMS.get(stream, 2) if token_dim is None else int(token_dim)
            self.buffers[stream] = FixedPageBuffer(stream=stream, page_size=self.page_size, token_dim=dim)
        return self.buffers[stream]

    def append_stream(self, stream: str, value: torch.Tensor, *, token_dim: int | None = None) -> None:
        self.buffer(stream, token_dim=token_dim).append_block(value)

    def materialize(self, stream: str) -> torch.Tensor | None:
        buf = self.buffers.get(stream)
        return None if buf is None else buf.materialize_contiguous()

    def descriptors(self) -> list[PageDescriptor]:
        out: list[PageDescriptor] = []
        for stream in sorted(self.buffers):
            out.extend(self.buffers[stream].descriptors())
        return out

    def stats(self) -> list[dict[str, int | str]]:
        return [self.buffers[stream].stats() for stream in sorted(self.buffers)]


def normalize_cache_backend(cache_backend: str | None) -> str:
    backend = str(cache_backend or os.environ.get("PATTERNKV_CACHE_BACKEND", CONTIGUOUS_CACHE_BACKEND)).strip().lower()
    aliases = {"fixed_page": FIXED_PAGE_CACHE_BACKEND, "page": FIXED_PAGE_CACHE_BACKEND}
    backend = aliases.get(backend, backend)
    if backend not in (CONTIGUOUS_CACHE_BACKEND, FIXED_PAGE_CACHE_BACKEND):
        raise ValueError("PATTERNKV_CACHE_BACKEND must be 'contiguous' or 'paged'")
    return backend


class ContiguousCapacityBuffer:
    """Preallocated contiguous storage for one logical token stream.

    Appends copy only new tokens while capacity is sufficient. `logical_view`
    returns a narrow view into storage and never calls `.contiguous()`.
    """

    def __init__(
        self,
        *,
        stream_name: str,
        shape_except_token: tuple[int, ...],
        token_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
        capacity: int = 0,
        chunk_tokens: int | None = None,
    ) -> None:
        if token_dim < 0:
            token_dim += len(shape_except_token) + 1
        if token_dim < 0 or token_dim > len(shape_except_token):
            raise ValueError(f"token_dim out of range: {token_dim}")
        self.stream_name = str(stream_name)
        self.shape_except_token = tuple(int(x) for x in shape_except_token)
        self.token_dim = int(token_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.length = 0
        self._capacity = 0
        self.chunk_tokens = int(chunk_tokens or 0)
        self.storage: torch.Tensor | None = None
        if capacity:
            self.reserve(int(capacity))

    def capacity(self) -> int:
        return int(self._capacity)

    def logical_length(self) -> int:
        return int(self.length)

    def remaining_capacity(self) -> int:
        return max(self.capacity() - self.logical_length(), 0)

    def _shape(self, tokens: int) -> tuple[int, ...]:
        shape = list(self.shape_except_token)
        shape.insert(self.token_dim, int(tokens))
        return tuple(shape)

    def _token_count(self, value: torch.Tensor) -> int:
        dim = self.token_dim if self.token_dim >= 0 else value.dim() + self.token_dim
        return int(value.shape[dim])

    def reserve(self, required_capacity: int) -> None:
        required_capacity = int(required_capacity)
        if required_capacity <= self._capacity:
            return
        old = self.storage
        old_bytes = tensor_bytes(old.narrow(self.token_dim, 0, self.length)) if old is not None and self.length else 0
        new_storage = torch.empty(self._shape(required_capacity), dtype=self.dtype, device=self.device)
        if old is not None and self.length:
            new_storage.narrow(self.token_dim, 0, self.length).copy_(old.narrow(self.token_dim, 0, self.length))
            _CAPACITY_COUNTERS["historical_old_bytes_copied"] += int(old_bytes)
            _CAPACITY_COUNTERS["capacity_growth_old_bytes_copied"] += int(old_bytes)
        self.storage = new_storage
        self._capacity = required_capacity
        _CAPACITY_COUNTERS["historical_realloc_events"] += 1
        _CAPACITY_COUNTERS["capacity_growth_events"] += 1

    def _grow_for(self, required: int) -> None:
        if required <= self._capacity:
            return
        if self.chunk_tokens > 0:
            new_capacity = ((required + self.chunk_tokens - 1) // self.chunk_tokens) * self.chunk_tokens
            self.reserve(new_capacity)
            return
        raise RuntimeError(f"{self.stream_name} capacity overflow: required={required}, capacity={self._capacity}")

    def append(self, value: torch.Tensor) -> None:
        self.append_block(value)

    def append_block(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError("ContiguousCapacityBuffer.append_block expects a tensor")
        tokens = self._token_count(value)
        if tokens == 0:
            return
        expected = self._shape(tokens)
        if tuple(value.shape) != expected:
            raise ValueError(f"{self.stream_name} append shape mismatch: expected tokenized shape {expected}, got {tuple(value.shape)}")
        if value.dtype != self.dtype or value.device != self.device:
            value = value.to(device=self.device, dtype=self.dtype)
        required = self.length + tokens
        self._grow_for(required)
        if self.storage is None:
            raise RuntimeError("capacity storage was not allocated")
        dst = self.storage.narrow(self.token_dim, self.length, tokens)
        dst.copy_(value.contiguous())
        self.length = required
        new_bytes = tensor_bytes(value)
        _CAPACITY_COUNTERS["historical_append_calls"] += 1
        _CAPACITY_COUNTERS["historical_new_bytes_written"] += int(new_bytes)

    def logical_view(self) -> torch.Tensor | None:
        if self.storage is None:
            return None
        return self.storage.narrow(self.token_dim, 0, self.length)

    def reset(self) -> None:
        self.length = 0

    def stats(self) -> dict[str, int | str | bool]:
        logical = self.logical_view()
        reserved = tensor_bytes(self.storage)
        valid = tensor_bytes(logical)
        unused = max(reserved - valid, 0)
        _CAPACITY_COUNTERS["reserved_capacity_bytes"] += int(reserved)
        _CAPACITY_COUNTERS["logical_valid_bytes"] += int(valid)
        _CAPACITY_COUNTERS["unused_capacity_bytes"] += int(unused)
        return {
            "stream": self.stream_name,
            "length": self.logical_length(),
            "capacity": self.capacity(),
            "remaining_capacity": self.remaining_capacity(),
            "token_dim": self.token_dim,
            "reserved_capacity_bytes": reserved,
            "logical_valid_bytes": valid,
            "unused_capacity_bytes": unused,
            "logical_is_contiguous": bool(logical.is_contiguous()) if logical is not None else True,
            "storage_offset": int(logical.storage_offset()) if logical is not None else 0,
            "stride": str(tuple(logical.stride())) if logical is not None else "",
        }


def normalize_cache_mode(cache_mode: str | None) -> str:
    mode = str(cache_mode or ROLLING_CACHE_MODE).strip().lower()
    aliases = {
        "segmented": ROLLING_CACHE_MODE,
        "rolling": ROLLING_CACHE_MODE,
        "chunked": CHUNKED_CACHE_MODE,
    }
    mode = aliases.get(mode, mode)
    if mode not in (CHUNKED_CACHE_MODE, ROLLING_CACHE_MODE):
        raise ValueError(f"unsupported segmented cache mode: {cache_mode!r}")
    return mode


def cache_validate_enabled() -> bool:
    return os.environ.get("PATTERNKV_CACHE_VALIDATE") == "1"


def segment_lengths(total_tokens: int, sink_length: int, recent_length: int) -> dict[str, int]:
    if total_tokens < 0 or sink_length < 0 or recent_length < 0:
        raise ValueError("cache token lengths must be non-negative")
    sink_tokens = min(total_tokens, sink_length)
    non_sink_tokens = max(total_tokens - sink_tokens, 0)
    recent_tokens = min(non_sink_tokens, recent_length)
    quantized_history_tokens = max(non_sink_tokens - recent_tokens, 0)
    return {
        "sink_tokens": sink_tokens,
        "quantized_history_tokens": quantized_history_tokens,
        "recent_tokens": recent_tokens,
        "total_tokens": total_tokens,
    }


def tensor_tokens(value: torch.Tensor | None) -> int:
    return int(value.shape[2]) if torch.is_tensor(value) else 0


def packed_last_dim_tokens(value: torch.Tensor | None, bits: int) -> int:
    return int(value.shape[-1] * (32 // bits)) if torch.is_tensor(value) else 0


def _empty_like_tokens(reference: torch.Tensor, tokens: int) -> torch.Tensor | None:
    if tokens == 0:
        return None
    return reference[:, :, :tokens, :].contiguous()


def _cat_token(a: torch.Tensor | None, b: torch.Tensor | None, *, category: str = "recent_pending") -> torch.Tensor | None:
    if a is None:
        return b
    if b is None:
        return a
    bytes_copied = tensor_bytes(a) + tensor_bytes(b)
    record_counter("cache_cat_events", bytes_copied=bytes_copied)
    record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
    with profile_range("cache_mutation", bytes_copied=bytes_copied):
        result = torch.cat([a, b], dim=2).contiguous()
    record_cache_mutation(category, a, b, result)
    return result


def _cat_packed_k(cache: QuantizedKVCache, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, tokens: int) -> None:
    if cache.packed_k is None:
        cache.packed_k = packed
        cache.packed_k_scale = scale
        cache.packed_k_zero = zero
    else:
        bytes_copied = tensor_bytes(cache.packed_k) + tensor_bytes(packed) + tensor_bytes(cache.packed_k_scale) + tensor_bytes(scale) + tensor_bytes(cache.packed_k_zero) + tensor_bytes(zero)
        record_counter("cache_cat_events", bytes_copied=bytes_copied)
        record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
        with profile_range("cache_mutation", bytes_copied=bytes_copied):
            old_packed, old_scale, old_zero = cache.packed_k, cache.packed_k_scale, cache.packed_k_zero
            cache.packed_k = torch.cat([old_packed, packed], dim=3)
            cache.packed_k_scale = torch.cat([old_scale, scale], dim=3)
            cache.packed_k_zero = torch.cat([old_zero, zero], dim=3)
        record_cache_mutation("packed_k_payload", old_packed, packed, cache.packed_k)
        record_cache_mutation("packed_k_scale", old_scale, scale, cache.packed_k_scale)
        record_cache_mutation("packed_k_zero", old_zero, zero, cache.packed_k_zero)
    cache.packed_k_tokens += int(tokens)
    cache.pack_count_k += 1


def _cat_packed_v(cache: QuantizedKVCache, packed: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, tokens: int) -> None:
    if isinstance(cache, PatternQuantizedKVCache) and _capacity_cache_enabled(cache):
        cache.packed_v = _append_capacity_stream(cache, "packed_v", packed, token_dim=2)
        cache.packed_v_scale = _append_capacity_stream(cache, "packed_v_scale", scale, token_dim=2)
        cache.packed_v_zero = _append_capacity_stream(cache, "packed_v_zero", zero, token_dim=2)
        cache.packed_v_tokens += int(tokens)
        cache.pack_count_v += 1
        return
    if cache.packed_v is None:
        cache.packed_v = packed
        cache.packed_v_scale = scale
        cache.packed_v_zero = zero
    else:
        bytes_copied = tensor_bytes(cache.packed_v) + tensor_bytes(packed) + tensor_bytes(cache.packed_v_scale) + tensor_bytes(scale) + tensor_bytes(cache.packed_v_zero) + tensor_bytes(zero)
        record_counter("cache_cat_events", bytes_copied=bytes_copied)
        record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
        with profile_range("cache_mutation", bytes_copied=bytes_copied):
            old_packed, old_scale, old_zero = cache.packed_v, cache.packed_v_scale, cache.packed_v_zero
            cache.packed_v = torch.cat([old_packed, packed], dim=2)
            cache.packed_v_scale = torch.cat([old_scale, scale], dim=2)
            cache.packed_v_zero = torch.cat([old_zero, zero], dim=2)
        record_cache_mutation("packed_v2_payload", old_packed, packed, cache.packed_v)
        record_cache_mutation("packed_v2_scale", old_scale, scale, cache.packed_v_scale)
        record_cache_mutation("packed_v2_zero", old_zero, zero, cache.packed_v_zero)
    cache.packed_v_tokens += int(tokens)
    cache.pack_count_v += 1


def _capacity_cache_enabled(cache: QuantizedKVCache) -> bool:
    if not isinstance(cache, PatternQuantizedKVCache):
        return False
    backend = normalize_capacity_growth_backend(getattr(cache, "capacity_backend", None))
    cache.capacity_backend = backend
    return backend in (CAPACITY_GROWTH_FIXED, CAPACITY_GROWTH_CHUNKED)


def _capacity_fixed_tokens() -> int:
    return int(os.environ.get("PATTERNKV_CACHE_FIXED_CAPACITY_TOKENS", "32768"))


def _capacity_chunk_tokens() -> int:
    return int(os.environ.get("PATTERNKV_CACHE_CHUNK_TOKENS", "4096"))


def _capacity_buffer(cache: PatternQuantizedKVCache, stream: str, value: torch.Tensor, *, token_dim: int) -> ContiguousCapacityBuffer:
    backend = normalize_capacity_growth_backend(getattr(cache, "capacity_backend", None))
    if cache.capacity_buffers is None:
        cache.capacity_buffers = {}
    buffers = cache.capacity_buffers
    if stream in buffers:
        return buffers[stream]
    logical_tokens = int(value.shape[token_dim])
    shape_except = list(value.shape)
    del shape_except[token_dim]
    if backend == CAPACITY_GROWTH_FIXED:
        capacity = max(_capacity_fixed_tokens(), logical_tokens)
        chunk = None
    elif backend == CAPACITY_GROWTH_CHUNKED:
        capacity = 0
        chunk = _capacity_chunk_tokens()
    else:
        raise RuntimeError("capacity buffer requested for baseline backend")
    buf = ContiguousCapacityBuffer(
        stream_name=stream,
        shape_except_token=tuple(shape_except),
        token_dim=token_dim,
        dtype=value.dtype,
        device=value.device,
        capacity=capacity,
        chunk_tokens=chunk,
    )
    buffers[stream] = buf
    return buf


def _append_capacity_stream(cache: PatternQuantizedKVCache, stream: str, value: torch.Tensor, *, token_dim: int) -> torch.Tensor:
    buf = _capacity_buffer(cache, stream, value, token_dim=token_dim)
    buf.append_block(value)
    logical = buf.logical_view()
    if logical is None:
        raise RuntimeError(f"{stream} capacity stream unexpectedly empty after append")
    return logical


def _cat_v_metadata(cache: PatternQuantizedKVCache, attr: str, value: torch.Tensor, *, category: str, compact_stream: str | None = None) -> torch.Tensor:
    if _capacity_cache_enabled(cache):
        stream = compact_stream or attr
        result = _append_capacity_stream(cache, stream, value, token_dim=2)
        setattr(cache, attr, result)
        return result
    result = _cat_assignment(getattr(cache, attr), value, category=category)
    setattr(cache, attr, result)
    return result


def _cat_v_precision_mask(cache: PatternQuantizedKVCache, value: torch.Tensor) -> torch.Tensor:
    if _capacity_cache_enabled(cache):
        result = _append_capacity_stream(cache, "v_precision_mask", value, token_dim=1)
        cache.v_precision_mask = result
        return result
    if cache.v_precision_mask is None:
        cache.v_precision_mask = value
    else:
        bytes_copied = tensor_bytes(cache.v_precision_mask) + tensor_bytes(value)
        record_counter("cache_cat_events", bytes_copied=bytes_copied)
        record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
        with profile_range("cache_mutation", bytes_copied=bytes_copied):
            old_mask = cache.v_precision_mask
            cache.v_precision_mask = torch.cat([old_mask, value], dim=1).contiguous()
        record_cache_mutation("precision_mask", old_mask, value, cache.v_precision_mask)
    return cache.v_precision_mask


def install_value_capacity_buffers(cache: PatternQuantizedKVCache) -> PatternQuantizedKVCache:
    """Adopt existing Value history tensors into capacity buffers.

    This is used by synthetic-past profiling helpers that construct a cache
    directly instead of going through the real quantization flush path.
    """
    cache.capacity_backend = normalize_capacity_growth_backend(getattr(cache, "capacity_backend", None))
    existing = {
        "packed_v": cache.packed_v,
        "packed_v_scale": cache.packed_v_scale,
        "packed_v_zero": cache.packed_v_zero,
        "packed_v4": cache.packed_v4,
        "packed_v4_scale": cache.packed_v4_scale,
        "packed_v4_zero": cache.packed_v4_zero,
        "v_precision_mask": cache.v_precision_mask,
        "v_pattern_mask": cache.v_pattern_mask,
        "v_assignment_idx": cache.v_assignment_idx,
        "v2_pattern_mask": cache.v2_pattern_mask,
        "v2_assignment_idx": cache.v2_assignment_idx,
        "v4_pattern_mask": cache.v4_pattern_mask,
        "v4_assignment_idx": cache.v4_assignment_idx,
    }
    if cache.v_precision_mask is not None and cache.v_pattern_mask is not None and cache.v_assignment_idx is not None:
        mask = cache.v_precision_mask[:, : cache.packed_v_tokens].bool()
        low_mask = ~mask[0]
        high_mask = mask[0]
        if existing["v2_pattern_mask"] is None and int(low_mask.sum().item()):
            existing["v2_pattern_mask"] = cache.v_pattern_mask[:, :, low_mask].to(torch.uint8).contiguous()
            existing["v2_assignment_idx"] = cache.v_assignment_idx[:, :, low_mask].to(torch.int32).contiguous()
        if existing["v4_pattern_mask"] is None and int(high_mask.sum().item()):
            existing["v4_pattern_mask"] = cache.v_pattern_mask[:, :, high_mask].to(torch.uint8).contiguous()
            existing["v4_assignment_idx"] = cache.v_assignment_idx[:, :, high_mask].to(torch.int32).contiguous()
    if existing["v2_pattern_mask"] is not None:
        cache.v2_pattern_mask = existing["v2_pattern_mask"]
    if existing["v2_assignment_idx"] is not None:
        cache.v2_assignment_idx = existing["v2_assignment_idx"]
    if existing["v4_pattern_mask"] is not None:
        cache.v4_pattern_mask = existing["v4_pattern_mask"]
    if existing["v4_assignment_idx"] is not None:
        cache.v4_assignment_idx = existing["v4_assignment_idx"]
    if not _capacity_cache_enabled(cache):
        validate_cache(cache)
        return cache
    stream_dims = {
        "packed_v": 2,
        "packed_v_scale": 2,
        "packed_v_zero": 2,
        "packed_v4": 2,
        "packed_v4_scale": 2,
        "packed_v4_zero": 2,
        "v_precision_mask": 1,
        "v_pattern_mask": 2,
        "v_assignment_idx": 2,
        "v2_pattern_mask": 2,
        "v2_assignment_idx": 2,
        "v4_pattern_mask": 2,
        "v4_assignment_idx": 2,
    }
    for stream, value in existing.items():
        if torch.is_tensor(value) and int(value.shape[stream_dims[stream]]) > 0:
            if "assignment_idx" in stream:
                value = value.to(torch.int32).contiguous()
            setattr(cache, stream, _append_capacity_stream(cache, stream, value, token_dim=stream_dims[stream]))
    cache.v_assignments = cache.v_pattern_mask
    validate_cache(cache)
    return cache


def _cat_v_payload(
    packed_current: torch.Tensor | None,
    scale_current: torch.Tensor | None,
    zero_current: torch.Tensor | None,
    packed: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    *,
    category_prefix: str = "packed_v2",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if packed_current is None:
        return packed, scale, zero
    if scale_current is None or zero_current is None:
        raise ValueError("existing V payload requires scale and zero")
    bytes_copied = tensor_bytes(packed_current) + tensor_bytes(packed) + tensor_bytes(scale_current) + tensor_bytes(scale) + tensor_bytes(zero_current) + tensor_bytes(zero)
    record_counter("cache_cat_events", bytes_copied=bytes_copied)
    record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
    with profile_range("cache_mutation", bytes_copied=bytes_copied):
        result = (
            torch.cat([packed_current, packed], dim=2).contiguous(),
            torch.cat([scale_current, scale], dim=2).contiguous(),
            torch.cat([zero_current, zero], dim=2).contiguous(),
        )
    record_cache_mutation(f"{category_prefix}_payload", packed_current, packed, result[0])
    record_cache_mutation(f"{category_prefix}_scale", scale_current, scale, result[1])
    record_cache_mutation(f"{category_prefix}_zero", zero_current, zero, result[2])
    return result


def _cat_assignment(current: torch.Tensor | None, value: torch.Tensor, *, category: str = "assignments") -> torch.Tensor:
    if current is None:
        return value
    bytes_copied = tensor_bytes(current) + tensor_bytes(value)
    record_counter("cache_cat_events", bytes_copied=bytes_copied)
    record_counter("cache_cat_largest_bytes", calls=0, bytes_copied=bytes_copied)
    with profile_range("cache_mutation", bytes_copied=bytes_copied):
        result = torch.cat([current, value], dim=2).contiguous()
    record_cache_mutation(category, current, value, result)
    return result


def _assign_minmax_hnk(x: torch.Tensor, centroids: torch.Tensor, block_k: int = 256) -> torch.Tensor:
    heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    best_dist = torch.full((heads, tokens), float("inf"), device=x.device, dtype=x.dtype)
    best_idx = torch.zeros((heads, tokens), device=x.device, dtype=torch.long)
    for start in range(0, centroids.shape[1], block_k):
        stop = min(start + block_k, centroids.shape[1])
        diff = x.unsqueeze(2) - centroids[:, start:stop, :].unsqueeze(1)
        distance = diff.amax(dim=-1) - diff.amin(dim=-1)
        cand, idx = distance.min(dim=-1)
        better = cand < best_dist
        best_dist[better] = cand[better]
        best_idx[better] = (start + idx)[better]
    return best_idx


def pattern_chebyshev_center_per_head(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [heads, tokens, dim], got {tuple(x.shape)}")
    return (x.amin(dim=1, keepdim=True) + x.amax(dim=1, keepdim=True)) * 0.5


def pattern_gather_centroids(idx: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if idx.dim() != 3 or centroids.dim() != 3:
        raise ValueError(f"expected idx [B,H,T] and centroids [H,M,D], got {tuple(idx.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens = idx.shape
    if centroids.shape[0] != heads:
        raise ValueError(f"centroid head mismatch: idx heads={heads}, centroids={centroids.shape[0]}")
    dim = centroids.shape[-1]
    expanded = centroids.unsqueeze(0).expand(bsz, -1, -1, -1)
    if idx.dtype != torch.long:
        idx = idx.long()
    return torch.gather(expanded, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dim))


def pattern_gather_request_centroids(idx: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if idx.dim() != 3 or centroids.dim() != 4:
        raise ValueError(f"expected idx [B,H,T] and centroids [B,H,M,D], got {tuple(idx.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens = idx.shape
    if centroids.shape[0] != bsz or centroids.shape[1] != heads:
        raise ValueError(f"request centroid shape mismatch: idx={tuple(idx.shape)} centroids={tuple(centroids.shape)}")
    dim = centroids.shape[-1]
    if idx.dtype != torch.long:
        idx = idx.long()
    return torch.gather(centroids, 2, idx.unsqueeze(-1).expand(-1, -1, -1, dim))


def _assign_minmax_bhnk(x: torch.Tensor, centroids: torch.Tensor, block_k: int = 256, counts: torch.Tensor | None = None) -> torch.Tensor:
    if x.dim() != 4 or centroids.dim() != 4:
        raise ValueError(f"expected x [B,H,T,D] and centroids [B,H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != bsz or centroids.shape[1] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    best_dist = torch.full((bsz, heads, tokens), float("inf"), device=x.device, dtype=x.dtype)
    best_idx = torch.zeros((bsz, heads, tokens), device=x.device, dtype=torch.long)
    for start in range(0, centroids.shape[2], block_k):
        stop = min(start + block_k, centroids.shape[2])
        diff = x.unsqueeze(3) - centroids[:, :, start:stop, :].unsqueeze(2)
        distance = diff.amax(dim=-1) - diff.amin(dim=-1)
        if counts is not None:
            positions = torch.arange(start, stop, dtype=counts.dtype, device=counts.device)
            valid = positions.unsqueeze(0) < counts.to(counts.device).unsqueeze(1)
            distance = torch.where(valid[:, None, None, :], distance, torch.full_like(distance, float("inf")))
        cand, idx = distance.min(dim=-1)
        better = cand < best_dist
        best_dist[better] = cand[better]
        best_idx[better] = (start + idx)[better]
    return best_idx.contiguous()


def pattern_nearest_v_centroid(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    if x.dim() != 4 or centroids.dim() != 3:
        raise ValueError(f"expected x [B,H,T,D] and centroids [H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    diff = x.unsqueeze(2) - centroids.unsqueeze(0).unsqueeze(3)
    distance = diff.amax(dim=-1) - diff.amin(dim=-1)
    return distance.argmin(dim=2).contiguous()


def pattern_nearest_request_v_centroid(x: torch.Tensor, centroids: torch.Tensor, counts: torch.Tensor | None = None) -> torch.Tensor:
    return _assign_minmax_bhnk(x, centroids, counts=counts)


def normalize_value_objective(value_objective: str | None) -> str:
    value = str(value_objective or "base").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "base",
        "minmax": "base",
        "range": "base",
        "dir": "v_dir",
        "direction": "v_dir",
        "hybrid": "v_hybrid",
    }
    value = aliases.get(value, value)
    if value not in {"base", "v_dir", "v_hybrid"}:
        raise ValueError(f"unsupported PatternKV Value objective: {value_objective!r}")
    return value


def normalize_value_precision_selector(selector: str | None) -> str:
    value = str(selector or "base_v2").strip().lower().replace("-", "_")
    aliases = {
        "base": "base_v2",
        "v2": "base_v2",
        "all_v2": "all_v2",
        "mixed_all_v2": "all_v2",
        "v4": "all_v4",
        "all_v4": "all_v4",
        "random": "random_v4",
        "random_v4": "random_v4",
        "causal": "causal_v4",
        "causal_importance": "causal_v4",
        "causal_importance_v4": "causal_v4",
        "oracle": "oracle_v4",
        "oracle_v4": "oracle_v4",
    }
    value = aliases.get(value, value)
    if value not in {"base_v2", "all_v2", "all_v4", "random_v4", "causal_v4", "oracle_v4"}:
        raise ValueError(f"unsupported Value precision selector: {selector!r}")
    return value


def value_precision_is_mixed(selector: str | None) -> bool:
    return normalize_value_precision_selector(selector) != "base_v2"


def pattern_v_threshold_and_mask(x: torch.Tensor, base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.shape != base.shape:
        raise ValueError(f"V threshold tensors must match: {tuple(x.shape)} != {tuple(base.shape)}")
    eps = 1e-12
    range_x = (x.amax(dim=-1) - x.amin(dim=-1)).clamp_min(eps)
    diff = x - base
    range_residual = (diff.amax(dim=-1) - diff.amin(dim=-1)).clamp_min(eps)
    rho = (range_residual / range_x).clamp_min(0.0)
    rho4 = rho * rho
    rho4 = rho4 * rho4
    f32 = torch.float32
    z = torch.sqrt(torch.tensor(2.0, dtype=f32, device=x.device)) * torch.erfinv(torch.tensor(0.9, dtype=f32, device=x.device))
    z = z.to(x.dtype)
    lhs = 1.0 - rho * rho
    rhs = (2.0 * z / torch.sqrt(torch.tensor(5.0 * float(x.shape[-1]), dtype=x.dtype, device=x.device))) * torch.sqrt(1.0 + rho4)
    return rho.unsqueeze(-1), lhs >= rhs


def affine_dequantize_last_dim_reference(x: torch.Tensor, group_size: int, bits: int) -> torch.Tensor:
    if x.shape[-1] % group_size != 0:
        raise ValueError(f"last dim {x.shape[-1]} must be divisible by group_size={group_size}")
    levels = float((1 << bits) - 1)
    grouped = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    zero = grouped.amin(dim=-1)
    mx = grouped.amax(dim=-1)
    scale = ((mx - zero) / levels).clamp_min(torch.finfo(x.dtype).eps)
    q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels)
    out = q.to(x.dtype) * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape_as(x).contiguous()


def pattern_v_candidate_reconstructions(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 4 or centroids.dim() != 3:
        raise ValueError(f"expected x [B,H,T,D] and centroids [H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    expanded_x = x.unsqueeze(2)
    expanded_c = centroids.unsqueeze(0).unsqueeze(3)
    _, mask = pattern_v_threshold_and_mask(expanded_x.expand(-1, -1, centroids.shape[1], -1, -1), expanded_c.expand(bsz, -1, -1, tokens, -1))
    adjusted = expanded_x - mask.unsqueeze(-1).to(x.dtype) * expanded_c
    dequant = affine_dequantize_last_dim_reference(adjusted.contiguous(), group_size, bits)
    restored = dequant + mask.unsqueeze(-1).to(x.dtype) * expanded_c
    base_score = (expanded_x - expanded_c).amax(dim=-1) - (expanded_x - expanded_c).amin(dim=-1)
    return restored, mask, base_score


def pattern_request_v_candidate_reconstructions(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.dim() != 4 or centroids.dim() != 4:
        raise ValueError(f"expected x [B,H,T,D] and centroids [B,H,M,D], got {tuple(x.shape)} {tuple(centroids.shape)}")
    bsz, heads, tokens, dim = x.shape
    if centroids.shape[0] != bsz or centroids.shape[1] != heads or centroids.shape[-1] != dim:
        raise ValueError(f"centroid shape mismatch: x={tuple(x.shape)} centroids={tuple(centroids.shape)}")
    expanded_x = x.unsqueeze(2)
    expanded_c = centroids.unsqueeze(3)
    _, mask = pattern_v_threshold_and_mask(expanded_x.expand(-1, -1, centroids.shape[2], -1, -1), expanded_c.expand(-1, -1, -1, tokens, -1))
    adjusted = expanded_x - mask.unsqueeze(-1).to(x.dtype) * expanded_c
    dequant = affine_dequantize_last_dim_reference(adjusted.contiguous(), group_size, bits)
    restored = dequant + mask.unsqueeze(-1).to(x.dtype) * expanded_c
    base_score = (expanded_x - expanded_c).amax(dim=-1) - (expanded_x - expanded_c).amin(dim=-1)
    return restored, mask, base_score


def _vector_direction_error(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    xf = x.float()
    yf = y.float()
    x_norm = xf.norm(dim=-1)
    y_norm = yf.norm(dim=-1)
    nre = (xf - yf).pow(2).sum(dim=-1) / xf.pow(2).sum(dim=-1).clamp_min(eps)
    valid = (x_norm >= eps) & (y_norm >= eps)
    cosine = (xf * yf).sum(dim=-1) / (x_norm.clamp_min(eps) * y_norm.clamp_min(eps))
    return torch.where(valid, 1.0 - cosine.clamp(-1.0, 1.0), nre)


def _vector_nre(x: torch.Tensor, y: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    xf = x.float()
    yf = y.float()
    return (xf - yf).pow(2).sum(dim=-1) / xf.pow(2).sum(dim=-1).clamp_min(eps)


def local_v2_v4_gain(
    v: torch.Tensor,
    centroid_per_token: torch.Tensor,
    pattern_mask: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    with profile_range("selector_gain", tokens=int(v.shape[2])):
        adjusted = v - pattern_mask.unsqueeze(-1).to(v.dtype) * centroid_per_token
        recon2 = affine_dequantize_last_dim_reference(adjusted, group_size, 2) + pattern_mask.unsqueeze(-1).to(v.dtype) * centroid_per_token
        recon4 = affine_dequantize_last_dim_reference(adjusted, group_size, 4) + pattern_mask.unsqueeze(-1).to(v.dtype) * centroid_per_token
        loss2 = _vector_nre(v, recon2, eps=eps) + _vector_direction_error(v, recon2, eps=eps)
        loss4 = _vector_nre(v, recon4, eps=eps) + _vector_direction_error(v, recon4, eps=eps)
        return (loss2 - loss4).mean(dim=1).clamp_min(0.0).contiguous()


def _budget_k(tokens: int, fraction: float, *, force_nonzero: bool = False) -> int:
    if tokens <= 0:
        return 0
    k = int(round(float(fraction) * float(tokens)))
    if force_nonzero and k == 0:
        k = 1
    return max(0, min(tokens, k))


def _stable_selector_hash(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _topk_mask(score: torch.Tensor, k: int, *, tie_break: torch.Tensor | None = None, largest: bool = True) -> torch.Tensor:
    with profile_range("selector_topk", tokens=int(score.shape[1])):
        bsz, tokens = score.shape
        out = torch.zeros(bsz, tokens, dtype=torch.bool, device=score.device)
        if k <= 0:
            return out
        for b in range(bsz):
            rows = []
            for t in range(tokens):
                primary = float(score[b, t].item())
                tie = float(tie_break[b, t].item()) if tie_break is not None else 0.0
                rows.append((primary, tie, t))
            if largest:
                rows.sort(key=lambda item: (-item[0], -item[1], item[2]))
            else:
                rows.sort(key=lambda item: (item[0], -item[1], item[2]))
            chosen = [t for _primary, _tie, t in rows[:k]]
            out[b, chosen] = True
        return out


def select_value_precision_mask(
    cache: PatternQuantizedKVCache,
    v_window: torch.Tensor,
    centroid_per_token: torch.Tensor,
    pattern_mask: torch.Tensor,
    *,
    absolute_start: int,
) -> torch.Tensor:
    with profile_range("selector_total", tokens=int(v_window.shape[2])):
        selector = normalize_value_precision_selector(getattr(cache, "v_precision_selector", "base_v2"))
        bsz, _heads, tokens, _dim = v_window.shape
        if selector in {"base_v2", "all_v2"}:
            return torch.zeros(bsz, tokens, dtype=torch.bool, device=v_window.device)
        if selector == "all_v4":
            return torch.ones(bsz, tokens, dtype=torch.bool, device=v_window.device)
        k = _budget_k(tokens, float(getattr(cache, "v4_budget_fraction", 0.125)), force_nonzero=False)
        gain = local_v2_v4_gain(v_window, centroid_per_token, pattern_mask, group_size=cache.group_size)
        if selector == "random_v4":
            with profile_range("selector_score", tokens=tokens):
                scores = torch.empty(bsz, tokens, dtype=torch.float64, device=v_window.device)
                task_key = str(getattr(cache, "selector_task_key", "task"))
                layer_idx = int(getattr(cache, "selector_layer_idx", -1))
                window_idx = int(getattr(cache, "pack_count_v", 0))
                seed = int(getattr(cache, "random_selector_seed", 20260809))
                for b in range(bsz):
                    for t in range(tokens):
                        h = _stable_selector_hash(task_key, layer_idx, window_idx, absolute_start + t, seed)
                        scores[b, t] = float(h) / float((1 << 64) - 1)
            return _topk_mask(scores, k, tie_break=gain, largest=False)
        abs_positions = torch.arange(absolute_start, absolute_start + tokens, device=v_window.device).unsqueeze(0).expand(bsz, -1)
        if selector == "causal_v4":
            with profile_range("selector_score", tokens=tokens):
                importance = torch.zeros(bsz, tokens, dtype=gain.dtype, device=v_window.device)
                causal = getattr(cache, "v_causal_importance", None)
                if torch.is_tensor(causal) and causal.shape[1] >= absolute_start + tokens:
                    importance = causal[:, absolute_start : absolute_start + tokens].to(gain.device, gain.dtype)
                score = (importance + 1e-8) * gain
            return _topk_mask(score, k, tie_break=gain - abs_positions.to(gain.dtype) * 0.0, largest=True)
        with profile_range("selector_score", tokens=tokens):
            oracle = getattr(cache, "v_oracle_importance", None)
            future = torch.zeros(bsz, tokens, dtype=gain.dtype, device=v_window.device)
            if torch.is_tensor(oracle) and oracle.shape[1] >= absolute_start + tokens:
                future = oracle[:, absolute_start : absolute_start + tokens].to(gain.device, gain.dtype)
            score = future * gain
        return _topk_mask(score, k, tie_break=gain, largest=True)


def _cat_mixed_packed_v(
    cache: PatternQuantizedKVCache,
    v_adjusted: torch.Tensor,
    precision_mask: torch.Tensor,
    tokens: int,
    *,
    v_assignment_idx: torch.Tensor,
    v_pattern_mask: torch.Tensor,
) -> None:
    with profile_range("pack_total", tokens=int(tokens)):
        from quant.page_batch import append_operator_ready_page_pools, build_operator_ready_page_pools, pack_mixed_v_pages

        chunk_page_cache = pack_mixed_v_pages(
            v_adjusted,
            precision_mask.bool().contiguous(),
            v_pattern_mask.to(torch.uint8).contiguous(),
            v_assignment_idx.contiguous(),
            cache.v_centroids,
            group_size=cache.group_size,
            nh=int(os.environ.get("PATTERNKV_RUNTIME_NH", "32")),
        )
        chunk_pools = build_operator_ready_page_pools(chunk_page_cache)
        cache.operator_ready_page_pools = append_operator_ready_page_pools(getattr(cache, "operator_ready_page_pools", None), chunk_pools)

        if v_adjusted.shape[0] != 1:
            cache.v_precision_mask = _cat_v_precision_mask(cache, precision_mask.to(torch.uint8))
            cache.packed_v_tokens += int(tokens)
            cache.packed_v4_tokens += int(precision_mask.bool().sum().item())
            cache.pack_count_v += 1
            return

        mask = precision_mask[0].bool()
        low = v_adjusted[:, :, ~mask, :].contiguous()
        high = v_adjusted[:, :, mask, :].contiguous()
        v2_pattern = v_pattern_mask[:, :, ~mask].to(torch.uint8).contiguous()
        v2_idx = v_assignment_idx[:, :, ~mask].contiguous()
        v4_pattern = v_pattern_mask[:, :, mask].to(torch.uint8).contiguous()
        v4_idx = v_assignment_idx[:, :, mask].contiguous()
        if low.shape[2]:
            packed2, scale2, zero2 = quantize_pack_v_reference(low, cache.group_size, 2)
            if _capacity_cache_enabled(cache):
                cache.packed_v = _append_capacity_stream(cache, "packed_v", packed2, token_dim=2)
                cache.packed_v_scale = _append_capacity_stream(cache, "packed_v_scale", scale2, token_dim=2)
                cache.packed_v_zero = _append_capacity_stream(cache, "packed_v_zero", zero2, token_dim=2)
            else:
                cache.packed_v, cache.packed_v_scale, cache.packed_v_zero = _cat_v_payload(
                    cache.packed_v,
                    cache.packed_v_scale,
                    cache.packed_v_zero,
                    packed2,
                    scale2,
                    zero2,
                    category_prefix="packed_v2",
                )
            cache.v2_pattern_mask = _cat_v_metadata(cache, "v2_pattern_mask", v2_pattern, category="v2_pattern_mask")
            cache.v2_assignment_idx = _cat_v_metadata(cache, "v2_assignment_idx", v2_idx, category="v2_assignment_idx")
        if high.shape[2]:
            packed4, scale4, zero4 = quantize_pack_v_reference(high, cache.group_size, 4)
            if _capacity_cache_enabled(cache):
                cache.packed_v4 = _append_capacity_stream(cache, "packed_v4", packed4, token_dim=2)
                cache.packed_v4_scale = _append_capacity_stream(cache, "packed_v4_scale", scale4, token_dim=2)
                cache.packed_v4_zero = _append_capacity_stream(cache, "packed_v4_zero", zero4, token_dim=2)
            else:
                cache.packed_v4, cache.packed_v4_scale, cache.packed_v4_zero = _cat_v_payload(
                    cache.packed_v4,
                    cache.packed_v4_scale,
                    cache.packed_v4_zero,
                    packed4,
                    scale4,
                    zero4,
                    category_prefix="packed_v4",
                )
            cache.v4_pattern_mask = _cat_v_metadata(cache, "v4_pattern_mask", v4_pattern, category="v4_pattern_mask")
            cache.v4_assignment_idx = _cat_v_metadata(cache, "v4_assignment_idx", v4_idx, category="v4_assignment_idx")
            cache.packed_v4_tokens += int(high.shape[2])
        mask_u8 = precision_mask.to(torch.uint8)
        _cat_v_precision_mask(cache, mask_u8)
        cache.packed_v_tokens += int(tokens)
        cache.pack_count_v += 1


def reconstruct_packed_v(cache: QuantizedKVCache) -> torch.Tensor | None:
    if not isinstance(cache, PatternQuantizedKVCache) or cache.v_precision_mask is None:
        packed_v = dequantize_v_reference(cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, cache.group_size, cache.v_bits)
        return packed_v[:, :, : cache.packed_v_tokens, :].contiguous() if packed_v is not None else None
    if cache.v_precision_mask.shape[0] != 1:
        raise ValueError("mixed Value precision currently requires batch size 1")
    mask = cache.v_precision_mask[:, : cache.packed_v_tokens].bool()
    low = dequantize_v_reference(cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, cache.group_size, 2)
    high = dequantize_v_reference(cache.packed_v4, cache.packed_v4_scale, cache.packed_v4_zero, cache.group_size, 4)
    template = high if high is not None else low
    if template is None:
        return None
    out = torch.empty(template.shape[0], template.shape[1], cache.packed_v_tokens, template.shape[-1], dtype=template.dtype, device=template.device)
    if low is not None:
        out[:, :, ~mask[0], :] = low[:, :, : int((~mask[0]).sum().item()), :]
    if high is not None:
        out[:, :, mask[0], :] = high[:, :, : int(mask[0].sum().item()), :]
    return out.contiguous()


def update_value_causal_importance(cache: PatternQuantizedKVCache, attn_weights: torch.Tensor) -> None:
    with profile_range("importance_update", tokens=int(attn_weights.shape[-1]) if attn_weights.dim() == 4 else 0):
        if attn_weights.dim() != 4:
            raise ValueError(f"expected attention weights [B,QH,Q,T], got {tuple(attn_weights.shape)}")
        mass = attn_weights.detach().float().mean(dim=1).sum(dim=1)
        if cache.v_causal_importance is None or cache.v_causal_importance.shape[1] < cache.total_tokens:
            new_state = torch.zeros(mass.shape[0], cache.total_tokens, dtype=torch.float32, device=mass.device)
            if torch.is_tensor(cache.v_causal_importance):
                old = cache.v_causal_importance.to(mass.device)
                new_state[:, : old.shape[1]] = old
                record_cache_mutation("causal_importance", old, None, new_state)
            else:
                record_cache_mutation("causal_importance", None, None, new_state)
            cache.v_causal_importance = new_state
        elif cache.v_causal_importance.device != mass.device or cache.v_causal_importance.dtype != torch.float32:
            cache.v_causal_importance = cache.v_causal_importance.to(device=mass.device, dtype=torch.float32)
        width = min(mass.shape[1], cache.total_tokens)
        cache.v_causal_importance[:, :width] += mass[:, :width]


def pattern_select_v_candidate(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    value_objective: str,
    group_size: int,
    bits: int,
    tie_atol: float = 1e-7,
    block_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    objective = normalize_value_objective(value_objective)
    if objective == "base":
        idx = pattern_nearest_v_centroid(x, centroids).to(torch.long)
        selected = pattern_gather_centroids(idx, centroids).to(x.dtype)
        _, mask = pattern_v_threshold_and_mask(x, selected)
        return idx, mask, {"base_score": torch.empty(0, device=x.device, dtype=x.dtype)}
    if x.shape[2] > block_tokens:
        idx_parts = []
        mask_parts = []
        for start in range(0, x.shape[2], block_tokens):
            stop = min(start + block_tokens, x.shape[2])
            idx_part, mask_part, _ = pattern_select_v_candidate(
                x[:, :, start:stop, :].contiguous(),
                centroids,
                value_objective=objective,
                group_size=group_size,
                bits=bits,
                tie_atol=tie_atol,
                block_tokens=block_tokens,
            )
            idx_parts.append(idx_part)
            mask_parts.append(mask_part)
        return torch.cat(idx_parts, dim=2), torch.cat(mask_parts, dim=2), {"score": torch.empty(0, device=x.device, dtype=torch.float32)}
    restored, masks, base_score = pattern_v_candidate_reconstructions(x, centroids, group_size=group_size, bits=bits)
    expanded_x = x.unsqueeze(2).expand_as(restored)
    direction = _vector_direction_error(expanded_x, restored)
    nre = _vector_nre(expanded_x, restored)
    score = direction if objective == "v_dir" else direction + nre
    best = score.min(dim=2).values
    eligible = score <= best.unsqueeze(2) + float(tie_atol)
    masked_base = torch.where(eligible, base_score.float(), torch.full_like(base_score.float(), float("inf")))
    idx = masked_base.argmin(dim=2).contiguous().to(torch.long)
    mask = torch.gather(masks, 2, idx.unsqueeze(2)).squeeze(2).contiguous()
    return idx, mask, {"score": score, "direction": direction, "nre": nre, "base_score": base_score}


def pattern_select_request_v_candidate(
    x: torch.Tensor,
    centroids: torch.Tensor,
    *,
    value_objective: str,
    group_size: int,
    bits: int,
    centroid_counts: torch.Tensor | None = None,
    tie_atol: float = 1e-7,
    block_tokens: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    objective = normalize_value_objective(value_objective)
    if objective == "base":
        idx = pattern_nearest_request_v_centroid(x, centroids, counts=centroid_counts).to(torch.long)
        selected = pattern_gather_request_centroids(idx, centroids).to(x.dtype)
        _, mask = pattern_v_threshold_and_mask(x, selected)
        return idx, mask, {"base_score": torch.empty(0, device=x.device, dtype=x.dtype)}
    if x.shape[2] > block_tokens:
        idx_parts = []
        mask_parts = []
        for start in range(0, x.shape[2], block_tokens):
            stop = min(start + block_tokens, x.shape[2])
            idx_part, mask_part, _ = pattern_select_request_v_candidate(
                x[:, :, start:stop, :].contiguous(),
                centroids,
                value_objective=objective,
                group_size=group_size,
                bits=bits,
                centroid_counts=centroid_counts,
                tie_atol=tie_atol,
                block_tokens=block_tokens,
            )
            idx_parts.append(idx_part)
            mask_parts.append(mask_part)
        return torch.cat(idx_parts, dim=2), torch.cat(mask_parts, dim=2), {"score": torch.empty(0, device=x.device, dtype=torch.float32)}
    restored, masks, base_score = pattern_request_v_candidate_reconstructions(x, centroids, group_size=group_size, bits=bits)
    expanded_x = x.unsqueeze(2).expand_as(restored)
    direction = _vector_direction_error(expanded_x, restored)
    nre = _vector_nre(expanded_x, restored)
    score = direction if objective == "v_dir" else direction + nre
    if centroid_counts is not None:
        positions = torch.arange(centroids.shape[2], dtype=centroid_counts.dtype, device=centroid_counts.device)
        valid = positions.unsqueeze(0) < centroid_counts.to(centroid_counts.device).unsqueeze(1)
        score = torch.where(valid[:, None, :, None], score, torch.full_like(score, float("inf")))
        base_score = torch.where(valid[:, None, :, None], base_score, torch.full_like(base_score, float("inf")))
    best = score.min(dim=2).values
    eligible = score <= best.unsqueeze(2) + float(tie_atol)
    masked_base = torch.where(eligible, base_score.float(), torch.full_like(base_score.float(), float("inf")))
    idx = masked_base.argmin(dim=2).contiguous().to(torch.long)
    mask = torch.gather(masks, 2, idx.unsqueeze(2)).squeeze(2).contiguous()
    return idx, mask, {"score": score, "direction": direction, "nre": nre, "base_score": base_score}


def quantize_pack_k_reference(k: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with profile_range("pack_k", tokens=int(k.shape[2])):
        if k.shape[2] % group_size != 0:
            raise ValueError(f"K token length {k.shape[2]} must be divisible by group_size={group_size}")
        if k.is_cuda:
            return triton_quantize_and_pack_along_last_dim(k.transpose(2, 3).contiguous(), group_size, bits)
        feat_per_int = 32 // bits
        legal_multiple = lcm(group_size, feat_per_int)
        pad_tokens = (-k.shape[2]) % legal_multiple
        if pad_tokens:
            pad = torch.zeros(k.shape[0], k.shape[1], pad_tokens, k.shape[3], dtype=k.dtype, device=k.device)
            k = torch.cat([k, pad], dim=2)
        transposed = k.transpose(2, 3).contiguous()
        bsz, heads, dim, tokens = transposed.shape
        levels = float((1 << bits) - 1)
        grouped = transposed.reshape(bsz, heads, dim, tokens // group_size, group_size)
        zero = grouped.amin(dim=-1)
        mx = grouped.amax(dim=-1)
        scale = ((mx - zero) / levels).clamp_min(torch.finfo(transposed.dtype).eps)
        q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels).to(torch.int32)
        q = q.reshape(bsz, heads, dim, tokens)
        return pack_tensor(q, bits, pack_dim=3), scale.contiguous(), zero.contiguous()


def quantize_pack_v_reference(v: torch.Tensor, group_size: int, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with profile_range(f"pack_v{bits}", tokens=int(v.shape[2])):
        if v.shape[-1] % group_size != 0:
            raise ValueError(f"V head_dim {v.shape[-1]} must be divisible by group_size={group_size}")
        if v.is_cuda:
            return triton_quantize_and_pack_along_last_dim(v.contiguous(), group_size, bits)
        bsz, heads, tokens, dim = v.shape
        levels = float((1 << bits) - 1)
        grouped = v.reshape(bsz, heads, tokens, dim // group_size, group_size)
        zero = grouped.amin(dim=-1)
        mx = grouped.amax(dim=-1)
        scale = ((mx - zero) / levels).clamp_min(torch.finfo(v.dtype).eps)
        q = torch.round((grouped - zero.unsqueeze(-1)) / scale.unsqueeze(-1)).clamp_(0, levels).to(torch.int32)
        q = q.reshape(bsz, heads, tokens, dim)
        return pack_tensor(q, bits, pack_dim=3), scale.contiguous(), zero.contiguous()


def dequantize_k_reference(packed: torch.Tensor | None, scale: torch.Tensor | None, zero: torch.Tensor | None, group_size: int, bits: int) -> torch.Tensor | None:
    if packed is None:
        return None
    if scale is None or zero is None:
        raise ValueError("packed K requires scale and zero")
    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, dim, tokens = q.shape
    grouped = q.reshape(bsz, heads, dim, tokens // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, dim, tokens).transpose(2, 3).contiguous()


def dequantize_v_reference(packed: torch.Tensor | None, scale: torch.Tensor | None, zero: torch.Tensor | None, group_size: int, bits: int) -> torch.Tensor | None:
    if packed is None:
        return None
    if scale is None or zero is None:
        raise ValueError("packed V requires scale and zero")
    q = unpack_tensor(packed, bits, pack_dim=3).to(scale.dtype)
    bsz, heads, tokens, dim = q.shape
    grouped = q.reshape(bsz, heads, tokens, dim // group_size, group_size)
    out = grouped * scale.unsqueeze(-1) + zero.unsqueeze(-1)
    return out.reshape(bsz, heads, tokens, dim).contiguous()


def _pack_raw_pending(cache: QuantizedKVCache, tokens: int) -> None:
    to_pack = cache.pending_k[:, :, :tokens, :].contiguous()
    packed, scale, zero = quantize_pack_k_reference(to_pack, cache.group_size, cache.k_bits)
    _cat_packed_k(cache, packed, scale, zero, tokens)
    cache.pending_k = cache.pending_k[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_k) > tokens else None
    if cache.pending_v is None or tensor_tokens(cache.pending_v) < tokens:
        raise ValueError("V pending must cover the same prefix as K pending")
    value_to_pack = cache.pending_v[:, :, :tokens, :].contiguous()
    packed_v, scale_v, zero_v = quantize_pack_v_reference(value_to_pack, cache.group_size, cache.v_bits)
    _cat_packed_v(cache, packed_v, scale_v, zero_v, tokens)
    cache.pending_v = cache.pending_v[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_v) > tokens else None


def _default_centroid_slots(bsz: int, device: torch.device) -> torch.Tensor:
    return torch.arange(int(bsz), dtype=torch.int64, device=device)


def _ensure_centroid_state_pool(cache: PatternQuantizedKVCache, bsz: int) -> PatternKVCentroidStatePool | None:
    if cache.k_centroids is None or cache.v_centroids is None:
        return None
    device = cache.k_centroids.device
    if cache.centroid_state_indices is None:
        cache.centroid_state_indices = _default_centroid_slots(bsz, device)
    if cache.centroid_state_pool is None:
        max_slots = max(int(os.environ.get("PATTERNKV_CENTROID_MAX_SLOTS", "16")), int(bsz))
        max_dynamic = int(os.environ.get("PATTERNKV_CENTROID_MAX_DYNAMIC", "512"))
        k_static = cache.k_centroids
        v_static = cache.v_centroids
        cache.centroid_state_pool = PatternKVCentroidStatePool.create(
            k_static,
            v_static,
            max_slots=max_slots,
            max_dynamic_centroids=max_dynamic,
            initial_slots=cache.centroid_state_indices,
        )
    cache.centroid_state_pool.allocate(cache.centroid_state_indices)
    return cache.centroid_state_pool


def _active_request_centroids(cache: PatternQuantizedKVCache, *, stream: str) -> torch.Tensor:
    pool = _ensure_centroid_state_pool(cache, int(cache.centroid_state_indices.numel()) if cache.centroid_state_indices is not None else 1)
    if pool is None or cache.centroid_state_indices is None:
        value = cache.k_centroids if stream == "k" else cache.v_centroids
        if value is None:
            raise ValueError(f"{stream} centroids are missing")
        return value
    slots = cache.centroid_state_indices.long()
    source = pool.k_centroid_pool if stream == "k" else pool.v_centroid_pool
    if slots.numel() == 1:
        updates = int(cache.centroid_updates_k if stream == "k" else cache.centroid_updates_v)
        static_count = int(pool.static_centroid_count if stream == "k" else (pool.static_v_centroid_count or pool.static_centroid_count))
        max_count = int(static_count + updates)
        centroids = source[slots, :, :max_count, :].contiguous()
        return centroids[0]
    return source[slots].contiguous()


def _active_centroid_counts(cache: PatternQuantizedKVCache, *, stream: str) -> torch.Tensor | None:
    pool = cache.centroid_state_pool
    if pool is None or cache.centroid_state_indices is None:
        return None
    counts = pool.k_counts if stream == "k" else pool.v_counts
    return counts[cache.centroid_state_indices.long()]


def _sync_cache_centroid_views(cache: PatternQuantizedKVCache) -> None:
    if cache.centroid_state_pool is None or cache.centroid_state_indices is None:
        return
    cache.k_centroids = _active_request_centroids(cache, stream="k")
    cache.v_centroids = _active_request_centroids(cache, stream="v")


def _append_dynamic_centroids(cache: PatternQuantizedKVCache, k_window: torch.Tensor, v_window: torch.Tensor) -> None:
    with profile_range("centroid_update", tokens=int(k_window.shape[2])):
        bsz, heads, tokens, dim = k_window.shape
        pool = _ensure_centroid_state_pool(cache, bsz)
        if pool is None or cache.centroid_state_indices is None:
            xk = k_window.permute(1, 0, 2, 3).reshape(heads, bsz * tokens, dim).contiguous()
            xv = v_window.permute(1, 0, 2, 3).reshape(heads, bsz * tokens, dim).contiguous()
            k_centroid = pattern_chebyshev_center_per_head(xk).to(cache.k_centroids.dtype)
            v_centroid = pattern_chebyshev_center_per_head(xv).to(cache.v_centroids.dtype)
            cache.k_centroids = torch.cat([cache.k_centroids, k_centroid], dim=1).contiguous()
            cache.v_centroids = torch.cat([cache.v_centroids, v_centroid], dim=1).contiguous()
            cache.centroid_updates_k += 1
            cache.centroid_updates_v += 1
            return
        flush_mask = cache.centroid_flush_mask
        if flush_mask is None:
            active_rows = torch.ones(bsz, dtype=torch.bool, device=k_window.device)
        else:
            active_rows = flush_mask.to(device=k_window.device, dtype=torch.bool)
        slots = cache.centroid_state_indices.to(device=k_window.device, dtype=torch.long)
        update_slots = slots[active_rows]
        if update_slots.numel() == 0:
            return
        k_centroid = ((k_window.amin(dim=2, keepdim=True) + k_window.amax(dim=2, keepdim=True)) * 0.5).to(pool.k_centroid_pool.dtype)
        v_centroid = ((v_window.amin(dim=2, keepdim=True) + v_window.amax(dim=2, keepdim=True)) * 0.5).to(pool.v_centroid_pool.dtype)
        k_pos = pool.k_counts[update_slots].long()
        v_pos = pool.v_counts[update_slots].long()
        pool.k_centroid_pool[update_slots, :, k_pos, :] = k_centroid[active_rows, :, 0, :]
        pool.v_centroid_pool[update_slots, :, v_pos, :] = v_centroid[active_rows, :, 0, :]
        pool.k_counts[update_slots] += 1
        pool.v_counts[update_slots] += 1
        pool.update_counts_k[update_slots] += 1
        pool.update_counts_v[update_slots] += 1
        pool.last_flush_pos[update_slots] = int(cache.packed_k_tokens + tokens)
        cache.centroid_updates_k += 1
        cache.centroid_updates_v += 1
        _sync_cache_centroid_views(cache)


def _pack_pattern_window(
    cache: PatternQuantizedKVCache,
    tokens: int,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool,
) -> None:
    with profile_range("pack_window", tokens=int(tokens)):
        _pack_pattern_window_impl(
            cache,
            tokens,
            k_assignments=k_assignments,
            v_assignment_idx=v_assignment_idx,
            v_pattern_mask=v_pattern_mask,
            dynamic_update=dynamic_update,
        )


def _pack_pattern_window_impl(
    cache: PatternQuantizedKVCache,
    tokens: int,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool,
) -> None:
    if cache.pending_k is None or cache.pending_v is None:
        return
    if cache.k_centroids is None or cache.v_centroids is None:
        _pack_raw_pending(cache, tokens)
        return
    k_window = cache.pending_k[:, :, :tokens, :].contiguous()
    v_window = cache.pending_v[:, :, :tokens, :].contiguous()
    _ensure_centroid_state_pool(cache, int(k_window.shape[0]))
    if dynamic_update:
        _append_dynamic_centroids(cache, k_window, v_window)
    bsz, heads, window_tokens, dim = k_window.shape
    k_centroids_active = _active_request_centroids(cache, stream="k")
    v_centroids_active = _active_request_centroids(cache, stream="v")
    k_counts_active = _active_centroid_counts(cache, stream="k")
    v_counts_active = _active_centroid_counts(cache, stream="v")
    if k_assignments is None:
        if k_centroids_active.dim() == 4:
            k_assignments = _assign_minmax_bhnk(k_window, k_centroids_active, counts=k_counts_active).to(torch.long)
        else:
            xk = k_window.permute(1, 0, 2, 3).reshape(heads, bsz * window_tokens, dim).contiguous()
            assign_hn = _assign_minmax_hnk(xk, k_centroids_active)
            k_assignments = assign_hn.view(heads, bsz, window_tokens).permute(1, 0, 2).contiguous().to(torch.long)
    else:
        k_assignments = k_assignments[:, :, :tokens].contiguous().to(torch.long)
    if v_assignment_idx is None:
        with profile_range("pattern_assignment", tokens=int(tokens)):
            if v_centroids_active.dim() == 4:
                v_assignment_idx, inferred_v_pattern_mask, _ = pattern_select_request_v_candidate(
                    v_window,
                    v_centroids_active,
                    value_objective=getattr(cache, "value_objective", "base"),
                    group_size=cache.group_size,
                    bits=cache.v_bits,
                    centroid_counts=v_counts_active,
                )
            else:
                v_assignment_idx, inferred_v_pattern_mask, _ = pattern_select_v_candidate(
                    v_window,
                    v_centroids_active,
                    value_objective=getattr(cache, "value_objective", "base"),
                    group_size=cache.group_size,
                    bits=cache.v_bits,
                )
        if v_pattern_mask is None:
            v_pattern_mask = inferred_v_pattern_mask
    else:
        v_assignment_idx = v_assignment_idx[:, :, :tokens].contiguous().to(torch.long)
    if k_centroids_active.dim() == 4:
        k_centroid_per_token = pattern_gather_request_centroids(k_assignments, k_centroids_active).to(k_window.dtype)
    else:
        k_centroid_per_token = pattern_gather_centroids(k_assignments, k_centroids_active).to(k_window.dtype)
    if v_centroids_active.dim() == 4:
        v_centroid_per_token = pattern_gather_request_centroids(v_assignment_idx, v_centroids_active).to(v_window.dtype)
    else:
        v_centroid_per_token = pattern_gather_centroids(v_assignment_idx, v_centroids_active).to(v_window.dtype)
    if v_pattern_mask is None:
        _, v_pattern_mask = pattern_v_threshold_and_mask(v_window, v_centroid_per_token)
    else:
        v_pattern_mask = v_pattern_mask[:, :, :tokens].contiguous().bool()
    k_residual = k_window - k_centroid_per_token
    v_adjusted = v_window - v_pattern_mask.unsqueeze(-1).to(v_window.dtype) * v_centroid_per_token
    v_assignment_idx_storage = v_assignment_idx.to(torch.int32).contiguous()
    if getattr(cache, "trace_layer_idx", None) is not None:
        try:
            from bench.patternkv_equivalence_reference import save_assignment_trace

            save_assignment_trace(
                mode=str(getattr(cache, "cache_mode", "segmented")),
                layer_idx=int(getattr(cache, "trace_layer_idx")),
                decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                k_window=k_window,
                v_window=v_window,
                k_centroids=cache.k_centroids,
                k_assignments=k_assignments,
                v_centroids=cache.v_centroids,
                v_assignment_idx=v_assignment_idx,
                v_gate=v_pattern_mask,
            )
        except Exception:
            if os.environ.get("PATTERNKV_EQUIV_TRACE_STRICT") == "1":
                raise
    packed_k, scale_k, zero_k = quantize_pack_k_reference(k_residual, cache.group_size, cache.k_bits)
    _cat_packed_k(cache, packed_k, scale_k, zero_k, tokens)
    if value_precision_is_mixed(getattr(cache, "v_precision_selector", "base_v2")):
        absolute_start = tensor_tokens(cache.sink_v) + int(cache.packed_v_tokens)
        precision_mask = select_value_precision_mask(cache, v_window, v_centroid_per_token, v_pattern_mask, absolute_start=absolute_start)
        _cat_mixed_packed_v(
            cache,
            v_adjusted,
            precision_mask,
            tokens,
            v_assignment_idx=v_assignment_idx_storage,
            v_pattern_mask=v_pattern_mask,
        )
    else:
        packed_v, scale_v, zero_v = quantize_pack_v_reference(v_adjusted, cache.group_size, cache.v_bits)
        _cat_packed_v(cache, packed_v, scale_v, zero_v, tokens)
        mask_u8_compact = v_pattern_mask.to(torch.uint8)
        cache.v2_pattern_mask = _cat_v_metadata(cache, "v2_pattern_mask", mask_u8_compact, category="v2_pattern_mask")
        cache.v2_assignment_idx = _cat_v_metadata(cache, "v2_assignment_idx", v_assignment_idx_storage, category="v2_assignment_idx")
    cache.k_assignments = _cat_assignment(cache.k_assignments, k_assignments, category="k_assignments")
    mask_u8 = v_pattern_mask.to(torch.uint8)
    if _capacity_cache_enabled(cache):
        cache.v_assignment_idx = _cat_v_metadata(cache, "v_assignment_idx", v_assignment_idx_storage, category="assignments")
        cache.v_pattern_mask = _cat_v_metadata(cache, "v_pattern_mask", mask_u8, category="pattern_mask")
    else:
        cache.v_assignment_idx = _cat_assignment(cache.v_assignment_idx, v_assignment_idx_storage, category="v_assignment_idx")
        cache.v_pattern_mask = _cat_assignment(cache.v_pattern_mask, mask_u8, category="pattern_mask")
    cache.v_assignments = cache.v_pattern_mask
    cache.pending_k = cache.pending_k[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_k) > tokens else None
    cache.pending_v = cache.pending_v[:, :, tokens:, :].contiguous() if tensor_tokens(cache.pending_v) > tokens else None


def flush_pending(
    cache: QuantizedKVCache,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool = True,
) -> None:
    pending_k_tokens = tensor_tokens(cache.pending_k)
    k_pack_tokens = (pending_k_tokens // cache.group_size) * cache.group_size
    if not k_pack_tokens:
        return
    if not isinstance(cache, PatternQuantizedKVCache):
        _pack_raw_pending(cache, k_pack_tokens)
        return
    if k_assignments is not None or v_assignment_idx is not None or v_pattern_mask is not None:
        _pack_pattern_window(
            cache,
            k_pack_tokens,
            k_assignments=k_assignments,
            v_assignment_idx=v_assignment_idx,
            v_pattern_mask=v_pattern_mask,
            dynamic_update=False,
        )
        return
    while tensor_tokens(cache.pending_k) >= cache.group_size:
        _pack_pattern_window(cache, cache.group_size, dynamic_update=dynamic_update)


def flush_chunked_buffer(
    cache: QuantizedKVCache,
    *,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    dynamic_update: bool = True,
) -> None:
    chunk_tokens = int(getattr(cache, "chunk_length", cache.group_size) or cache.group_size)
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_length must be positive, got {chunk_tokens}")
    if chunk_tokens % cache.group_size != 0:
        raise ValueError(f"chunk_length={chunk_tokens} must be divisible by group_size={cache.group_size}")
    while tensor_tokens(cache.pending_k) >= chunk_tokens:
        if isinstance(cache, PatternQuantizedKVCache):
            _pack_pattern_window(
                cache,
                chunk_tokens,
                k_assignments=k_assignments,
                v_assignment_idx=v_assignment_idx,
                v_pattern_mask=v_pattern_mask,
                dynamic_update=dynamic_update and k_assignments is None and v_assignment_idx is None and v_pattern_mask is None,
            )
            if k_assignments is not None:
                k_assignments = k_assignments[:, :, chunk_tokens:].contiguous() if tensor_tokens(k_assignments) > chunk_tokens else None
            if v_assignment_idx is not None:
                v_assignment_idx = v_assignment_idx[:, :, chunk_tokens:].contiguous() if tensor_tokens(v_assignment_idx) > chunk_tokens else None
            if v_pattern_mask is not None:
                v_pattern_mask = v_pattern_mask[:, :, chunk_tokens:].contiguous() if tensor_tokens(v_pattern_mask) > chunk_tokens else None
        else:
            _pack_raw_pending(cache, chunk_tokens)


def build_cache_from_prefill(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    sink_length: int,
    recent_length: int,
    group_size: int,
    k_bits: int,
    v_bits: int,
    pattern: bool = False,
    k_centroids: torch.Tensor | None = None,
    v_centroids: torch.Tensor | None = None,
    k_assignments: torch.Tensor | None = None,
    v_assignment_idx: torch.Tensor | None = None,
    v_pattern_mask: torch.Tensor | None = None,
    cache_mode: str = ROLLING_CACHE_MODE,
    chunk_length: int | None = None,
    value_objective: str = "base",
    v_precision_selector: str = "base_v2",
    v4_budget_fraction: float = 0.0,
    random_selector_seed: int = 20260809,
    selector_task_key: str | None = None,
    selector_layer_idx: int | None = None,
    v_causal_importance: torch.Tensor | None = None,
    v_oracle_importance: torch.Tensor | None = None,
    centroid_state_indices: torch.Tensor | None = None,
    centroid_flush_mask: torch.Tensor | None = None,
) -> QuantizedKVCache:
    cache_mode = normalize_cache_mode(cache_mode)
    cache_cls = PatternQuantizedKVCache if pattern else QuantizedKVCache
    cache = cache_cls(
        total_tokens=int(key_states.shape[2]),
        sink_length=int(sink_length),
        recent_length=0 if cache_mode == CHUNKED_CACHE_MODE else int(recent_length),
        group_size=int(group_size),
        k_bits=int(k_bits),
        v_bits=int(v_bits),
        cache_mode=cache_mode,
        chunk_length=int(chunk_length if chunk_length is not None else group_size),
    )
    total = cache.total_tokens
    sink_end = 0 if cache_mode == CHUNKED_CACHE_MODE else min(total, sink_length)
    recent_start = total if cache_mode == CHUNKED_CACHE_MODE else max(sink_end, total - recent_length)
    cache.sink_k = _empty_like_tokens(key_states, sink_end)
    cache.sink_v = _empty_like_tokens(value_states, sink_end)
    history_k = key_states[:, :, sink_end:recent_start, :].contiguous()
    history_v = value_states[:, :, sink_end:recent_start, :].contiguous()
    cache.recent_k = key_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.recent_v = value_states[:, :, recent_start:, :].contiguous() if recent_start < total else None
    cache.pending_k = history_k if history_k.shape[2] else None
    cache.pending_v = history_v if history_v.shape[2] else None
    if isinstance(cache, PatternQuantizedKVCache):
        cache.capacity_backend = normalize_capacity_growth_backend(None)
        cache.capacity_buffers = {} if _capacity_cache_enabled(cache) else None
        cache.k_centroids = k_centroids
        cache.v_centroids = v_centroids
        cache.centroid_state_indices = centroid_state_indices.to(dtype=torch.int64, device=key_states.device).contiguous() if centroid_state_indices is not None else _default_centroid_slots(key_states.shape[0], key_states.device)
        cache.centroid_flush_mask = centroid_flush_mask.to(dtype=torch.bool, device=key_states.device).contiguous() if centroid_flush_mask is not None else None
        _ensure_centroid_state_pool(cache, int(key_states.shape[0]))
        _sync_cache_centroid_views(cache)
        cache.value_objective = normalize_value_objective(value_objective)
        cache.v_precision_selector = normalize_value_precision_selector(v_precision_selector)
        cache.v4_budget_fraction = float(v4_budget_fraction)
        cache.random_selector_seed = int(random_selector_seed)
        if selector_task_key is not None:
            cache.selector_task_key = str(selector_task_key)
        if selector_layer_idx is not None:
            cache.selector_layer_idx = int(selector_layer_idx)
        cache.v_causal_importance = v_causal_importance
        cache.v_oracle_importance = v_oracle_importance
        history_k_assignments = k_assignments[:, :, sink_end:recent_start].contiguous() if k_assignments is not None and recent_start > sink_end else None
        history_v_assignment_idx = v_assignment_idx[:, :, sink_end:recent_start].contiguous() if v_assignment_idx is not None and recent_start > sink_end else None
        history_v_pattern_mask = v_pattern_mask[:, :, sink_end:recent_start].contiguous() if v_pattern_mask is not None and recent_start > sink_end else None
        if cache.cache_mode == CHUNKED_CACHE_MODE:
            flush_chunked_buffer(
                cache,
                k_assignments=history_k_assignments,
                v_assignment_idx=history_v_assignment_idx,
                v_pattern_mask=history_v_pattern_mask,
                dynamic_update=False,
            )
        else:
            flush_pending(
                cache,
                k_assignments=history_k_assignments,
                v_assignment_idx=history_v_assignment_idx,
                v_pattern_mask=history_v_pattern_mask,
                dynamic_update=False,
            )
    else:
        if cache.cache_mode == CHUNKED_CACHE_MODE:
            flush_chunked_buffer(cache)
        else:
            flush_pending(cache)
    validate_cache(cache)
    return cache


def _ragged_rows_to_padded_batch(rows: list[torch.Tensor]) -> torch.Tensor | None:
    max_tokens = max((int(row.shape[2]) for row in rows), default=0)
    if max_tokens <= 0:
        return None
    padded = []
    for row in rows:
        if int(row.shape[2]) == max_tokens:
            padded.append(row.contiguous())
            continue
        pad_shape = (row.shape[0], row.shape[1], max_tokens - int(row.shape[2]), row.shape[3])
        pad = torch.zeros(pad_shape, dtype=row.dtype, device=row.device)
        padded.append(torch.cat([row, pad], dim=2).contiguous())
    return torch.cat(padded, dim=0)


def _empty_row_like(reference: torch.Tensor, row: int) -> torch.Tensor:
    return reference[row : row + 1, :, :0, :]


def _roll_ragged_recent_overflow(
    cache: QuantizedKVCache,
    old_segments: dict[str, torch.Tensor],
    *,
    old_recent_physical: int,
    recent_append_tokens: int,
) -> None:
    if cache.recent_k is None or cache.recent_v is None:
        return
    bsz = cache_batch_size(cache)
    appended_k = cache.recent_k[:, :, old_recent_physical : old_recent_physical + recent_append_tokens, :]
    appended_v = cache.recent_v[:, :, old_recent_physical : old_recent_physical + recent_append_tokens, :]
    pending_k_rows = []
    pending_v_rows = []
    recent_k_rows = []
    recent_v_rows = []
    for row in range(bsz):
        old_recent = int(old_segments["recent"][row].item())
        old_pending = int(old_segments["pending"][row].item())
        recent_k_valid = cache.recent_k[row : row + 1, :, :old_recent, :]
        recent_v_valid = cache.recent_v[row : row + 1, :, :old_recent, :]
        if recent_append_tokens:
            recent_k_valid = torch.cat([recent_k_valid, appended_k[row : row + 1]], dim=2)
            recent_v_valid = torch.cat([recent_v_valid, appended_v[row : row + 1]], dim=2)
        overflow = max(int(recent_k_valid.shape[2]) - int(cache.recent_length), 0)
        overflow_k = recent_k_valid[:, :, :overflow, :]
        overflow_v = recent_v_valid[:, :, :overflow, :]
        if cache.pending_k is not None:
            pending_k_valid = cache.pending_k[row : row + 1, :, :old_pending, :]
        else:
            pending_k_valid = _empty_row_like(cache.recent_k, row)
        if cache.pending_v is not None:
            pending_v_valid = cache.pending_v[row : row + 1, :, :old_pending, :]
        else:
            pending_v_valid = _empty_row_like(cache.recent_v, row)
        pending_k_rows.append(torch.cat([pending_k_valid, overflow_k], dim=2))
        pending_v_rows.append(torch.cat([pending_v_valid, overflow_v], dim=2))
        recent_k_rows.append(recent_k_valid[:, :, overflow:, :])
        recent_v_rows.append(recent_v_valid[:, :, overflow:, :])
    cache.pending_k = _ragged_rows_to_padded_batch(pending_k_rows)
    cache.pending_v = _ragged_rows_to_padded_batch(pending_v_rows)
    cache.recent_k = _ragged_rows_to_padded_batch(recent_k_rows)
    cache.recent_v = _ragged_rows_to_padded_batch(recent_v_rows)


def append_decode_rolling(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    append_tokens = int(key_states.shape[2])
    ragged_segments = None
    old_recent_physical = tensor_tokens(cache.recent_k)
    if torch.is_tensor(getattr(cache, "request_total_tokens", None)):
        ragged_segments = k_segment_valid_lengths(cache, device=key_states.device)
    sink_capacity = max(int(cache.sink_length) - tensor_tokens(cache.sink_k), 0)
    sink_fill = min(sink_capacity, append_tokens)
    if sink_fill:
        cache.sink_k = _cat_token(cache.sink_k, key_states[:, :, :sink_fill, :].contiguous(), category="sink")
        cache.sink_v = _cat_token(cache.sink_v, value_states[:, :, :sink_fill, :].contiguous(), category="sink")
    if sink_fill < append_tokens:
        cache.recent_k = _cat_token(cache.recent_k, key_states[:, :, sink_fill:, :].contiguous(), category="recent_pending")
        cache.recent_v = _cat_token(cache.recent_v, value_states[:, :, sink_fill:, :].contiguous(), category="recent_pending")
    cache.total_tokens += int(key_states.shape[2])
    if torch.is_tensor(getattr(cache, "request_total_tokens", None)):
        advance_request_total_tokens(cache, int(key_states.shape[2]))
    overflow = max(tensor_tokens(cache.recent_k) - cache.recent_length, 0)
    if overflow:
        if ragged_segments is not None:
            _roll_ragged_recent_overflow(
                cache,
                ragged_segments,
                old_recent_physical=old_recent_physical,
                recent_append_tokens=append_tokens - sink_fill,
            )
        else:
            cache.pending_k = _cat_token(cache.pending_k, cache.recent_k[:, :, :overflow, :].contiguous(), category="recent_pending")
            cache.pending_v = _cat_token(cache.pending_v, cache.recent_v[:, :, :overflow, :].contiguous(), category="recent_pending")
            cache.recent_k = cache.recent_k[:, :, overflow:, :].contiguous()
            cache.recent_v = cache.recent_v[:, :, overflow:, :].contiguous()
    flush_pending(cache)
    if isinstance(cache, PatternQuantizedKVCache):
        reference = cache.sink_k
        if reference is None:
            reference = cache.recent_k
        if reference is None:
            reference = cache.pending_k
        if reference is not None:
            bsz, heads = reference.shape[0], reference.shape[1]
            device = reference.device
            if cache.packed_k_tokens and (cache.k_assignments is None or cache.k_assignments.shape[2] != cache.packed_k_tokens):
                cache.k_assignments = torch.zeros(bsz, heads, cache.packed_k_tokens, dtype=torch.long, device=device)
            if cache.packed_v_tokens and (cache.v_assignment_idx is None or cache.v_assignment_idx.shape[2] != cache.packed_v_tokens):
                cache.v_assignment_idx = torch.zeros(bsz, heads, cache.packed_v_tokens, dtype=torch.long, device=device)
                cache.v_assignments = torch.zeros(bsz, heads, cache.packed_v_tokens, dtype=torch.uint8, device=device)
    validate_cache(cache)
    return cache


def append_decode_chunked(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    append_decode_chunked_buffer_only(cache, key_states, value_states)
    flush_chunked_buffer(cache)
    validate_cache(cache)
    return cache


def append_decode_chunked_buffer_only(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    if tensor_tokens(cache.sink_k) or tensor_tokens(cache.recent_k):
        raise ValueError("chunked cache must not contain sink or rolling recent tokens")
    cache.pending_k = _cat_token(cache.pending_k, key_states, category="recent_pending")
    cache.pending_v = _cat_token(cache.pending_v, value_states, category="recent_pending")
    cache.total_tokens += int(key_states.shape[2])
    if torch.is_tensor(getattr(cache, "request_total_tokens", None)):
        advance_request_total_tokens(cache, int(key_states.shape[2]))
    return cache


def append_decode(cache: QuantizedKVCache, key_states: torch.Tensor, value_states: torch.Tensor) -> QuantizedKVCache:
    mode = normalize_cache_mode(getattr(cache, "cache_mode", ROLLING_CACHE_MODE))
    if mode == CHUNKED_CACHE_MODE:
        return append_decode_chunked(cache, key_states, value_states)
    return append_decode_rolling(cache, key_states, value_states)


def reconstruct_full_k(cache: QuantizedKVCache) -> torch.Tensor | None:
    packed_k = dequantize_k_reference(cache.packed_k, cache.packed_k_scale, cache.packed_k_zero, cache.group_size, cache.k_bits)
    if packed_k is not None:
        packed_k = packed_k[:, :, : cache.packed_k_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.k_centroids is not None and cache.k_assignments is not None:
            packed_k = packed_k + pattern_gather_centroids(cache.k_assignments[:, :, : cache.packed_k_tokens], cache.k_centroids).to(packed_k.dtype)
    parts = [
        cache.sink_k,
        packed_k,
        cache.pending_k,
        cache.recent_k,
    ]
    parts = [part for part in parts if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def reconstruct_full_v(cache: QuantizedKVCache) -> torch.Tensor | None:
    packed_v = reconstruct_packed_v(cache)
    if packed_v is not None:
        packed_v = packed_v[:, :, : cache.packed_v_tokens, :].contiguous()
        if isinstance(cache, PatternQuantizedKVCache) and cache.v_centroids is not None and cache.v_assignment_idx is not None:
            mask = cache.v_pattern_mask if cache.v_pattern_mask is not None else cache.v_assignments
            if mask is not None:
                centroids = pattern_gather_centroids(cache.v_assignment_idx[:, :, : cache.packed_v_tokens], cache.v_centroids).to(packed_v.dtype)
                packed_v = packed_v + mask[:, :, : cache.packed_v_tokens].unsqueeze(-1).to(packed_v.dtype) * centroids
    parts = [
        cache.sink_v,
        packed_v,
        cache.pending_v,
        cache.recent_v,
    ]
    parts = [part for part in parts if torch.is_tensor(part)]
    return torch.cat(parts, dim=2).contiguous() if parts else None


def cache_segment_stats(cache: QuantizedKVCache | None) -> dict[str, int | None]:
    if cache is None:
        return {
            "sink_tokens": 0,
            "packed_history_tokens": 0,
            "pending_history_tokens": 0,
            "recent_tokens": 0,
            "total_tokens": 0,
            "k_assignment_tokens": None,
            "v_assignment_tokens": None,
        }
    k_assignments = getattr(cache, "k_assignments", None)
    v_assignment_idx = getattr(cache, "v_assignment_idx", None)
    v_pattern_mask = getattr(cache, "v_pattern_mask", None)
    if v_pattern_mask is None:
        v_pattern_mask = getattr(cache, "v_assignments", None)
    return {
        "sink_tokens": tensor_tokens(cache.sink_k),
        "packed_history_tokens": int(cache.packed_k_tokens),
        "pending_history_tokens": tensor_tokens(cache.pending_k),
        "recent_tokens": tensor_tokens(cache.recent_k),
        "chunk_tokens": tensor_tokens(cache.pending_k) if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE else 0,
        "total_tokens": int(cache.total_tokens),
        "cache_mode": getattr(cache, "cache_mode", ROLLING_CACHE_MODE),
        "chunk_length": int(getattr(cache, "chunk_length", cache.group_size)),
        "k_assignment_tokens": tensor_tokens(k_assignments) if torch.is_tensor(k_assignments) else None,
        "v_assignment_tokens": tensor_tokens(v_assignment_idx) if torch.is_tensor(v_assignment_idx) else None,
        "v_pattern_mask_tokens": tensor_tokens(v_pattern_mask) if torch.is_tensor(v_pattern_mask) else None,
    }


def validate_cache(cache: QuantizedKVCache) -> None:
    cache.cache_mode = normalize_cache_mode(getattr(cache, "cache_mode", ROLLING_CACHE_MODE))
    request_lengths = get_total_tokens_per_request(cache)
    ragged_lengths = bool(request_lengths.numel() > 1 and not torch.equal(request_lengths, torch.full_like(request_lengths, int(request_lengths[0].item()))))
    if not int(getattr(cache, "chunk_length", 0) or 0):
        cache.chunk_length = int(cache.group_size)
    sink_tokens = tensor_tokens(cache.sink_k)
    recent_tokens = tensor_tokens(cache.recent_k)
    pending_tokens = tensor_tokens(cache.pending_k)
    if cache.cache_mode == CHUNKED_CACHE_MODE:
        if sink_tokens or recent_tokens or cache.sink_length or cache.recent_length:
            raise ValueError("chunked cache requires empty sink/recent and zero sink/recent lengths")
        if pending_tokens >= cache.chunk_length:
            raise ValueError(f"chunked pending buffer not flushed: {pending_tokens} >= {cache.chunk_length}")
        if cache.packed_k_tokens != (cache.total_tokens // cache.chunk_length) * cache.chunk_length:
            raise ValueError("chunked packed token cadence mismatch")
        if pending_tokens != cache.total_tokens % cache.chunk_length:
            raise ValueError("chunked buffer token cadence mismatch")
    if sink_tokens > cache.sink_length:
        raise ValueError(f"sink exceeds configured length: {sink_tokens} > {cache.sink_length}")
    if recent_tokens > cache.recent_length:
        raise ValueError(f"recent exceeds configured length: {recent_tokens} > {cache.recent_length}")
    if tensor_tokens(cache.sink_v) != sink_tokens:
        raise ValueError("sink K/V token mismatch")
    if tensor_tokens(cache.pending_v) != pending_tokens:
        raise ValueError("pending K/V token mismatch")
    if tensor_tokens(cache.recent_v) != recent_tokens:
        raise ValueError("recent K/V token mismatch")
    if cache.packed_k_tokens != cache.packed_v_tokens:
        raise ValueError(f"packed K/V token mismatch: {cache.packed_k_tokens} != {cache.packed_v_tokens}")
    counted = sink_tokens + cache.packed_k_tokens + pending_tokens + recent_tokens
    if not ragged_lengths and counted != cache.total_tokens:
        raise ValueError(f"cache token conservation failed: counted={counted}, total={cache.total_tokens}")
    if cache.cache_mode != CHUNKED_CACHE_MODE and not ragged_lengths:
        expected = segment_lengths(cache.total_tokens, cache.sink_length, cache.recent_length)
        if sink_tokens != expected["sink_tokens"]:
            raise ValueError(f"sink token count mismatch: {sink_tokens} != {expected['sink_tokens']}")
        if recent_tokens != expected["recent_tokens"]:
            raise ValueError(f"recent token count mismatch: {recent_tokens} != {expected['recent_tokens']}")
        if cache.packed_k_tokens + pending_tokens != expected["quantized_history_tokens"]:
            raise ValueError("history token count mismatch")
    if isinstance(cache, PatternQuantizedKVCache):
        if cache.v_pattern_mask is None and cache.v_assignments is not None:
            cache.v_pattern_mask = cache.v_assignments
        if cache.v_assignments is None and cache.v_pattern_mask is not None:
            cache.v_assignments = cache.v_pattern_mask
        assignment_tokens = tensor_tokens(cache.k_assignments)
        if assignment_tokens not in (0, cache.packed_k_tokens):
            raise ValueError(f"Pattern K assignment tokens must match packed history: {assignment_tokens} != {cache.packed_k_tokens}")
        v_assignment_tokens = tensor_tokens(cache.v_assignment_idx)
        if v_assignment_tokens not in (0, cache.packed_v_tokens):
            raise ValueError(f"Pattern V assignment tokens must match packed history: {v_assignment_tokens} != {cache.packed_v_tokens}")
        v_mask_tokens = tensor_tokens(cache.v_pattern_mask)
        if v_mask_tokens not in (0, cache.packed_v_tokens):
            raise ValueError(f"Pattern V gate tokens must match packed history: {v_mask_tokens} != {cache.packed_v_tokens}")
        if cache.v_precision_mask is not None:
            if cache.v_precision_mask.dim() != 2:
                raise ValueError(f"V precision mask must be [batch, packed_tokens], got {tuple(cache.v_precision_mask.shape)}")
            if cache.v_precision_mask.shape[1] != cache.packed_v_tokens:
                raise ValueError(f"V precision mask tokens must match packed history: {cache.v_precision_mask.shape[1]} != {cache.packed_v_tokens}")
            selected_by_request = cache.v_precision_mask.bool().sum(dim=1).to(dtype=torch.long)
            if torch.is_tensor(getattr(cache, "request_packed_v4_tokens", None)):
                packed_v4_by_request = _request_vector(cache, "request_packed_v4_tokens", int(cache.packed_v4_tokens), device=selected_by_request.device)
                if packed_v4_by_request.numel() == selected_by_request.numel() and not torch.equal(selected_by_request, packed_v4_by_request):
                    raise ValueError(f"V4 payload token mismatch: mask selected={selected_by_request.tolist()}, payload={packed_v4_by_request.tolist()}")
            elif int(selected_by_request.sum().item()) != int(cache.packed_v4_tokens):
                raise ValueError(f"V4 payload token mismatch: mask selected={int(selected_by_request.sum().item())}, payload={cache.packed_v4_tokens}")
            selected = int(selected_by_request.max().item()) if selected_by_request.numel() else 0
            if cache.v_precision_mask.shape[0] > 1 and getattr(cache, "operator_ready_page_pools", None) is not None:
                pools = cache.operator_ready_page_pools
                if torch.is_tensor(getattr(cache, "request_packed_v_tokens", None)):
                    packed_v_by_request = _request_vector(cache, "request_packed_v_tokens", int(cache.packed_v_tokens), device=pools.metadata.seq_lens.device)
                    if not torch.equal(pools.metadata.seq_lens.to(dtype=torch.long), packed_v_by_request):
                        raise ValueError("operator-ready page pool sequence length mismatch")
                elif int(pools.metadata.seq_lens.min().item()) != cache.packed_v_tokens or int(pools.metadata.seq_lens.max().item()) != cache.packed_v_tokens:
                    raise ValueError("operator-ready page pool sequence length mismatch")
                return
            v2_tokens = cache.packed_v_tokens - selected
            if tensor_tokens(cache.packed_v) != v2_tokens:
                raise ValueError(f"V2 payload token mismatch: mask low={v2_tokens}, payload={tensor_tokens(cache.packed_v)}")
            if tensor_tokens(cache.packed_v4) != selected:
                raise ValueError(f"V4 payload tensor token mismatch: mask selected={selected}, payload={tensor_tokens(cache.packed_v4)}")
            if cache.v2_pattern_mask is not None and tensor_tokens(cache.v2_pattern_mask) != v2_tokens:
                raise ValueError(f"V2 compact mask token mismatch: {tensor_tokens(cache.v2_pattern_mask)} != {v2_tokens}")
            if cache.v2_assignment_idx is not None and tensor_tokens(cache.v2_assignment_idx) != v2_tokens:
                raise ValueError(f"V2 compact assignment token mismatch: {tensor_tokens(cache.v2_assignment_idx)} != {v2_tokens}")
            if cache.v4_pattern_mask is not None and tensor_tokens(cache.v4_pattern_mask) != selected:
                raise ValueError(f"V4 compact mask token mismatch: {tensor_tokens(cache.v4_pattern_mask)} != {selected}")
            if cache.v4_assignment_idx is not None and tensor_tokens(cache.v4_assignment_idx) != selected:
                raise ValueError(f"V4 compact assignment token mismatch: {tensor_tokens(cache.v4_assignment_idx)} != {selected}")
        if torch.is_tensor(cache.k_centroids):
            if cache.k_centroids.dim() not in (3, 4):
                raise ValueError(f"K centroids must be [kv_heads, centroids, head_dim] or [B,kv_heads,centroids,head_dim], got {tuple(cache.k_centroids.shape)}")
            k_head_dim = 0 if cache.k_centroids.dim() == 3 else 1
            k_count_dim = 1 if cache.k_centroids.dim() == 3 else 2
            if cache.k_assignments is not None and cache.k_assignments.shape[1] != cache.k_centroids.shape[k_head_dim]:
                raise ValueError("K assignment KV heads must match K centroid heads")
            if cache.k_assignments is not None and cache.k_assignments.numel() and int(cache.k_assignments.max().item()) >= cache.k_centroids.shape[k_count_dim]:
                raise ValueError("K assignment index exceeds K centroid bank")
        if torch.is_tensor(cache.v_centroids):
            if cache.v_centroids.dim() not in (3, 4):
                raise ValueError(f"V centroids must be [kv_heads, centroids, head_dim] or [B,kv_heads,centroids,head_dim], got {tuple(cache.v_centroids.shape)}")
            v_head_dim = 0 if cache.v_centroids.dim() == 3 else 1
            v_count_dim = 1 if cache.v_centroids.dim() == 3 else 2
            if cache.v_assignment_idx is not None and cache.v_assignment_idx.shape[1] != cache.v_centroids.shape[v_head_dim]:
                raise ValueError("V assignment KV heads must match V centroid heads")
            if cache.v_assignment_idx is not None and cache.v_assignment_idx.numel() and int(cache.v_assignment_idx.max().item()) >= cache.v_centroids.shape[v_count_dim]:
                raise ValueError("V assignment index exceeds V centroid bank")


def _pad_tensor_tokens(value: torch.Tensor | None, length: int, max_length: int, token_dim: int) -> torch.Tensor | None:
    if value is None:
        return None
    dim = token_dim if token_dim >= 0 else value.dim() + token_dim
    slices = [slice(None)] * value.dim()
    slices[dim] = slice(0, int(length))
    trimmed = value[tuple(slices)].contiguous()
    pad = int(max_length) - int(trimmed.shape[dim])
    if pad <= 0:
        return trimmed
    shape = list(trimmed.shape)
    shape[dim] = pad
    filler = torch.zeros(shape, dtype=trimmed.dtype, device=trimmed.device)
    return torch.cat([trimmed, filler], dim=dim).contiguous()


def _cat_optional_by_batch(items: list[torch.Tensor | None], lengths: list[int], token_dim: int) -> torch.Tensor | None:
    if not any(torch.is_tensor(item) for item in items):
        return None
    template = next(item for item in items if torch.is_tensor(item))
    max_length = max(int(x) for x in lengths) if lengths else 0
    rows = []
    for item, length in zip(items, lengths):
        if item is None:
            shape = list(template.shape)
            shape[0] = 1
            dim = token_dim if token_dim >= 0 else len(shape) + token_dim
            shape[dim] = max_length
            rows.append(torch.zeros(shape, dtype=template.dtype, device=template.device))
        else:
            rows.append(_pad_tensor_tokens(item, int(length), max_length, token_dim))
    return torch.cat(rows, dim=0).contiguous()


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _request_cache_from_any(value: Any) -> QuantizedKVCache:
    cache = deserialize_cache(value, pattern=True)
    if cache_batch_size(cache) != 1:
        raise ValueError("ragged cache assembly expects per-request B1 caches")
    return cache


def _merge_centroid_pools(caches: list[PatternQuantizedKVCache]) -> tuple[PatternKVCentroidStatePool | None, torch.Tensor | None]:
    pools = [getattr(cache, "centroid_state_pool", None) for cache in caches]
    slots = [getattr(cache, "centroid_state_indices", None) for cache in caches]
    if not all(pool is not None and slot is not None for pool, slot in zip(pools, slots)):
        return None, None
    first_pool = pools[0]
    new_pool = PatternKVCentroidStatePool.create(
        first_pool.k_centroid_pool[int(slots[0][0].item())],
        first_pool.v_centroid_pool[int(slots[0][0].item())],
        max_slots=len(caches),
        max_dynamic_centroids=int(first_pool.k_centroid_pool.shape[2] - first_pool.static_centroid_count),
        initial_slots=torch.arange(len(caches), dtype=torch.long, device=first_pool.k_centroid_pool.device),
    )
    new_slots = torch.arange(len(caches), dtype=torch.long, device=first_pool.k_centroid_pool.device)
    for out_slot, (pool, slot_tensor) in enumerate(zip(pools, slots)):
        src_slot = int(slot_tensor[0].item())
        k_count = int(pool.k_counts[src_slot].item())
        v_count = int(pool.v_counts[src_slot].item())
        new_pool.k_centroid_pool[out_slot, :, :k_count, :] = pool.k_centroid_pool[src_slot, :, :k_count, :]
        new_pool.v_centroid_pool[out_slot, :, :v_count, :] = pool.v_centroid_pool[src_slot, :, :v_count, :]
        new_pool.k_counts[out_slot] = k_count
        new_pool.v_counts[out_slot] = v_count
        new_pool.update_counts_k[out_slot] = int(pool.update_counts_k[src_slot].item())
        new_pool.update_counts_v[out_slot] = int(pool.update_counts_v[src_slot].item())
        new_pool.last_flush_pos[out_slot] = int(pool.last_flush_pos[src_slot].item())
    new_pool.allocate(new_slots)
    return new_pool, new_slots


def _merge_operator_ready_page_pools(caches: list[PatternQuantizedKVCache], centroids: torch.Tensor | None) -> Any | None:
    pools = [getattr(cache, "operator_ready_page_pools", None) for cache in caches]
    if not all(pool is not None for pool in pools):
        return None

    from quant.page_batch import PatternKVBatchMetadata, PatternKVOperatorReadyPagePools

    first = pools[0]
    device = first.metadata.seq_lens.device
    pages_per_request = [int(pool.metadata.num_pages[0].item()) for pool in pools]
    max_pages = max(pages_per_request) if pages_per_request else 0

    def _shift_nonnegative(value: torch.Tensor, delta: int) -> torch.Tensor:
        delta_t = torch.tensor(int(delta), dtype=value.dtype, device=value.device)
        return torch.where(value >= 0, value + delta_t, value)

    def _pad_page_table(row: torch.Tensor, pages: int, delta: int) -> torch.Tensor:
        row = _shift_nonnegative(row[:pages].to(device=device), delta).to(torch.int32)
        if int(row.numel()) == max_pages:
            return row.contiguous()
        pad = torch.full((max_pages - int(row.numel()),), -1, dtype=torch.int32, device=device)
        return torch.cat([row, pad], dim=0).contiguous()

    def _cat_pool_attr(name: str) -> torch.Tensor:
        return torch.cat([getattr(pool, name) for pool in pools], dim=1).contiguous()

    v2_page_offset = 0
    v4_page_offset = 0
    metadata_page_offset = 0
    v2_token_offset = 0
    v4_token_offset = 0
    v2_tables = []
    v4_tables = []
    metadata_tables = []
    v2_offsets = []
    v4_offsets = []
    seq_lens = []
    num_pages = []
    precision_bitmap = []
    v2_counts = []
    v4_counts = []
    valid_tokens = []
    v4_prefix_counts = []
    request_indptr = [0]

    for pool, pages in zip(pools, pages_per_request):
        meta = pool.metadata
        v2_tables.append(_pad_page_table(meta.v2_page_table[0], pages, v2_page_offset))
        v4_tables.append(_pad_page_table(meta.v4_page_table[0], pages, v4_page_offset))
        metadata_tables.append(_pad_page_table(meta.metadata_page_table[0], pages, metadata_page_offset))
        v2_offsets.append(_shift_nonnegative(pool.v2_page_offsets, v2_token_offset))
        v4_offsets.append(_shift_nonnegative(pool.v4_page_offsets, v4_token_offset))
        seq_lens.append(meta.seq_lens[0].to(device=device, dtype=torch.int32))
        num_pages.append(torch.tensor(pages, dtype=torch.int32, device=device))
        precision_bitmap.append(meta.precision_bitmap[:pages].to(device=device))
        v2_counts.append(meta.v2_counts[:pages].to(device=device))
        v4_counts.append(meta.v4_counts[:pages].to(device=device))
        valid_tokens.append(meta.valid_tokens[:pages].to(device=device))
        v4_prefix_counts.append(meta.v4_prefix_counts[:pages].to(device=device))
        request_indptr.append(request_indptr[-1] + pages)
        v2_page_offset += int(pool.v2_page_offsets.numel())
        v4_page_offset += int(pool.v4_page_offsets.numel())
        metadata_page_offset += pages
        v2_token_offset += int(pool.v2_payload_pool.shape[1])
        v4_token_offset += int(pool.v4_payload_pool.shape[1])

    metadata = PatternKVBatchMetadata(
        request_indptr=torch.tensor(request_indptr, dtype=torch.int32, device=device),
        seq_lens=torch.stack(seq_lens).contiguous(),
        num_pages=torch.stack(num_pages).contiguous(),
        v2_page_table=torch.stack(v2_tables).contiguous(),
        v4_page_table=torch.stack(v4_tables).contiguous(),
        metadata_page_table=torch.stack(metadata_tables).contiguous(),
        precision_bitmap=torch.cat(precision_bitmap, dim=0).contiguous(),
        v2_counts=torch.cat(v2_counts, dim=0).contiguous(),
        v4_counts=torch.cat(v4_counts, dim=0).contiguous(),
        valid_tokens=torch.cat(valid_tokens, dim=0).contiguous(),
        v4_prefix_counts=torch.cat(v4_prefix_counts, dim=0).contiguous(),
    )
    return PatternKVOperatorReadyPagePools(
        v2_payload_pool=_cat_pool_attr("v2_payload_pool"),
        v4_payload_pool=_cat_pool_attr("v4_payload_pool"),
        v2_scale_pool=_cat_pool_attr("v2_scale_pool"),
        v2_zero_pool=_cat_pool_attr("v2_zero_pool"),
        v4_scale_pool=_cat_pool_attr("v4_scale_pool"),
        v4_zero_pool=_cat_pool_attr("v4_zero_pool"),
        v2_pattern_pool=_cat_pool_attr("v2_pattern_pool"),
        v4_pattern_pool=_cat_pool_attr("v4_pattern_pool"),
        v2_assignment_pool=_cat_pool_attr("v2_assignment_pool"),
        v4_assignment_pool=_cat_pool_attr("v4_assignment_pool"),
        v2_page_offsets=torch.cat(v2_offsets, dim=0).contiguous(),
        v4_page_offsets=torch.cat(v4_offsets, dim=0).contiguous(),
        metadata=metadata,
        centroids=centroids if torch.is_tensor(centroids) else first.centroids,
        group_size=int(first.group_size),
        nh=int(first.nh),
        nh_kv=int(first.nh_kv),
        head_dim=int(first.head_dim),
        page_size=int(first.page_size),
        historical_materialized_bytes=sum(int(getattr(pool, "historical_materialized_bytes", 0)) for pool in pools),
        page_value_materialized_bytes=sum(int(getattr(pool, "page_value_materialized_bytes", 0)) for pool in pools),
        python_page_dispatches=sum(int(getattr(pool, "python_page_dispatches", 0)) for pool in pools),
        gpu_tensor_item_calls=sum(int(getattr(pool, "gpu_tensor_item_calls", 0)) for pool in pools),
    )


def assemble_ragged_patternkv_cache(per_request_caches: list[Any]) -> PatternQuantizedKVCache:
    caches = [_request_cache_from_any(item) for item in per_request_caches]
    if not caches:
        raise ValueError("cannot assemble empty ragged cache")
    if not all(isinstance(cache, PatternQuantizedKVCache) for cache in caches):
        raise TypeError("ragged PatternKV assembly requires PatternQuantizedKVCache inputs")
    first = caches[0]
    packed_k_lengths = [int(cache.packed_k_tokens) for cache in caches]
    packed_k_payload_lengths = [_ceil_div(length, 32 // int(first.k_bits)) for length in packed_k_lengths]
    packed_k_scale_lengths = [_ceil_div(length, int(first.group_size)) for length in packed_k_lengths]
    packed_v_lengths = [int(cache.packed_v_tokens) for cache in caches]
    packed_v4_lengths = [int(getattr(cache, "packed_v4_tokens", 0) or 0) for cache in caches]
    sink_lengths = [tensor_tokens(cache.sink_k) for cache in caches]
    pending_lengths = [tensor_tokens(cache.pending_k) for cache in caches]
    recent_lengths = [tensor_tokens(cache.recent_k) for cache in caches]
    total_lengths = [int(get_total_tokens_per_request(cache)[0].item()) for cache in caches]
    centroid_pool, centroid_slots = _merge_centroid_pools(caches)
    assembled = PatternQuantizedKVCache(
        sink_k=_cat_optional_by_batch([cache.sink_k for cache in caches], sink_lengths, 2),
        sink_v=_cat_optional_by_batch([cache.sink_v for cache in caches], sink_lengths, 2),
        packed_k=_cat_optional_by_batch([cache.packed_k for cache in caches], packed_k_payload_lengths, 3),
        packed_k_scale=_cat_optional_by_batch([cache.packed_k_scale for cache in caches], packed_k_scale_lengths, 3),
        packed_k_zero=_cat_optional_by_batch([cache.packed_k_zero for cache in caches], packed_k_scale_lengths, 3),
        packed_v=_cat_optional_by_batch([cache.packed_v for cache in caches], packed_v_lengths, 2),
        packed_v_scale=_cat_optional_by_batch([cache.packed_v_scale for cache in caches], packed_v_lengths, 2),
        packed_v_zero=_cat_optional_by_batch([cache.packed_v_zero for cache in caches], packed_v_lengths, 2),
        pending_k=_cat_optional_by_batch([cache.pending_k for cache in caches], pending_lengths, 2),
        pending_v=_cat_optional_by_batch([cache.pending_v for cache in caches], pending_lengths, 2),
        recent_k=_cat_optional_by_batch([cache.recent_k for cache in caches], recent_lengths, 2),
        recent_v=_cat_optional_by_batch([cache.recent_v for cache in caches], recent_lengths, 2),
        total_tokens=max(total_lengths),
        packed_k_tokens=max(packed_k_lengths),
        packed_v_tokens=max(packed_v_lengths),
        sink_length=int(first.sink_length),
        recent_length=int(first.recent_length),
        group_size=int(first.group_size),
        k_bits=int(first.k_bits),
        v_bits=int(first.v_bits),
        pack_count_k=max(int(cache.pack_count_k) for cache in caches),
        pack_count_v=max(int(cache.pack_count_v) for cache in caches),
        cache_mode=getattr(first, "cache_mode", ROLLING_CACHE_MODE),
        chunk_length=int(getattr(first, "chunk_length", first.group_size)),
        cache_backend=getattr(first, "cache_backend", CONTIGUOUS_CACHE_BACKEND),
        k_assignments=_cat_optional_by_batch([cache.k_assignments for cache in caches], packed_k_lengths, 2),
        v_assignments=_cat_optional_by_batch([cache.v_assignments for cache in caches], packed_v_lengths, 2),
        v_assignment_idx=_cat_optional_by_batch([cache.v_assignment_idx for cache in caches], packed_v_lengths, 2),
        v_pattern_mask=_cat_optional_by_batch([cache.v_pattern_mask for cache in caches], packed_v_lengths, 2),
        k_centroids=None,
        v_centroids=None,
        centroid_updates_k=max(int(cache.centroid_updates_k) for cache in caches),
        centroid_updates_v=max(int(cache.centroid_updates_v) for cache in caches),
        value_objective=getattr(first, "value_objective", "base"),
        v_precision_selector=getattr(first, "v_precision_selector", "base_v2"),
        v4_budget_fraction=float(getattr(first, "v4_budget_fraction", 0.0)),
        random_selector_seed=int(getattr(first, "random_selector_seed", 20260809)),
        v_precision_mask=_cat_optional_by_batch([cache.v_precision_mask for cache in caches], packed_v_lengths, 1),
        packed_v4=_cat_optional_by_batch([cache.packed_v4 for cache in caches], packed_v4_lengths, 2),
        packed_v4_scale=_cat_optional_by_batch([cache.packed_v4_scale for cache in caches], packed_v4_lengths, 2),
        packed_v4_zero=_cat_optional_by_batch([cache.packed_v4_zero for cache in caches], packed_v4_lengths, 2),
        packed_v4_tokens=max(packed_v4_lengths),
        v_causal_importance=_cat_optional_by_batch([cache.v_causal_importance for cache in caches], total_lengths, 1),
        v_oracle_importance=_cat_optional_by_batch([cache.v_oracle_importance for cache in caches], total_lengths, 1),
        v2_pattern_mask=_cat_optional_by_batch([cache.v2_pattern_mask for cache in caches], [int(cache.packed_v_tokens) - int(getattr(cache, "packed_v4_tokens", 0) or 0) for cache in caches], 2),
        v2_assignment_idx=_cat_optional_by_batch([cache.v2_assignment_idx for cache in caches], [int(cache.packed_v_tokens) - int(getattr(cache, "packed_v4_tokens", 0) or 0) for cache in caches], 2),
        v4_pattern_mask=_cat_optional_by_batch([cache.v4_pattern_mask for cache in caches], packed_v4_lengths, 2),
        v4_assignment_idx=_cat_optional_by_batch([cache.v4_assignment_idx for cache in caches], packed_v4_lengths, 2),
        capacity_backend=getattr(first, "capacity_backend", CAPACITY_GROWTH_BASELINE),
        capacity_buffers=None,
        operator_ready_page_pools=None,
        centroid_state_pool=centroid_pool,
        centroid_state_indices=centroid_slots,
    )
    set_request_total_tokens(assembled, total_lengths)
    assembled.request_packed_k_tokens = torch.tensor(packed_k_lengths, dtype=torch.long, device=get_total_tokens_per_request(assembled).device)
    assembled.request_packed_v_tokens = torch.tensor(packed_v_lengths, dtype=torch.long, device=get_total_tokens_per_request(assembled).device)
    assembled.request_packed_v4_tokens = torch.tensor(packed_v4_lengths, dtype=torch.long, device=get_total_tokens_per_request(assembled).device)
    if centroid_pool is not None:
        assembled.k_centroids = centroid_pool.current_k(centroid_slots)
        assembled.v_centroids = centroid_pool.current_v(centroid_slots)
    assembled.operator_ready_page_pools = _merge_operator_ready_page_pools(caches, assembled.v_centroids)
    validate_cache(assembled)
    return assembled


def serialize_cache(cache: QuantizedKVCache) -> tuple[Any, ...]:
    base = (
        "patternkv_segmented_cache_v1" if isinstance(cache, PatternQuantizedKVCache) else "quantized_segmented_cache_v1",
        cache.sink_k,
        cache.sink_v,
        cache.packed_k,
        cache.packed_k_scale,
        cache.packed_k_zero,
        cache.packed_v,
        cache.packed_v_scale,
        cache.packed_v_zero,
        cache.pending_k,
        cache.pending_v,
        cache.recent_k,
        cache.recent_v,
        int(cache.total_tokens),
        int(cache.packed_k_tokens),
        int(cache.packed_v_tokens),
        int(cache.sink_length),
        int(cache.recent_length),
        int(cache.group_size),
        int(cache.k_bits),
        int(cache.v_bits),
        int(cache.pack_count_k),
        int(cache.pack_count_v),
    )
    if isinstance(cache, PatternQuantizedKVCache):
        return base + (
            cache.k_assignments,
            cache.v_assignments,
            cache.v_assignment_idx,
            cache.v_pattern_mask,
            cache.k_centroids,
            cache.v_centroids,
            int(cache.centroid_updates_k),
            int(cache.centroid_updates_v),
            cache.cache_mode,
            int(cache.chunk_length),
            cache.value_objective,
            cache.v_precision_selector,
            float(cache.v4_budget_fraction),
            int(cache.random_selector_seed),
            cache.v_precision_mask,
            cache.packed_v4,
            cache.packed_v4_scale,
            cache.packed_v4_zero,
            int(cache.packed_v4_tokens),
            cache.v_causal_importance,
            cache.v_oracle_importance,
            cache.v2_pattern_mask,
            cache.v2_assignment_idx,
            cache.v4_pattern_mask,
            cache.v4_assignment_idx,
            cache.capacity_backend,
            cache.capacity_buffers,
            cache.operator_ready_page_pools,
            cache.centroid_state_pool,
            cache.centroid_state_indices,
            cache.centroid_flush_mask,
            cache.request_total_tokens,
            cache.request_packed_k_tokens,
            cache.request_packed_v_tokens,
            cache.request_packed_v4_tokens,
        )
    return base + (cache.cache_mode, int(cache.chunk_length), cache.request_total_tokens)


def deserialize_cache(value: Any, *, pattern: bool = False) -> QuantizedKVCache:
    if isinstance(value, QuantizedKVCache):
        return value
    if not isinstance(value, tuple) or not value:
        raise TypeError("cache must be a segmented cache tuple")
    tag = value[0]
    if tag not in ("quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"):
        raise TypeError(f"unsupported cache tag: {tag!r}")
    cls = PatternQuantizedKVCache if tag == "patternkv_segmented_cache_v1" or pattern else QuantizedKVCache
    cache = cls(
        sink_k=value[1],
        sink_v=value[2],
        packed_k=value[3],
        packed_k_scale=value[4],
        packed_k_zero=value[5],
        packed_v=value[6],
        packed_v_scale=value[7],
        packed_v_zero=value[8],
        pending_k=value[9],
        pending_v=value[10],
        recent_k=value[11],
        recent_v=value[12],
        total_tokens=int(value[13]),
        packed_k_tokens=int(value[14]),
        packed_v_tokens=int(value[15]),
        sink_length=int(value[16]),
        recent_length=int(value[17]),
        group_size=int(value[18]),
        k_bits=int(value[19]),
        v_bits=int(value[20]),
        pack_count_k=int(value[21]),
        pack_count_v=int(value[22]),
        cache_mode=ROLLING_CACHE_MODE,
        chunk_length=int(value[18]),
    )
    if not isinstance(cache, PatternQuantizedKVCache) and len(value) >= 25 and isinstance(value[23], str):
        cache.cache_mode = normalize_cache_mode(value[23])
        cache.chunk_length = int(value[24])
    pattern_offset = 23
    if isinstance(cache, PatternQuantizedKVCache) and len(value) >= pattern_offset + 7:
        cache.k_assignments = value[pattern_offset]
        cache.v_assignments = value[pattern_offset + 1]
        cache.v_assignment_idx = value[pattern_offset + 2]
        if len(value) >= pattern_offset + 8:
            cache.v_pattern_mask = value[pattern_offset + 3]
            cache.k_centroids = value[pattern_offset + 4]
            cache.v_centroids = value[pattern_offset + 5]
            cache.centroid_updates_k = int(value[pattern_offset + 6])
            cache.centroid_updates_v = int(value[pattern_offset + 7])
        else:
            cache.v_pattern_mask = cache.v_assignments
            cache.k_centroids = value[pattern_offset + 3]
            cache.v_centroids = value[pattern_offset + 4]
            cache.centroid_updates_k = int(value[pattern_offset + 5])
            cache.centroid_updates_v = int(value[pattern_offset + 6])
        if len(value) >= pattern_offset + 10 and isinstance(value[pattern_offset + 8], str):
            cache.cache_mode = normalize_cache_mode(value[pattern_offset + 8])
            cache.chunk_length = int(value[pattern_offset + 9])
        if len(value) >= pattern_offset + 11 and isinstance(value[pattern_offset + 10], str):
            cache.value_objective = normalize_value_objective(value[pattern_offset + 10])
        if len(value) >= pattern_offset + 21 and isinstance(value[pattern_offset + 11], str):
            cache.v_precision_selector = normalize_value_precision_selector(value[pattern_offset + 11])
            cache.v4_budget_fraction = float(value[pattern_offset + 12])
            cache.random_selector_seed = int(value[pattern_offset + 13])
            cache.v_precision_mask = value[pattern_offset + 14]
            cache.packed_v4 = value[pattern_offset + 15]
            cache.packed_v4_scale = value[pattern_offset + 16]
            cache.packed_v4_zero = value[pattern_offset + 17]
            cache.packed_v4_tokens = int(value[pattern_offset + 18])
            cache.v_causal_importance = value[pattern_offset + 19]
            cache.v_oracle_importance = value[pattern_offset + 20]
            if len(value) >= pattern_offset + 27:
                cache.v2_pattern_mask = value[pattern_offset + 21]
                cache.v2_assignment_idx = value[pattern_offset + 22]
                cache.v4_pattern_mask = value[pattern_offset + 23]
                cache.v4_assignment_idx = value[pattern_offset + 24]
                cache.capacity_backend = normalize_capacity_growth_backend(value[pattern_offset + 25])
                cache.capacity_buffers = value[pattern_offset + 26]
            if len(value) >= pattern_offset + 28:
                cache.operator_ready_page_pools = value[pattern_offset + 27]
            if len(value) >= pattern_offset + 31:
                cache.centroid_state_pool = value[pattern_offset + 28]
                cache.centroid_state_indices = value[pattern_offset + 29]
                cache.centroid_flush_mask = value[pattern_offset + 30]
            if len(value) >= pattern_offset + 32:
                cache.request_total_tokens = value[pattern_offset + 31]
            if len(value) >= pattern_offset + 35:
                cache.request_packed_k_tokens = value[pattern_offset + 32]
                cache.request_packed_v_tokens = value[pattern_offset + 33]
                cache.request_packed_v4_tokens = value[pattern_offset + 34]
    if not isinstance(cache, PatternQuantizedKVCache) and len(value) >= 26:
        cache.request_total_tokens = value[25]
    validate_cache(cache)
    return cache


def maybe_validate_cache(cache: QuantizedKVCache) -> None:
    if cache_validate_enabled():
        validate_cache(cache)
