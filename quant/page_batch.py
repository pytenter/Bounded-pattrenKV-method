from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from models.segmented_cache import dequantize_v_reference, pattern_gather_centroids, quantize_pack_v_reference
from quant.patternkv_profile import profile_range, record_counter, record_temp_allocation


PAGE_SIZE = 128

_PAGE_BATCH_COUNTERS = {
    "page_batch_decode_calls": 0,
    "page_batch_pack_calls": 0,
    "logical_pages_processed": 0,
    "v2_pages_processed": 0,
    "v4_pages_processed": 0,
    "v2_tokens_processed": 0,
    "v4_tokens_processed": 0,
    "page_value_materialization_calls": 0,
    "page_value_materialized_bytes": 0,
    "historical_v_materialization_bytes": 0,
    "python_serial_b1_dispatches": 0,
    "host_sync_item_calls": 0,
    "gpu_tensor_item_calls": 0,
    "matmul_calls": 0,
    "accumulate_calls": 0,
    "attention_slice_calls": 0,
    "repeat_kv_calls": 0,
}

_REAL_DECODE_COUNTERS = {
    "real_decode_steps": 0,
    "fused_page_operator_calls": 0,
    "legacy_mixed_v_operator_calls": 0,
    "serial_b1_dispatches": 0,
    "operator_ready_pool_full_rebuilds": 0,
    "operator_ready_pool_incremental_updates": 0,
    "new_pages_allocated": 0,
    "page_value_materialization_bytes": 0,
    "historical_v_materialization_bytes": 0,
    "gpu_tensor_item_calls_hot_path": 0,
    "python_page_dispatches": 0,
}


def reset_patternkv_page_batch_counters() -> None:
    for key in _PAGE_BATCH_COUNTERS:
        _PAGE_BATCH_COUNTERS[key] = 0


def get_patternkv_page_batch_counters() -> dict[str, int]:
    return dict(_PAGE_BATCH_COUNTERS)


def reset_patternkv_real_decode_counters() -> None:
    for key in _REAL_DECODE_COUNTERS:
        _REAL_DECODE_COUNTERS[key] = 0


def get_patternkv_real_decode_counters() -> dict[str, int]:
    return dict(_REAL_DECODE_COUNTERS)


def record_patternkv_real_decode_counter(key: str, amount: int = 1) -> None:
    if key not in _REAL_DECODE_COUNTERS:
        raise KeyError(f"unknown real decode counter: {key}")
    _REAL_DECODE_COUNTERS[key] += int(amount)


@dataclass
class PatternKVBatchMetadata:
    request_indptr: torch.Tensor
    seq_lens: torch.Tensor
    num_pages: torch.Tensor
    v2_page_table: torch.Tensor
    v4_page_table: torch.Tensor
    metadata_page_table: torch.Tensor
    precision_bitmap: torch.Tensor
    v2_counts: torch.Tensor
    v4_counts: torch.Tensor
    valid_tokens: torch.Tensor
    v4_prefix_counts: torch.Tensor


@dataclass
class PatternKVPageBatchCache:
    metadata: PatternKVBatchMetadata
    v2_payload: list[torch.Tensor | None]
    v2_scale: list[torch.Tensor | None]
    v2_zero: list[torch.Tensor | None]
    v4_payload: list[torch.Tensor | None]
    v4_scale: list[torch.Tensor | None]
    v4_zero: list[torch.Tensor | None]
    v2_pattern_mask: list[torch.Tensor | None]
    v2_assignment_idx: list[torch.Tensor | None]
    v4_pattern_mask: list[torch.Tensor | None]
    v4_assignment_idx: list[torch.Tensor | None]
    centroids: torch.Tensor
    group_size: int
    nh: int
    nh_kv: int
    head_dim: int
    page_size: int = PAGE_SIZE
    historical_materialization_calls: int = 0
    historical_materialized_bytes: int = 0


@dataclass
class PatternKVOperatorReadyPagePools:
    v2_payload_pool: torch.Tensor
    v4_payload_pool: torch.Tensor
    v2_scale_pool: torch.Tensor
    v2_zero_pool: torch.Tensor
    v4_scale_pool: torch.Tensor
    v4_zero_pool: torch.Tensor
    v2_pattern_pool: torch.Tensor
    v4_pattern_pool: torch.Tensor
    v2_assignment_pool: torch.Tensor
    v4_assignment_pool: torch.Tensor
    v2_page_offsets: torch.Tensor
    v4_page_offsets: torch.Tensor
    metadata: PatternKVBatchMetadata
    centroids: torch.Tensor
    group_size: int
    nh: int
    nh_kv: int
    head_dim: int
    page_size: int = PAGE_SIZE
    historical_materialized_bytes: int = 0
    page_value_materialized_bytes: int = 0
    python_page_dispatches: int = 0
    gpu_tensor_item_calls: int = 0


def _tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


def _item_int(value: torch.Tensor, component: str) -> int:
    _PAGE_BATCH_COUNTERS["host_sync_item_calls"] += 1
    if value.is_cuda:
        _PAGE_BATCH_COUNTERS["gpu_tensor_item_calls"] += 1
    record_counter("page_batch_item_calls", calls=1)
    record_counter(f"{component}_item_calls", calls=1)
    return int(value.item())


def _pack_precision_bitmap(mask: torch.Tensor, *, page_size: int = PAGE_SIZE) -> torch.Tensor:
    if mask.dim() != 1:
        raise ValueError(f"precision page mask must be 1D, got {tuple(mask.shape)}")
    out = torch.zeros(4, dtype=torch.int64, device=mask.device)
    mask_i64 = mask.to(torch.int64)
    for word in range(4):
        start = word * 32
        stop = min(start + 32, int(mask.shape[0]))
        if start >= stop:
            continue
        shifts = torch.arange(stop - start, dtype=torch.int64, device=mask.device)
        out[word] = ((mask_i64[start:stop] & 1) << shifts).sum()
    return out.to(torch.int32)


def _hash_tensor(tensor: torch.Tensor) -> str:
    data = tensor.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _empty_page_payload_lists() -> tuple[list[torch.Tensor | None], list[torch.Tensor | None], list[torch.Tensor | None]]:
    return [], [], []


def pack_mixed_v_pages(
    v_adjusted: torch.Tensor,
    precision_mask: torch.Tensor,
    v_pattern_mask: torch.Tensor,
    v_assignment_idx: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int = 128,
    nh: int = 32,
    page_size: int = PAGE_SIZE,
) -> PatternKVPageBatchCache:
    """Pack a fixed-length batch into page-centric independent V2/V4 streams."""

    with profile_range("page_batch_pack"):
        if page_size != PAGE_SIZE:
            raise ValueError("S6-B.2 MVP fixes page_size=128")
        if v_adjusted.dim() != 4:
            raise ValueError(f"v_adjusted must be [B,Hkv,T,D], got {tuple(v_adjusted.shape)}")
        bsz, nh_kv, tokens, head_dim = v_adjusted.shape
        if bsz not in (1, 2, 4):
            raise ValueError("S6-B.2 MVP only supports B=1, B=2, and B=4")
        if precision_mask.shape != (bsz, tokens):
            raise ValueError(f"precision_mask must be [B,T], got {tuple(precision_mask.shape)}")
        if v_pattern_mask.shape != (bsz, nh_kv, tokens):
            raise ValueError(f"v_pattern_mask must be [B,Hkv,T], got {tuple(v_pattern_mask.shape)}")
        if v_assignment_idx.shape != (bsz, nh_kv, tokens):
            raise ValueError(f"v_assignment_idx must be [B,Hkv,T], got {tuple(v_assignment_idx.shape)}")
        if centroids.dim() == 3:
            centroid_heads = int(centroids.shape[0])
        elif centroids.dim() == 4:
            if int(centroids.shape[0]) != bsz:
                raise ValueError(f"request-local centroids batch must match V batch, got {tuple(centroids.shape)}")
            centroid_heads = int(centroids.shape[1])
        else:
            raise ValueError(f"centroids must be [Hkv,M,D] or [B,Hkv,M,D], got {tuple(centroids.shape)}")
        if centroid_heads != nh_kv or centroids.shape[-1] != head_dim:
            raise ValueError(f"centroids must be [Hkv,M,D] or [B,Hkv,M,D], got {tuple(centroids.shape)}")
        if head_dim % group_size != 0:
            raise ValueError("head_dim must be divisible by group_size")

        _PAGE_BATCH_COUNTERS["page_batch_pack_calls"] += 1
        device = v_adjusted.device
        num_pages_per_request = (tokens + page_size - 1) // page_size
        total_pages = bsz * num_pages_per_request
        request_indptr = torch.arange(0, total_pages + 1, num_pages_per_request, dtype=torch.int32, device=device)
        seq_lens = torch.full((bsz,), tokens, dtype=torch.int32, device=device)
        num_pages = torch.full((bsz,), num_pages_per_request, dtype=torch.int32, device=device)
        v2_page_table = torch.empty((bsz, num_pages_per_request), dtype=torch.int32, device=device)
        v4_page_table = torch.empty((bsz, num_pages_per_request), dtype=torch.int32, device=device)
        metadata_page_table = torch.arange(total_pages, dtype=torch.int32, device=device).view(bsz, num_pages_per_request)
        precision_bitmap = torch.zeros((total_pages, 4), dtype=torch.int32, device=device)
        v2_counts = torch.zeros((total_pages,), dtype=torch.int16, device=device)
        v4_counts = torch.zeros((total_pages,), dtype=torch.int16, device=device)
        valid_tokens = torch.zeros((total_pages,), dtype=torch.int16, device=device)
        v4_prefix_counts = torch.zeros((total_pages, page_size + 1), dtype=torch.int16, device=device)

        v2_payload, v2_scale, v2_zero = _empty_page_payload_lists()
        v4_payload, v4_scale, v4_zero = _empty_page_payload_lists()
        v2_pattern_pages: list[torch.Tensor | None] = []
        v2_idx_pages: list[torch.Tensor | None] = []
        v4_pattern_pages: list[torch.Tensor | None] = []
        v4_idx_pages: list[torch.Tensor | None] = []

        for b in range(bsz):
            for page in range(num_pages_per_request):
                start = page * page_size
                stop = min(start + page_size, tokens)
                valid = stop - start
                metadata_page = b * num_pages_per_request + page
                page_precision = precision_mask[b, start:stop].bool().contiguous()
                prefix = torch.zeros(page_size + 1, dtype=torch.int16, device=device)
                prefix[1 : valid + 1] = page_precision.to(torch.int16).cumsum(dim=0)
                precision_bitmap[metadata_page] = _pack_precision_bitmap(page_precision, page_size=page_size)
                v4_count = int(page_precision.sum().item())
                v2_count = int(valid - v4_count)
                v2_counts[metadata_page] = v2_count
                v4_counts[metadata_page] = v4_count
                valid_tokens[metadata_page] = valid
                v4_prefix_counts[metadata_page] = prefix

                local_v = v_adjusted[b : b + 1, :, start:stop, :]
                local_mask = v_pattern_mask[b : b + 1, :, start:stop]
                local_idx = v_assignment_idx[b : b + 1, :, start:stop].to(torch.int32)

                if v2_count:
                    v2_values = local_v[:, :, ~page_precision, :].contiguous()
                    p2, s2, z2 = quantize_pack_v_reference(v2_values, group_size, 2)
                    v2_page_id = len(v2_payload)
                    v2_payload.append(p2)
                    v2_scale.append(s2)
                    v2_zero.append(z2)
                    v2_pattern_pages.append(local_mask[:, :, ~page_precision].to(torch.uint8).contiguous())
                    v2_idx_pages.append(local_idx[:, :, ~page_precision].contiguous())
                else:
                    v2_page_id = -1
                    v2_payload.append(None)
                    v2_scale.append(None)
                    v2_zero.append(None)
                    v2_pattern_pages.append(None)
                    v2_idx_pages.append(None)
                if v4_count:
                    v4_values = local_v[:, :, page_precision, :].contiguous()
                    p4, s4, z4 = quantize_pack_v_reference(v4_values, group_size, 4)
                    v4_page_id = len(v4_payload)
                    v4_payload.append(p4)
                    v4_scale.append(s4)
                    v4_zero.append(z4)
                    v4_pattern_pages.append(local_mask[:, :, page_precision].to(torch.uint8).contiguous())
                    v4_idx_pages.append(local_idx[:, :, page_precision].contiguous())
                else:
                    v4_page_id = -1
                    v4_payload.append(None)
                    v4_scale.append(None)
                    v4_zero.append(None)
                    v4_pattern_pages.append(None)
                    v4_idx_pages.append(None)

                v2_page_table[b, page] = v2_page_id
                v4_page_table[b, page] = v4_page_id

        metadata = PatternKVBatchMetadata(
            request_indptr=request_indptr,
            seq_lens=seq_lens,
            num_pages=num_pages,
            v2_page_table=v2_page_table,
            v4_page_table=v4_page_table,
            metadata_page_table=metadata_page_table,
            precision_bitmap=precision_bitmap,
            v2_counts=v2_counts,
            v4_counts=v4_counts,
            valid_tokens=valid_tokens,
            v4_prefix_counts=v4_prefix_counts,
        )
        return PatternKVPageBatchCache(
            metadata=metadata,
            v2_payload=v2_payload,
            v2_scale=v2_scale,
            v2_zero=v2_zero,
            v4_payload=v4_payload,
            v4_scale=v4_scale,
            v4_zero=v4_zero,
            v2_pattern_mask=v2_pattern_pages,
            v2_assignment_idx=v2_idx_pages,
            v4_pattern_mask=v4_pattern_pages,
            v4_assignment_idx=v4_idx_pages,
            centroids=centroids,
            group_size=group_size,
            nh=nh,
            nh_kv=nh_kv,
            head_dim=head_dim,
            page_size=page_size,
        )


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    _PAGE_BATCH_COUNTERS["repeat_kv_calls"] += 1
    with profile_range("page_batch_repeat_kv"):
        if n_rep == 1:
            return hidden_states
        bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(bsz, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


def _restore_page_values(
    payload: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    pattern_mask: torch.Tensor,
    assignment_idx: torch.Tensor,
    centroids: torch.Tensor,
    *,
    bits: int,
    group_size: int,
) -> torch.Tensor:
    with profile_range(f"v{bits}_page_restore"):
        with profile_range(f"v{bits}_dequant"):
            values = dequantize_v_reference(payload, scale, zero, group_size, bits)
        if values is None:
            raise RuntimeError("missing page payload")
        _PAGE_BATCH_COUNTERS["page_value_materialization_calls"] += 1
        _PAGE_BATCH_COUNTERS["page_value_materialized_bytes"] += _tensor_bytes(values)
        with profile_range("page_temp_allocation", bytes_copied=_tensor_bytes(values)):
            record_temp_allocation(f"page_batch_v{bits}_page_values", values)
        with profile_range(f"v{bits}_centroid_gather"):
            gathered = pattern_gather_centroids(assignment_idx.to(torch.long), centroids).to(values.dtype)
        with profile_range(f"v{bits}_restore_combine"):
            return values + pattern_mask.unsqueeze(-1).to(values.dtype) * gathered


def patternkv_page_batch_decode(attn: torch.Tensor, cache: PatternKVPageBatchCache) -> torch.Tensor:
    """Page-centric batched mixed-V decode API.

    The MVP keeps K untouched and consumes request-local compact V2/V4 pages.
    It never calls the legacy B=1 mixed-V operator and never reconstructs the
    full historical Value tensor; only the current physical Value page is
    expanded for the page-local accumulation step.
    """

    with profile_range("page_batch_decode_total"):
        if attn.dim() != 4 or attn.shape[2] != 1:
            raise ValueError(f"attn must be [B,Hq,1,T], got {tuple(attn.shape)}")
        bsz, nh, _q, tokens = attn.shape
        if bsz not in (1, 2, 4):
            raise ValueError("S6-B.2 MVP only supports B=1, B=2, and B=4")
        if nh != cache.nh:
            raise ValueError(f"attention heads mismatch: {nh} != {cache.nh}")
        max_len = _item_int(cache.metadata.seq_lens.max(), "metadata_seq_lens")
        min_len = _item_int(cache.metadata.seq_lens.min(), "metadata_seq_lens")
        if max_len != tokens or min_len != tokens:
            raise ValueError("S6-B.2 MVP requires equal sequence lengths matching attention width")

        _PAGE_BATCH_COUNTERS["page_batch_decode_calls"] += 1
        record_counter("page_batch_decode_calls", calls=1)
        out = torch.zeros((bsz, nh, 1, cache.head_dim), dtype=torch.float32, device=attn.device)
        n_rep = cache.nh // cache.nh_kv
        num_pages = _item_int(cache.metadata.num_pages[0], "metadata_num_pages")
        metadata_pages = cache.metadata.metadata_page_table.reshape(-1)
        v2_pages = cache.metadata.v2_page_table.reshape(-1)
        v4_pages = cache.metadata.v4_page_table.reshape(-1)
        total_pages = int(metadata_pages.numel())
        for flat_page in range(total_pages):
            with profile_range("page_metadata_lookup"):
                b = flat_page // num_pages
                page = flat_page - b * num_pages
                metadata_page = _item_int(metadata_pages[flat_page], "metadata_page")
                valid = _item_int(cache.metadata.valid_tokens[metadata_page], "metadata_valid_tokens")
            if valid <= 0:
                continue
            start = page * cache.page_size
            stop = start + valid
            with profile_range("page_metadata_lookup"):
                v2_page_id = _item_int(v2_pages[flat_page], "metadata_v2_page")
                v4_page_id = _item_int(v4_pages[flat_page], "metadata_v4_page")
                v2_count = _item_int(cache.metadata.v2_counts[metadata_page], "metadata_v2_count")
                v4_count = _item_int(cache.metadata.v4_counts[metadata_page], "metadata_v4_count")
            if v2_count + v4_count != valid:
                raise RuntimeError("invalid page counts")

            _PAGE_BATCH_COUNTERS["logical_pages_processed"] += 1
            _PAGE_BATCH_COUNTERS["v2_tokens_processed"] += v2_count
            _PAGE_BATCH_COUNTERS["v4_tokens_processed"] += v4_count
            with profile_range("page_precision_reconstruct"):
                prefix = cache.metadata.v4_prefix_counts[metadata_page]
                page_precision = (prefix[1 : valid + 1] > prefix[:valid]).bool()
            with profile_range("page_attn_slice"):
                page_attn = attn[b : b + 1, :, :, start:stop]
            if v2_count:
                _PAGE_BATCH_COUNTERS["v2_pages_processed"] += 1
                v2_values = _restore_page_values(
                    cache.v2_payload[v2_page_id],
                    cache.v2_scale[v2_page_id],
                    cache.v2_zero[v2_page_id],
                    cache.v2_pattern_mask[v2_page_id],
                    cache.v2_assignment_idx[v2_page_id],
                    cache.centroids,
                    bits=2,
                    group_size=cache.group_size,
                )
                with profile_range("v2_attn_slice"):
                    _PAGE_BATCH_COUNTERS["attention_slice_calls"] += 1
                    attn2 = page_attn[:, :, :, ~page_precision].contiguous()
                v2_repeated = _repeat_kv(v2_values, n_rep)
                with profile_range("v2_matmul"):
                    _PAGE_BATCH_COUNTERS["matmul_calls"] += 1
                    part2 = torch.matmul(attn2, v2_repeated).float()
                with profile_range("v2_accumulate"):
                    _PAGE_BATCH_COUNTERS["accumulate_calls"] += 1
                    out[b : b + 1] += part2
            if v4_count:
                _PAGE_BATCH_COUNTERS["v4_pages_processed"] += 1
                v4_values = _restore_page_values(
                    cache.v4_payload[v4_page_id],
                    cache.v4_scale[v4_page_id],
                    cache.v4_zero[v4_page_id],
                    cache.v4_pattern_mask[v4_page_id],
                    cache.v4_assignment_idx[v4_page_id],
                    cache.centroids,
                    bits=4,
                    group_size=cache.group_size,
                )
                with profile_range("v4_attn_slice"):
                    _PAGE_BATCH_COUNTERS["attention_slice_calls"] += 1
                    attn4 = page_attn[:, :, :, page_precision].contiguous()
                v4_repeated = _repeat_kv(v4_values, n_rep)
                with profile_range("v4_matmul"):
                    _PAGE_BATCH_COUNTERS["matmul_calls"] += 1
                    part4 = torch.matmul(attn4, v4_repeated).float()
                with profile_range("v4_accumulate"):
                    _PAGE_BATCH_COUNTERS["accumulate_calls"] += 1
                    out[b : b + 1] += part4
        return out.to(attn.dtype)


def patternkv_page_batched_v_decode(attn: torch.Tensor, cache: PatternKVPageBatchCache) -> torch.Tensor:
    return patternkv_page_batch_decode(attn, cache)


def _cat_existing_pages(pages: list[torch.Tensor | None], *, stream: str, nh_kv: int, tail: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    offsets: list[int] = []
    live_pages: list[torch.Tensor] = []
    cursor = 0
    for page in pages:
        if page is None:
            offsets.append(-1)
            continue
        if page.shape[0] != 1 or page.shape[1] != nh_kv or page.shape[-1] != tail:
            raise ValueError(f"{stream} page shape mismatch: got {tuple(page.shape)}")
        offsets.append(cursor)
        live_pages.append(page.squeeze(0).contiguous())
        cursor += int(page.shape[2])
    if live_pages:
        pool = torch.cat(live_pages, dim=1).contiguous()
    else:
        pool = torch.empty((nh_kv, 0, tail), dtype=dtype, device=device)
    return pool, torch.tensor(offsets, dtype=torch.int32, device=device)


def _cat_existing_metadata_pages(pages: list[torch.Tensor | None], *, stream: str, nh_kv: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    live_pages: list[torch.Tensor] = []
    for page in pages:
        if page is None:
            continue
        if page.shape[0] != 1 or page.shape[1] != nh_kv:
            raise ValueError(f"{stream} metadata page shape mismatch: got {tuple(page.shape)}")
        live_pages.append(page.squeeze(0).contiguous())
    if live_pages:
        return torch.cat(live_pages, dim=1).contiguous().to(dtype)
    return torch.empty((nh_kv, 0), dtype=dtype, device=device)


def build_operator_ready_page_pools(cache: PatternKVPageBatchCache) -> PatternKVOperatorReadyPagePools:
    """Create flat GPU pools suitable for a single page-aware operator launch."""

    device = cache.centroids.device
    v2_payload_pool, v2_offsets = _cat_existing_pages(
        cache.v2_payload,
        stream="v2_payload",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // 16,
        dtype=torch.int32,
        device=device,
    )
    v4_payload_pool, v4_offsets = _cat_existing_pages(
        cache.v4_payload,
        stream="v4_payload",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // 8,
        dtype=torch.int32,
        device=device,
    )
    v2_scale_pool, _ = _cat_existing_pages(
        cache.v2_scale,
        stream="v2_scale",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // cache.group_size,
        dtype=cache.centroids.dtype,
        device=device,
    )
    v2_zero_pool, _ = _cat_existing_pages(
        cache.v2_zero,
        stream="v2_zero",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // cache.group_size,
        dtype=cache.centroids.dtype,
        device=device,
    )
    v4_scale_pool, _ = _cat_existing_pages(
        cache.v4_scale,
        stream="v4_scale",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // cache.group_size,
        dtype=cache.centroids.dtype,
        device=device,
    )
    v4_zero_pool, _ = _cat_existing_pages(
        cache.v4_zero,
        stream="v4_zero",
        nh_kv=cache.nh_kv,
        tail=cache.head_dim // cache.group_size,
        dtype=cache.centroids.dtype,
        device=device,
    )
    return PatternKVOperatorReadyPagePools(
        v2_payload_pool=v2_payload_pool,
        v4_payload_pool=v4_payload_pool,
        v2_scale_pool=v2_scale_pool,
        v2_zero_pool=v2_zero_pool,
        v4_scale_pool=v4_scale_pool,
        v4_zero_pool=v4_zero_pool,
        v2_pattern_pool=_cat_existing_metadata_pages(cache.v2_pattern_mask, stream="v2_pattern", nh_kv=cache.nh_kv, dtype=torch.uint8, device=device),
        v4_pattern_pool=_cat_existing_metadata_pages(cache.v4_pattern_mask, stream="v4_pattern", nh_kv=cache.nh_kv, dtype=torch.uint8, device=device),
        v2_assignment_pool=_cat_existing_metadata_pages(cache.v2_assignment_idx, stream="v2_assignment", nh_kv=cache.nh_kv, dtype=torch.int32, device=device),
        v4_assignment_pool=_cat_existing_metadata_pages(cache.v4_assignment_idx, stream="v4_assignment", nh_kv=cache.nh_kv, dtype=torch.int32, device=device),
        v2_page_offsets=v2_offsets,
        v4_page_offsets=v4_offsets,
        metadata=cache.metadata,
        centroids=cache.centroids,
        group_size=cache.group_size,
        nh=cache.nh,
        nh_kv=cache.nh_kv,
        head_dim=cache.head_dim,
        page_size=cache.page_size,
    )


def append_operator_ready_page_pools(
    existing: PatternKVOperatorReadyPagePools | None,
    chunk: PatternKVOperatorReadyPagePools,
) -> PatternKVOperatorReadyPagePools:
    """Append newly packed page pools without rebuilding previous pages."""

    if existing is None:
        record_patternkv_real_decode_counter("operator_ready_pool_incremental_updates", 1)
        record_patternkv_real_decode_counter("new_pages_allocated", int(chunk.metadata.v4_prefix_counts.shape[0]))
        return chunk

    if existing.nh != chunk.nh or existing.nh_kv != chunk.nh_kv or existing.head_dim != chunk.head_dim:
        raise ValueError("operator-ready page pool geometry mismatch")
    if existing.group_size != chunk.group_size or existing.page_size != chunk.page_size:
        raise ValueError("operator-ready page pool ABI mismatch")
    if existing.metadata.v2_page_table.shape[0] != chunk.metadata.v2_page_table.shape[0]:
        raise ValueError("operator-ready page pool batch mismatch")
    def centroid_geometry(centroids: torch.Tensor) -> tuple[int, ...]:
        if centroids.dim() == 3:
            return (3, int(centroids.shape[0]), int(centroids.shape[2]))
        if centroids.dim() == 4:
            return (4, int(centroids.shape[0]), int(centroids.shape[1]), int(centroids.shape[3]))
        raise ValueError("operator-ready page pool centroids must be [H,M,D] or [B,H,M,D]")

    def centroid_bank_size(centroids: torch.Tensor) -> int:
        return int(centroids.shape[1] if centroids.dim() == 3 else centroids.shape[2])

    if centroid_geometry(existing.centroids) != centroid_geometry(chunk.centroids):
        raise ValueError("operator-ready page pool centroid geometry mismatch")
    if centroid_bank_size(chunk.centroids) < centroid_bank_size(existing.centroids):
        raise ValueError("operator-ready page pool centroid bank shrank")

    old_v2_pages = int(existing.v2_page_offsets.numel())
    old_v4_pages = int(existing.v4_page_offsets.numel())
    old_meta_pages = int(existing.metadata.v4_prefix_counts.shape[0])
    old_v2_tokens = int(existing.v2_payload_pool.shape[1])
    old_v4_tokens = int(existing.v4_payload_pool.shape[1])
    pages_per_request = int(existing.metadata.v2_page_table.shape[1] + chunk.metadata.v2_page_table.shape[1])
    bsz = int(existing.metadata.v2_page_table.shape[0])

    def shift_table(table: torch.Tensor, delta: int) -> torch.Tensor:
        delta_t = torch.tensor(delta, dtype=table.dtype, device=table.device)
        return torch.where(table >= 0, table + delta_t, table)

    def shift_offsets(offsets: torch.Tensor, delta: int) -> torch.Tensor:
        delta_t = torch.tensor(delta, dtype=offsets.dtype, device=offsets.device)
        return torch.where(offsets >= 0, offsets + delta_t, offsets)

    metadata = PatternKVBatchMetadata(
        request_indptr=torch.arange(0, (bsz + 1) * pages_per_request, pages_per_request, dtype=torch.int32, device=existing.metadata.request_indptr.device),
        seq_lens=existing.metadata.seq_lens + chunk.metadata.seq_lens,
        num_pages=existing.metadata.num_pages + chunk.metadata.num_pages,
        v2_page_table=torch.cat([existing.metadata.v2_page_table, shift_table(chunk.metadata.v2_page_table, old_v2_pages)], dim=1).contiguous(),
        v4_page_table=torch.cat([existing.metadata.v4_page_table, shift_table(chunk.metadata.v4_page_table, old_v4_pages)], dim=1).contiguous(),
        metadata_page_table=torch.cat([existing.metadata.metadata_page_table, shift_table(chunk.metadata.metadata_page_table, old_meta_pages)], dim=1).contiguous(),
        precision_bitmap=torch.cat([existing.metadata.precision_bitmap, chunk.metadata.precision_bitmap], dim=0).contiguous(),
        v2_counts=torch.cat([existing.metadata.v2_counts, chunk.metadata.v2_counts], dim=0).contiguous(),
        v4_counts=torch.cat([existing.metadata.v4_counts, chunk.metadata.v4_counts], dim=0).contiguous(),
        valid_tokens=torch.cat([existing.metadata.valid_tokens, chunk.metadata.valid_tokens], dim=0).contiguous(),
        v4_prefix_counts=torch.cat([existing.metadata.v4_prefix_counts, chunk.metadata.v4_prefix_counts], dim=0).contiguous(),
    )
    record_patternkv_real_decode_counter("operator_ready_pool_incremental_updates", 1)
    record_patternkv_real_decode_counter("new_pages_allocated", int(chunk.metadata.v4_prefix_counts.shape[0]))
    return PatternKVOperatorReadyPagePools(
        v2_payload_pool=torch.cat([existing.v2_payload_pool, chunk.v2_payload_pool], dim=1).contiguous(),
        v4_payload_pool=torch.cat([existing.v4_payload_pool, chunk.v4_payload_pool], dim=1).contiguous(),
        v2_scale_pool=torch.cat([existing.v2_scale_pool, chunk.v2_scale_pool], dim=1).contiguous(),
        v2_zero_pool=torch.cat([existing.v2_zero_pool, chunk.v2_zero_pool], dim=1).contiguous(),
        v4_scale_pool=torch.cat([existing.v4_scale_pool, chunk.v4_scale_pool], dim=1).contiguous(),
        v4_zero_pool=torch.cat([existing.v4_zero_pool, chunk.v4_zero_pool], dim=1).contiguous(),
        v2_pattern_pool=torch.cat([existing.v2_pattern_pool, chunk.v2_pattern_pool], dim=1).contiguous(),
        v4_pattern_pool=torch.cat([existing.v4_pattern_pool, chunk.v4_pattern_pool], dim=1).contiguous(),
        v2_assignment_pool=torch.cat([existing.v2_assignment_pool, chunk.v2_assignment_pool], dim=1).contiguous(),
        v4_assignment_pool=torch.cat([existing.v4_assignment_pool, chunk.v4_assignment_pool], dim=1).contiguous(),
        v2_page_offsets=torch.cat([existing.v2_page_offsets, shift_offsets(chunk.v2_page_offsets, old_v2_tokens)], dim=0).contiguous(),
        v4_page_offsets=torch.cat([existing.v4_page_offsets, shift_offsets(chunk.v4_page_offsets, old_v4_tokens)], dim=0).contiguous(),
        metadata=metadata,
        centroids=chunk.centroids,
        group_size=existing.group_size,
        nh=existing.nh,
        nh_kv=existing.nh_kv,
        head_dim=existing.head_dim,
        page_size=existing.page_size,
    )


def patternkv_fused_page_batch_decode(attn: torch.Tensor, pools: PatternKVOperatorReadyPagePools) -> torch.Tensor:
    """Single-launch fused page-centric mixed-V Value operator MVP."""

    from quant.matmul import patternkv_gemv

    record_patternkv_real_decode_counter("fused_page_operator_calls", 1)
    out = patternkv_gemv.attn_v_forward_cuda_page_mixed_pool(
        attn.to(torch.float16).contiguous(),
        pools.v2_payload_pool,
        pools.v4_payload_pool,
        pools.v2_scale_pool,
        pools.v2_zero_pool,
        pools.v4_scale_pool,
        pools.v4_zero_pool,
        pools.v2_pattern_pool,
        pools.v4_pattern_pool,
        pools.v2_assignment_pool,
        pools.v4_assignment_pool,
        pools.centroids,
        pools.v2_page_offsets,
        pools.v4_page_offsets,
        pools.metadata.v2_page_table,
        pools.metadata.v4_page_table,
        pools.metadata.metadata_page_table,
        pools.metadata.v4_prefix_counts,
        int(pools.group_size),
        int(pools.nh),
        int(pools.nh_kv),
        int(pools.page_size),
    )
    return out.to(attn.dtype)


def validate_page_mapping(cache: PatternKVPageBatchCache) -> dict[str, Any]:
    metadata = cache.metadata
    bsz = int(metadata.seq_lens.shape[0])
    num_pages = int(metadata.num_pages[0].item())
    rows = []
    ok = True
    for b in range(bsz):
        valid_sum = 0
        for p in range(num_pages):
            mp = int(metadata.metadata_page_table[b, p].item())
            valid = int(metadata.valid_tokens[mp].item())
            v2 = int(metadata.v2_counts[mp].item())
            v4 = int(metadata.v4_counts[mp].item())
            prefix_end = int(metadata.v4_prefix_counts[mp, valid].item())
            page_ok = (v2 + v4 == valid) and (prefix_end == v4)
            ok = ok and page_ok
            valid_sum += valid
            rows.append({"request": b, "page": p, "valid_tokens": valid, "v2_count": v2, "v4_count": v4, "ok": page_ok})
        ok = ok and valid_sum == int(metadata.seq_lens[b].item())
    return {"page_size": cache.page_size, "mapping_valid": bool(ok), "pages": rows}


def selector_isolation_summary(precision_mask: torch.Tensor) -> dict[str, Any]:
    rows = []
    for b in range(precision_mask.shape[0]):
        ids = torch.nonzero(precision_mask[b].bool(), as_tuple=False).flatten().to(torch.int32)
        rows.append(
            {
                "request": b,
                "v4_count": int(ids.numel()),
                "selected_hash": _hash_tensor(ids),
                "selected_first16": ids[:16].detach().cpu().tolist(),
            }
        )
    passed = len({row["selected_hash"] for row in rows}) == len(rows) if len(rows) > 1 else True
    return {"selector_isolation_pass": passed, "requests": rows}


def cache_isolation_summary(
    attn: torch.Tensor,
    v_adjusted: torch.Tensor,
    precision_mask: torch.Tensor,
    v_pattern_mask: torch.Tensor,
    v_assignment_idx: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int = 128,
    nh: int = 32,
) -> dict[str, Any]:
    if precision_mask.shape[0] < 2:
        return {"cache_isolation_pass": True, "reason": "B=1"}
    base_cache = pack_mixed_v_pages(v_adjusted, precision_mask, v_pattern_mask, v_assignment_idx, centroids, group_size=group_size, nh=nh)
    base = patternkv_page_batch_decode(attn, base_cache)
    mutated_precision = precision_mask.clone()
    mutated_precision[0] = ~mutated_precision[0]
    mutated_cache = pack_mixed_v_pages(v_adjusted, mutated_precision, v_pattern_mask, v_assignment_idx, centroids, group_size=group_size, nh=nh)
    mutated = patternkv_page_batch_decode(attn, mutated_cache)
    unchanged = torch.equal(base[1:].detach().cpu(), mutated[1:].detach().cpu())
    return {
        "cache_isolation_pass": bool(unchanged),
        "request0_changed_max_abs": float((base[0].float() - mutated[0].float()).abs().max().item()),
        "other_requests_max_abs": float((base[1:].float() - mutated[1:].float()).abs().max().item()),
        "other_requests_checksum_before": _hash_tensor(base[1:]),
        "other_requests_checksum_after": _hash_tensor(mutated[1:]),
    }


def correctness_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    diff = (candidate.float() - reference.float()).abs()
    rel_l2 = torch.linalg.vector_norm((candidate - reference).float()) / torch.linalg.vector_norm(reference.float()).clamp_min(1e-8)
    cosine = torch.nn.functional.cosine_similarity(candidate.float().flatten(), reference.float().flatten(), dim=0)
    return {
        "max_abs": float(diff.max().item()),
        "relative_l2": float(rel_l2.item()),
        "cosine": float(cosine.item()),
        "nan": int(torch.isnan(candidate).sum().item()),
        "inf": int(torch.isinf(candidate).sum().item()),
    }
