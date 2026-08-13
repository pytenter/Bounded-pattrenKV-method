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
}


def reset_patternkv_page_batch_counters() -> None:
    for key in _PAGE_BATCH_COUNTERS:
        _PAGE_BATCH_COUNTERS[key] = 0


def get_patternkv_page_batch_counters() -> dict[str, int]:
    return dict(_PAGE_BATCH_COUNTERS)


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


def _tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else int(value.numel() * value.element_size())


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
        if centroids.shape[0] != nh_kv or centroids.shape[-1] != head_dim:
            raise ValueError(f"centroids must be [Hkv,M,D], got {tuple(centroids.shape)}")
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
    values = dequantize_v_reference(payload, scale, zero, group_size, bits)
    if values is None:
        raise RuntimeError("missing page payload")
    _PAGE_BATCH_COUNTERS["page_value_materialization_calls"] += 1
    _PAGE_BATCH_COUNTERS["page_value_materialized_bytes"] += _tensor_bytes(values)
    record_temp_allocation(f"page_batch_v{bits}_page_values", values)
    gathered = pattern_gather_centroids(assignment_idx.to(torch.long), centroids).to(values.dtype)
    return values + pattern_mask.unsqueeze(-1).to(values.dtype) * gathered


def patternkv_page_batch_decode(attn: torch.Tensor, cache: PatternKVPageBatchCache) -> torch.Tensor:
    """Page-centric batched mixed-V decode API.

    The MVP keeps K untouched and consumes request-local compact V2/V4 pages.
    It never calls the legacy B=1 mixed-V operator and never reconstructs the
    full historical Value tensor; only the current physical Value page is
    expanded for the page-local accumulation step.
    """

    with profile_range("page_batch_decode"):
        if attn.dim() != 4 or attn.shape[2] != 1:
            raise ValueError(f"attn must be [B,Hq,1,T], got {tuple(attn.shape)}")
        bsz, nh, _q, tokens = attn.shape
        if bsz not in (1, 2, 4):
            raise ValueError("S6-B.2 MVP only supports B=1, B=2, and B=4")
        if nh != cache.nh:
            raise ValueError(f"attention heads mismatch: {nh} != {cache.nh}")
        if int(cache.metadata.seq_lens.max().item()) != tokens or int(cache.metadata.seq_lens.min().item()) != tokens:
            raise ValueError("S6-B.2 MVP requires equal sequence lengths matching attention width")

        _PAGE_BATCH_COUNTERS["page_batch_decode_calls"] += 1
        record_counter("page_batch_decode_calls", calls=1)
        out = torch.zeros((bsz, nh, 1, cache.head_dim), dtype=torch.float32, device=attn.device)
        n_rep = cache.nh // cache.nh_kv
        num_pages = int(cache.metadata.num_pages[0].item())
        metadata_pages = cache.metadata.metadata_page_table.reshape(-1)
        v2_pages = cache.metadata.v2_page_table.reshape(-1)
        v4_pages = cache.metadata.v4_page_table.reshape(-1)
        total_pages = int(metadata_pages.numel())
        for flat_page in range(total_pages):
            b = flat_page // num_pages
            page = flat_page - b * num_pages
            metadata_page = int(metadata_pages[flat_page].item())
            valid = int(cache.metadata.valid_tokens[metadata_page].item())
            if valid <= 0:
                continue
            start = page * cache.page_size
            stop = start + valid
            v2_page_id = int(v2_pages[flat_page].item())
            v4_page_id = int(v4_pages[flat_page].item())
            v2_count = int(cache.metadata.v2_counts[metadata_page].item())
            v4_count = int(cache.metadata.v4_counts[metadata_page].item())
            if v2_count + v4_count != valid:
                raise RuntimeError("invalid page counts")

            _PAGE_BATCH_COUNTERS["logical_pages_processed"] += 1
            _PAGE_BATCH_COUNTERS["v2_tokens_processed"] += v2_count
            _PAGE_BATCH_COUNTERS["v4_tokens_processed"] += v4_count
            prefix = cache.metadata.v4_prefix_counts[metadata_page]
            page_precision = (prefix[1 : valid + 1] > prefix[:valid]).bool()
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
                attn2 = page_attn[:, :, :, ~page_precision].contiguous()
                out[b : b + 1] += torch.matmul(attn2, _repeat_kv(v2_values, n_rep)).float()
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
                attn4 = page_attn[:, :, :, page_precision].contiguous()
                out[b : b + 1] += torch.matmul(attn4, _repeat_kv(v4_values, n_rep)).float()
        return out.to(attn.dtype)


def patternkv_page_batched_v_decode(attn: torch.Tensor, cache: PatternKVPageBatchCache) -> torch.Tensor:
    return patternkv_page_batch_decode(attn, cache)


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
