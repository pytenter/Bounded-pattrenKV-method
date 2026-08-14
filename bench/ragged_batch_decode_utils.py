from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class RaggedBatchMetadata:
    request_ids: tuple[str, ...]
    seq_lens: tuple[int, ...]
    position_ids: tuple[int, ...]
    total_tokens: tuple[int, ...]
    packed_k_tokens: tuple[int, ...]
    packed_v_tokens: tuple[int, ...]
    packed_v4_tokens: tuple[int, ...]
    centroid_state_indices: tuple[int, ...]
    page_indptr: tuple[int, ...]
    page_counts: tuple[int, ...]
    last_page_valid_tokens: tuple[int, ...]

    @property
    def request_count(self) -> int:
        return len(self.request_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "request_ids": list(self.request_ids),
            "seq_lens": list(self.seq_lens),
            "position_ids": list(self.position_ids),
            "total_tokens": list(self.total_tokens),
            "packed_k_tokens": list(self.packed_k_tokens),
            "packed_v_tokens": list(self.packed_v_tokens),
            "packed_v4_tokens": list(self.packed_v4_tokens),
            "centroid_state_indices": list(self.centroid_state_indices),
            "page_indptr": list(self.page_indptr),
            "page_counts": list(self.page_counts),
            "last_page_valid_tokens": list(self.last_page_valid_tokens),
        }


def build_ragged_metadata(
    *,
    request_ids: Iterable[str],
    total_tokens: Iterable[int],
    packed_k_tokens: Iterable[int],
    packed_v_tokens: Iterable[int],
    packed_v4_tokens: Iterable[int],
    centroid_state_indices: Iterable[int],
    page_counts: Iterable[int],
    last_page_valid_tokens: Iterable[int],
) -> RaggedBatchMetadata:
    request_ids_t = tuple(str(x) for x in request_ids)
    total_t = tuple(int(x) for x in total_tokens)
    page_counts_t = tuple(int(x) for x in page_counts)
    indptr = [0]
    for count in page_counts_t:
        indptr.append(indptr[-1] + int(count))
    metadata = RaggedBatchMetadata(
        request_ids=request_ids_t,
        seq_lens=total_t,
        position_ids=total_t,
        total_tokens=total_t,
        packed_k_tokens=tuple(int(x) for x in packed_k_tokens),
        packed_v_tokens=tuple(int(x) for x in packed_v_tokens),
        packed_v4_tokens=tuple(int(x) for x in packed_v4_tokens),
        centroid_state_indices=tuple(int(x) for x in centroid_state_indices),
        page_indptr=tuple(indptr),
        page_counts=page_counts_t,
        last_page_valid_tokens=tuple(int(x) for x in last_page_valid_tokens),
    )
    validate_ragged_metadata(metadata)
    return metadata


def validate_ragged_metadata(metadata: RaggedBatchMetadata) -> None:
    n = metadata.request_count
    fields = (
        metadata.seq_lens,
        metadata.position_ids,
        metadata.total_tokens,
        metadata.packed_k_tokens,
        metadata.packed_v_tokens,
        metadata.packed_v4_tokens,
        metadata.centroid_state_indices,
        metadata.page_counts,
        metadata.last_page_valid_tokens,
    )
    if any(len(value) != n for value in fields):
        raise ValueError("all ragged metadata vectors must match request_count")
    if len(metadata.page_indptr) != n + 1:
        raise ValueError("page_indptr must have request_count + 1 entries")
    if metadata.page_indptr[0] != 0:
        raise ValueError("page_indptr must start at zero")
    for left, right in zip(metadata.page_indptr, metadata.page_indptr[1:]):
        if int(right) < int(left):
            raise ValueError("page_indptr must be monotonic")
    for idx, count in enumerate(metadata.page_counts):
        if metadata.page_indptr[idx + 1] - metadata.page_indptr[idx] != int(count):
            raise ValueError("page_indptr deltas must match page_counts")
    if any(int(x) < 0 for vector in fields for x in vector):
        raise ValueError("ragged metadata counts must be non-negative")
    if len(set(metadata.centroid_state_indices)) != n:
        raise ValueError("ragged centroid slots must be unique")
    for packed_k, packed_v in zip(metadata.packed_k_tokens, metadata.packed_v_tokens):
        if int(packed_k) != int(packed_v):
            raise ValueError("packed K/V logical token counts must match")


def ragged_position_ids_from_lengths(total_tokens: Iterable[int], *, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor([[int(x)] for x in total_tokens], dtype=torch.long, device=device)


def increment_ragged_total_tokens(total_tokens: Iterable[int], increments: Iterable[int] | None = None) -> tuple[int, ...]:
    base = tuple(int(x) for x in total_tokens)
    inc = tuple(1 for _ in base) if increments is None else tuple(int(x) for x in increments)
    if len(base) != len(inc):
        raise ValueError("increments must match total_tokens length")
    return tuple(x + y for x, y in zip(base, inc))


def page_count_for_tokens(tokens: int, *, page_size: int = 128) -> int:
    if tokens < 0:
        raise ValueError("tokens must be non-negative")
    return (int(tokens) + int(page_size) - 1) // int(page_size)


def last_page_valid_for_tokens(tokens: int, *, page_size: int = 128) -> int:
    if tokens <= 0:
        return 0
    rem = int(tokens) % int(page_size)
    return int(page_size) if rem == 0 else rem


def validate_assignment_index_range(indices: torch.Tensor | None, centroid_count: int | None) -> bool:
    if indices is None:
        return True
    if centroid_count is None or int(centroid_count) <= 0:
        return False
    if indices.numel() == 0:
        return True
    return bool(((indices >= 0) & (indices < int(centroid_count))).all().item())


def validate_v4_budget(packed_v4_tokens: int, packed_v_tokens: int, budget_fraction: float, *, slack: int = 1) -> bool:
    if packed_v4_tokens < 0 or packed_v_tokens < 0:
        return False
    budget = int(math.ceil(float(packed_v_tokens) * float(budget_fraction))) + int(slack)
    return int(packed_v4_tokens) <= budget


def ragged_k_masked_scores(query: torch.Tensor, key: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-1, -2))
    positions = torch.arange(key.shape[-2], device=key.device).view(1, 1, 1, -1)
    mask = positions >= valid_lengths.to(device=key.device).view(-1, 1, 1, 1)
    return scores.masked_fill(mask, torch.finfo(scores.dtype).min)


def page_range_for_request(metadata: RaggedBatchMetadata, request_index: int) -> tuple[int, int]:
    return int(metadata.page_indptr[request_index]), int(metadata.page_indptr[request_index + 1])


def ragged_metadata_from_layer_summaries(summaries: list[dict[str, Any]]) -> RaggedBatchMetadata:
    return build_ragged_metadata(
        request_ids=[item["request_id"] for item in summaries],
        total_tokens=[item["total_tokens"] for item in summaries],
        packed_k_tokens=[item["packed_k_tokens"] for item in summaries],
        packed_v_tokens=[item["packed_v_tokens"] for item in summaries],
        packed_v4_tokens=[item["packed_v4_tokens"] for item in summaries],
        centroid_state_indices=[item["centroid_state_slot"] for item in summaries],
        page_counts=[item["page_count"] for item in summaries],
        last_page_valid_tokens=[item["last_page_valid_tokens"] for item in summaries],
    )


def current_first_ragged_blocker() -> dict[str, Any]:
    return {
        "first_ragged_blocker": "RAGGED_CACHE_ASSEMBLY_UNSUPPORTED",
        "secondary_blockers": [
            "RAGGED_POSITION_IDS_UNSUPPORTED",
            "RAGGED_K_LENGTH_UNSUPPORTED",
            "RAGGED_V_PAGE_METADATA_UNSUPPORTED",
            "RAGGED_ATTENTION_MASK_UNSUPPORTED",
        ],
        "evidence": [
            "PatternQuantizedKVCache.total_tokens is serialized as a batch-global scalar.",
            "LlamaModel_PatternKV derives decode position_ids from the first layer cache length.",
            "Segmented PatternKV attention validates [B,H,Q,cache.total_tokens] with one global cache length.",
            "quant.page_batch patternkv_page_batch_decode currently requires equal seq_lens.",
        ],
    }
