from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from models.segmented_cache import quantize_pack_v_reference
from quant.matmul import cuda_attn_v_mixed_fused_with_base
from quant.page_batch import (
    PAGE_SIZE,
    cache_isolation_summary,
    correctness_metrics,
    pack_mixed_v_pages,
    patternkv_page_batched_v_decode,
    selector_isolation_summary,
    validate_page_mapping,
)


def reference_batch_mixed_v(
    attn: torch.Tensor,
    v_adjusted: torch.Tensor,
    precision_mask: torch.Tensor,
    v_pattern_mask: torch.Tensor,
    v_assignment_idx: torch.Tensor,
    centroids: torch.Tensor,
    *,
    group_size: int = 128,
    nh: int = 32,
    nh_kv: int = 8,
) -> torch.Tensor:
    """Golden reference: serial B=1 dispatch of the frozen mixed-V operator."""

    outs = []
    for b in range(attn.shape[0]):
        precision_b = precision_mask[b : b + 1].bool().contiguous()
        low = v_adjusted[b : b + 1, :, ~precision_b[0], :].contiguous()
        high = v_adjusted[b : b + 1, :, precision_b[0], :].contiguous()
        p2 = quantize_pack_v_reference(low, group_size, 2) if low.shape[2] else (None, None, None)
        p4 = quantize_pack_v_reference(high, group_size, 4) if high.shape[2] else (None, None, None)
        outs.append(
            cuda_attn_v_mixed_fused_with_base(
                group_size,
                attn[b : b + 1].contiguous(),
                p2[0],
                p2[1],
                p2[2],
                p4[0],
                p4[1],
                p4[2],
                precision_b,
                centroids,
                v_pattern_mask[b : b + 1].contiguous(),
                v_assignment_idx[b : b + 1].contiguous(),
                nh,
                nh_kv,
            )
        )
    return torch.cat(outs, dim=0).contiguous()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def time_cuda_callable(fn, *, warmup: int = 10, measured: int = 50) -> float | None:
    if not torch.cuda.is_available():
        return None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(measured):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1000.0 / measured)


def time_cpu_callable(fn, *, warmup: int = 2, measured: int = 5) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(measured):
        fn()
    return (time.perf_counter() - start) * 1_000_000.0 / measured
