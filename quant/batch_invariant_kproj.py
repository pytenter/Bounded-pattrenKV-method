from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised only on Triton-less systems
    triton = None
    tl = None


@dataclass
class BatchInvariantKProjCounters:
    calls: int = 0
    rows: int = 0
    kernel_launches: int = 0
    serial_request_dispatches: int = 0
    fallback_calls: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "bi_kproj_calls": self.calls,
            "bi_kproj_rows": self.rows,
            "bi_kproj_kernel_launches": self.kernel_launches,
            "bi_kproj_serial_request_dispatches": self.serial_request_dispatches,
            "bi_kproj_fallback_calls": self.fallback_calls,
        }


COUNTERS = BatchInvariantKProjCounters()


def reset_batch_invariant_kproj_counters() -> None:
    COUNTERS.calls = 0
    COUNTERS.rows = 0
    COUNTERS.kernel_launches = 0
    COUNTERS.serial_request_dispatches = 0
    COUNTERS.fallback_calls = 0


def batch_invariant_kproj_counters() -> dict[str, int]:
    return COUNTERS.as_dict()


def batch_invariant_kproj_available() -> bool:
    return triton is not None and tl is not None and torch.cuda.is_available()


if triton is not None and tl is not None:

    @triton.jit
    def _bi_linear_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        n_block = tl.program_id(1)
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)
            k_mask = k_offsets < K
            x = tl.load(x_ptr + row * K + k_offsets, mask=k_mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + n_offsets[:, None] * K + k_offsets[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)
        if HAS_BIAS:
            acc += tl.load(bias_ptr + n_offsets, mask=n_mask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * N + n_offsets, acc, mask=n_mask)


def _choose_block_n(n: int) -> int:
    if n <= 16:
        return 16
    if n <= 32:
        return 32
    return 64


def batch_invariant_k_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_k: int = 64,
) -> torch.Tensor:
    if not batch_invariant_kproj_available():
        raise RuntimeError("Triton/CUDA batch-invariant K projection is unavailable")
    if x.dim() not in (2, 3):
        raise ValueError(f"x must be [M,K] or [B,T,K], got shape {tuple(x.shape)}")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [N,K], got shape {tuple(weight.shape)}")
    if bias is not None and (bias.dim() != 1 or bias.shape[0] != weight.shape[0]):
        raise ValueError("bias must be None or [N]")
    original_shape = x.shape
    k = int(weight.shape[1])
    n = int(weight.shape[0])
    if int(x.shape[-1]) != k:
        raise ValueError(f"x last dim {x.shape[-1]} does not match weight K {k}")
    x_2d = x.reshape(-1, k).contiguous()
    weight_2d = weight.contiguous()
    bias_1d = bias.contiguous() if bias is not None else torch.empty((1,), device=x.device, dtype=x.dtype)
    m = int(x_2d.shape[0])
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_n = _choose_block_n(n)
    COUNTERS.calls += 1
    COUNTERS.rows += m
    COUNTERS.kernel_launches += 1
    grid = (m, triton.cdiv(n, block_n))
    with torch.cuda.device(x.device):
        _bi_linear_kernel[grid](
            x_2d,
            weight_2d,
            bias_1d,
            out,
            M=m,
            K=k,
            N=n,
            HAS_BIAS=bias is not None,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
        )
    if len(original_shape) == 3:
        return out.reshape(original_shape[0], original_shape[1], n)
    return out


def flag_enabled(env: dict[str, str] | None = None) -> bool:
    source: Any = env if env is not None else __import__("os").environ
    return str(source.get("PATTERNKV_BATCH_INVARIANT_KPROJ", "0")).strip() == "1"
