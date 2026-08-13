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
    v1_calls: int = 0
    v2_calls: int = 0
    rows: int = 0
    kernel_launches: int = 0
    serial_request_dispatches: int = 0
    fallback_calls: int = 0
    weight_copy_bytes: int = 0
    input_copy_bytes: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "bi_kproj_calls": self.calls,
            "bi_kproj_v1_calls": self.v1_calls,
            "bi_kproj_v2_calls": self.v2_calls,
            "bi_kproj_rows": self.rows,
            "bi_kproj_kernel_launches": self.kernel_launches,
            "bi_kproj_serial_request_dispatches": self.serial_request_dispatches,
            "bi_kproj_fallback_calls": self.fallback_calls,
            "bi_kproj_weight_copy_bytes": self.weight_copy_bytes,
            "bi_kproj_input_copy_bytes": self.input_copy_bytes,
        }


COUNTERS = BatchInvariantKProjCounters()


def reset_batch_invariant_kproj_counters() -> None:
    COUNTERS.calls = 0
    COUNTERS.v1_calls = 0
    COUNTERS.v2_calls = 0
    COUNTERS.rows = 0
    COUNTERS.kernel_launches = 0
    COUNTERS.serial_request_dispatches = 0
    COUNTERS.fallback_calls = 0
    COUNTERS.weight_copy_bytes = 0
    COUNTERS.input_copy_bytes = 0


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

    @triton.jit
    def _bi_linear_persistent_kernel(
        x_ptr,
        w_ptr,
        bias_ptr,
        y_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        NUM_TILES: tl.constexpr,
        NUM_SMS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        start_pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        tiles_per_group = GROUP_M * num_pid_n
        tile_id = start_pid
        while tile_id < NUM_TILES:
            group_id = tile_id // tiles_per_group
            first_pid_m = group_id * GROUP_M
            group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
            tile_in_group = tile_id % tiles_per_group
            pid_m = first_pid_m + (tile_in_group % group_size_m)
            pid_n = tile_in_group // group_size_m

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, K, BLOCK_K):
                k_offsets = k_start + offs_k
                a = tl.load(
                    x_ptr + offs_m[:, None] * K + k_offsets[None, :],
                    mask=(offs_m[:, None] < M) & (k_offsets[None, :] < K),
                    other=0.0,
                )
                b = tl.load(
                    w_ptr + offs_n[None, :] * K + k_offsets[:, None],
                    mask=(offs_n[None, :] < N) & (k_offsets[:, None] < K),
                    other=0.0,
                )
                acc = tl.dot(a, b, acc, input_precision="ieee")
            if HAS_BIAS:
                acc += tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)[None, :].to(tl.float32)
            tl.store(
                y_ptr + offs_m[:, None] * N + offs_n[None, :],
                acc,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
            )
            tile_id += NUM_SMS


def _choose_block_n(n: int) -> int:
    if n <= 16:
        return 16
    if n <= 32:
        return 32
    return 64


def _validate_inputs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> tuple[torch.Size, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    x_2d = x.reshape(-1, k)
    if not x_2d.is_contiguous():
        COUNTERS.input_copy_bytes += x_2d.numel() * x_2d.element_size()
        x_2d = x_2d.contiguous()
    weight_2d = weight
    if not weight_2d.is_contiguous():
        COUNTERS.weight_copy_bytes += weight_2d.numel() * weight_2d.element_size()
        weight_2d = weight_2d.contiguous()
    bias_1d = bias.contiguous() if bias is not None else torch.empty((1,), device=x.device, dtype=x.dtype)
    return original_shape, k, n, x_2d, weight_2d, bias_1d


def _restore_shape(out: torch.Tensor, original_shape: torch.Size, n: int) -> torch.Tensor:
    if len(original_shape) == 3:
        return out.reshape(original_shape[0], original_shape[1], n)
    return out


def batch_invariant_k_projection_v1(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_k: int = 64,
) -> torch.Tensor:
    original_shape, k, n, x_2d, weight_2d, bias_1d = _validate_inputs(x, weight, bias)
    m = int(x_2d.shape[0])
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    block_n = _choose_block_n(n)
    COUNTERS.calls += 1
    COUNTERS.v1_calls += 1
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
    return _restore_shape(out, original_shape, n)


def batch_invariant_k_projection_v2(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    group_m: int = 8,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    original_shape, k, n, x_2d, weight_2d, bias_1d = _validate_inputs(x, weight, bias)
    m = int(x_2d.shape[0])
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid_sms = min(num_sms, num_tiles)
    COUNTERS.calls += 1
    COUNTERS.v2_calls += 1
    COUNTERS.rows += m
    COUNTERS.kernel_launches += 1
    with torch.cuda.device(x.device):
        _bi_linear_persistent_kernel[(grid_sms,)](
            x_2d,
            weight_2d,
            bias_1d,
            out,
            M=m,
            K=k,
            N=n,
            HAS_BIAS=bias is not None,
            NUM_TILES=num_tiles,
            NUM_SMS=grid_sms,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return _restore_shape(out, original_shape, n)


def batch_invariant_k_projection(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    backend: str = "v2",
    block_k: int = 64,
    block_m: int = 128,
    block_n: int = 128,
    group_m: int = 8,
    num_warps: int = 8,
    num_stages: int = 3,
) -> torch.Tensor:
    normalized = backend.strip().lower()
    if normalized in {"v1", "v1_rowwise", "rowwise"}:
        return batch_invariant_k_projection_v1(x, weight, bias, block_k=block_k)
    if normalized in {"v2", "v2_persistent", "persistent"}:
        return batch_invariant_k_projection_v2(
            x,
            weight,
            bias,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            group_m=group_m,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    raise ValueError(f"Unsupported batch-invariant K projection backend: {backend!r}")


def flag_enabled(env: dict[str, str] | None = None) -> bool:
    source: Any = env if env is not None else __import__("os").environ
    return str(source.get("PATTERNKV_BATCH_INVARIANT_KPROJ", "0")).strip() == "1"
