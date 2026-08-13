from __future__ import annotations

import pytest
import torch

from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_k_projection_v1,
    batch_invariant_k_projection_v2,
    batch_invariant_kproj_available,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)


pytestmark = pytest.mark.skipif(not batch_invariant_kproj_available(), reason="Triton/CUDA batch-invariant K projection unavailable")


def _data(batch: int = 4, tokens: int = 17, hidden: int = 65, out: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda")
    g.manual_seed(20260210)
    x = torch.randn((batch, tokens, hidden), device="cuda", dtype=torch.float16, generator=g)
    w = torch.randn((out, hidden), device="cuda", dtype=torch.float16, generator=g)
    return x, w


def test_bi_kproj_v2_b1_b2_exact() -> None:
    x, w = _data(batch=2)
    assert torch.equal(batch_invariant_k_projection_v2(x[0:1], w), batch_invariant_k_projection_v2(x, w)[0:1])


def test_bi_kproj_v2_b1_b4_exact() -> None:
    x, w = _data(batch=4)
    assert torch.equal(batch_invariant_k_projection_v2(x[0:1], w), batch_invariant_k_projection_v2(x, w)[0:1])


def test_bi_kproj_v2_request_reorder() -> None:
    x, w = _data(batch=4)
    ref = batch_invariant_k_projection_v2(x[0:1], w)
    assert torch.equal(ref, batch_invariant_k_projection_v2(torch.cat([x[1:2], x[0:1]], dim=0), w)[1:2])
    assert torch.equal(ref, batch_invariant_k_projection_v2(torch.cat([x[2:3], x[3:4], x[0:1], x[1:2]], dim=0), w)[2:3])


def test_bi_kproj_v2_batch_composition() -> None:
    x, w = _data(batch=4)
    ref = batch_invariant_k_projection_v2(x[0:1], w)
    for other in (x[1:2], x[2:3], x[3:4], x[1:4]):
        assert torch.equal(ref, batch_invariant_k_projection_v2(torch.cat([x[0:1], other], dim=0), w)[0:1])


def test_bi_kproj_v2_non_aligned_m() -> None:
    x, w = _data(batch=3, tokens=65, hidden=128, out=256)
    assert torch.equal(batch_invariant_k_projection_v2(x[0:1], w), batch_invariant_k_projection_v2(x, w)[0:1])


def test_bi_kproj_v2_non_aligned_n() -> None:
    x, w = _data(batch=2, tokens=17, hidden=128, out=257)
    assert torch.equal(batch_invariant_k_projection_v2(x[0:1], w), batch_invariant_k_projection_v2(x, w)[0:1])


def test_bi_kproj_v2_non_aligned_k() -> None:
    x, w = _data(batch=2, tokens=17, hidden=73, out=128)
    assert torch.equal(batch_invariant_k_projection_v2(x[0:1], w), batch_invariant_k_projection_v2(x, w)[0:1])


def test_bi_kproj_v2_matches_fp32_reference() -> None:
    x, w = _data(batch=2, tokens=9, hidden=96, out=192)
    y = batch_invariant_k_projection_v2(x, w)
    ref = torch.matmul(x.float(), w.float().t()).to(torch.float16)
    rel = torch.linalg.vector_norm((y - ref).float()) / torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)
    assert float(rel.item()) < 5e-4


def test_bi_kproj_v2_no_weight_transpose_copy() -> None:
    x, w = _data(batch=2, tokens=9, hidden=64, out=128)
    reset_batch_invariant_kproj_counters()
    batch_invariant_k_projection_v2(x, w)
    counters = batch_invariant_kproj_counters()
    assert counters["bi_kproj_weight_copy_bytes"] == 0
    assert counters["bi_kproj_input_copy_bytes"] == 0


def test_bi_kproj_v2_no_serial_dispatch() -> None:
    x, w = _data(batch=4)
    reset_batch_invariant_kproj_counters()
    batch_invariant_k_projection_v2(x, w)
    counters = batch_invariant_kproj_counters()
    assert counters["bi_kproj_v2_calls"] == 1
    assert counters["bi_kproj_serial_request_dispatches"] == 0
    assert counters["bi_kproj_fallback_calls"] == 0
    assert counters["bi_kproj_kernel_launches"] == 1


def test_bi_kproj_backend_dispatch() -> None:
    x, w = _data(batch=2, tokens=9, hidden=64, out=128)
    reset_batch_invariant_kproj_counters()
    y = batch_invariant_k_projection(x, w, backend="v2_persistent")
    counters = batch_invariant_kproj_counters()
    assert y.shape == (2, 9, 128)
    assert counters["bi_kproj_v2_calls"] == 1


def test_bi_kproj_v1_preserved() -> None:
    x, w = _data(batch=2, tokens=9, hidden=64, out=128)
    reset_batch_invariant_kproj_counters()
    y = batch_invariant_k_projection_v1(x, w)
    counters = batch_invariant_kproj_counters()
    assert y.shape == (2, 9, 128)
    assert counters["bi_kproj_v1_calls"] == 1
