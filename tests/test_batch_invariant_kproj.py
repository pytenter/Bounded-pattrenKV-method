from __future__ import annotations

import pytest
import torch

from quant.batch_invariant_kproj import (
    batch_invariant_k_projection,
    batch_invariant_kproj_available,
    batch_invariant_kproj_counters,
    flag_enabled,
    reset_batch_invariant_kproj_counters,
)


pytestmark = pytest.mark.skipif(not batch_invariant_kproj_available(), reason="Triton/CUDA batch-invariant K projection unavailable")


def _data(batch: int = 4, tokens: int = 17, hidden: int = 65, out: int = 19) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cuda")
    g.manual_seed(20260209)
    x = torch.randn((batch, tokens, hidden), device="cuda", dtype=torch.float16, generator=g)
    w = torch.randn((out, hidden), device="cuda", dtype=torch.float16, generator=g)
    return x, w


def test_batch_invariant_kproj_b1_b2() -> None:
    x, w = _data(batch=2)

    y1 = batch_invariant_k_projection(x[0:1], w)
    y2 = batch_invariant_k_projection(x[0:2], w)[0:1]

    assert torch.equal(y1, y2)


def test_batch_invariant_kproj_b1_b4() -> None:
    x, w = _data(batch=4)

    y1 = batch_invariant_k_projection(x[0:1], w)
    y4 = batch_invariant_k_projection(x, w)[0:1]

    assert torch.equal(y1, y4)


def test_batch_invariant_kproj_request_reorder() -> None:
    x, w = _data(batch=4)
    a = x[0:1]

    ref = batch_invariant_k_projection(a, w)
    row0 = batch_invariant_k_projection(torch.cat([x[0:1], x[1:2]], dim=0), w)[0:1]
    row1 = batch_invariant_k_projection(torch.cat([x[1:2], x[0:1]], dim=0), w)[1:2]
    row2 = batch_invariant_k_projection(torch.cat([x[2:3], x[3:4], x[0:1], x[1:2]], dim=0), w)[2:3]
    row3 = batch_invariant_k_projection(torch.cat([x[1:2], x[2:3], x[3:4], x[0:1]], dim=0), w)[3:4]

    assert torch.equal(ref, row0)
    assert torch.equal(ref, row1)
    assert torch.equal(ref, row2)
    assert torch.equal(ref, row3)


def test_batch_invariant_kproj_batch_composition() -> None:
    x, w = _data(batch=4)
    a = x[0:1]

    ref = batch_invariant_k_projection(a, w)
    for other in (x[1:2], x[2:3], x[3:4], x[1:4]):
        got = batch_invariant_k_projection(torch.cat([a, other], dim=0), w)[0:1]
        assert torch.equal(ref, got)


def test_batch_invariant_kproj_non_aligned_shapes() -> None:
    for tokens in (1, 17, 127, 128, 129, 511, 512, 513):
        for batch in (1, 2, 3, 4):
            x, w = _data(batch=batch, tokens=tokens, hidden=73, out=37)
            ref = batch_invariant_k_projection(x[0:1], w)
            got = batch_invariant_k_projection(x, w)[0:1]
            assert torch.equal(ref, got)


def test_batch_invariant_kproj_bias_none() -> None:
    x, w = _data(batch=2)

    y = batch_invariant_k_projection(x, w, bias=None)

    assert y.shape == (2, 17, 19)


def test_batch_invariant_kproj_matches_reference() -> None:
    x, w = _data(batch=2, tokens=9, hidden=64, out=21)

    y = batch_invariant_k_projection(x, w)
    ref = torch.matmul(x.float(), w.float().t()).to(torch.float16)
    rel = torch.linalg.vector_norm((y - ref).float()) / torch.linalg.vector_norm(ref.float()).clamp_min(1e-12)

    assert float(rel.item()) < 5e-4


def test_batch_invariant_kproj_flag_disabled() -> None:
    assert flag_enabled({}) is False
    assert flag_enabled({"PATTERNKV_BATCH_INVARIANT_KPROJ": "0"}) is False
    assert flag_enabled({"PATTERNKV_BATCH_INVARIANT_KPROJ": "1"}) is True


def test_batch_invariant_kproj_no_serial_dispatch() -> None:
    x, w = _data(batch=4)
    reset_batch_invariant_kproj_counters()

    batch_invariant_k_projection(x, w)
    counters = batch_invariant_kproj_counters()

    assert counters["bi_kproj_calls"] == 1
    assert counters["bi_kproj_rows"] == 4 * 17
    assert counters["bi_kproj_kernel_launches"] == 1
    assert counters["bi_kproj_serial_request_dispatches"] == 0
    assert counters["bi_kproj_fallback_calls"] == 0
