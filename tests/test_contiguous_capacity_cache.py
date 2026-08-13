import pytest
import torch

from models.segmented_cache import (
    CAPACITY_GROWTH_BASELINE,
    ContiguousCapacityBuffer,
    normalize_capacity_growth_backend,
)
from quant.matmul import cuda_attn_v_fused_with_base, patternkv_gqa_v_backend, patternkv_page_v_reader_backend


def _buffer(tokens: int, *, capacity: int | None = None, chunk: int | None = None, token_dim: int = 2) -> ContiguousCapacityBuffer:
    return ContiguousCapacityBuffer(
        stream_name="test",
        shape_except_token=(1, 2, 4),
        token_dim=token_dim,
        dtype=torch.float32,
        device="cpu",
        capacity=capacity or 0,
        chunk_tokens=chunk,
    )


def _values(tokens: int) -> torch.Tensor:
    return torch.arange(1 * 2 * tokens * 4, dtype=torch.float32).view(1, 2, tokens, 4)


def test_capacity_buffer_single_append():
    buf = _buffer(1, capacity=4)
    x = _values(1)
    buf.append(x)
    torch.testing.assert_close(buf.logical_view(), x)
    assert buf.logical_length() == 1


def test_capacity_buffer_block_append():
    buf = _buffer(3, capacity=8)
    a = _values(2)
    b = _values(3) + 100
    buf.append_block(a)
    buf.append_block(b)
    torch.testing.assert_close(buf.logical_view(), torch.cat([a, b], dim=2))


def test_capacity_buffer_exact_fill():
    buf = _buffer(4, capacity=4)
    x = _values(4)
    buf.append_block(x)
    torch.testing.assert_close(buf.logical_view(), x)
    assert buf.remaining_capacity() == 0
    assert buf.logical_view().is_contiguous()


def test_capacity_buffer_overflow_safe():
    buf = _buffer(2, capacity=2)
    buf.append_block(_values(2))
    with pytest.raises(RuntimeError):
        buf.append_block(_values(1))


def test_chunked_capacity_growth():
    buf = _buffer(0, chunk=4)
    buf.append_block(_values(5))
    assert buf.capacity() == 8
    assert buf.logical_length() == 5


def test_chunked_capacity_preserves_old_values():
    buf = _buffer(0, chunk=4)
    first = _values(3)
    second = _values(3) + 50
    buf.append_block(first)
    buf.append_block(second)
    torch.testing.assert_close(buf.logical_view(), torch.cat([first, second], dim=2))


def test_capacity_logical_view_matches_baseline():
    buf = _buffer(0, chunk=8)
    parts = [_values(3), _values(5) + 10, _values(1) + 99]
    for part in parts:
        buf.append_block(part)
    torch.testing.assert_close(buf.logical_view(), torch.cat(parts, dim=2))


def test_capacity_logical_view_no_materialization():
    buf = _buffer(0, chunk=8)
    buf.append_block(_values(3))
    view = buf.logical_view()
    assert view.untyped_storage().data_ptr() == buf.storage.untyped_storage().data_ptr()


def test_capacity_view_contiguous_for_current_layout():
    exact = _buffer(0, capacity=4)
    exact.append_block(_values(4))
    assert exact.logical_view().is_contiguous()
    slack = _buffer(0, capacity=8)
    slack.append_block(_values(4))
    assert not slack.logical_view().is_contiguous()


def test_capacity_unused_slots_not_read():
    buf = _buffer(0, capacity=8)
    x = _values(3)
    buf.append_block(x)
    buf.storage.narrow(2, 3, 5).fill_(float("nan"))
    torch.testing.assert_close(buf.logical_view(), x)
    assert torch.isnan(buf.storage.narrow(2, 3, 5)).all()


def test_capacity_v2_compact_order():
    buf = ContiguousCapacityBuffer(stream_name="packed_v", shape_except_token=(1, 2, 8), token_dim=2, dtype=torch.int32, device="cpu", capacity=8)
    first = torch.arange(1 * 2 * 3 * 8, dtype=torch.int32).view(1, 2, 3, 8)
    second = torch.arange(1 * 2 * 2 * 8, dtype=torch.int32).view(1, 2, 2, 8) + 1000
    buf.append_block(first)
    buf.append_block(second)
    assert torch.equal(buf.logical_view(), torch.cat([first, second], dim=2))


def test_capacity_v4_compact_order():
    buf = ContiguousCapacityBuffer(stream_name="packed_v4", shape_except_token=(1, 2, 16), token_dim=2, dtype=torch.int32, device="cpu", capacity=8)
    x = torch.arange(1 * 2 * 5 * 16, dtype=torch.int32).view(1, 2, 5, 16)
    buf.append_block(x)
    assert torch.equal(buf.logical_view(), x)


def test_capacity_v4_identity():
    buf = ContiguousCapacityBuffer(stream_name="precision", shape_except_token=(1,), token_dim=1, dtype=torch.uint8, device="cpu", capacity=257)
    mask = torch.zeros(1, 257, dtype=torch.uint8)
    mask[:, [1, 4, 128, 129, 256]] = 1
    buf.append_block(mask)
    assert torch.equal(buf.logical_view().bool().nonzero()[:, 1], torch.tensor([1, 4, 128, 129, 256]))


def test_capacity_scale_zero():
    buf = ContiguousCapacityBuffer(stream_name="scale", shape_except_token=(1, 2, 1), token_dim=2, dtype=torch.float16, device="cpu", capacity=8)
    scale = torch.arange(10, dtype=torch.float16).view(1, 2, 5, 1)
    buf.append_block(scale)
    torch.testing.assert_close(buf.logical_view(), scale)


def test_capacity_pattern_metadata():
    idx = ContiguousCapacityBuffer(stream_name="idx", shape_except_token=(1, 2), token_dim=2, dtype=torch.long, device="cpu", capacity=8)
    mask = ContiguousCapacityBuffer(stream_name="mask", shape_except_token=(1, 2), token_dim=2, dtype=torch.uint8, device="cpu", capacity=8)
    idx_value = (torch.arange(10).view(1, 2, 5) % 4).long()
    mask_value = (idx_value % 2).to(torch.uint8)
    idx.append_block(idx_value)
    mask.append_block(mask_value)
    assert torch.equal(idx.logical_view(), idx_value)
    assert torch.equal(mask.logical_view(), mask_value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for capacity attention smoke")
def test_capacity_attention_matches_baseline():
    from tests.test_mixed_v_fused_kernel import GROUP_SIZE, NH, NH_KV, _build_case

    data = _build_case("all_v2", 128)
    p2 = data["p2"]
    out = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0],
        p2[1],
        p2[2],
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    clone = cuda_attn_v_fused_with_base(
        GROUP_SIZE,
        data["attn"],
        p2[0].clone(),
        p2[1].clone(),
        p2[2].clone(),
        2,
        data["centroids"],
        data["v_pattern_mask"],
        data["v_idx"],
        NH,
        NH_KV,
    )
    torch.testing.assert_close(out, clone, rtol=5e-3, atol=5e-3)


def test_capacity_default_backend_safe(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_GROWTH_BACKEND", raising=False)
    assert normalize_capacity_growth_backend(None) == CAPACITY_GROWTH_BASELINE


def test_failed_paged_reader_remains_nondefault(monkeypatch):
    monkeypatch.delenv("PATTERNKV_PAGE_V_READER", raising=False)
    assert patternkv_page_v_reader_backend() == "contiguous"


def test_failed_gqa_backend_remains_nondefault(monkeypatch):
    monkeypatch.delenv("PATTERNKV_GQA_V_BACKEND", raising=False)
    assert patternkv_gqa_v_backend() == "baseline"
