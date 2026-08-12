import os

import torch

from models.segmented_cache import (
    DEFAULT_PAGE_SIZE,
    FixedPageBuffer,
    FixedPageCacheStorage,
    RecentRingBuffer,
    normalize_cache_backend,
)


def _tokens(tokens: int, *, heads: int = 2, dim: int = 4, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    values = torch.arange(tokens, dtype=dtype).view(1, 1, tokens, 1)
    return values.expand(1, heads, tokens, dim).contiguous()


def _append_in_chunks(buffer: FixedPageBuffer, total: int, chunks: list[int]) -> torch.Tensor:
    parts = []
    start = 0
    for chunk in chunks:
        block = _tokens(chunk) + float(start)
        buffer.append_block(block)
        parts.append(block)
        start += chunk
    return torch.cat(parts, dim=2).contiguous()


def test_fixed_page_append_single_token():
    buf = FixedPageBuffer(stream="recent_k")
    expected = _append_in_chunks(buf, 3, [1, 1, 1])
    torch.testing.assert_close(buf.materialize_contiguous(), expected)
    assert buf.page_count() == 1
    assert buf.old_bytes_copied == 0


def test_fixed_page_append_block():
    buf = FixedPageBuffer(stream="packed_v")
    block = _tokens(200)
    buf.append_block(block)
    torch.testing.assert_close(buf.materialize_contiguous(), block)
    assert buf.page_count() == 2
    assert buf.last_page_fill() == 72


def test_fixed_page_cross_boundary_127_128_129():
    for tokens in (127, 128, 129):
        buf = FixedPageBuffer(stream=f"boundary_{tokens}")
        expected = _append_in_chunks(buf, tokens, [tokens])
        torch.testing.assert_close(buf.materialize_contiguous(), expected)
        assert buf.logical_length() == tokens


def test_fixed_page_cross_boundary_255_256_257():
    for tokens in (255, 256, 257):
        buf = FixedPageBuffer(stream=f"boundary_{tokens}")
        expected = _append_in_chunks(buf, tokens, [64, tokens - 64])
        torch.testing.assert_close(buf.materialize_contiguous(), expected)
        assert buf.logical_length() == tokens


def test_page_logical_order():
    buf = FixedPageBuffer(stream="logical")
    expected = _append_in_chunks(buf, 257, [17, 111, 1, 128])
    out = buf.materialize_contiguous()
    torch.testing.assert_close(out, expected)
    assert torch.equal(out[0, 0, :, 0], torch.arange(257, dtype=out.dtype))


def test_page_descriptor_valid_length():
    buf = FixedPageBuffer(stream="descriptor")
    buf.append_block(_tokens(257))
    descriptors = buf.descriptors()
    assert [d.valid_tokens for d in descriptors] == [128, 128, 1]
    assert [d.logical_start_token for d in descriptors] == [0, 128, 256]


def test_recent_ring_order():
    ring = RecentRingBuffer(capacity=DEFAULT_PAGE_SIZE, stream="recent")
    ring.append_block(_tokens(64))
    ring.append_block(_tokens(64) + 64)
    out = ring.materialize_contiguous()
    assert torch.equal(out[0, 0, :, 0], torch.arange(128, dtype=out.dtype))


def test_recent_ring_flush():
    ring = RecentRingBuffer(capacity=DEFAULT_PAGE_SIZE, stream="recent")
    ring.append_block(_tokens(128))
    flushed = ring.append_block(_tokens(1) + 128)
    assert flushed is not None
    assert torch.equal(flushed[0, 0, :, 0], torch.tensor([0.0]))
    out = ring.materialize_contiguous()
    assert torch.equal(out[0, 0, :, 0], torch.arange(1, 129, dtype=out.dtype))


def test_sink_preserved():
    sink = _tokens(16)
    paged_sink = sink.contiguous()
    torch.testing.assert_close(paged_sink, sink)


def test_v2_compact_order_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    v2 = _tokens(96, dim=8)
    storage.append_stream("packed_v", v2)
    torch.testing.assert_close(storage.materialize("packed_v"), v2)


def test_v4_compact_order_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    v4 = _tokens(33, dim=16)
    storage.append_stream("packed_v4", v4)
    torch.testing.assert_close(storage.materialize("packed_v4"), v4)


def test_v4_token_identity_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    precision = torch.zeros(1, 257, dtype=torch.uint8)
    precision[:, [1, 4, 128, 129, 256]] = 1
    storage.append_stream("v_precision_mask", precision)
    out = storage.materialize("v_precision_mask")
    assert torch.equal(out.bool().nonzero()[:, 1], torch.tensor([1, 4, 128, 129, 256]))


def test_scale_zero_metadata_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    scale = torch.arange(257, dtype=torch.float16).view(1, 1, 257, 1)
    zero = -scale
    storage.append_stream("packed_v_scale", scale)
    storage.append_stream("packed_v_zero", zero)
    torch.testing.assert_close(storage.materialize("packed_v_scale"), scale)
    torch.testing.assert_close(storage.materialize("packed_v_zero"), zero)


def test_pattern_mask_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    mask = (torch.arange(257).view(1, 1, 257) % 3 == 0).to(torch.uint8)
    storage.append_stream("v_pattern_mask", mask)
    assert torch.equal(storage.materialize("v_pattern_mask"), mask)


def test_pattern_assignment_preserved():
    storage = FixedPageCacheStorage(page_size=128)
    idx = (torch.arange(257).view(1, 1, 257) % 16).to(torch.long)
    storage.append_stream("v_assignment_idx", idx)
    assert torch.equal(storage.materialize("v_assignment_idx"), idx)


def test_contiguous_materialization_matches_baseline():
    storage = FixedPageCacheStorage(page_size=128)
    baseline = torch.cat([_tokens(127), _tokens(130) + 127], dim=2).contiguous()
    storage.append_stream("packed_v", baseline[:, :, :127, :])
    storage.append_stream("packed_v", baseline[:, :, 127:, :])
    torch.testing.assert_close(storage.materialize("packed_v"), baseline)


def test_default_cache_backend_safe(monkeypatch):
    monkeypatch.delenv("PATTERNKV_CACHE_BACKEND", raising=False)
    assert normalize_cache_backend(None) == "contiguous"
    monkeypatch.setenv("PATTERNKV_CACHE_BACKEND", "paged")
    assert normalize_cache_backend(None) == "paged"
