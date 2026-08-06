from __future__ import annotations

import torch

from bench.aime24_int2_wave1 import split_sink_quant_recent


def test_sink_recent_split_boundaries() -> None:
    assert split_sink_quant_recent(32, 64, 256) == {"sink_tokens": 32, "quantized_tokens": 0, "recent_tokens": 0, "total_tokens": 32}
    assert split_sink_quant_recent(128, 64, 256) == {"sink_tokens": 64, "quantized_tokens": 0, "recent_tokens": 64, "total_tokens": 128}
    assert split_sink_quant_recent(512, 64, 256) == {"sink_tokens": 64, "quantized_tokens": 192, "recent_tokens": 256, "total_tokens": 512}


def test_token_order_reference_layout() -> None:
    tokens = torch.arange(20)
    split = split_sink_quant_recent(len(tokens), 4, 6)
    sink = tokens[: split["sink_tokens"]]
    quantized = tokens[split["sink_tokens"] : split["sink_tokens"] + split["quantized_tokens"]]
    recent = tokens[-split["recent_tokens"] :]
    restored = torch.cat([sink, quantized, recent])
    assert torch.equal(restored, tokens)


def test_recent_window_rolls_without_quantizing_sink() -> None:
    sink_length = 3
    recent_length = 4
    for total in range(1, 18):
        split = split_sink_quant_recent(total, sink_length, recent_length)
        assert split["sink_tokens"] == min(total, sink_length)
        assert split["sink_tokens"] + split["quantized_tokens"] + split["recent_tokens"] == total
        if total > sink_length:
            assert split["recent_tokens"] == min(total - sink_length, recent_length)


def test_small_attention_reference_shape() -> None:
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 7, 8)
    v = torch.randn(1, 2, 7, 8)
    weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / (8**0.5), dim=-1)
    out = torch.matmul(weights, v)
    assert out.shape == (1, 2, 1, 8)
    assert torch.isfinite(out).all()
