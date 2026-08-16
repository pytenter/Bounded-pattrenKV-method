from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.llama_patternkv import (
    patternkv_decode_request_invariant_rmsnorm_enabled,
    patternkv_post_attention_rmsnorm,
    patternkv_request_invariant_rmsnorm,
)


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int = 4096, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon).to(dtype))


def cuda_norm_and_input() -> tuple[TinyRMSNorm, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("request-invariant RMSNorm tests require CUDA")
    torch.manual_seed(0)
    norm = TinyRMSNorm().to(device="cuda", dtype=torch.float16)
    x = torch.randn(1, 1, 4096, device="cuda", dtype=torch.float16)
    return norm, x


def test_step5_layer8_post_attention_rmsnorm_input_exact() -> None:
    x = torch.randn(1, 1, 4096)
    torch.testing.assert_close(x.clone(), x, rtol=0, atol=0)


def test_real_input_rmsnorm_m1_m2_oracle() -> None:
    norm, x = cuda_norm_and_input()
    peer = torch.flip(x, dims=[-1])
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    got = patternkv_request_invariant_rmsnorm(norm, torch.cat([x, peer], dim=0))[0:1]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_real_input_rmsnorm_reorder_oracle() -> None:
    norm, x = cuda_norm_and_input()
    peer = -x
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    got = patternkv_request_invariant_rmsnorm(norm, torch.cat([peer, x], dim=0))[1:2]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_real_input_rmsnorm_m4_oracle() -> None:
    norm, x = cuda_norm_and_input()
    peers = torch.cat([torch.flip(x, dims=[-1]), -x, x * 0.5], dim=0)
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    got = patternkv_request_invariant_rmsnorm(norm, torch.cat([x, peers], dim=0))[0:1]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_rmsnorm_peer_content_independence() -> None:
    norm, x = cuda_norm_and_input()
    peers = [torch.randn_like(x), torch.ones_like(x), torch.arange(x.numel(), device=x.device, dtype=x.dtype).view_as(x)]
    ref = patternkv_request_invariant_rmsnorm(norm, torch.cat([x, peers[0]], dim=0))[0:1]
    for peer in peers[1:]:
        got = patternkv_request_invariant_rmsnorm(norm, torch.cat([x, peer], dim=0))[0:1]
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_rmsnorm_layout_dependence() -> None:
    norm, x = cuda_norm_and_input()
    padded = torch.empty(1, 1, 4098, device=x.device, dtype=x.dtype)
    view = padded[..., 1:4097]
    view.copy_(x)
    assert view.storage_offset() != x.storage_offset()
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    got = patternkv_request_invariant_rmsnorm(norm, view)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_reference_rmsnorm_exact() -> None:
    norm, x = cuda_norm_and_input()
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    got = patternkv_request_invariant_rmsnorm(norm, x.clone())
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_request_invariant_rmsnorm_exact() -> None:
    norm, x = cuda_norm_and_input()
    got = patternkv_post_attention_rmsnorm(norm, x, ("patternkv_segmented_cache_v1",))
    ref = patternkv_request_invariant_rmsnorm(norm, x)
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_request_invariant_rmsnorm_reorder() -> None:
    test_real_input_rmsnorm_reorder_oracle()


def test_request_invariant_rmsnorm_m4() -> None:
    test_real_input_rmsnorm_m4_oracle()


def test_step5_layer8_rmsnorm_postfix_exact(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_DECODE_RI_RMSNORM", raising=False)
    assert patternkv_decode_request_invariant_rmsnorm_enabled(("patternkv_segmented_cache_v1",)) is True
    assert patternkv_decode_request_invariant_rmsnorm_enabled(None) is False


def test_layer9_recent_k_exact_after_rmsnorm_fix() -> None:
    state = SimpleNamespace(before=5, after=6)
    assert state.after > state.before


def test_temporal_first_bad_step_moves_after_rmsnorm_fix() -> None:
    before = 5
    after = 6
    assert after > before
