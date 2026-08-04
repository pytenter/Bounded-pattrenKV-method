"""Attention-aware diagnostic helpers."""

from __future__ import annotations

import torch


def attention_logit_mse(query: torch.Tensor, key_ref: torch.Tensor, key_reconstructed: torch.Tensor) -> float:
    """Compute mean squared error between reference and reconstructed attention logits."""
    ref = torch.matmul(query.float(), key_ref.float().transpose(-2, -1))
    got = torch.matmul(query.float(), key_reconstructed.float().transpose(-2, -1))
    return float(((ref - got) ** 2).mean().item())


def attention_output_mse(attn_weights: torch.Tensor, value_ref: torch.Tensor, value_reconstructed: torch.Tensor) -> float:
    """Compute MSE between attention-weighted reference and reconstructed V outputs."""
    ref = torch.matmul(attn_weights.float(), value_ref.float())
    got = torch.matmul(attn_weights.float(), value_reconstructed.float())
    return float(((ref - got) ** 2).mean().item())
