from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


ABSOLUTE_WINDOWS = (16, 32, 64, 128)


@dataclass(frozen=True)
class RegionSlice:
    name: str
    start: int
    stop: int

    @property
    def tokens(self) -> int:
        return max(self.stop - self.start, 0)


def repeat_kv_for_gqa(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if num_key_value_groups == 1:
        return hidden_states
    batch, kv_heads, tokens, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, num_key_value_groups, tokens, head_dim)
    return hidden_states.reshape(batch, kv_heads * num_key_value_groups, tokens, head_dim)


def absolute_regions(context_tokens: int, recent_length: int = 128) -> dict[str, RegionSlice]:
    regions: dict[str, RegionSlice] = {}
    for window in ABSOLUTE_WINDOWS:
        regions[f"E{window}"] = RegionSlice(f"E{window}", 0, min(window, context_tokens))
    regions["R128"] = RegionSlice("R128", max(context_tokens - recent_length, 0), context_tokens)
    middle_start = min(128, context_tokens)
    middle_stop = max(context_tokens - recent_length, middle_start)
    regions["middle"] = RegionSlice("middle", middle_start, middle_stop)
    return regions


def cache_segment_regions(*, sink_tokens: int, packed_history_tokens: int, pending_tokens: int, recent_tokens: int) -> dict[str, RegionSlice]:
    start = 0
    out = {}
    for name, tokens in (
        ("protected_sink", sink_tokens),
        ("packed_history", packed_history_tokens),
        ("pending_history", pending_tokens),
        ("recent", recent_tokens),
    ):
        out[name] = RegionSlice(name, start, start + int(tokens))
        start += int(tokens)
    return out


def shadow_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.shape[-1])
    if attention_mask is not None:
        scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(probs, value)
    return {"scores": scores, "probs": probs, "output": output}


def region_mass(probs: torch.Tensor, regions: dict[str, RegionSlice]) -> dict[str, torch.Tensor]:
    out = {}
    for name, region in regions.items():
        if region.tokens == 0:
            out[name] = torch.zeros(probs.shape[:-1], dtype=probs.dtype, device=probs.device)
        else:
            out[name] = probs[..., region.start : region.stop].sum(dim=-1)
    return out


def enrichment(mass: torch.Tensor, *, region_tokens: int, context_tokens: int) -> torch.Tensor:
    expected = float(region_tokens) / max(float(context_tokens), 1.0)
    if expected <= 0:
        return torch.zeros_like(mass)
    return mass / expected


def region_contributions(probs: torch.Tensor, value: torch.Tensor, regions: dict[str, RegionSlice]) -> dict[str, torch.Tensor]:
    out = {}
    for name, region in regions.items():
        if region.tokens == 0:
            out[name] = torch.zeros(*probs.shape[:-1], value.shape[-1], dtype=value.dtype, device=value.device)
        else:
            out[name] = torch.matmul(probs[..., region.start : region.stop], value[..., region.start : region.stop, :])
    return out


def tensor_pair_metrics(left: torch.Tensor, right: torch.Tensor, *, eps: float = 1e-12) -> dict[str, float]:
    left_f = left.detach().float().reshape(-1)
    right_f = right.detach().float().reshape(-1)
    diff = left_f - right_f
    denom = right_f.norm().clamp_min(eps)
    mse = torch.mean(diff * diff)
    return {
        "cosine": float(F.cosine_similarity(left_f, right_f, dim=0, eps=eps).item()),
        "relative_l2": float((diff.norm() / denom).item()),
        "relative_mse": float((mse / torch.mean(right_f * right_f).clamp_min(eps)).item()),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
    }


def probability_metrics(left_probs: torch.Tensor, right_probs: torch.Tensor, *, eps: float = 1e-12) -> dict[str, float]:
    left = left_probs.detach().float().clamp_min(eps)
    right = right_probs.detach().float().clamp_min(eps)
    kl = (right * (right.log() - left.log())).sum(dim=-1).mean()
    return {
        "kl_ref_to_quant": float(kl.item()),
        "cosine": tensor_pair_metrics(left, right)["cosine"],
        "mass_l1": float((left - right).abs().sum(dim=-1).mean().item()),
    }


def summarize_heads(values: torch.Tensor) -> dict[str, float | int]:
    vals = values.detach().float().reshape(-1)
    if vals.numel() == 0:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0.0, "top_head": -1}
    sorted_vals = torch.sort(vals).values
    p90_idx = max(0, min(vals.numel() - 1, math.ceil(0.9 * vals.numel()) - 1))
    return {
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "p90": float(sorted_vals[p90_idx].item()),
        "max": float(vals.max().item()),
        "top_head": int(vals.argmax().item()),
    }


def routing_value_decomposition(
    *,
    fp16_probs: torch.Tensor,
    quant_probs: torch.Tensor,
    fp16_value: torch.Tensor,
    quant_value: torch.Tensor,
    fp16_output: torch.Tensor,
) -> dict[str, dict[str, float]]:
    routing_output = torch.matmul(quant_probs, fp16_value)
    value_output = torch.matmul(fp16_probs, quant_value)
    full_output = torch.matmul(quant_probs, quant_value)
    return {
        "routing_only": tensor_pair_metrics(routing_output, fp16_output),
        "value_only": tensor_pair_metrics(value_output, fp16_output),
        "full": tensor_pair_metrics(full_output, fp16_output),
    }


def clone_signature(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return (tuple(obj.shape), str(obj.dtype), str(obj.device), float(obj.detach().float().sum().item()) if obj.numel() else 0.0)
    if isinstance(obj, (tuple, list)):
        return tuple(clone_signature(item) for item in obj)
    if isinstance(obj, dict):
        return {key: clone_signature(value) for key, value in obj.items()}
    return obj
