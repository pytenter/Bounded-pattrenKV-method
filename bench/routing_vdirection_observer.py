from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


EPS = 1e-12
SCHEMA_VERSION = "routing_vdirection_observer_v1"
ABSOLUTE_WINDOWS = (16, 32, 64, 128)


@dataclass(frozen=True)
class Region:
    name: str
    start: int
    stop: int

    @property
    def tokens(self) -> int:
        return max(0, int(self.stop) - int(self.start))


def gqa_kv_head_for_query_head(query_head: int, num_attention_heads: int, num_key_value_heads: int) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    return int(query_head) // (int(num_attention_heads) // int(num_key_value_heads))


def repeat_kv_for_gqa(hidden_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    if int(num_key_value_groups) == 1:
        return hidden_states
    batch, kv_heads, tokens, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, kv_heads, int(num_key_value_groups), tokens, head_dim)
    return hidden_states.reshape(batch, kv_heads * int(num_key_value_groups), tokens, head_dim)


def current_query(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4:
        raise ValueError(f"expected [B,H,T,D], got {tuple(tensor.shape)}")
    return tensor[:, :, -1:, :].contiguous()


def vector_errors(left: torch.Tensor, right: torch.Tensor, *, eps: float = EPS) -> dict[str, torch.Tensor]:
    left_f = left.detach().float()
    right_f = right.detach().float()
    if left_f.shape != right_f.shape:
        limit = min(left_f.shape[-2], right_f.shape[-2])
        left_f = left_f[..., :limit, :]
        right_f = right_f[..., :limit, :]
    dot = (left_f * right_f).sum(dim=-1)
    ln = torch.linalg.vector_norm(left_f, dim=-1)
    rn = torch.linalg.vector_norm(right_f, dim=-1)
    cosine = (dot / (ln * rn).clamp_min(eps)).clamp(-1.0, 1.0)
    rel_l2 = torch.linalg.vector_norm(left_f - right_f, dim=-1) / rn.clamp_min(eps)
    return {"direction_error": 1.0 - cosine, "relative_L2": rel_l2}


def quantile(values: torch.Tensor, q: float) -> float:
    vals = values.detach().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return 0.0
    return float(torch.quantile(vals, float(q)).item())


def summarize_tensor(values: torch.Tensor) -> dict[str, float | int]:
    vals = values.detach().float().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    if vals.numel() == 0:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "n_samples": 0}
    return {
        "mean": float(vals.mean().item()),
        "p50": quantile(vals, 0.50),
        "p90": quantile(vals, 0.90),
        "p95": quantile(vals, 0.95),
        "p99": quantile(vals, 0.99),
        "max": float(vals.max().item()),
        "n_samples": int(vals.numel()),
    }


def qk_logits(query: torch.Tensor, key: torch.Tensor, *, num_key_value_groups: int, scale: float | None = None) -> torch.Tensor:
    q = query.detach().float()
    k = key.detach().float()
    if q.ndim != 4 or k.ndim != 4:
        raise ValueError("query/key must be [B,H,T,D]")
    if q.shape[-2] != 1:
        q = current_query(q)
    kg = repeat_kv_for_gqa(k, int(num_key_value_groups))
    if q.shape[1] != kg.shape[1]:
        raise ValueError(f"GQA head mismatch: q={q.shape[1]} repeated_k={kg.shape[1]}")
    factor = float(scale) if scale is not None else 1.0 / math.sqrt(float(q.shape[-1]))
    return torch.matmul(q, kg.transpose(-2, -1)) * factor


def logit_metrics(quant_logits: torch.Tensor, fp_logits: torch.Tensor, *, eps: float = EPS) -> dict[str, float]:
    q = quant_logits.detach().float()
    f = fp_logits.detach().float()
    diff = q - f
    denom = torch.linalg.vector_norm(f.reshape(-1)).clamp_min(eps)
    cosine = F.cosine_similarity(q.reshape(-1), f.reshape(-1), dim=0, eps=eps)
    abs_diff = diff.abs().reshape(-1)
    return {
        "relative_L2": float((torch.linalg.vector_norm(diff.reshape(-1)) / denom).item()),
        "cosine_loss": float((1.0 - cosine).item()),
        "max_abs_diff": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs_diff": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "p95_abs_diff": quantile(abs_diff, 0.95),
        "p99_abs_diff": quantile(abs_diff, 0.99),
    }


def topk_overlap(quant_logits: torch.Tensor, fp_logits: torch.Tensor, k: int) -> torch.Tensor:
    q = quant_logits.detach().float()
    f = fp_logits.detach().float()
    valid = min(int(k), int(q.shape[-1]), int(f.shape[-1]))
    if valid <= 0:
        return torch.zeros(q.shape[:-1], dtype=torch.float32)
    q_idx = torch.topk(q, valid, dim=-1).indices
    f_idx = torch.topk(f, valid, dim=-1).indices
    return (q_idx[..., :, None] == f_idx[..., None, :]).any(dim=-1).float().mean(dim=-1)


def top1_agreement(quant_logits: torch.Tensor, fp_logits: torch.Tensor) -> torch.Tensor:
    return (quant_logits.detach().float().argmax(dim=-1) == fp_logits.detach().float().argmax(dim=-1)).float()


def attention_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits.detach().float(), dim=-1)


def probability_metrics(quant_probs: torch.Tensor, fp_probs: torch.Tensor, *, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    q = quant_probs.detach().float().clamp_min(eps)
    f = fp_probs.detach().float().clamp_min(eps)
    m = 0.5 * (q + f)
    kl_fp_quant = (f * (f.log() - q.log())).sum(dim=-1)
    kl_quant_fp = (q * (q.log() - f.log())).sum(dim=-1)
    js = 0.5 * (f * (f.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)
    l1 = (q - f).abs().sum(dim=-1)
    tv = 0.5 * l1
    cosine = F.cosine_similarity(q, f, dim=-1, eps=eps)
    h_fp = -(f * f.log()).sum(dim=-1)
    h_quant = -(q * q.log()).sum(dim=-1)
    return {
        "kl_fp_quant": kl_fp_quant,
        "kl_quant_fp": kl_quant_fp,
        "js": js,
        "tv": tv,
        "l1": l1,
        "cosine_loss": 1.0 - cosine,
        "entropy_fp": h_fp,
        "entropy_quant": h_quant,
        "entropy_delta": h_quant - h_fp,
    }


def attention_regions(context_tokens: int, *, recent_length: int = 128) -> dict[str, Region]:
    total = int(context_tokens)
    regions: dict[str, Region] = {}
    for window in ABSOLUTE_WINDOWS:
        regions[f"E{window}"] = Region(f"E{window}", 0, min(int(window), total))
    regions["Recent128"] = Region("Recent128", max(total - int(recent_length), 0), total)
    return regions


def region_mass(probs: torch.Tensor, regions: dict[str, Region]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, region in regions.items():
        if region.tokens <= 0:
            out[name] = torch.zeros(probs.shape[:-1], dtype=probs.dtype, device=probs.device)
        else:
            out[name] = probs[..., region.start : region.stop].sum(dim=-1)
    return out


def attention_weighted_vector_errors(
    quant_value: torch.Tensor,
    fp_value: torch.Tensor,
    fp_probs: torch.Tensor,
    quant_probs: torch.Tensor | None = None,
    *,
    eps: float = EPS,
) -> dict[str, torch.Tensor]:
    errors = vector_errors(quant_value, fp_value, eps=eps)
    fp_w = fp_probs.detach().float().squeeze(-2)
    out = {
        "weighted_direction_error_fp": (fp_w * errors["direction_error"]).sum(dim=-1),
        "weighted_relative_L2_fp": (fp_w * errors["relative_L2"]).sum(dim=-1),
    }
    if quant_probs is not None:
        q_w = quant_probs.detach().float().squeeze(-2)
        out["weighted_direction_error_quant"] = (q_w * errors["direction_error"]).sum(dim=-1)
        out["weighted_relative_L2_quant"] = (q_w * errors["relative_L2"]).sum(dim=-1)
    return out


def oracle_outputs(
    *,
    fp_probs: torch.Tensor,
    quant_probs: torch.Tensor,
    fp_value: torch.Tensor,
    quant_value: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "O_FP": torch.matmul(fp_probs.detach().float(), fp_value.detach().float()),
        "O_Q": torch.matmul(quant_probs.detach().float(), quant_value.detach().float()),
        "ROUTING_ONLY_OUTPUT": torch.matmul(quant_probs.detach().float(), fp_value.detach().float()),
        "VALUE_ONLY_OUTPUT": torch.matmul(fp_probs.detach().float(), quant_value.detach().float()),
    }


def output_error(output: torch.Tensor, fp_output: torch.Tensor, *, eps: float = EPS) -> dict[str, float]:
    left = output.detach().float().reshape(-1)
    right = fp_output.detach().float().reshape(-1)
    diff = left - right
    rel = torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(right).clamp_min(eps)
    cosine = F.cosine_similarity(left, right, dim=0, eps=eps)
    return {"relative_L2": float(rel.item()), "cosine_loss": float((1.0 - cosine).item())}


def oracle_error_metrics(outputs: dict[str, torch.Tensor], *, eps: float = EPS) -> dict[str, float]:
    actual = output_error(outputs["O_Q"], outputs["O_FP"], eps=eps)
    routing = output_error(outputs["ROUTING_ONLY_OUTPUT"], outputs["O_FP"], eps=eps)
    value = output_error(outputs["VALUE_ONLY_OUTPUT"], outputs["O_FP"], eps=eps)
    e_actual = actual["relative_L2"]
    e_routing = routing["relative_L2"]
    e_value = value["relative_L2"]
    routing_recovery_raw = 1.0 - e_value / (e_actual + eps)
    value_recovery_raw = 1.0 - e_routing / (e_actual + eps)
    return {
        "actual_relative_L2": e_actual,
        "actual_cosine_loss": actual["cosine_loss"],
        "routing_only_relative_L2": e_routing,
        "routing_only_cosine_loss": routing["cosine_loss"],
        "value_only_relative_L2": e_value,
        "value_only_cosine_loss": value["cosine_loss"],
        "routing_oracle_recovery_raw": routing_recovery_raw,
        "routing_oracle_recovery_clamped_0_1": min(1.0, max(0.0, routing_recovery_raw)),
        "value_oracle_recovery_raw": value_recovery_raw,
        "value_oracle_recovery_clamped_0_1": min(1.0, max(0.0, value_recovery_raw)),
        "interaction_residual": e_actual - e_routing - e_value,
    }


def all_finite(payload: Any) -> bool:
    if torch.is_tensor(payload):
        return bool(torch.isfinite(payload.detach()).all().item())
    if isinstance(payload, dict):
        return all(all_finite(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(all_finite(value) for value in payload)
    if isinstance(payload, (float, int)):
        return math.isfinite(float(payload))
    return True
