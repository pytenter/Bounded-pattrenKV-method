"""Read-only tensor metrics used by PatternKV Insight model hooks."""

from __future__ import annotations

from typing import Any

import torch

from insight.dynamic_metrics import relative_gain, selected_fraction
from insight.gate_metrics import confusion_from_decisions
from insight.pattern_metrics import normalized_entropy, range_contraction, relative_benefit
from insight.quant_reference import mse, quantize_dequant_k_token_groups, quantize_dequant_v_head_dim
from insight.runtime import get_active_observer
from insight.sampling import sample_indices


def _to_int_list(x: torch.Tensor, limit: int = 100_000) -> list[int]:
    flat = x.detach().reshape(-1)
    if flat.numel() > limit:
        flat = flat[:limit]
    return [int(v) for v in flat.to("cpu").tolist()]


def _range(x: torch.Tensor) -> float:
    y = x.detach().float()
    return float((y.amax() - y.amin()).item()) if y.numel() else 0.0


def _sample_token_tensor(x: torch.Tensor, indices: list[int]) -> torch.Tensor:
    if not indices:
        return x[:, :, :0, :]
    idx = torch.tensor(indices, device=x.device, dtype=torch.long)
    return x.index_select(2, idx)


def _sample_k_groups(x: torch.Tensor, indices: list[int], group_size: int) -> torch.Tensor:
    if x.shape[2] < group_size or not indices:
        return _sample_token_tensor(x, indices)
    starts: list[int] = []
    for idx in indices:
        start = (idx // group_size) * group_size
        if start + group_size <= x.shape[2] and start not in starts:
            starts.append(start)
    chunks = [x[:, :, start : start + group_size, :] for start in starts]
    return torch.cat(chunks, dim=2) if chunks else x[:, :, :0, :]


def record_prefill_k_metrics(
    *,
    key_states: torch.Tensor,
    key_states_quant: torch.Tensor,
    assignments: torch.Tensor,
    k_base: torch.Tensor,
    key_states_full: torch.Tensor,
    layer_idx: int | None,
    bits: int,
    group_size: int,
) -> None:
    """Record prefill K assignment and pattern benefit metrics."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    with torch.no_grad():
        layer = int(layer_idx if layer_idx is not None else -1)
        prefix = f"prefill.k.layer{layer}"
        assign_flat = assignments.detach().reshape(-1)
        pattern_count = int(k_base.shape[-2]) if k_base.dim() >= 4 else int(assign_flat.max().item() + 1 if assign_flat.numel() else 0)
        entropy = normalized_entropy(_to_int_list(assign_flat), pattern_count)
        for name, value in entropy.items():
            observer.add_scalar(f"{prefix}.{name}", float(value))
        observer.add_histogram(f"{prefix}.assignment_histogram", assign_flat)

        raw = key_states_full.detach()
        residual = key_states_quant.detach()
        sampled = sample_indices(raw.shape[2], observer.config.sample_tokens, observer.metadata, layer, 0, "prefill_k", None, observer.config.seed)
        raw_s = _sample_token_tensor(raw, sampled)
        residual_s = _sample_token_tensor(residual, sampled)
        raw_q_input = _sample_k_groups(raw, sampled, group_size)
        residual_q_input = _sample_k_groups(residual, sampled, group_size)
        baseline_dq = quantize_dequant_k_token_groups(raw_q_input, bits=bits, group_size=group_size).dequant if raw_q_input.shape[2] >= group_size else raw_q_input
        pattern_dq = quantize_dequant_k_token_groups(residual_q_input, bits=bits, group_size=group_size).dequant if residual_q_input.shape[2] >= group_size else residual_q_input
        raw_mse = float(mse(raw_q_input, baseline_dq).item()) if raw_q_input.numel() else 0.0
        pat_mse = float(mse(residual_q_input, pattern_dq).item()) if residual_q_input.numel() else 0.0
        raw_range = _range(raw_s)
        residual_range = _range(residual_s)
        observer.add_scalar(f"{prefix}.raw_range", raw_range)
        observer.add_scalar(f"{prefix}.residual_range", residual_range)
        observer.add_scalar(f"{prefix}.range_contraction", range_contraction(raw_range, residual_range))
        observer.add_scalar(f"{prefix}.raw_mse", raw_mse)
        observer.add_scalar(f"{prefix}.pattern_mse", pat_mse)
        observer.add_scalar(f"{prefix}.relative_benefit", relative_benefit(raw_mse, pat_mse))
        observer.add_scalar(f"{prefix}.harmful", 1.0 if pat_mse > raw_mse else 0.0)
        observer.add_sample_record({"hook": "prefill_k", "layer_idx": layer, "sampled_tokens": len(sampled), "raw_mse": raw_mse, "pattern_mse": pat_mse})


def record_prefill_v_metrics(
    *,
    value_states_quant: torch.Tensor,
    idx_q: torch.Tensor,
    v_centroids: torch.Tensor,
    v_mask_q: torch.Tensor,
    value_states_full: torch.Tensor | None,
    layer_idx: int | None,
    bits: int,
    group_size: int,
) -> None:
    """Record prefill V gate and selected-pattern oracle gap metrics."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    with torch.no_grad():
        layer = int(layer_idx if layer_idx is not None else -1)
        prefix = f"prefill.v.layer{layer}"
        observer.add_histogram(f"{prefix}.assignment_histogram", idx_q.detach())
        gate = v_mask_q.detach().bool().reshape(-1)
        observer.add_scalar(f"{prefix}.gate_acceptance", float(gate.float().mean().item()) if gate.numel() else 0.0)

        residual = value_states_quant.detach()
        sampled = sample_indices(residual.shape[2], observer.config.sample_tokens, observer.metadata, layer, 0, "prefill_v", None, observer.config.seed)
        residual_s = _sample_token_tensor(residual, sampled)
        pattern_dq = quantize_dequant_v_head_dim(residual_s, bits=bits, group_size=group_size).dequant if residual_s.numel() else residual_s
        pat_mse = float(mse(residual_s, pattern_dq).item()) if residual_s.numel() else 0.0
        observer.add_scalar(f"{prefix}.current_selected_pattern_mse", pat_mse)
        observer.add_scalar(f"{prefix}.pattern_range", _range(residual_s))

        if value_states_full is not None:
            raw_s = _sample_token_tensor(value_states_full.detach(), sampled)
            raw_dq = quantize_dequant_v_head_dim(raw_s, bits=bits, group_size=group_size).dequant if raw_s.numel() else raw_s
            raw_mse = float(mse(raw_s, raw_dq).item()) if raw_s.numel() else 0.0
            observer.add_scalar(f"{prefix}.raw_mse", raw_mse)
            observer.add_scalar(f"{prefix}.raw_range", _range(raw_s))
            observer.add_scalar(f"{prefix}.relative_benefit", relative_benefit(raw_mse, pat_mse))
            current = gate.to("cpu").tolist()
            oracle = [pat_mse <= raw_mse for _ in current]
            cm = confusion_from_decisions([bool(x) for x in current], [bool(x) for x in oracle])
            observer.add_confusion(
                f"{prefix}.gate_vs_selected_oracle",
                true_positive=cm.true_positive,
                true_negative=cm.true_negative,
                false_positive=cm.false_positive,
                false_negative=cm.false_negative,
            )
        observer.add_sample_record({"hook": "prefill_v", "layer_idx": layer, "sampled_tokens": len(sampled), "pattern_mse": pat_mse})


def record_decode_k_window_metrics(
    *,
    old_count: int,
    new_count: int,
    old_mse: float,
    new_mse: float,
    selected_count: int,
    total_count: int,
    layer_idx: int | None,
    window_idx: int,
) -> None:
    """Record decode K dynamic-pattern utility scalars."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    prefix = f"decode.k.layer{layer}"
    observer.add_scalar(f"{prefix}.old_pattern_count", int(old_count))
    observer.add_scalar(f"{prefix}.new_pattern_count", int(new_count))
    observer.add_scalar(f"{prefix}.mse_gain", relative_gain(float(old_mse), float(new_mse)))
    observer.add_scalar(f"{prefix}.candidate_selected_fraction", selected_fraction(int(selected_count), int(total_count)))
    observer.add_sample_record({"hook": "decode_k", "layer_idx": layer, "window_idx": int(window_idx), "old_mse": float(old_mse), "new_mse": float(new_mse)})


def record_decode_v_window_metrics(
    *,
    old_count: int,
    new_count: int,
    old_mse: float,
    new_mse: float,
    selected_count: int,
    total_count: int,
    layer_idx: int | None,
    window_idx: int,
    gate_confusion: dict[str, Any] | None = None,
) -> None:
    """Record decode V dynamic-pattern and gate scalars."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    prefix = f"decode.v.layer{layer}"
    observer.add_scalar(f"{prefix}.old_pattern_count", int(old_count))
    observer.add_scalar(f"{prefix}.new_pattern_count", int(new_count))
    observer.add_scalar(f"{prefix}.mse_gain", relative_gain(float(old_mse), float(new_mse)))
    observer.add_scalar(f"{prefix}.candidate_selected_fraction", selected_fraction(int(selected_count), int(total_count)))
    if gate_confusion:
        observer.add_confusion(f"{prefix}.gate_confusion", **{k: int(v) for k, v in gate_confusion.items() if k in {"true_positive", "true_negative", "false_positive", "false_negative"}})
    observer.add_sample_record({"hook": "decode_v", "layer_idx": layer, "window_idx": int(window_idx), "old_mse": float(old_mse), "new_mse": float(new_mse)})
