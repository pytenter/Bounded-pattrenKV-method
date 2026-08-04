"""Read-only tensor metrics used by PatternKV Insight model hooks."""

from __future__ import annotations

from typing import Any

import torch

from insight.dynamic_metrics import relative_gain, selected_fraction
from insight.errors import InsightHookError, tensor_shapes
from insight.pattern_metrics import normalized_entropy, range_contraction, relative_benefit
from insight.quant_reference import mse, quantize_dequant_k_token_groups, quantize_dequant_v_head_dim
from insight.runtime import get_active_observer
from insight.sampling import sample_indices


def _bucket(pos: int, total: int) -> str:
    if total <= 1:
        return "all"
    frac = pos / max(total - 1, 1)
    if frac < 1 / 3:
        return "first"
    if frac < 2 / 3:
        return "middle"
    return "last"


def _range(x: torch.Tensor) -> float:
    y = x.detach().float()
    return float((y.amax() - y.amin()).item()) if y.numel() else 0.0


def _scalar_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(mse(a, b).item()) if a.numel() else 0.0


def _gather_k_patterns(assignments: torch.Tensor, k_base: torch.Tensor, head: int) -> torch.Tensor:
    """Return per-position K patterns for one head: [B,T,D]."""
    base = k_base[head].unsqueeze(0).expand(assignments.shape[0], -1, -1)
    assign_head = assignments[:, 0, :] if assignments.shape[1] == 1 else assignments[:, head, :]
    return torch.gather(base, 1, assign_head.unsqueeze(-1).expand(-1, -1, base.shape[-1]))


def _gather_v_patterns(idx: torch.Tensor, centroids: torch.Tensor, head: int) -> torch.Tensor:
    """Return per-token V patterns for one head: [B,T,D]."""
    base = centroids[head].unsqueeze(0).expand(idx.shape[0], -1, -1)
    idx_head = idx[:, 0, :] if idx.shape[1] == 1 else idx[:, head, :]
    return torch.gather(base, 1, idx_head.unsqueeze(-1).expand(-1, -1, base.shape[-1]))


def _minmax_assign_tokens(tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    residual = tokens[:, None, :].float() - centroids[None, :, :].float()
    ranges = residual.amax(dim=-1) - residual.amin(dim=-1)
    return ranges.argmin(dim=-1)


def _l2_assign_tokens(tokens: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    dist = ((tokens[:, None, :].float() - centroids[None, :, :].float()) ** 2).sum(dim=-1)
    return dist.argmin(dim=-1)


def _v_token_mse(raw: torch.Tensor, pattern: torch.Tensor | None, *, bits: int, group_size: int) -> torch.Tensor:
    """Return per-token V reconstruction MSE for [N,D]."""
    if raw.numel() == 0:
        return torch.empty(raw.shape[0], device=raw.device)
    if pattern is None:
        dq = quantize_dequant_v_head_dim(raw.view(1, 1, raw.shape[0], raw.shape[1]), bits=bits, group_size=group_size).dequant.view_as(raw)
        rec = dq
    else:
        residual = raw - pattern
        dq = quantize_dequant_v_head_dim(residual.view(1, 1, residual.shape[0], residual.shape[1]), bits=bits, group_size=group_size).dequant.view_as(residual)
        rec = pattern + dq
    return ((raw.float() - rec.float()) ** 2).mean(dim=-1)


def _record_scalar(observer: Any, prefix: str, metric: str, value: float) -> None:
    observer.add_scalar(f"{prefix}.{metric}", float(value))
    observer.add_sample_record({"phase": prefix.split(".")[0], "kv_type": prefix.split(".")[1], "metric": metric, "value": float(value)})


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
    """Record prefill K gain by layer/head/128-token group."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    try:
        with torch.no_grad():
            raw = key_states_full.detach()
            residual = key_states_quant.detach()
            if raw.dim() != 4 or residual.shape != raw.shape or assignments.shape[:3] != raw.shape[:3]:
                raise ValueError("K hook tensors must be raw/residual [B,H,T,D] and assignments [B,H,T]")
            B, H, T, D = raw.shape
            full_groups = T // group_size
            pattern_count = int(k_base.shape[1])
            for head in range(H):
                h_assign = assignments[:, head, : full_groups * group_size].detach()
                entropy = normalized_entropy([int(x) for x in h_assign.reshape(-1).to("cpu").tolist()], pattern_count)
                prefix_head = f"prefill.k.layer{layer}.head{head}"
                observer.add_histogram(f"{prefix_head}.assignment_histogram", h_assign)
                for name, value in entropy.items():
                    observer.add_scalar(f"{prefix_head}.{name}", float(value))
                for group_idx in range(full_groups):
                    start = group_idx * group_size
                    end = start + group_size
                    bucket = _bucket(start + group_size // 2, T)
                    raw_group = raw[:, head : head + 1, start:end, :]
                    residual_group = residual[:, head : head + 1, start:end, :]
                    raw_dq = quantize_dequant_k_token_groups(raw_group, bits=bits, group_size=group_size).dequant
                    pat_dq = quantize_dequant_k_token_groups(residual_group, bits=bits, group_size=group_size).dequant
                    raw_mse = _scalar_mse(raw_group, raw_dq)
                    pattern_mse = _scalar_mse(residual_group, pat_dq)
                    raw_range = _range(raw_group)
                    residual_range = _range(residual_group)
                    prefix = f"{prefix_head}.bucket{bucket}"
                    observer.add_scalar(f"{prefix}.raw_mse", raw_mse)
                    observer.add_scalar(f"{prefix}.pattern_mse", pattern_mse)
                    observer.add_scalar(f"{prefix}.relative_benefit", relative_benefit(raw_mse, pattern_mse))
                    observer.add_scalar(f"{prefix}.harmful", 1.0 if pattern_mse > raw_mse else 0.0)
                    observer.add_scalar(f"{prefix}.raw_range", raw_range)
                    observer.add_scalar(f"{prefix}.pattern_residual_range", residual_range)
                    observer.add_scalar(f"{prefix}.range_contraction", range_contraction(raw_range, residual_range))
                    observer.add_sample_record(
                        {
                            "hook": "prefill_k",
                            "phase": "prefill",
                            "kv_type": "k",
                            "layer_idx": layer,
                            "kv_head": head,
                            "position_bucket": bucket,
                            "group_start_token": start,
                            "group_end_token": end - 1,
                            "raw_mse": raw_mse,
                            "pattern_mse": pattern_mse,
                            "relative_benefit": relative_benefit(raw_mse, pattern_mse),
                            "harmful": pattern_mse > raw_mse,
                        }
                    )
                if observer.config.level == "oracle" and layer in observer.config.oracle_layers and full_groups:
                    _record_k_conditional_oracle(observer, raw, assignments, k_base, layer, head, bits=bits, group_size=group_size)
    except Exception as exc:
        if isinstance(exc, InsightHookError):
            raise
        raise InsightHookError(
            "prefill K hook failed",
            hook_name="record_prefill_k_metrics",
            phase="prefill",
            kv_type="k",
            layer_idx=layer,
            tensor_shapes=tensor_shapes(
                {
                    "key_states": key_states,
                    "key_states_quant": key_states_quant,
                    "assignments": assignments,
                    "k_base": k_base,
                    "key_states_full": key_states_full,
                }
            ),
            cause=exc,
        ) from exc


def _record_k_conditional_oracle(observer: Any, raw: torch.Tensor, assignments: torch.Tensor, k_base: torch.Tensor, layer: int, head: int, *, bits: int, group_size: int) -> None:
    B, H, T, D = raw.shape
    sampled = sample_indices(T, observer.config.sample_tokens, observer.metadata, layer, head, "prefill_k_oracle", None, observer.config.seed)
    for token in sampled[: observer.config.sample_tokens]:
        group_start = (token // group_size) * group_size
        if group_start + group_size > T:
            continue
        local = token - group_start
        raw_group = raw[:, head : head + 1, group_start : group_start + group_size, :]
        cur_assign = assignments[:, head, group_start : group_start + group_size].clone()
        patterns = _gather_k_patterns(assignments[:, :, group_start : group_start + group_size], k_base, head).unsqueeze(1)
        current_residual = raw_group - patterns
        current_dq = quantize_dequant_k_token_groups(current_residual, bits=bits, group_size=group_size).dequant
        current_mse = _scalar_mse(raw_group, current_dq + patterns)
        raw_dq = quantize_dequant_k_token_groups(raw_group, bits=bits, group_size=group_size).dequant
        raw_mse = _scalar_mse(raw_group, raw_dq)
        token_vec = raw_group[:, 0, local, :].reshape(-1, D)
        l2 = int(_l2_assign_tokens(token_vec, k_base[head]).reshape(-1)[0].item())
        minmax = int(_minmax_assign_tokens(token_vec, k_base[head]).reshape(-1)[0].item())
        best_idx = int(cur_assign[0, local].item())
        best_mse = current_mse
        minmax_mse = None
        for cand in range(k_base.shape[1]):
            cand_assign = cur_assign.clone()
            cand_assign[:, local] = cand
            cand_patterns = _gather_k_patterns(cand_assign.unsqueeze(1), k_base, head).unsqueeze(1)
            residual = raw_group - cand_patterns
            dq = quantize_dequant_k_token_groups(residual, bits=bits, group_size=group_size).dequant
            cand_mse = _scalar_mse(raw_group, dq + cand_patterns)
            if cand == minmax:
                minmax_mse = cand_mse
            if cand_mse < best_mse:
                best_mse = cand_mse
                best_idx = cand
        observer.add_sample_record(
            {
                "hook": "k_conditional_oracle",
                "phase": "prefill",
                "kv_type": "k",
                "layer_idx": layer,
                "kv_head": head,
                "position_bucket": _bucket(token, T),
                "token_idx": token,
                "group_start_token": group_start,
                "current_assignment": int(cur_assign[0, local].item()),
                "l2_assignment": l2,
                "minmax_assignment": minmax,
                "conditional_oracle_assignment": best_idx,
                "raw_group_mse": raw_mse,
                "current_group_mse": current_mse,
                "minmax_group_mse": float(minmax_mse if minmax_mse is not None else current_mse),
                "conditional_oracle_group_mse": best_mse,
                "current_conditional_oracle_gap": current_mse - best_mse,
                "minmax_conditional_oracle_gap": float(minmax_mse if minmax_mse is not None else current_mse) - best_mse,
            }
        )
        prefix = f"prefill.k.layer{layer}.head{head}.bucket{_bucket(token, T)}"
        observer.add_scalar(f"{prefix}.conditional_oracle_gap", current_mse - best_mse)


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
    rho: torch.Tensor | None = None,
) -> None:
    """Record prefill V per-token gate confusion and optional matching oracle."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    try:
        with torch.no_grad():
            if value_states_full is None:
                return
            raw_all = value_states_full.detach()
            residual_actual = value_states_quant.detach()
            if raw_all.dim() != 4 or residual_actual.shape != raw_all.shape:
                raise ValueError("V hook tensors must be [B,H,T,D]")
            B, H, T, D = raw_all.shape
            sampled_by_head = {
                head: sample_indices(T, observer.config.sample_tokens, observer.metadata, layer, head, "prefill_v", None, observer.config.seed)
                for head in range(H)
            }
            for head, sampled in sampled_by_head.items():
                prefix_head = f"prefill.v.layer{layer}.head{head}"
                observer.add_histogram(f"{prefix_head}.assignment_histogram", idx_q[:, head, :].detach())
                gate_head = v_mask_q[:, head, :].detach().bool()
                observer.add_scalar(f"{prefix_head}.gate_acceptance", float(gate_head.float().mean().item()) if gate_head.numel() else 0.0)
                for token in sampled:
                    bucket = _bucket(token, T)
                    raw = raw_all[:, head, token, :]
                    current_pattern = _gather_v_patterns(idx_q[:, :, token : token + 1], v_centroids, head)[:, 0, :]
                    gate = v_mask_q[:, head, token].detach().bool()
                    raw_mse_vec = _v_token_mse(raw, None, bits=bits, group_size=group_size)
                    pattern_mse_vec = _v_token_mse(raw, current_pattern, bits=bits, group_size=group_size)
                    oracle = pattern_mse_vec < raw_mse_vec
                    actual = torch.where(gate, pattern_mse_vec, raw_mse_vec)
                    tp = int((gate & oracle).sum().item())
                    tn = int((~gate & ~oracle).sum().item())
                    fp = int((gate & ~oracle).sum().item())
                    fn = int((~gate & oracle).sum().item())
                    prefix = f"{prefix_head}.bucket{bucket}"
                    observer.add_confusion(f"{prefix}.gate_vs_mse_oracle", true_positive=tp, true_negative=tn, false_positive=fp, false_negative=fn)
                    raw_mse = float(raw_mse_vec.mean().item())
                    pat_mse = float(pattern_mse_vec.mean().item())
                    actual_mse = float(actual.mean().item())
                    observer.add_scalar(f"{prefix}.raw_mse", raw_mse)
                    observer.add_scalar(f"{prefix}.pattern_candidate_mse", pat_mse)
                    observer.add_scalar(f"{prefix}.actual_selected_path_mse", actual_mse)
                    observer.add_scalar(f"{prefix}.relative_candidate_benefit", relative_benefit(raw_mse, pat_mse))
                    observer.add_sample_record(
                        {
                            "hook": "prefill_v",
                            "phase": "prefill",
                            "kv_type": "v",
                            "layer_idx": layer,
                            "kv_head": head,
                            "position_bucket": bucket,
                            "token_idx": token,
                            "raw_mse": raw_mse,
                            "pattern_candidate_mse": pat_mse,
                            "actual_selected_path_mse": actual_mse,
                            "relative_candidate_benefit": relative_benefit(raw_mse, pat_mse),
                            "gate_current": bool(gate.reshape(-1)[0].item()),
                            "gate_oracle": bool(oracle.reshape(-1)[0].item()),
                            "rho": float(rho[:, head, token, :].float().mean().item()) if rho is not None and rho.dim() == 4 else None,
                            "false_positive_penalty": float(torch.clamp(pattern_mse_vec - raw_mse_vec, min=0).mean().item()),
                            "false_negative_opportunity": float(torch.clamp(raw_mse_vec - pattern_mse_vec, min=0).mean().item()),
                            "centroid_idx": int(idx_q[0, head, token].item()),
                        }
                    )
                if observer.config.level == "oracle" and layer in observer.config.oracle_layers:
                    _record_v_matching_oracle(observer, raw_all, idx_q, v_centroids, layer, head, sampled, bits=bits, group_size=group_size)
    except Exception as exc:
        if isinstance(exc, InsightHookError):
            raise
        raise InsightHookError(
            "prefill V hook failed",
            hook_name="record_prefill_v_metrics",
            phase="prefill",
            kv_type="v",
            layer_idx=layer,
            tensor_shapes=tensor_shapes(
                {
                    "value_states_quant": value_states_quant,
                    "idx_q": idx_q,
                    "v_centroids": v_centroids,
                    "v_mask_q": v_mask_q,
                    "value_states_full": value_states_full,
                    "rho": rho,
                }
            ),
            cause=exc,
        ) from exc


def _record_v_matching_oracle(observer: Any, raw_all: torch.Tensor, idx_q: torch.Tensor, centroids: torch.Tensor, layer: int, head: int, sampled: list[int], *, bits: int, group_size: int) -> None:
    T = raw_all.shape[2]
    for token in sampled[: observer.config.sample_tokens]:
        raw = raw_all[:, head, token, :].reshape(-1, raw_all.shape[-1])
        cand_mses = []
        for cand in range(centroids.shape[1]):
            pattern = centroids[head, cand].view(1, -1).expand_as(raw)
            cand_mses.append(_v_token_mse(raw, pattern, bits=bits, group_size=group_size))
        losses = torch.stack(cand_mses, dim=-1).mean(dim=0)
        oracle_idx = int(losses.argmin().item())
        current_idx = int(idx_q[0, head, token].item())
        l2_idx = int(_l2_assign_tokens(raw, centroids[head]).reshape(-1)[0].item())
        minmax_idx = int(_minmax_assign_tokens(raw, centroids[head]).reshape(-1)[0].item())
        oracle_mse = float(losses[oracle_idx].item())
        current_mse = float(losses[current_idx].item())
        minmax_mse = float(losses[minmax_idx].item())
        l2_mse = float(losses[l2_idx].item())
        bucket = _bucket(token, T)
        observer.add_sample_record(
            {
                "hook": "v_matching_oracle",
                "phase": "prefill",
                "kv_type": "v",
                "layer_idx": layer,
                "kv_head": head,
                "position_bucket": bucket,
                "token_idx": token,
                "l2_assignment": l2_idx,
                "minmax_assignment": minmax_idx,
                "current_assignment": current_idx,
                "mse_oracle_assignment": oracle_idx,
                "l2_minmax_mismatch": l2_idx != minmax_idx,
                "current_mse_mismatch": current_idx != oracle_idx,
                "minmax_mse_mismatch": minmax_idx != oracle_idx,
                "l2_mse": l2_mse,
                "minmax_mse": minmax_mse,
                "current_pattern_mse": current_mse,
                "oracle_mse": oracle_mse,
                "range_regret": minmax_mse - oracle_mse,
                "current_oracle_gap": current_mse - oracle_mse,
                "minmax_oracle_gap": minmax_mse - oracle_mse,
            }
        )
        prefix = f"prefill.v.layer{layer}.head{head}.bucket{bucket}"
        observer.add_scalar(f"{prefix}.current_oracle_gap", current_mse - oracle_mse)
        observer.add_scalar(f"{prefix}.minmax_oracle_gap", minmax_mse - oracle_mse)


def record_decode_k_window_metrics(
    *,
    window_raw: torch.Tensor,
    old_k_base: torch.Tensor,
    new_k_base: torch.Tensor,
    layer_idx: int | None,
    window_idx: int,
    bits: int,
    group_size: int,
) -> None:
    """Record decode K dynamic utility from true old/new reconstruction MSE."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    try:
        with torch.no_grad():
            B, H, T, D = window_raw.shape
            Xw = window_raw.permute(1, 0, 2, 3).reshape(H, B * T, D).contiguous()
            for head in range(H):
                old_assign_flat = _minmax_assign_tokens(Xw[head], old_k_base[head])
                new_assign_flat = _minmax_assign_tokens(Xw[head], new_k_base[head])
                old_assign = old_assign_flat.view(B, T)
                new_assign = new_assign_flat.view(B, T)
                old_patterns = _gather_k_patterns(old_assign.unsqueeze(1), old_k_base, head).unsqueeze(1)
                new_patterns = _gather_k_patterns(new_assign.unsqueeze(1), new_k_base, head).unsqueeze(1)
                raw = window_raw[:, head : head + 1, :, :]
                old_residual = raw - old_patterns
                new_residual = raw - new_patterns
                old_dq = quantize_dequant_k_token_groups(old_residual, bits=bits, group_size=group_size).dequant
                new_dq = quantize_dequant_k_token_groups(new_residual, bits=bits, group_size=group_size).dequant
                old_mse = _scalar_mse(raw, old_dq + old_patterns)
                new_mse = _scalar_mse(raw, new_dq + new_patterns)
                old_range = _range(old_residual)
                new_range = _range(new_residual)
                candidate_count = int((new_assign == (new_k_base.shape[1] - 1)).sum().item())
                prefix = f"decode.k.layer{layer}.head{head}.bucket{_bucket(0, T)}"
                observer.add_scalar(f"{prefix}.old_mse", old_mse)
                observer.add_scalar(f"{prefix}.new_mse", new_mse)
                observer.add_scalar(f"{prefix}.relative_mse_gain", relative_gain(old_mse, new_mse))
                observer.add_scalar(f"{prefix}.relative_range_gain", relative_gain(old_range, new_range))
                observer.add_scalar(f"{prefix}.candidate_assignment_fraction", selected_fraction(candidate_count, int(new_assign.numel())))
                observer.add_sample_record(
                    {
                        "hook": "decode_k",
                        "phase": "decode",
                        "kv_type": "k",
                        "layer_idx": layer,
                        "kv_head": head,
                        "window_idx": int(window_idx),
                        "old_mse": old_mse,
                        "new_mse": new_mse,
                        "relative_mse_gain": relative_gain(old_mse, new_mse),
                        "relative_range_gain": relative_gain(old_range, new_range),
                        "candidate_assignment_count": candidate_count,
                        "candidate_assignment_fraction": selected_fraction(candidate_count, int(new_assign.numel())),
                    }
                )
    except Exception as exc:
        if isinstance(exc, InsightHookError):
            raise
        raise InsightHookError(
            "decode K hook failed",
            hook_name="record_decode_k_window_metrics",
            phase="decode",
            kv_type="k",
            layer_idx=layer,
            tensor_shapes=tensor_shapes({"window_raw": window_raw, "old_k_base": old_k_base, "new_k_base": new_k_base}),
            cause=exc,
        ) from exc


def record_decode_v_window_metrics(
    *,
    window_raw: torch.Tensor,
    old_v_centroids: torch.Tensor,
    new_v_centroids: torch.Tensor,
    old_idx: torch.Tensor,
    new_idx: torch.Tensor,
    old_mask: torch.Tensor,
    new_mask: torch.Tensor,
    layer_idx: int | None,
    window_idx: int,
    bits: int,
    group_size: int,
) -> None:
    """Record decode V dynamic utility with assignment and applied-gate ratios."""
    observer = get_active_observer()
    if observer is None or not observer.enabled:
        return
    layer = int(layer_idx if layer_idx is not None else -1)
    try:
        with torch.no_grad():
            B, H, T, D = window_raw.shape
            for head in range(H):
                raw = window_raw[:, head, :, :].reshape(B * T, D)
                old_patterns = _gather_v_patterns(old_idx, old_v_centroids, head).reshape(B * T, D)
                new_patterns = _gather_v_patterns(new_idx, new_v_centroids, head).reshape(B * T, D)
                raw_mse = _v_token_mse(raw, None, bits=bits, group_size=group_size).view(B, T)
                old_candidate = _v_token_mse(raw, old_patterns, bits=bits, group_size=group_size).view(B, T)
                new_candidate = _v_token_mse(raw, new_patterns, bits=bits, group_size=group_size).view(B, T)
                old_actual = torch.where(old_mask[:, head, :].bool(), old_candidate, raw_mse)
                new_actual = torch.where(new_mask[:, head, :].bool(), new_candidate, raw_mse)
                candidate_selected = new_idx[:, head, :] == (new_v_centroids.shape[1] - 1)
                candidate_gate = candidate_selected & new_mask[:, head, :].bool()
                prefix = f"decode.v.layer{layer}.head{head}.bucket{_bucket(0, T)}"
                old_assignment_mse = float(old_candidate.mean().item())
                new_assignment_mse = float(new_candidate.mean().item())
                old_actual_mse = float(old_actual.mean().item())
                new_actual_mse = float(new_actual.mean().item())
                observer.add_scalar(f"{prefix}.old_assignment_mse", old_assignment_mse)
                observer.add_scalar(f"{prefix}.new_assignment_mse", new_assignment_mse)
                observer.add_scalar(f"{prefix}.old_actual_gate_mse", old_actual_mse)
                observer.add_scalar(f"{prefix}.new_actual_gate_mse", new_actual_mse)
                observer.add_scalar(f"{prefix}.candidate_assignment_fraction", float(candidate_selected.float().mean().item()))
                observer.add_scalar(f"{prefix}.candidate_gate_accepted_fraction", float(candidate_gate.float().mean().item()))
                observer.add_sample_record(
                    {
                        "hook": "decode_v",
                        "phase": "decode",
                        "kv_type": "v",
                        "layer_idx": layer,
                        "kv_head": head,
                        "window_idx": int(window_idx),
                        "old_assignment_mse": old_assignment_mse,
                        "new_assignment_mse": new_assignment_mse,
                        "old_actual_gate_mse": old_actual_mse,
                        "new_actual_gate_mse": new_actual_mse,
                        "candidate_assignment_fraction": float(candidate_selected.float().mean().item()),
                        "candidate_gate_accepted_fraction": float(candidate_gate.float().mean().item()),
                    }
                )
    except Exception as exc:
        if isinstance(exc, InsightHookError):
            raise
        raise InsightHookError(
            "decode V hook failed",
            hook_name="record_decode_v_window_metrics",
            phase="decode",
            kv_type="v",
            layer_idx=layer,
            tensor_shapes=tensor_shapes(
                {
                    "window_raw": window_raw,
                    "old_v_centroids": old_v_centroids,
                    "new_v_centroids": new_v_centroids,
                    "old_idx": old_idx,
                    "new_idx": new_idx,
                    "old_mask": old_mask,
                    "new_mask": new_mask,
                }
            ),
            cause=exc,
        ) from exc
