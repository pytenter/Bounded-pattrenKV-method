from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class TensorComparison:
    shape_equal: bool
    ref_shape: tuple[int, ...] | None
    got_shape: tuple[int, ...] | None
    comparable: bool
    exact: bool | None
    difference_rate: float | None
    relative_l2: float | None
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape_equal": self.shape_equal,
            "ref_shape": list(self.ref_shape) if self.ref_shape is not None else None,
            "got_shape": list(self.got_shape) if self.got_shape is not None else None,
            "comparable": self.comparable,
            "exact": self.exact,
            "difference_rate": self.difference_rate,
            "relative_l2": self.relative_l2,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "cosine": self.cosine,
        }


def compare_tensors(ref: torch.Tensor | None, got: torch.Tensor | None) -> dict[str, Any]:
    if ref is None and got is None:
        return TensorComparison(True, None, None, True, True, 0.0, 0.0, 0.0, 0.0, 1.0).as_dict()
    if ref is None or got is None:
        return TensorComparison(False, tuple(ref.shape) if ref is not None else None, tuple(got.shape) if got is not None else None, False, None, None, None, None, None, None).as_dict()
    ref_c = ref.detach().cpu().contiguous()
    got_c = got.detach().cpu().contiguous()
    if tuple(ref_c.shape) != tuple(got_c.shape):
        return TensorComparison(False, tuple(ref_c.shape), tuple(got_c.shape), False, None, None, None, None, None, None).as_dict()
    if ref_c.numel() == 0:
        return TensorComparison(True, tuple(ref_c.shape), tuple(got_c.shape), True, True, 0.0, 0.0, 0.0, 0.0, 1.0).as_dict()
    exact = bool(torch.equal(ref_c, got_c))
    diff_rate = float((ref_c != got_c).sum().item()) / float(ref_c.numel())
    ref_f = ref_c.float()
    got_f = got_c.float()
    diff = got_f - ref_f
    ref_norm = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    got_norm = torch.linalg.vector_norm(got_f).clamp_min(1e-12)
    cosine = torch.sum(ref_f * got_f) / (ref_norm * got_norm)
    return TensorComparison(
        True,
        tuple(ref_c.shape),
        tuple(got_c.shape),
        True,
        exact,
        diff_rate,
        float((torch.linalg.vector_norm(diff) / ref_norm).item()),
        float(diff.abs().max().item()),
        float(diff.abs().mean().item()),
        float(cosine.item()),
    ).as_dict()


def logical_slice(value: torch.Tensor | None, token_axis: int | None, logical_length: int | None) -> torch.Tensor | None:
    if value is None or token_axis is None or logical_length is None:
        return value
    axis = token_axis if token_axis >= 0 else value.dim() + token_axis
    slices = [slice(None)] * value.dim()
    slices[axis] = slice(0, int(logical_length))
    return value[tuple(slices)].contiguous()


def canonical_request_state_from_fields(
    fields: dict[str, torch.Tensor | None],
    *,
    token_axes: dict[str, int | None] | None = None,
    logical_lengths: dict[str, int | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    token_axes = token_axes or {}
    logical_lengths = logical_lengths or {}
    canonical = {}
    for name, value in fields.items():
        canonical[name] = logical_slice(value, token_axes.get(name), logical_lengths.get(name))
    return {"fields": canonical, "metadata": dict(metadata or {})}


def compare_canonical_states(ref: dict[str, Any], got: dict[str, Any], fields: Iterable[str] | None = None) -> dict[str, Any]:
    names = sorted(set(fields or (set(ref.get("fields", {})) | set(got.get("fields", {})))))
    field_results = {name: compare_tensors(ref.get("fields", {}).get(name), got.get("fields", {}).get(name)) for name in names}
    metadata_ref = dict(ref.get("metadata", {}))
    metadata_got = dict(got.get("metadata", {}))
    metadata_exact = metadata_ref == metadata_got
    comparable = all(item["comparable"] for item in field_results.values())
    exact = metadata_exact and comparable and all(item["exact"] for item in field_results.values())
    return {
        "exact": exact,
        "comparable": comparable,
        "metadata_exact": metadata_exact,
        "field_results": field_results,
    }


def first_non_exact(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in rows:
        if not row.get("exact", False):
            return row
    return None

