"""Dynamic pattern utility metrics."""

from __future__ import annotations


def relative_gain(before: float, after: float, eps: float = 1e-12) -> float:
    """Return relative gain from replacing ``before`` with ``after``."""
    return (before - after) / (before + eps)


def selected_fraction(selected_count: int, total_count: int) -> float:
    """Return selected fraction with zero-safe denominator."""
    return selected_count / total_count if total_count else 0.0
