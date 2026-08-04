"""Pattern usage and gain metric helpers."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable


def relative_benefit(raw_mse: float, pattern_mse: float, eps: float = 1e-12) -> float:
    """Compute normalized MSE reduction."""
    return (raw_mse - pattern_mse) / (raw_mse + eps)


def normalized_entropy(assignments: Iterable[int], pattern_count: int) -> dict[str, float | int]:
    """Compute assignment entropy and dead-pattern statistics."""
    counts = Counter(int(x) for x in assignments)
    total = sum(counts.values())
    if total == 0 or pattern_count <= 0:
        return {"entropy": 0.0, "normalized_entropy": 0.0, "dead_pattern_count": pattern_count, "dead_pattern_fraction": 1.0, "top1_pattern_share": 0.0, "top4_pattern_share": 0.0}
    probs = [c / total for c in counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    norm = entropy / math.log(pattern_count) if pattern_count > 1 else 0.0
    shares = sorted(probs, reverse=True)
    dead = max(pattern_count - len(counts), 0)
    return {
        "entropy": entropy,
        "normalized_entropy": max(0.0, min(1.0, norm)),
        "dead_pattern_count": dead,
        "dead_pattern_fraction": dead / pattern_count,
        "top1_pattern_share": shares[0] if shares else 0.0,
        "top4_pattern_share": sum(shares[:4]),
    }


def range_contraction(raw_range: float, residual_range: float, eps: float = 1e-12) -> float:
    """Compute residual range divided by raw range."""
    return residual_range / (raw_range + eps)
