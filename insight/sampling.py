"""Deterministic token sampling helpers for PatternKV Insight."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def _stable_seed(metadata: dict[str, Any], layer_idx: int, kv_head: int, phase: str, window_idx: int | None, seed: int) -> int:
    payload = {
        "metadata": metadata,
        "layer_idx": int(layer_idx),
        "kv_head": int(kv_head),
        "phase": phase,
        "window_idx": window_idx,
        "seed": int(seed),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def sample_indices(
    total_tokens: int,
    sample_count: int,
    metadata: dict[str, Any],
    layer_idx: int,
    kv_head: int,
    phase: str,
    window_idx: int | None,
    seed: int,
) -> list[int]:
    """Return deterministic token indices without touching global RNG state.

    The selection always includes boundary/midpoint coverage when available,
    then fills the remaining slots using an independent local RNG.
    """
    if total_tokens <= 0 or sample_count <= 0:
        return []
    sample_count = min(int(sample_count), int(total_tokens))
    anchors = [0, total_tokens // 2, total_tokens - 1]
    if phase == "decode" and window_idx is not None:
        anchors.append(max(0, min(total_tokens - 1, int(window_idx) * 128)))
    chosen: list[int] = []
    for idx in anchors:
        if 0 <= idx < total_tokens and idx not in chosen:
            chosen.append(idx)
        if len(chosen) >= sample_count:
            return sorted(chosen)
    remaining = [idx for idx in range(total_tokens) if idx not in set(chosen)]
    rng = random.Random(_stable_seed(metadata, layer_idx, kv_head, phase, window_idx, seed))
    rng.shuffle(remaining)
    chosen.extend(remaining[: sample_count - len(chosen)])
    return sorted(chosen)
