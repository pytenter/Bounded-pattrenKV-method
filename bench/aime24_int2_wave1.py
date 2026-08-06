from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

from bench.aime_utils import effective_seed, task_key


CONFIGS: tuple[dict[str, Any], ...] = (
    {"gpu": 0, "config_name": "kivi_k2v2_s0_r128", "method": "kivi_official", "k_bits": 2, "v_bits": 2, "sink_length": 0, "recent_length": 128},
    {"gpu": 1, "config_name": "pattern_k2v2_s0_r128", "method": "patternkv", "k_bits": 2, "v_bits": 2, "sink_length": 0, "recent_length": 128},
    {"gpu": 2, "config_name": "kivi_k2v2_s64_r256", "method": "kivi_official", "k_bits": 2, "v_bits": 2, "sink_length": 64, "recent_length": 256},
    {"gpu": 3, "config_name": "pattern_k2v2_s64_r256", "method": "patternkv", "k_bits": 2, "v_bits": 2, "sink_length": 64, "recent_length": 256},
    {"gpu": 4, "config_name": "pattern_k4v2_s0_r128", "method": "patternkv", "k_bits": 4, "v_bits": 2, "sink_length": 0, "recent_length": 128},
    {"gpu": 5, "config_name": "pattern_k2v4_s0_r128", "method": "patternkv", "k_bits": 2, "v_bits": 4, "sink_length": 0, "recent_length": 128},
)

BLOCKED_WAVE1B_CONFIGS: tuple[dict[str, Any], ...] = (
    {"gpu": 6, "config_name": "pattern_magnitude_kmix_v2_s0_r128", "method": "patternkv", "k_bits": 2, "v_bits": 2, "sink_length": 0, "recent_length": 128, "mixed_key_mode": "magnitude", "status": "blocked_wave1b"},
    {"gpu": 7, "config_name": "pattern_queryaware_kmix_v2_s0_r128", "method": "patternkv", "k_bits": 2, "v_bits": 2, "sink_length": 0, "recent_length": 128, "mixed_key_mode": "query_aware", "status": "blocked_wave1b"},
)


def stable_hash(payload: Any, length: int = 16) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def task_key3(problem_id: int, sample_id: int, seed: int) -> str:
    return f"{task_key(problem_id, sample_id)}:seed{seed}"


def split_sink_quant_recent(total_tokens: int, sink_length: int, recent_length: int) -> dict[str, int]:
    if total_tokens < 0 or sink_length < 0 or recent_length < 0:
        raise ValueError("token lengths must be non-negative")
    sink = min(total_tokens, sink_length)
    remaining = max(total_tokens - sink, 0)
    recent = min(remaining, recent_length)
    quantized = max(total_tokens - sink - recent, 0)
    return {"sink_tokens": sink, "quantized_tokens": quantized, "recent_tokens": recent, "total_tokens": total_tokens}


def quantize_dequantize_per_channel(x: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        raise ValueError("bits must be positive")
    levels = float((1 << bits) - 1)
    mn = x.amin(dim=-2, keepdim=True)
    mx = x.amax(dim=-2, keepdim=True)
    scale = (mx - mn).clamp_min(1e-6) / levels
    q = torch.round((x - mn) / scale).clamp(0, levels)
    return q * scale + mn


def mixed_key_qk_reference(query: torch.Tensor, key: torch.Tensor, mask: torch.Tensor, int2_bits: int = 2, int4_bits: int = 4) -> torch.Tensor:
    if query.shape[-1] != key.shape[-1] or mask.shape[-1] != key.shape[-1]:
        raise ValueError("query, key and mask must share head_dim")
    mask_bool = mask.to(dtype=torch.bool, device=key.device)
    key_int4 = quantize_dequantize_per_channel(key[..., mask_bool], int4_bits)
    key_int2 = quantize_dequantize_per_channel(key[..., ~mask_bool], int2_bits)
    return torch.matmul(query[..., mask_bool], key_int4.transpose(-2, -1)) + torch.matmul(query[..., ~mask_bool], key_int2.transpose(-2, -1))


def kv_head_query_groups(num_attention_heads: int, num_key_value_heads: int) -> list[list[int]]:
    if num_attention_heads % num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    group = num_attention_heads // num_key_value_heads
    return [list(range(kv_head * group, (kv_head + 1) * group)) for kv_head in range(num_key_value_heads)]


def aggregate_query_importance(query_abs_mean: torch.Tensor, num_key_value_heads: int) -> torch.Tensor:
    if query_abs_mean.dim() != 3:
        raise ValueError("query_abs_mean must be [layers, query_heads, head_dim]")
    layers, query_heads, head_dim = query_abs_mean.shape
    groups = kv_head_query_groups(query_heads, num_key_value_heads)
    out = torch.empty(layers, num_key_value_heads, head_dim, dtype=query_abs_mean.dtype, device=query_abs_mean.device)
    for kv_head, query_group in enumerate(groups):
        out[:, kv_head, :] = query_abs_mean[:, query_group, :].mean(dim=1)
    return out


def make_channel_mask(scores: torch.Tensor, ratio: float = 0.125) -> torch.Tensor:
    if scores.dim() != 3:
        raise ValueError("scores must be [layers, kv_heads, head_dim]")
    if not 0 < ratio <= 1:
        raise ValueError("ratio must be in (0, 1]")
    keep = max(1, int(round(scores.shape[-1] * ratio)))
    idx = torch.topk(scores, keep, dim=-1, largest=True, sorted=True).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask


def mask_hash(mask: torch.Tensor) -> str:
    cpu = mask.detach().to(torch.bool).cpu().numpy().tobytes()
    return hashlib.sha256(cpu).hexdigest()[:16]


def jaccard_by_layer(mask_a: torch.Tensor, mask_b: torch.Tensor) -> list[dict[str, Any]]:
    if mask_a.shape != mask_b.shape:
        raise ValueError("mask shapes must match")
    rows = []
    for layer in range(mask_a.shape[0]):
        a = mask_a[layer].reshape(-1)
        b = mask_b[layer].reshape(-1)
        inter = int((a & b).sum().item())
        union = int((a | b).sum().item())
        rows.append(
            {
                "layer": layer,
                "overlap_count": inter,
                "jaccard": inter / union if union else 1.0,
                "query_aware_only": int((b & ~a).sum().item()),
                "magnitude_only": int((a & ~b).sum().item()),
            }
        )
    return rows


@dataclass(frozen=True)
class BitwidthConfig:
    method: str
    total_tokens: int
    sink_length: int
    recent_length: int
    k_bits: float
    v_bits: float
    group_size: int = 128
    pattern_centroids: int = 32
    head_dim: int = 128
    mixed_ratio: float = 0.0


def effective_bitwidth(cfg: BitwidthConfig) -> dict[str, float]:
    split = split_sink_quant_recent(cfg.total_tokens, cfg.sink_length, cfg.recent_length)
    quant = split["quantized_tokens"]
    fp16 = split["sink_tokens"] + split["recent_tokens"]
    denom = max(cfg.total_tokens, 1)
    avg_payload = ((quant * ((cfg.k_bits + cfg.v_bits) / 2.0)) + fp16 * 16.0) / denom
    metadata = 0.0 if quant == 0 else 32.0 / max(cfg.group_size, 1)
    pattern_centroid = 0.0
    pattern_assignment = 0.0
    if cfg.method.startswith("pattern"):
        pattern_centroid = (cfg.pattern_centroids * cfg.head_dim * 16.0) / max(quant * cfg.head_dim, 1)
        pattern_assignment = math.ceil(math.log2(max(cfg.pattern_centroids, 2))) / max(cfg.head_dim, 1)
    mixed_mask = cfg.mixed_ratio / max(denom, 1)
    return {
        "payload_bits_per_scalar": avg_payload,
        "metadata_bits_per_scalar": metadata * quant / denom,
        "sink_recent_overhead_bits_per_scalar": max(avg_payload - ((cfg.k_bits + cfg.v_bits) / 2.0), 0.0),
        "pattern_centroid_overhead_bits_per_scalar": pattern_centroid,
        "pattern_assignment_overhead_bits_per_scalar": pattern_assignment,
        "mixed_mask_overhead_bits_per_scalar": mixed_mask,
        "total_effective_bits_per_scalar": avg_payload + metadata * quant / denom + pattern_centroid + pattern_assignment + mixed_mask,
    }


def read_result_files(root: Path, method: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / method).glob("*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            rows.append({"path": str(path), "error": repr(exc), "stop_reason": "invalid_json"})
    return rows


def classify_failure(row: dict[str, Any]) -> str:
    if row.get("error"):
        return "runtime_error"
    if row.get("stop_reason") == "length" or row.get("length_truncated"):
        return "length_truncation"
    if row.get("parsed_answer") is None or row.get("parser_error"):
        return "parser_failure"
    text = str(row.get("generated_text") or "")
    words = text.split()
    if len(words) >= 80 and len(set(words[-80:])) < 25:
        return "repetition_loop"
    return "reasoning_error"


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize_config(rows: list[dict[str, Any]], planned: int) -> dict[str, Any]:
    parsed = [r for r in rows if r.get("parsed_answer") is not None and not r.get("error")]
    correct = [r for r in rows if r.get("is_correct") and not r.get("error")]
    generated = [int(r.get("generated_tokens") or 0) for r in rows]
    walls = [float(r.get("wall_time_seconds") or 0) for r in rows if r.get("wall_time_seconds") is not None]
    mem = [int(r.get("peak_memory_reserved_bytes") or 0) for r in rows if r.get("peak_memory_reserved_bytes")]
    return {
        "completed": len(rows),
        "valid_parsed": len(parsed),
        "correct": len(correct),
        "strict_accuracy": len(correct) / planned if planned else None,
        "valid_only_accuracy": len(correct) / len(parsed) if parsed else None,
        "parser_success_rate": len(parsed) / len(rows) if rows else None,
        "eos_count": sum(1 for r in rows if r.get("stop_reason") == "eos"),
        "length_stop_count": sum(1 for r in rows if r.get("stop_reason") == "length"),
        "error_oom_count": sum(1 for r in rows if r.get("error") or r.get("stop_reason") == "oom"),
        "average_generated_tokens": statistics.mean(generated) if generated else None,
        "median_generated_tokens": statistics.median(generated) if generated else None,
        "p95_generated_tokens": sorted(generated)[max(0, math.ceil(0.95 * len(generated)) - 1)] if generated else None,
        "average_wall_time": statistics.mean(walls) if walls else None,
        "peak_memory": max(mem) if mem else None,
    }
