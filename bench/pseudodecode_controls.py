from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


ACCUMULATION_METRIC_SCHEMA_VERSION = "matched_path_accumulation_v1"
MATCHED_PATH_CONTROL_VERSION = "v1"


class ExecutionPath(str, Enum):
    STATIC = "static"
    PSEUDO = "pseudo"


@dataclass(frozen=True)
class MatchedMetricRecord:
    task_key: str
    trajectory_sha256: str
    config: str
    execution_mode: ExecutionPath
    matched_reference_mode: ExecutionPath
    checkpoint_generated_tokens: int
    absolute_sequence_position: int
    layer: str
    metric_name: str
    quantized_value: float
    fp16_reference_value: float
    degradation_value: float
    matched_reference_valid: bool
    source_commit: str
    schema_version: str = ACCUMULATION_METRIC_SCHEMA_VERSION


def matched_reference_valid(execution_mode: str, matched_reference_mode: str) -> bool:
    return ExecutionPath(execution_mode) == ExecutionPath(matched_reference_mode)


def ensure_matched_reference(execution_mode: str, matched_reference_mode: str) -> None:
    if not matched_reference_valid(execution_mode, matched_reference_mode):
        raise ValueError(f"cross-path reference fallback is forbidden: mode={execution_mode} reference={matched_reference_mode}")


def hidden_cosine_to_loss(hidden_cosine: float) -> float:
    return 1.0 - float(hidden_cosine)


def top1_agreement_to_disagreement(top1_agreement: bool | int | float) -> float:
    return 1.0 - float(bool(top1_agreement))


def clamp_divergence(value: float) -> float:
    return max(float(value), 0.0)


def compute_matched_degradation(
    *,
    metric_name: str,
    quantized_value: float | bool,
    fp16_reference_value: float | bool,
    execution_mode: str,
    matched_reference_mode: str,
) -> float:
    ensure_matched_reference(execution_mode, matched_reference_mode)
    if metric_name == "hidden_cosine":
        return hidden_cosine_to_loss(float(quantized_value))
    if metric_name == "top1_agreement":
        return top1_agreement_to_disagreement(bool(quantized_value))
    if metric_name in {"next_token_KL", "next_token_JS"}:
        return clamp_divergence(float(quantized_value))
    if metric_name.endswith("_delta") or metric_name.endswith("_error"):
        return float(quantized_value)
    return float(quantized_value)


def compute_accumulation_gap(*, pseudo_degradation: float, static_degradation: float) -> float:
    return float(pseudo_degradation) - float(static_degradation)


def validate_match_alignment(
    *,
    static_task_key: str,
    pseudo_task_key: str,
    static_trajectory_sha256: str,
    pseudo_trajectory_sha256: str,
    static_checkpoint: int,
    pseudo_checkpoint: int,
    static_next_token_id: int | None,
    pseudo_next_token_id: int | None,
    static_absolute_position: int,
    pseudo_absolute_position: int,
) -> bool:
    return all(
        [
            static_task_key == pseudo_task_key,
            static_trajectory_sha256 == pseudo_trajectory_sha256,
            int(static_checkpoint) == int(pseudo_checkpoint),
            static_next_token_id == pseudo_next_token_id,
            int(static_absolute_position) == int(pseudo_absolute_position),
        ]
    )


def path_baseline_not_double_subtracted(*, pseudo_degradation: float, static_degradation: float, execution_path_baseline: float | None = None) -> float:
    del execution_path_baseline
    return compute_accumulation_gap(pseudo_degradation=pseudo_degradation, static_degradation=static_degradation)


def row_dict(record: MatchedMetricRecord) -> dict[str, Any]:
    return {
        "task_key": record.task_key,
        "trajectory_sha256": record.trajectory_sha256,
        "config": record.config,
        "execution_mode": record.execution_mode.value,
        "matched_reference_mode": record.matched_reference_mode.value,
        "checkpoint_generated_tokens": record.checkpoint_generated_tokens,
        "absolute_sequence_position": record.absolute_sequence_position,
        "layer": record.layer,
        "metric_name": record.metric_name,
        "quantized_value": record.quantized_value,
        "fp16_reference_value": record.fp16_reference_value,
        "degradation_value": record.degradation_value,
        "matched_reference_valid": record.matched_reference_valid,
        "source_commit": record.source_commit,
        "schema_version": record.schema_version,
    }
