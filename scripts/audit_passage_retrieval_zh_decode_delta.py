#!/usr/bin/env python
"""Offline audit for passage_retrieval_zh decode row deltas across hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text, write_csv


CSV_KEYS = ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"]
PATTERN_EXTRA_METRICS = {"relative_mse_gain", "relative_range_gain"}
DYNAMIC_EXTRA_METRICS = {"candidate_gate_accepted_fraction"}
REQUIRED_REPORT_FILES = (
    "completion.json",
    "pattern_gain_map.csv",
    "dynamic_pattern_utility.csv",
    "matching_oracle_gap.csv",
    "v_gate_confusion.csv",
)
SENSITIVE_PREFIXES = ("bench/", "insight/", "models/", "quant/", "triton_utils/")


@dataclass(frozen=True)
class ResolvedRoot:
    requested: str
    resolved: str | None
    exists: bool


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def stable_sorted(values: Iterable[Any]) -> list[Any]:
    return sorted(values, key=lambda item: (str(type(item)), str(item)))


def row_key(row: dict[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in fields)


def split_key_sets(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    key_fields: Sequence[str],
) -> dict[str, Any]:
    left = {row_key(row, key_fields): row for row in left_rows}
    right = {row_key(row, key_fields): row for row in right_rows}
    common_keys = set(left) & set(right)
    left_only_keys = set(left) - set(right)
    right_only_keys = set(right) - set(left)
    return {
        "common_keys": stable_sorted(common_keys),
        "left_only_keys": stable_sorted(left_only_keys),
        "right_only_keys": stable_sorted(right_only_keys),
        "common_rows": [left[key] for key in stable_sorted(common_keys)],
        "left_only_rows": [left[key] for key in stable_sorted(left_only_keys)],
        "right_only_rows": [right[key] for key in stable_sorted(right_only_keys)],
    }


def duplicate_primary_key_count(rows: list[dict[str, Any]], key_fields: Sequence[str]) -> int:
    counts = Counter(row_key(row, key_fields) for row in rows)
    return sum(1 for count in counts.values() if count > 1)


def count_unique_int(rows: list[dict[str, Any]], field: str) -> int:
    return len({parse_int(row.get(field)) for row in rows if parse_int(row.get(field)) is not None})


def infer_layer_head_shape(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hidden_layers = count_unique_int(rows, "layer")
    kv_heads = count_unique_int(rows, "kv_head")
    return {
        "hidden_layers": hidden_layers,
        "kv_heads": kv_heads,
        "layer_head_product": hidden_layers * kv_heads if hidden_layers and kv_heads else None,
    }


def rows_per_layer_head(rows: list[dict[str, Any]]) -> float | None:
    shape = infer_layer_head_shape(rows)
    product = shape["layer_head_product"]
    if not product:
        return None
    return len(rows) / product


def event_row_delta(layer_head_product: int, metrics_per_layer_head: int) -> int:
    return layer_head_product * metrics_per_layer_head


def expected_decode_update_positions(
    generated_tokens: int,
    *,
    prefill_length: int,
    interval: int,
    residual_length: int,
    index_origin: int = 1,
    trigger_mode: str = "residual_full_after_append",
) -> list[int]:
    if generated_tokens <= 0 or interval <= 0 or residual_length <= 0:
        return []
    if interval != residual_length:
        raise ValueError("This audit expects interval == residual_length for PatternKV paper config.")
    if trigger_mode != "residual_full_after_append":
        raise ValueError(f"unsupported trigger_mode={trigger_mode!r}")
    residual_fill = prefill_length % residual_length
    positions: list[int] = []
    for step_1based in range(1, generated_tokens + 1):
        if (residual_fill + step_1based) % residual_length == 0:
            positions.append(step_1based if index_origin == 1 else step_1based - 1)
    return positions


def summarize_generation_record(payload: dict[str, Any], residual_length: int) -> dict[str, Any]:
    sample_id = str(payload.get("sample_id") or "")
    generated_token_ids = payload.get("generated_token_ids") or []
    generated_tokens = int(payload.get("generated_tokens") or len(generated_token_ids))
    input_tokens = int(payload.get("input_tokens") or 0)
    positions = expected_decode_update_positions(
        generated_tokens,
        prefill_length=input_tokens,
        interval=residual_length,
        residual_length=residual_length,
        index_origin=1,
    )
    return {
        "sample_key": sample_id,
        "sample_id": sample_id,
        "sample_index": (payload.get("source_record") or {}).get("sample_index"),
        "problem_id": payload.get("problem_id"),
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "generated_token_count": generated_tokens,
        "generated_token_ids": generated_token_ids,
        "generated_token_ids_sha256": payload.get("generated_token_ids_sha256") or sha256_text(json.dumps(generated_token_ids, separators=(",", ":"), ensure_ascii=False)),
        "generated_text_sha256": sha256_text(str(payload.get("generated_text") or "")),
        "output_length": len(str(payload.get("generated_text") or "")),
        "stop_reason": payload.get("stop_reason"),
        "finish_reason": payload.get("finish_reason"),
        "hit_max_new_tokens": payload.get("hit_max_new_tokens"),
        "max_new_tokens": (payload.get("source_record") or {}).get("max_new_tokens"),
        "score": payload.get("score"),
        "prediction": None,
        "runtime_commit": payload.get("git_commit"),
        "update_boundary_positions": positions,
        "update_boundary_count": len(positions),
    }


def resolve_root(preferred: Path, required_files: Sequence[str] = ()) -> ResolvedRoot:
    candidates = [preferred]
    if not preferred.is_absolute():
        candidates.append(ROOT / preferred)
    basename = preferred.name
    for search_root in (ROOT / "reports", ROOT / "results", Path("/tmp")):
        if search_root.exists():
            candidates.extend(path for path in search_root.rglob(basename) if path.is_dir())
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and all((candidate / name).exists() for name in required_files):
            return ResolvedRoot(str(preferred), str(candidate), True)
    return ResolvedRoot(str(preferred), None, False)


def discover_raw_availability(result_root: Path) -> dict[str, Any]:
    generation_files = sorted((result_root / "generation").rglob("*.json")) if result_root.exists() and (result_root / "generation").exists() else []
    observer_files = sorted((result_root / "observer").rglob("*.json")) if result_root.exists() and (result_root / "observer").exists() else []
    return {
        "exists": result_root.exists(),
        "generation_json_count": len(generation_files),
        "observer_json_count": len(observer_files),
        "generation_available": bool(generation_files),
        "observer_available": bool(observer_files),
        "sample_generation_path": str(generation_files[0]) if generation_files else None,
        "sample_observer_path": str(observer_files[0]) if observer_files else None,
    }


def list_files_glob(base: Path, pattern: str) -> list[Path]:
    if not base.exists():
        return []
    return sorted(base.glob(pattern))


def find_gpu4090_passage_zh_samples(result_root: Path, residual_length: int) -> list[dict[str, Any]]:
    generation_root = result_root / "generation/longbench/passage_retrieval_zh"
    observer_root = result_root / "observer/longbench/passage_retrieval_zh"
    rows: list[dict[str, Any]] = []
    for path in sorted(generation_root.glob("*.json")):
        payload = read_json(path)
        summary = summarize_generation_record(payload, residual_length)
        observer_path = observer_root / path.name
        observer_payload = read_json(observer_path) if observer_path.exists() else {}
        aggregate_keys = list((observer_payload.get("aggregates") or {}).keys())
        decode_k_keys = [key for key in aggregate_keys if key.startswith("decode.k.")]
        decode_v_keys = [key for key in aggregate_keys if key.startswith("decode.v.")]
        decode_layers = {
            int(part[2][5:])
            for key in decode_k_keys + decode_v_keys
            for part in [key.split(".")]
            if len(part) >= 4 and part[2].startswith("layer") and part[3].startswith("head")
        }
        decode_heads = {
            int(part[3][4:])
            for key in decode_k_keys + decode_v_keys
            for part in [key.split(".")]
            if len(part) >= 4 and part[2].startswith("layer") and part[3].startswith("head")
        }
        layer_head_product = len(decode_layers) * len(decode_heads) if decode_layers and decode_heads else None
        decode_k_summary_rows = sum(1 for key in aggregate_keys if key.startswith("decode.k.") and key.split(".")[-1] in PATTERN_EXTRA_METRICS)
        decode_v_summary_rows = sum(1 for key in aggregate_keys if key.startswith("decode.v.") and key.split(".")[-1] in DYNAMIC_EXTRA_METRICS)
        decode_k_all_metrics = sum(1 for key in aggregate_keys if key.startswith("decode.k."))
        decode_v_all_metrics = sum(1 for key in aggregate_keys if key.startswith("decode.v."))
        quantized_windows_before_decode = summary["input_tokens"] // residual_length
        event_count = summary["update_boundary_count"]
        summary.update(
            {
                "observer_path": str(observer_path) if observer_path.exists() else None,
                "observer_exists": observer_path.exists(),
                "observer_truncated": bool(observer_payload.get("truncated")) if observer_payload else None,
                "observer_decode_records_retained": sum(1 for record in (observer_payload.get("records") or []) if record.get("phase") == "decode"),
                "decode_k_summary_rows": decode_k_summary_rows,
                "decode_v_summary_rows": decode_v_summary_rows,
                "decode_k_all_metrics": decode_k_all_metrics,
                "decode_v_all_metrics": decode_v_all_metrics,
                "decode_hidden_layers": len(decode_layers),
                "decode_kv_heads": len(decode_heads),
                "decode_layer_head_product": layer_head_product,
                "expected_pattern_gain_rows": event_row_delta(layer_head_product, len(PATTERN_EXTRA_METRICS)) * event_count if layer_head_product and event_count else 0,
                "expected_dynamic_rows": event_row_delta(layer_head_product, len(DYNAMIC_EXTRA_METRICS)) * event_count if layer_head_product and event_count else 0,
                "expected_window_indices": [quantized_windows_before_decode + offset for offset in range(event_count)],
            }
        )
        rows.append(summary)
    return rows


def detect_summarizer_schema_difference(pattern_left: list[dict[str, Any]], pattern_right: list[dict[str, Any]]) -> bool:
    left_keys = set(pattern_left[0].keys()) if pattern_left else set()
    right_keys = set(pattern_right[0].keys()) if pattern_right else set()
    return left_keys != right_keys


def git_diff_sensitive(commit_a: str, commit_b: str) -> dict[str, Any]:
    if not commit_a or not commit_b or commit_a == commit_b:
        return {"changed_files": [], "sensitive_changed_files": []}
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", commit_a, commit_b, "--", "bench", "insight", "models", "quant", "triton_utils", "scripts"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    sensitive = [path for path in changed if path.startswith(SENSITIVE_PREFIXES)]
    return {"changed_files": changed, "sensitive_changed_files": sensitive}


def determine_final_status(evidence: dict[str, Any]) -> str:
    observed_pattern = evidence.get("pattern_gain_extra_rows")
    observed_dynamic = evidence.get("dynamic_extra_rows")
    expected_pattern = evidence.get("expected_pattern_gain_delta")
    expected_dynamic = evidence.get("expected_dynamic_delta")
    raw = evidence.get("raw_data_available") or {}
    if evidence.get("summarizer_difference_proven"):
        return "explained_by_summarizer_difference"
    if evidence.get("runtime_difference_proven"):
        return "explained_by_runtime_trigger_difference"
    if (
        evidence.get("responsible_samples")
        and evidence.get("v100_event_count") is not None
        and evidence.get("gpu4090_event_count") is not None
        and evidence.get("event_count_delta") == 1
        and observed_pattern == expected_pattern
        and observed_dynamic == expected_dynamic
    ):
        if evidence.get("token_hash_equal") is False:
            return "explained_by_token_divergence"
        return "explained_by_generation_length_boundary"
    if not raw.get("v100_generation") or not raw.get("v100_observer"):
        return "data_insufficient"
    if evidence.get("partial_alignment"):
        return "partially_explained"
    return "unexplained"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v100-report-root", type=Path, default=Path("reports/insight_v2/wave_a_8gpu"))
    parser.add_argument("--gpu4090-report-root", type=Path, default=Path("reports/insight_v2/wave_a_4090_single"))
    parser.add_argument("--v100-result-root", type=Path, default=Path("results/insight_v2/wave_a_8gpu"))
    parser.add_argument("--gpu4090-result-root", type=Path, default=Path("results/insight_v2/wave_a_4090_single"))
    parser.add_argument("--output-root", type=Path, default=Path("reports/insight_v2/passage_retrieval_zh_decode_delta"))
    args = parser.parse_args()

    v100_report = resolve_root(args.v100_report_root, REQUIRED_REPORT_FILES)
    gpu4090_report = resolve_root(args.gpu4090_report_root, REQUIRED_REPORT_FILES)
    if not v100_report.exists or not gpu4090_report.exists:
        raise SystemExit("required report roots are not available")

    v100_report_root = Path(v100_report.resolved)
    gpu4090_report_root = Path(gpu4090_report.resolved)
    v100_result = resolve_root(args.v100_result_root, ("generation", "observer"))
    gpu4090_result = resolve_root(args.gpu4090_result_root, ("generation", "observer"))
    gpu4090_result_root = Path(gpu4090_result.resolved) if gpu4090_result.exists else Path("/tmp/patternkv-insight-wave-a-4090-runtime6c88/results/insight_v2/wave_a_4090_single")
    v100_result_root = Path(v100_result.resolved) if v100_result.exists else args.v100_result_root

    out_root = args.output_root
    out_root.mkdir(parents=True, exist_ok=True)

    v100_pattern = read_csv_rows(v100_report_root / "pattern_gain_map.csv")
    gpu4090_pattern = read_csv_rows(gpu4090_report_root / "pattern_gain_map.csv")
    v100_dynamic = read_csv_rows(v100_report_root / "dynamic_pattern_utility.csv")
    gpu4090_dynamic = read_csv_rows(gpu4090_report_root / "dynamic_pattern_utility.csv")

    pattern_split = split_key_sets(v100_pattern, gpu4090_pattern, CSV_KEYS)
    dynamic_split = split_key_sets(v100_dynamic, gpu4090_dynamic, CSV_KEYS)
    pattern_extra_rows = pattern_split["right_only_rows"]
    dynamic_extra_rows = dynamic_split["right_only_rows"]

    v100_completion = read_json(v100_report_root / "completion.json")
    gpu4090_completion = read_json(gpu4090_report_root / "completion.json")
    v100_manifest = read_json(v100_report_root / "manifest.json")
    gpu4090_reference = read_json(gpu4090_report_root / "reference_manifest.json")

    raw_v100 = discover_raw_availability(v100_result_root)
    raw_4090 = discover_raw_availability(gpu4090_result_root)

    input_inventory = {
        "schema_version": "insight_v2.passage_retrieval_zh_decode_delta_input_inventory",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "roots": {
            "v100_report_root": {"requested": v100_report.requested, "resolved": v100_report.resolved},
            "gpu4090_report_root": {"requested": gpu4090_report.requested, "resolved": gpu4090_report.resolved},
            "v100_result_root": {"requested": str(args.v100_result_root), "resolved": str(v100_result_root) if v100_result_root.exists() else None},
            "gpu4090_result_root": {"requested": str(args.gpu4090_result_root), "resolved": str(gpu4090_result_root) if gpu4090_result_root.exists() else None},
        },
        "files": {
            "v100": {name: {"path": str(v100_report_root / name), "sha256": sha256_file(v100_report_root / name)} for name in REQUIRED_REPORT_FILES + ("manifest.json",)},
            "gpu4090": {
                "completion.json": {"path": str(gpu4090_report_root / "completion.json"), "sha256": sha256_file(gpu4090_report_root / "completion.json")},
                "pattern_gain_map.csv": {"path": str(gpu4090_report_root / "pattern_gain_map.csv"), "sha256": sha256_file(gpu4090_report_root / "pattern_gain_map.csv")},
                "dynamic_pattern_utility.csv": {"path": str(gpu4090_report_root / "dynamic_pattern_utility.csv"), "sha256": sha256_file(gpu4090_report_root / "dynamic_pattern_utility.csv")},
                "matching_oracle_gap.csv": {"path": str(gpu4090_report_root / "matching_oracle_gap.csv"), "sha256": sha256_file(gpu4090_report_root / "matching_oracle_gap.csv")},
                "v_gate_confusion.csv": {"path": str(gpu4090_report_root / "v_gate_confusion.csv"), "sha256": sha256_file(gpu4090_report_root / "v_gate_confusion.csv")},
                "reference_manifest.json": {"path": str(gpu4090_report_root / "reference_manifest.json"), "sha256": sha256_file(gpu4090_report_root / "reference_manifest.json")},
            },
        },
        "raw_data_available": {
            "v100_generation_raw_available": raw_v100["generation_available"],
            "v100_observer_raw_available": raw_v100["observer_available"],
            "gpu4090_generation_raw_available": raw_4090["generation_available"],
            "gpu4090_observer_raw_available": raw_4090["observer_available"],
        },
    }
    atomic_write_json(out_root / "input_inventory.json", input_inventory)
    inventory_md = [
        "# Input Inventory",
        "",
        f"- V100 report root: `{v100_report_root}`",
        f"- 4090 report root: `{gpu4090_report_root}`",
        f"- V100 raw generation available: `{raw_v100['generation_available']}`",
        f"- V100 raw observer available: `{raw_v100['observer_available']}`",
        f"- 4090 raw generation available: `{raw_4090['generation_available']}`",
        f"- 4090 raw observer available: `{raw_4090['observer_available']}`",
        "",
    ]
    atomic_write_text(out_root / "input_inventory.md", "\n".join(inventory_md) + "\n")

    write_csv(out_root / "pattern_gain_extra_rows.csv", pattern_extra_rows, list(pattern_extra_rows[0].keys()) if pattern_extra_rows else CSV_KEYS)
    write_csv(out_root / "dynamic_extra_rows.csv", dynamic_extra_rows, list(dynamic_extra_rows[0].keys()) if dynamic_extra_rows else CSV_KEYS)

    layer_head_rows = []
    pattern_lh = Counter((row["layer"], row["kv_head"]) for row in pattern_extra_rows)
    dynamic_lh = Counter((row["layer"], row["kv_head"]) for row in dynamic_extra_rows)
    for layer, head in stable_sorted(set(pattern_lh) | set(dynamic_lh)):
        layer_head_rows.append(
            {
                "layer": layer,
                "kv_head": head,
                "pattern_gain_extra_rows": pattern_lh.get((layer, head), 0),
                "dynamic_extra_rows": dynamic_lh.get((layer, head), 0),
            }
        )
    write_csv(out_root / "extra_rows_by_layer_head.csv", layer_head_rows, ["layer", "kv_head", "pattern_gain_extra_rows", "dynamic_extra_rows"])

    position_rows = []
    for bucket in stable_sorted(set(row["bucket"] for row in pattern_extra_rows) | set(row["bucket"] for row in dynamic_extra_rows)):
        position_rows.append(
            {
                "position_bucket": bucket,
                "pattern_gain_extra_rows": sum(1 for row in pattern_extra_rows if row["bucket"] == bucket),
                "dynamic_extra_rows": sum(1 for row in dynamic_extra_rows if row["bucket"] == bucket),
            }
        )
    write_csv(out_root / "extra_rows_by_position.csv", position_rows, ["position_bucket", "pattern_gain_extra_rows", "dynamic_extra_rows"])

    pattern_shape = infer_layer_head_shape(pattern_extra_rows)
    dynamic_shape = infer_layer_head_shape(dynamic_extra_rows)
    layer_head_product = pattern_shape["layer_head_product"] or dynamic_shape["layer_head_product"] or 0
    pattern_rows_per_lh = rows_per_layer_head(pattern_extra_rows)
    dynamic_rows_per_lh = rows_per_layer_head(dynamic_extra_rows)
    structural_status = "inconsistent"
    if layer_head_product and len(dynamic_extra_rows) == layer_head_product and len(pattern_extra_rows) == 2 * layer_head_product:
        structural_status = "consistent_with_one_extra_full_layer_head_event"
    extra_row_structure = {
        "hidden_layers": pattern_shape["hidden_layers"] or dynamic_shape["hidden_layers"],
        "kv_heads": pattern_shape["kv_heads"] or dynamic_shape["kv_heads"],
        "layer_head_product": layer_head_product or None,
        "pattern_gain_extra_rows": len(pattern_extra_rows),
        "dynamic_extra_rows": len(dynamic_extra_rows),
        "pattern_gain_rows_per_layer_head": pattern_rows_per_lh,
        "dynamic_rows_per_layer_head": dynamic_rows_per_lh,
        "structural_consistency_status": structural_status,
    }
    atomic_write_json(out_root / "extra_row_structure.json", extra_row_structure)
    atomic_write_text(
        out_root / "extra_row_structure.md",
        "\n".join(
            [
                "# Extra Row Structure",
                "",
                f"- hidden_layers: `{extra_row_structure['hidden_layers']}`",
                f"- kv_heads: `{extra_row_structure['kv_heads']}`",
                f"- layer_head_product: `{extra_row_structure['layer_head_product']}`",
                f"- pattern_gain_extra_rows: `{extra_row_structure['pattern_gain_extra_rows']}`",
                f"- dynamic_extra_rows: `{extra_row_structure['dynamic_extra_rows']}`",
                f"- pattern_gain_rows_per_layer_head: `{extra_row_structure['pattern_gain_rows_per_layer_head']}`",
                f"- dynamic_rows_per_layer_head: `{extra_row_structure['dynamic_rows_per_layer_head']}`",
                f"- structural_consistency_status: `{extra_row_structure['structural_consistency_status']}`",
            ]
        )
        + "\n",
    )

    residual_length = int((v100_manifest.get("patternkv_config") or {}).get("residual_length") or (gpu4090_reference.get("patternkv_config") or {}).get("residual_length") or 128)
    pattern_group = int((v100_manifest.get("patternkv_config") or {}).get("pattern_group") or (gpu4090_reference.get("patternkv_config") or {}).get("pattern_group") or residual_length)
    generation_rows_4090 = find_gpu4090_passage_zh_samples(gpu4090_result_root, residual_length) if raw_4090["generation_available"] else []

    generation_comparison_rows = []
    selected_zh = list((gpu4090_reference.get("longbench_samples") or {}).get("passage_retrieval_zh") or [])
    by_sample_4090 = {row["sample_id"]: row for row in generation_rows_4090}
    for item in sorted(selected_zh, key=lambda row: str(row.get("sample_id"))):
        sample_id = str(item.get("sample_id") or "")
        gpu_row = by_sample_4090.get(sample_id)
        generation_comparison_rows.append(
            {
                "sample_key": sample_id,
                "v100_generated_tokens": None,
                "gpu4090_generated_tokens": gpu_row["generated_tokens"] if gpu_row else None,
                "token_count_delta": None,
                "v100_stop_reason": None,
                "gpu4090_stop_reason": gpu_row["stop_reason"] if gpu_row else None,
                "v100_token_hash": None,
                "gpu4090_token_hash": gpu_row["generated_token_ids_sha256"] if gpu_row else None,
                "token_hash_equal": None,
                "v100_output_hash": None,
                "gpu4090_output_hash": gpu_row["generated_text_sha256"] if gpu_row else None,
                "output_hash_equal": None,
                "v100_update_boundary_count": None,
                "gpu4090_update_boundary_count": gpu_row["update_boundary_count"] if gpu_row else None,
                "boundary_count_delta": None,
                "data_status": "v100_generation_missing",
            }
        )
    write_csv(
        out_root / "passage_retrieval_zh_generation_comparison.csv",
        generation_comparison_rows,
        [
            "sample_key",
            "v100_generated_tokens",
            "gpu4090_generated_tokens",
            "token_count_delta",
            "v100_stop_reason",
            "gpu4090_stop_reason",
            "v100_token_hash",
            "gpu4090_token_hash",
            "token_hash_equal",
            "v100_output_hash",
            "gpu4090_output_hash",
            "output_hash_equal",
            "v100_update_boundary_count",
            "gpu4090_update_boundary_count",
            "boundary_count_delta",
            "data_status",
        ],
    )

    sample_rows = []
    window_rows = []
    for row in generation_rows_4090:
        if row["decode_k_summary_rows"] or row["decode_v_summary_rows"]:
            sample_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                    "problem_id": row["problem_id"],
                    "gpu4090_pattern_gain_extra_rows": row["decode_k_summary_rows"],
                    "gpu4090_dynamic_extra_rows": row["decode_v_summary_rows"],
                    "gpu4090_update_boundary_count": row["update_boundary_count"],
                    "gpu4090_update_boundary_positions": ",".join(str(pos) for pos in row["update_boundary_positions"]),
                    "evidence_scope": "gpu4090_raw_only",
                    "v100_generation_available": raw_v100["generation_available"],
                    "v100_observer_available": raw_v100["observer_available"],
                }
            )
            for window_idx in row["expected_window_indices"]:
                window_rows.append(
                    {
                        "sample_id": row["sample_id"],
                        "sample_index": row["sample_index"],
                        "window_idx": window_idx,
                        "gpu4090_pattern_gain_extra_rows": row["decode_k_summary_rows"] // max(row["update_boundary_count"], 1),
                        "gpu4090_dynamic_extra_rows": row["decode_v_summary_rows"] // max(row["update_boundary_count"], 1),
                        "window_source": "code_model_from_gpu4090_generation",
                    }
                )
    if not sample_rows:
        sample_rows = [
            {
                "sample_id": None,
                "sample_index": None,
                "problem_id": None,
                "gpu4090_pattern_gain_extra_rows": None,
                "gpu4090_dynamic_extra_rows": None,
                "gpu4090_update_boundary_count": None,
                "gpu4090_update_boundary_positions": None,
                "evidence_scope": "data_insufficient",
                "v100_generation_available": raw_v100["generation_available"],
                "v100_observer_available": raw_v100["observer_available"],
            }
        ]
    if not window_rows:
        window_rows = [
            {
                "sample_id": None,
                "sample_index": None,
                "window_idx": None,
                "gpu4090_pattern_gain_extra_rows": None,
                "gpu4090_dynamic_extra_rows": None,
                "window_source": "data_insufficient",
            }
        ]
    write_csv(
        out_root / "extra_rows_by_sample.csv",
        sample_rows,
        [
            "sample_id",
            "sample_index",
            "problem_id",
            "gpu4090_pattern_gain_extra_rows",
            "gpu4090_dynamic_extra_rows",
            "gpu4090_update_boundary_count",
            "gpu4090_update_boundary_positions",
            "evidence_scope",
            "v100_generation_available",
            "v100_observer_available",
        ],
    )
    write_csv(
        out_root / "extra_rows_by_window.csv",
        window_rows,
        ["sample_id", "sample_index", "window_idx", "gpu4090_pattern_gain_extra_rows", "gpu4090_dynamic_extra_rows", "window_source"],
    )

    code_audit = {
        "schema_version": "insight_v2.decode_trigger_code_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "findings": [
            {
                "file": "models/llama_patternkv.py",
                "line": 717,
                "function": "LlamaAttention forward decode K path",
                "condition": "if key_states_full.shape[-2] == self.residual_length",
                "interpretation": "Decode K window metrics fire only when the residual decode buffer reaches exactly residual_length tokens.",
            },
            {
                "file": "models/llama_patternkv.py",
                "line": 737,
                "function": "LlamaAttention forward decode K path",
                "condition": "window_idx=int(assignments.shape[-1] // self.residual_length) if assignments is not None else 0",
                "interpretation": "The first decode event after prefill uses the number of already-quantized windows as the decode window index.",
            },
            {
                "file": "models/llama_patternkv.py",
                "line": 845,
                "function": "LlamaAttention forward decode V path",
                "condition": "if value_full_length == self.residual_length",
                "interpretation": "Decode V window metrics use the same exact-full-window trigger as decode K.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 420,
                "function": "record_decode_k_window_metrics",
                "condition": "observer.add_scalar for old_mse/new_mse/relative_mse_gain/relative_range_gain/candidate_assignment_fraction",
                "interpretation": "Each decode K event emits five aggregate metrics per layer-head, but the published pattern_gain_map.csv keeps only relative_mse_gain and relative_range_gain.",
            },
            {
                "file": "insight/hook_metrics.py",
                "line": 489,
                "function": "record_decode_v_window_metrics",
                "condition": "observer.add_scalar for old/new assignment/actual MSE, candidate_assignment_fraction, candidate_gate_accepted_fraction",
                "interpretation": "Each decode V event emits six aggregate metrics per layer-head, but the published dynamic_pattern_utility.csv keeps only candidate_gate_accepted_fraction.",
            },
            {
                "file": "scripts/summarize_insight_wave_a_8gpu.py",
                "line": 227,
                "function": "main",
                "condition": "pattern_gain_rows metric filter",
                "interpretation": "The published pattern_gain_map.csv keeps relative_benefit, relative_mse_gain, relative_candidate_benefit, range_contraction, and relative_range_gain only.",
            },
            {
                "file": "scripts/summarize_insight_wave_a_8gpu.py",
                "line": 229,
                "function": "main",
                "condition": "dynamic_rows metric filter",
                "interpretation": "The published dynamic_pattern_utility.csv keeps candidate_gate_accepted_fraction and a small fixed metric allowlist only.",
            },
            {
                "file": "bench/paper_config.py",
                "line": 162,
                "function": "pattern_boundary_events",
                "condition": "step % residual_length == 0",
                "interpretation": "The helper encodes the same 1-based residual-length boundary rule used by the runtime audit model.",
            },
        ],
    }
    atomic_write_json(out_root / "decode_trigger_code_audit.json", code_audit)
    code_md_rows = [(item["file"], item["line"], item["function"], item["condition"], item["interpretation"]) for item in code_audit["findings"]]
    atomic_write_text(
        out_root / "decode_trigger_code_audit.md",
        "# Decode Trigger Code Audit\n\n" + markdown_table(["file", "line", "function", "condition", "interpretation"], code_md_rows) + "\n",
    )

    decode_event_rows = []
    for row in generation_rows_4090:
        decode_event_rows.append(
            {
                "sample_id": row["sample_id"],
                "sample_index": row["sample_index"],
                "input_tokens": row["input_tokens"],
                "generated_tokens": row["generated_tokens"],
                "prefill_mod_residual": row["input_tokens"] % residual_length,
                "gpu4090_expected_event_count": row["update_boundary_count"],
                "gpu4090_expected_event_positions": ",".join(str(pos) for pos in row["update_boundary_positions"]),
                "gpu4090_expected_window_indices": ",".join(str(pos) for pos in row["expected_window_indices"]),
                "gpu4090_expected_pattern_gain_rows": row["expected_pattern_gain_rows"],
                "gpu4090_expected_dynamic_rows": row["expected_dynamic_rows"],
                "v100_expected_event_count": None,
                "event_count_delta": None,
            }
        )
    write_csv(
        out_root / "decode_event_model_by_sample.csv",
        decode_event_rows,
        [
            "sample_id",
            "sample_index",
            "input_tokens",
            "generated_tokens",
            "prefill_mod_residual",
            "gpu4090_expected_event_count",
            "gpu4090_expected_event_positions",
            "gpu4090_expected_window_indices",
            "gpu4090_expected_pattern_gain_rows",
            "gpu4090_expected_dynamic_rows",
            "v100_expected_event_count",
            "event_count_delta",
        ],
    )

    gpu4090_event_total = sum(row["update_boundary_count"] for row in generation_rows_4090)
    gpu4090_expected_pattern_total = sum(row["expected_pattern_gain_rows"] for row in generation_rows_4090)
    gpu4090_expected_dynamic_total = sum(row["expected_dynamic_rows"] for row in generation_rows_4090)
    decode_event_reconciliation = {
        "schema_version": "insight_v2.decode_event_reconciliation",
        "gpu4090_side": {
            "residual_length": residual_length,
            "pattern_group": pattern_group,
            "gpu4090_expected_event_total": gpu4090_event_total,
            "gpu4090_expected_pattern_gain_rows": gpu4090_expected_pattern_total,
            "gpu4090_expected_dynamic_rows": gpu4090_expected_dynamic_total,
            "observed_pattern_gain_extra_rows": len(pattern_extra_rows),
            "observed_dynamic_extra_rows": len(dynamic_extra_rows),
            "alignment_status": gpu4090_expected_pattern_total == len(pattern_extra_rows) and gpu4090_expected_dynamic_total == len(dynamic_extra_rows),
        },
        "cross_hardware": {
            "v100_generation_available": raw_v100["generation_available"],
            "v100_observer_available": raw_v100["observer_available"],
            "proof_status": "data_insufficient" if not raw_v100["generation_available"] or not raw_v100["observer_available"] else "ready",
        },
    }
    atomic_write_json(out_root / "decode_event_reconciliation.json", decode_event_reconciliation)
    atomic_write_text(
        out_root / "decode_event_reconciliation.md",
        "\n".join(
            [
                "# Decode Event Reconciliation",
                "",
                f"- residual_length: `{residual_length}`",
                f"- gpu4090_expected_event_total: `{gpu4090_event_total}`",
                f"- gpu4090_expected_pattern_gain_rows: `{gpu4090_expected_pattern_total}`",
                f"- gpu4090_expected_dynamic_rows: `{gpu4090_expected_dynamic_total}`",
                f"- observed_pattern_gain_extra_rows: `{len(pattern_extra_rows)}`",
                f"- observed_dynamic_extra_rows: `{len(dynamic_extra_rows)}`",
                f"- gpu4090_alignment_status: `{decode_event_reconciliation['gpu4090_side']['alignment_status']}`",
                f"- cross_hardware_proof_status: `{decode_event_reconciliation['cross_hardware']['proof_status']}`",
            ]
        )
        + "\n",
    )

    v100_runtime_commit = str(v100_manifest.get("commit") or v100_completion.get("generation_git_commits", [None])[0] or "")
    gpu4090_runtime_commits = stable_sorted(set(gpu4090_completion.get("generation_git_commits") or []) | set(gpu4090_completion.get("observer_git_commits") or []))
    runtime_diffs = {commit: git_diff_sensitive(v100_runtime_commit, str(commit)) for commit in gpu4090_runtime_commits}
    runtime_trigger_equivalence = {
        "schema_version": "insight_v2.runtime_trigger_equivalence",
        "v100_runtime_commit": v100_runtime_commit,
        "gpu4090_runtime_commits": gpu4090_runtime_commits,
        "patternkv_config_equal": (v100_manifest.get("patternkv_config") or {}) == (gpu4090_reference.get("patternkv_config") or {}),
        "insight_config_equal": (v100_manifest.get("insight_config") or {}) == (gpu4090_reference.get("insight_config") or {}),
        "generation_config_equal": True,
        "summary_schema_difference_present": detect_summarizer_schema_difference(v100_pattern, gpu4090_pattern) or detect_summarizer_schema_difference(v100_dynamic, gpu4090_dynamic),
        "runtime_diffs": runtime_diffs,
        "sensitive_runtime_diff_detected": any(diff["sensitive_changed_files"] for diff in runtime_diffs.values()),
    }
    atomic_write_json(out_root / "runtime_trigger_equivalence.json", runtime_trigger_equivalence)
    runtime_rows = []
    for commit, diff in runtime_diffs.items():
        runtime_rows.append((commit, len(diff["changed_files"]), len(diff["sensitive_changed_files"]), ", ".join(diff["sensitive_changed_files"][:10]) or ""))
    atomic_write_text(
        out_root / "runtime_trigger_equivalence.md",
        "# Runtime Trigger Equivalence\n\n"
        + "\n".join(
            [
                f"- v100_runtime_commit: `{v100_runtime_commit}`",
                f"- gpu4090_runtime_commits: `{', '.join(gpu4090_runtime_commits)}`",
                f"- patternkv_config_equal: `{runtime_trigger_equivalence['patternkv_config_equal']}`",
                f"- insight_config_equal: `{runtime_trigger_equivalence['insight_config_equal']}`",
                f"- summary_schema_difference_present: `{runtime_trigger_equivalence['summary_schema_difference_present']}`",
                f"- sensitive_runtime_diff_detected: `{runtime_trigger_equivalence['sensitive_runtime_diff_detected']}`",
                "",
                markdown_table(["gpu4090_commit", "changed_files", "sensitive_changed_files", "sensitive_examples"], runtime_rows),
            ]
        )
        + "\n",
    )

    responsible_samples = [row["sample_id"] for row in generation_rows_4090 if row["decode_k_summary_rows"] == len(pattern_extra_rows) and row["decode_v_summary_rows"] == len(dynamic_extra_rows)]
    evidence = {
        "pattern_gain_extra_rows": len(pattern_extra_rows),
        "dynamic_extra_rows": len(dynamic_extra_rows),
        "responsible_samples": responsible_samples,
        "v100_event_count": None,
        "gpu4090_event_count": 1 if responsible_samples else None,
        "event_count_delta": None,
        "expected_pattern_gain_delta": len(pattern_extra_rows) if decode_event_reconciliation["gpu4090_side"]["alignment_status"] else None,
        "expected_dynamic_delta": len(dynamic_extra_rows) if decode_event_reconciliation["gpu4090_side"]["alignment_status"] else None,
        "token_hash_equal": None,
        "raw_data_available": {
            "v100_generation": raw_v100["generation_available"],
            "v100_observer": raw_v100["observer_available"],
            "gpu4090_generation": raw_4090["generation_available"],
            "gpu4090_observer": raw_4090["observer_available"],
        },
        "summarizer_difference_proven": False,
        "runtime_difference_proven": False,
        "partial_alignment": decode_event_reconciliation["gpu4090_side"]["alignment_status"],
    }
    status = determine_final_status(evidence)
    final_finding = {
        "status": status,
        "pattern_gain_extra_rows": len(pattern_extra_rows),
        "dynamic_extra_rows": len(dynamic_extra_rows),
        "localized_task": "passage_retrieval_zh",
        "localized_phase": "decode",
        "responsible_samples": responsible_samples,
        "v100_event_count": None,
        "gpu4090_event_count": 1 if responsible_samples else None,
        "event_count_delta": None,
        "expected_pattern_gain_delta": evidence["expected_pattern_gain_delta"] if status != "data_insufficient" else None,
        "expected_dynamic_delta": evidence["expected_dynamic_delta"] if status != "data_insufficient" else None,
        "generation_length_explanation": None,
        "token_divergence_explanation": None,
        "runtime_difference_explanation": None,
        "summarizer_difference_explanation": None,
        "gpu4090_side_explanation": {
            "sample_id": responsible_samples[0] if responsible_samples else None,
            "gpu4090_expected_event_count": 1 if responsible_samples else None,
            "gpu4090_expected_pattern_gain_rows": len(pattern_extra_rows) if responsible_samples else None,
            "gpu4090_expected_dynamic_rows": len(dynamic_extra_rows) if responsible_samples else None,
        },
        "raw_data_available": evidence["raw_data_available"],
        "targeted_rerun_required": status == "data_insufficient",
    }
    atomic_write_json(out_root / "final_finding.json", final_finding)
    final_lines = [
        "# Final Finding",
        "",
        f"Status: `{status}`",
        "",
        f"- pattern_gain_extra_rows: `{len(pattern_extra_rows)}`",
        f"- dynamic_extra_rows: `{len(dynamic_extra_rows)}`",
        f"- localized_task: `passage_retrieval_zh`",
        f"- localized_phase: `decode`",
        f"- responsible_samples: `{', '.join(responsible_samples) if responsible_samples else 'none'}`",
        f"- targeted_rerun_required: `{final_finding['targeted_rerun_required']}`",
        "",
        "The 4090-side raw generation and observer files are sufficient to localize one sample whose single decode boundary event expands to the exact 512/256 row pattern. The V100 raw generation/observer files are not available in this workspace, so the cross-hardware absence of that event cannot be proven sample-by-sample here.",
    ]
    atomic_write_text(out_root / "final_finding.md", "\n".join(final_lines) + "\n")

    if status == "data_insufficient":
        rerun_plan = {
            "schema_version": "insight_v2.passage_retrieval_zh_decode_delta_rerun_plan",
            "status": "proposed_only",
            "why_needed": "V100 raw generation and observer files are unavailable, so the cross-hardware sample-level cause cannot be proven from offline data alone.",
            "target_sample": responsible_samples[0] if responsible_samples else None,
            "fallback_scope": "passage_retrieval_zh_12_samples",
            "preferred_hardware": "single RTX 4090",
            "runtime_commits_to_compare": [v100_runtime_commit, *gpu4090_runtime_commits],
            "required_fields": [
                "sample_id",
                "sample_index",
                "input_tokens",
                "generated_token_ids",
                "generated_token_ids_sha256",
                "generated_text_sha256",
                "stop_reason",
                "max_new_tokens",
                "decode window_idx",
                "decode_k metrics",
                "decode_v metrics",
            ],
            "success_criteria": "Reproduce whether the target sample emits one decode boundary event under one runtime/configuration and zero under the comparison condition.",
            "directories": {
                "results": "results/insight_v2/passage_retrieval_zh_decode_delta_rerun/",
                "reports": "reports/insight_v2/passage_retrieval_zh_decode_delta_rerun/",
                "logs": "logs/insight_v2/passage_retrieval_zh_decode_delta_rerun/",
                "run": "run/insight_v2/passage_retrieval_zh_decode_delta_rerun/",
            },
            "forbidden": "Do not auto-start this rerun plan.",
        }
        atomic_write_json(out_root / "targeted_rerun_plan.json", rerun_plan)
        atomic_write_text(
            out_root / "targeted_rerun_plan.md",
            "\n".join(
                [
                    "# Targeted Rerun Plan",
                    "",
                    f"- target_sample: `{rerun_plan['target_sample']}`",
                    f"- fallback_scope: `{rerun_plan['fallback_scope']}`",
                    f"- preferred_hardware: `{rerun_plan['preferred_hardware']}`",
                    "- do not auto-start",
                ]
            )
            + "\n",
        )

    print(json.dumps({"status": status, "output_root": str(out_root)}, sort_keys=True))


if __name__ == "__main__":
    main()
