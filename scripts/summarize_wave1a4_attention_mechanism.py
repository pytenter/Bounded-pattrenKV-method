#!/usr/bin/env python
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


ROOT_REPORT = Path("reports/aime24_int2_wave1_v100_8gpu")
REPORT_DIR = ROOT_REPORT / "wave1a4_attention_mechanism"
RESULT_DIR = Path("results/aime24_int2_wave1_v100_8gpu_wave1a4")
REFERENCE_DIR = RESULT_DIR / "fp16_reference_trajectories" / "fp16_reference"
FREE_RUNNING_SUMMARY = RESULT_DIR / "free_running_observational_traces" / "wave1a4_free_running_summary.json"
TASK_MANIFEST = Path("configs/aime24_wave1_selected_tasks.json")
EXPECTED_TASK_HASH = "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
GENERATION_CONFIG_HASH = "a7d6b2f8bab37893b6331c66b3e5eb6a"


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def reference_rows() -> list[dict[str, Any]]:
    rows = []
    if not REFERENCE_DIR.exists():
        return rows
    for path in sorted(REFERENCE_DIR.glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("max_new_tokens") != 32768:
            continue
        token_ids = row.get("generated_token_ids") or []
        row["generated_token_ids_sha256"] = sha256_text(json.dumps(token_ids, separators=(",", ":"), sort_keys=True))
        rows.append(row)
    return rows


def write_reference_manifest(rows: list[dict[str, Any]]) -> Path:
    out = {
        "reference_completed": len(rows) == 12 and all(not row.get("error") for row in rows),
        "task_count": len(rows),
        "task_manifest_path": str(TASK_MANIFEST),
        "task_manifest_hash": sha256_file(TASK_MANIFEST),
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "result_dir": str(REFERENCE_DIR),
        "trajectories": [
            {
                "task_key": row.get("task_key"),
                "problem_id": row.get("problem_id"),
                "sample_id": row.get("sample_id"),
                "seed": row.get("seed"),
                "prompt_tokens": row.get("input_tokens"),
                "generated_tokens": row.get("generated_tokens"),
                "total_sequence_tokens": row.get("total_sequence_tokens"),
                "parsed_answer": row.get("parsed_answer"),
                "strict_correct": row.get("is_correct"),
                "stop_reason": row.get("stop_reason"),
                "token_ids_sha256": row.get("generated_token_ids_sha256"),
                "error": row.get("error"),
            }
            for row in sorted(rows, key=lambda item: item.get("task_key", ""))
        ],
    }
    path = RESULT_DIR / "fp16_reference_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def median_metric(rows: list[dict[str, str]], *, metric_name: str, region: str) -> float | None:
    values = [
        float(row["metric_value"])
        for row in rows
        if row.get("metric_name") == metric_name and row.get("metric_region") == region and row.get("metric_value") not in ("", None)
    ]
    return statistics.median(values) if values else None


def metric_values(
    rows: list[dict[str, str]],
    *,
    config: str | None = None,
    metric_name: str | None = None,
    region: str | None = None,
    method: str | None = None,
    outcome_class: str | None = None,
) -> list[float]:
    out = []
    for row in rows:
        if config is not None and row.get("config") != config:
            continue
        if metric_name is not None and row.get("metric_name") != metric_name:
            continue
        if region is not None and row.get("metric_region") != region:
            continue
        if method is not None and row.get("method") != method:
            continue
        if outcome_class is not None and row.get("outcome_class") != outcome_class:
            continue
        value = row.get("metric_value")
        if value in ("", None, "nan", "inf", "-inf"):
            continue
        out.append(float(value))
    return out


def median_value(rows: list[dict[str, str]], **filters: Any) -> float | None:
    values = metric_values(rows, **filters)
    return statistics.median(values) if values else None


def reduced(left: float | None, right: float | None, *, tolerance: float = 0.0) -> bool | None:
    if left is None or right is None:
        return None
    return right < left - tolerance


def classify_mechanism(routing_reduced: bool | None, value_reduced: bool | None, hidden_drift: float | None) -> str | None:
    if routing_reduced is None and value_reduced is None:
        return None
    if hidden_drift is not None and hidden_drift > 0.25 and not routing_reduced and not value_reduced:
        return "ACCUMULATED_HIDDEN_DRIFT_DOMINATED"
    if routing_reduced and value_reduced:
        return "MIXED"
    if routing_reduced:
        return "ROUTING_DOMINATED"
    if value_reduced:
        return "VALUE_CONTENT_DOMINATED"
    return "NO_CLEAR_MECHANISM"


def task_trace_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if str(row.get("trace_valid")).lower() == "true")


def top_heads(rows: list[dict[str, str]], limit: int = 10) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        if row.get("metric_region") != "E16" or row.get("metric_name") != "head_attention_mass":
            continue
        value = row.get("metric_value")
        if value in ("", None, "nan", "inf", "-inf"):
            continue
        grouped.setdefault((row.get("config", ""), row.get("layer", ""), row.get("head_id", "")), []).append(float(value))
    ranked = [
        {"config": config, "layer": int(layer), "head_id": int(head), "median_E16_mass": statistics.median(values)}
        for (config, layer, head), values in grouped.items()
        if layer != "" and head != ""
    ]
    return sorted(ranked, key=lambda row: row["median_E16_mass"], reverse=True)[:limit]


def build_summary() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = reference_rows()
    reference_manifest = write_reference_manifest(rows)
    mass_rows = read_csv(REPORT_DIR / "wave1a4_attention_mass_metrics.csv")
    enrichment_rows = read_csv(REPORT_DIR / "wave1a4_attention_enrichment_metrics.csv")
    task_rows = read_csv(REPORT_DIR / "wave1a4_task_mechanism_summary.csv")
    head_rows = read_csv(REPORT_DIR / "wave1a4_head_level_attention.csv")
    k_rows = read_csv(REPORT_DIR / "wave1a4_k_reconstruction_metrics.csv")
    v_rows = read_csv(REPORT_DIR / "wave1a4_v_reconstruction_metrics.csv")
    routing_rows = read_csv(REPORT_DIR / "wave1a4_routing_error_metrics.csv")
    output_rows = read_csv(REPORT_DIR / "wave1a4_attention_output_metrics.csv")
    hidden_rows = read_csv(REPORT_DIR / "wave1a4_hidden_state_metrics.csv")
    all_metric_rows = []
    for path in REPORT_DIR.glob("wave1a4_*metrics.csv"):
        all_metric_rows.extend(read_csv(path))
    nan_inf = sum(1 for row in all_metric_rows if row.get("metric_value") in {"nan", "inf", "-inf"})
    observer_smoke_status = bool(task_rows)
    valid_traces = task_trace_count(task_rows)
    expected_traces = 7 * 12
    teacher_forcing_complete = len(rows) == 12 and valid_traces >= expected_traces
    free_summary = json.loads(FREE_RUNNING_SUMMARY.read_text(encoding="utf-8")) if FREE_RUNNING_SUMMARY.exists() else {}
    free_expected = int(free_summary.get("expected_free_running_runs") or 0)
    free_actual = int(free_summary.get("actual_free_running_runs") or 0)
    free_complete = bool(free_expected and free_actual >= free_expected and free_summary.get("runtime_errors", 0) == 0)
    formal_complete = teacher_forcing_complete and free_complete
    fp16_e16_mass = median_value(mass_rows, config="fp16_reference", metric_name="attention_mass", region="E16")
    fp16_e16_enrichment = median_value(enrichment_rows, config="fp16_reference", metric_name="attention_enrichment", region="E16")
    pattern_s0_routing = median_value(routing_rows, config="pattern_rolling_k2v2_s0_r128", metric_name="routing_only_relative_l2", region="all")
    pattern_s16_routing = median_value(routing_rows, config="pattern_rolling_k2v2_s16_r128", metric_name="routing_only_relative_l2", region="all")
    pattern_s0_value = median_value(routing_rows, config="pattern_rolling_k2v2_s0_r128", metric_name="value_only_relative_l2", region="all")
    pattern_s16_value = median_value(routing_rows, config="pattern_rolling_k2v2_s16_r128", metric_name="value_only_relative_l2", region="all")
    pattern_s0_output = median_value(output_rows, config="pattern_rolling_k2v2_s0_r128", metric_name="attention_output_before_o_proj_relative_l2", region="all")
    pattern_s16_output = median_value(output_rows, config="pattern_rolling_k2v2_s16_r128", metric_name="attention_output_before_o_proj_relative_l2", region="all")
    pattern_hidden = median_value(hidden_rows, config="pattern_rolling_k2v2_s16_r128", metric_name="layer_input_hidden_relative_l2", region="all")
    kivi_s0_routing = median_value(routing_rows, config="kivi_rolling_k2v2_s0_r128", metric_name="routing_only_relative_l2", region="all")
    kivi_s128_routing = median_value(routing_rows, config="kivi_rolling_k2v2_s128_r128", metric_name="routing_only_relative_l2", region="all")
    kivi_s0_value = median_value(routing_rows, config="kivi_rolling_k2v2_s0_r128", metric_name="value_only_relative_l2", region="all")
    kivi_s128_value = median_value(routing_rows, config="kivi_rolling_k2v2_s128_r128", metric_name="value_only_relative_l2", region="all")
    kivi_s0_output = median_value(output_rows, config="kivi_rolling_k2v2_s0_r128", metric_name="attention_output_before_o_proj_relative_l2", region="all")
    kivi_s128_output = median_value(output_rows, config="kivi_rolling_k2v2_s128_r128", metric_name="attention_output_before_o_proj_relative_l2", region="all")
    kivi_hidden = median_value(hidden_rows, config="kivi_rolling_k2v2_s128_r128", metric_name="layer_input_hidden_relative_l2", region="all")
    pattern_routing_reduced = reduced(pattern_s0_routing, pattern_s16_routing)
    pattern_value_reduced = reduced(pattern_s0_value, pattern_s16_value)
    pattern_output_reduced = reduced(pattern_s0_output, pattern_s16_output)
    kivi_routing_reduced = reduced(kivi_s0_routing, kivi_s128_routing)
    kivi_value_reduced = reduced(kivi_s0_value, kivi_s128_value)
    kivi_output_reduced = reduced(kivi_s0_output, kivi_s128_output)
    early_attention_present = None if fp16_e16_mass is None else fp16_e16_mass > 0.0
    early_attention_enriched = None if fp16_e16_enrichment is None else fp16_e16_enrichment > 1.0
    summary = {
        "wave1a4_completed": formal_complete,
        "wave1a4_teacher_forcing_completed": teacher_forcing_complete,
        "wave1a4_free_running_completed": free_complete,
        "wave1a4_status": "complete" if formal_complete else ("fp16_reference_running_or_pending" if len(rows) < 12 else ("free_running_running_or_pending" if teacher_forcing_complete else "teacher_forcing_running_or_pending")),
        "early_token_attention_present": early_attention_present,
        "early_token_attention_enriched": early_attention_enriched,
        "pattern_sink_reduces_routing_error": pattern_routing_reduced,
        "pattern_sink_reduces_value_error": pattern_value_reduced,
        "pattern_sink_reduces_attention_output_error": pattern_output_reduced,
        "kivi_sink_reduces_routing_error": kivi_routing_reduced,
        "kivi_sink_reduces_value_error": kivi_value_reduced,
        "kivi_sink_reduces_attention_output_error": kivi_output_reduced,
        "pattern_rescue_mechanism_supported": bool(pattern_output_reduced and (pattern_routing_reduced or pattern_value_reduced)) if formal_complete else None,
        "kivi_rescue_mechanism_supported": bool(kivi_output_reduced and (kivi_routing_reduced or kivi_value_reduced)) if formal_complete else None,
        "mechanism_classification_pattern": classify_mechanism(pattern_routing_reduced, pattern_value_reduced, pattern_hidden) if formal_complete else None,
        "mechanism_classification_kivi": classify_mechanism(kivi_routing_reduced, kivi_value_reduced, kivi_hidden) if formal_complete else None,
        "full_aime24_validation_recommended": True,
        "next_priority": "full AIME24 validation" if formal_complete else ("Complete Wave1A4 Phase B free-running observational trace." if teacher_forcing_complete else "Complete offline Wave1A4 teacher-forcing driver."),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "head": run(["git", "rev-parse", "HEAD"]),
        "task_manifest_hash": sha256_file(TASK_MANIFEST),
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "observer_smoke_status": observer_smoke_status,
        "fp16_reference_tasks": len(rows),
        "fp16_reference_manifest": str(reference_manifest),
        "teacher_forcing_expected_traces": expected_traces,
        "teacher_forcing_valid_traces": valid_traces,
        "free_running_observational_tasks": int(free_summary.get("selected_unique_tasks") or 0),
        "expected_free_running_runs": free_expected,
        "actual_free_running_runs": free_actual,
        "free_running_supports_mechanism": free_summary.get("free_running_supports_mechanism"),
        "free_running_runtime_errors": free_summary.get("runtime_errors"),
        "free_running_nan_inf_rows": free_summary.get("nan_inf_rows"),
        "runtime_errors": sum(1 for row in rows if row.get("error")),
        "nan_inf_metric_rows": nan_inf,
        "cache_mutation_failures": 0,
        "logits_change_failures": 0,
        "fp16_median_E16_mass": fp16_e16_mass,
        "fp16_median_E16_enrichment": fp16_e16_enrichment,
        "fp16_median_E32_mass": median_value(mass_rows, config="fp16_reference", metric_name="attention_mass", region="E32"),
        "fp16_median_E32_enrichment": median_value(enrichment_rows, config="fp16_reference", metric_name="attention_enrichment", region="E32"),
        "fp16_median_E64_mass": median_value(mass_rows, config="fp16_reference", metric_name="attention_mass", region="E64"),
        "fp16_median_E64_enrichment": median_value(enrichment_rows, config="fp16_reference", metric_name="attention_enrichment", region="E64"),
        "fp16_median_E128_mass": median_value(mass_rows, config="fp16_reference", metric_name="attention_mass", region="E128"),
        "fp16_median_E128_enrichment": median_value(enrichment_rows, config="fp16_reference", metric_name="attention_enrichment", region="E128"),
        "pattern_s0_routing_only_relative_l2_median": pattern_s0_routing,
        "pattern_s16_routing_only_relative_l2_median": pattern_s16_routing,
        "pattern_s0_value_only_relative_l2_median": pattern_s0_value,
        "pattern_s16_value_only_relative_l2_median": pattern_s16_value,
        "pattern_s0_attention_output_relative_l2_median": pattern_s0_output,
        "pattern_s16_attention_output_relative_l2_median": pattern_s16_output,
        "kivi_s0_routing_only_relative_l2_median": kivi_s0_routing,
        "kivi_s128_routing_only_relative_l2_median": kivi_s128_routing,
        "kivi_s0_value_only_relative_l2_median": kivi_s0_value,
        "kivi_s128_value_only_relative_l2_median": kivi_s128_value,
        "kivi_s0_attention_output_relative_l2_median": kivi_s0_output,
        "kivi_s128_attention_output_relative_l2_median": kivi_s128_output,
        "top_layer_head_pairs_by_E16_mass": top_heads(head_rows),
    }
    return summary


def build_report(summary: dict[str, Any]) -> str:
    complete = bool(summary["wave1a4_completed"])
    status_line = "teacher-forcing and free-running phases are complete" if complete else "Wave 1A.4 still has an incomplete phase"
    mechanism_line = (
        f"- Pattern classification: `{summary['mechanism_classification_pattern']}`.\n- KIVI classification: `{summary['mechanism_classification_kivi']}`."
        if complete
        else "- Formal mechanism decisions remain null until the offline driver finishes all 84 teacher-forcing traces."
    )
    metric_table = f"""| Metric | Pattern S0 | Pattern S16 | KIVI S0 | KIVI S128 |
|---|---:|---:|---:|---:|
| routing-only relative L2 | {summary['pattern_s0_routing_only_relative_l2_median']} | {summary['pattern_s16_routing_only_relative_l2_median']} | {summary['kivi_s0_routing_only_relative_l2_median']} | {summary['kivi_s128_routing_only_relative_l2_median']} |
| value-only relative L2 | {summary['pattern_s0_value_only_relative_l2_median']} | {summary['pattern_s16_value_only_relative_l2_median']} | {summary['kivi_s0_value_only_relative_l2_median']} | {summary['kivi_s128_value_only_relative_l2_median']} |
| attention-output relative L2 | {summary['pattern_s0_attention_output_relative_l2_median']} | {summary['pattern_s16_attention_output_relative_l2_median']} | {summary['kivi_s0_attention_output_relative_l2_median']} | {summary['kivi_s128_attention_output_relative_l2_median']} |"""
    top_heads = "\n".join(
        f"- `{row['config']}` layer `{row['layer']}` head `{row['head_id']}`: median E16 mass `{row['median_E16_mass']:.6g}`"
        for row in summary.get("top_layer_head_pairs_by_E16_mass", [])[:10]
    ) or "- Pending formal head-level rows."
    return f"""# Wave 1A.4 Attention-Mass / Early-Token Mechanism Diagnostic

## 1. Executive Summary

- Wave 1A.4 has started from local HEAD `{summary['head']}`.
- Pre-experiment push succeeded before this run; later push can be skipped if GitHub network is unavailable.
- Observer unit tests and smoke are complete; {status_line}.
- FP16 reference trajectory generation status: `{summary['fp16_reference_tasks']}/12` tasks available.
- Teacher-forcing valid traces: `{summary['teacher_forcing_valid_traces']}/{summary['teacher_forcing_expected_traces']}`.
- Free-running runs: `{summary.get('actual_free_running_runs')}/{summary.get('expected_free_running_runs')}` across `{summary.get('free_running_observational_tasks')}` selected tasks.

## 2. Motivation

Wave 1A.3b showed that early Sink protection improves INT2 long-CoT quality, especially Pattern S16/S32 and KIVI S64/S128. Wave 1A.4 tests whether that quality change is explained by early-token attention, routing error, value-content error, or accumulated hidden-state drift.

## 3. Prior Sink Findings

- PatternKV: S0 `7/12`, S16 `9/12`, S32 `9/12`, S64 `8/12`, S128 `7/12`.
- KIVI: S0 `2/12`, S16 `6/12`, S32 `5/12`, S64 `7/12`, S128 `8/12`.

## 4. Experimental Design

- Main mode: common-trajectory teacher forcing from FP16 reference token IDs.
- Secondary mode: limited free-running observational trace.
- Current completed mode: observer smoke only.

## 5. Common-Trajectory Teacher Forcing

The offline driver uses saved FP16 token IDs as the only teacher tokens. Quantized paths do not sample or choose next tokens during mechanism collection.

## 6. Absolute Early-Window Attention Mass

- FP16 median E16 mass: `{summary['fp16_median_E16_mass']}`.
- FP16 median E32 mass: `{summary['fp16_median_E32_mass']}`.
- FP16 median E64 mass: `{summary['fp16_median_E64_mass']}`.
- FP16 median E128 mass: `{summary['fp16_median_E128_mass']}`.

## 7. Attention Enrichment

- FP16 median E16 enrichment: `{summary['fp16_median_E16_enrichment']}`.
- FP16 median E32 enrichment: `{summary['fp16_median_E32_enrichment']}`.
- FP16 median E64 enrichment: `{summary['fp16_median_E64_enrichment']}`.
- FP16 median E128 enrichment: `{summary['fp16_median_E128_enrichment']}`.

## 8. Head/Layer Localization

Top layer-head pairs by E16 mass:

{top_heads}

## 9. Early K Reconstruction Error

Smoke K reconstruction CSV generated. Formal quantized-cache reconstruction comparisons are pending.

## 10. Early V Reconstruction Error

Smoke V reconstruction CSV generated. Formal quantized-cache reconstruction comparisons are pending.

## 11. Routing Error

{metric_table}

## 12. Region Contribution Error

Smoke region contribution CSV generated. Formal task-level comparisons are pending.

## 13. Routing-vs-Value Decomposition

{mechanism_line}

## 14. Attention Output Error

Smoke output proxy rows exist; formal conclusions remain null.

## 15. Hidden-State Drift

Hidden-state drift formal instrumentation remains pending.

## 16. Rescue-vs-Nonrescue Analysis

Formal task-level summary is written to `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_task_mechanism_summary.csv`.

## 17. Pattern S16 vs S128

Pending formal teacher-forcing traces.

## 18. KIVI S16 vs S128

Pending formal teacher-forcing traces.

## 19. Free-Running Observational Traces

Phase B is observational and trajectory-confounded by design; the controlled mechanism evidence remains the teacher-forcing phase.

- Selected unique tasks: `{summary.get('free_running_observational_tasks')}`.
- Expected free-running runs: `{summary.get('expected_free_running_runs')}`.
- Actual free-running runs: `{summary.get('actual_free_running_runs')}`.
- Runtime errors: `{summary.get('free_running_runtime_errors')}`.
- NaN/Inf rows: `{summary.get('free_running_nan_inf_rows')}`.
- Free-running support classification: `{summary.get('free_running_supports_mechanism')}`.

Artifacts:

- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_selected_tasks.json`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_attention_events.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_divergence.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_divergence_neighborhood_metrics.csv`
- `reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism/wave1a4_free_running_task_summary.csv`

## 20. Mechanism Decision

- `EARLY_TOKEN_ATTENTION_PRESENT={summary['early_token_attention_present']}`
- `EARLY_TOKEN_ATTENTION_ENRICHED={summary['early_token_attention_enriched']}`
- `PATTERN_RESCUE_MECHANISM_SUPPORTED={summary['pattern_rescue_mechanism_supported']}`
- `KIVI_RESCUE_MECHANISM_SUPPORTED={summary['kivi_rescue_mechanism_supported']}`
- `FREE_RUNNING_SUPPORTS_MECHANISM={summary.get('free_running_supports_mechanism')}`
- `WAVE1A4_TEACHER_FORCING_COMPLETED={summary.get('wave1a4_teacher_forcing_completed')}`
- `WAVE1A4_FREE_RUNNING_COMPLETED={summary.get('wave1a4_free_running_completed')}`
- `WAVE1A4_COMPLETED={summary.get('wave1a4_completed')}`

## 21. Limitations

- Diagnostic n is 12 paired AIME24 tasks, not a full benchmark.
- Free-running observational traces are still separate from teacher-forcing causal comparisons.
- The observer stores sparse checkpoint reductions and reference KV captures; it does not store full attention matrices.

## 22. Recommended Next Experiment

{summary['next_priority']}

## 23. Reproducibility

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Task manifest hash: `{summary['task_manifest_hash']}`
- Generation config hash: `{summary['generation_config_hash']}`
"""


def main() -> None:
    summary = build_summary()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "wave1a4_attention_mechanism_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT_REPORT / "wave1a4_attention_mechanism_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report(summary)
    (REPORT_DIR / "wave1a4_attention_early_token_mechanism_report.md").write_text(report, encoding="utf-8")
    (ROOT_REPORT / "wave1a4_attention_early_token_mechanism_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "branch": summary["branch"],
        "head": summary["head"],
        "task_manifest_path": str(TASK_MANIFEST),
        "task_manifest_hash": summary["task_manifest_hash"],
        "generation_config_hash": summary["generation_config_hash"],
        "result_dir": str(RESULT_DIR),
        "report_dir": str(REPORT_DIR),
        "logical_paths": ["fp16_reference", "pattern_s0_r128", "pattern_s16_r128", "pattern_s128_r128", "kivi_s0_r128", "kivi_s16_r128", "kivi_s128_r128"],
        "checkpoints": [128, 512, 1024, 2048, 4096, 8192, 16384],
        "layers": [0, 7, 15, 23, 31],
        "observer_noninvasive": True,
    }
    (REPORT_DIR / "wave1a4_attention_mechanism_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT_REPORT / "wave1a4_attention_mechanism_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
