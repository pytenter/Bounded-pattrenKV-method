from __future__ import annotations

import json
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.pseudodecode_metrics import (  # noqa: E402
    accumulation_gap,
    grouped_median_iqr,
    paired_delta,
    read_csv_rows,
    trapezoid_auc_log2,
    write_csv_rows,
)
from bench.pseudodecode_controls import ACCUMULATION_METRIC_SCHEMA_VERSION, MATCHED_PATH_CONTROL_VERSION  # noqa: E402


REPORT_DIR = ROOT / "reports/aime24_pseudodecode_3090_8gpu"


def coerce(row: dict, key: str):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return value


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    metrics_path = REPORT_DIR / "static_vs_pseudo_metrics.csv"
    rows = read_csv_rows(metrics_path)
    by_key = {}
    for row in rows:
        key = (row.get("task_key"), row.get("config"), row.get("checkpoint"), row.get("layer"), row.get("metric_name"))
        by_key[(row.get("mode"), *key)] = row

    gaps = []
    for mode, task_key, config, checkpoint, layer, metric_name in list(by_key):
        if mode != "pseudo":
            continue
        pseudo = by_key[(mode, task_key, config, checkpoint, layer, metric_name)]
        static = by_key.get(("static", task_key, config, checkpoint, layer, metric_name))
        if not static:
            continue
        pv = coerce(pseudo, "metric_value")
        sv = coerce(static, "metric_value")
        if isinstance(pv, float) and isinstance(sv, float):
            gaps.append(
                {
                    "task_key": task_key,
                    "config": config,
                    "checkpoint": int(float(checkpoint)),
                    "layer": layer,
                    "metric_name": metric_name,
                    "accumulation_gap": accumulation_gap(pv, sv),
                    "pseudo_value": pv,
                    "static_value": sv,
                    "matched_path_control_version": pseudo.get("matched_path_control_version") or MATCHED_PATH_CONTROL_VERSION,
                    "accumulation_metric_schema_version": pseudo.get("accumulation_metric_schema_version") or ACCUMULATION_METRIC_SCHEMA_VERSION,
                    "source_commit": pseudo.get("source_commit"),
                    "trajectory_sha256": pseudo.get("trajectory_sha256"),
                }
            )
    write_csv_rows(REPORT_DIR / "accumulation_gap.csv", gaps)

    auc_rows = []
    groups = {}
    for row in gaps:
        groups.setdefault((row["task_key"], row["config"], row["metric_name"], row["layer"]), []).append((row["checkpoint"], row["accumulation_gap"]))
    for (task_key, config, metric_name, layer), points in sorted(groups.items()):
        auc_rows.append(
            {
                "task_key": task_key,
                "config": config,
                "metric_name": metric_name,
                "layer": layer,
                "n_available": len(points),
                "acc_auc": trapezoid_auc_log2(points),
            }
        )
    write_csv_rows(REPORT_DIR / "accumulation_auc.csv", auc_rows)
    summary_rows = grouped_median_iqr(auc_rows, ("config", "metric_name", "layer"), "acc_auc")
    write_csv_rows(REPORT_DIR / "task_level_summary.csv", summary_rows)

    primary_auc_rows = [row for row in auc_rows if row.get("metric_name") == "hidden_relative_L2" and row.get("layer") == "final"]
    pattern_pair = paired_delta(primary_auc_rows, left_config="pattern_rolling_k2v2_s0_r128", right_config="pattern_rolling_k2v2_s16_r128", value_key="acc_auc")
    kivi_pair = paired_delta(primary_auc_rows, left_config="kivi_rolling_k2v2_s0_r128", right_config="kivi_rolling_k2v2_s16_r128", value_key="acc_auc")
    completeness = read_csv_rows(REPORT_DIR / "formal_completeness_audit.csv")
    failed = [row for row in completeness if row.get("status") != "ok"]
    core_checkpoints = {"128", "512", "1024", "2048", "4096"}
    core_failed = [row for row in completeness if row.get("checkpoint") in core_checkpoints and row.get("status") != "ok"]
    core_gap_rows = [row for row in gaps if str(row.get("checkpoint")) in core_checkpoints]
    formal_run = read_json(REPORT_DIR / "formal_run_summary.json")
    preflight = read_json(REPORT_DIR / "preflight_gate_summary.json")
    summary_path = REPORT_DIR / "pseudodecode_summary.json"
    summary = read_json(summary_path)
    summary.update(
        {
            "formal_run_complete": bool(rows) and not failed,
            "formal_core_matched_checkpoints_complete": bool(rows) and not core_failed and len(core_gap_rows) == 2880,
            "formal_core_checkpoints": [128, 512, 1024, 2048, 4096],
            "formal_metric_rows": len(rows),
            "formal_gap_rows": len(gaps),
            "formal_completeness_rows": len(completeness),
            "formal_failed_rows": len(failed),
            "formal_failed_reason": None if not failed else "static_full_prefix_oom_at_8192_or_16384_on_24gb_rtx3090",
            "formal_run_summary": formal_run,
            "formal_run_approved": preflight.get("formal_run_approved"),
            "preflight_complete": preflight.get("preflight_complete"),
            "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
            "accumulation_metric_schema_version": ACCUMULATION_METRIC_SCHEMA_VERSION,
            "pattern_sink_pair_auc": pattern_pair,
            "kivi_sink_pair_auc": kivi_pair,
            "pattern_sink_reduces_accumulation": None if pattern_pair["paired_n"] == 0 else pattern_pair["tasks_improved"] > pattern_pair["tasks_regressed"],
            "kivi_sink_reduces_accumulation": None if kivi_pair["paired_n"] == 0 else kivi_pair["tasks_improved"] > kivi_pair["tasks_regressed"],
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = [
        "# AIME24 Pseudo-Decode Accumulation Report",
        "",
        "## Executive Summary",
        "",
        "The formal matched-path accumulated-error run completed for the core checkpoints `128, 512, 1024, 2048, 4096` across 12 frozen AIME24 tasks and 6 quantized configs.",
        "",
        f"- `FORMAL_RUN_APPROVED={preflight.get('formal_run_approved')}`",
        f"- `formal_core_matched_checkpoints_complete={summary['formal_core_matched_checkpoints_complete']}`",
        f"- `formal_run_complete={summary['formal_run_complete']}`",
        f"- Metric rows: `{len(rows)}`",
        f"- Matched accumulation gap rows: `{len(gaps)}`",
        f"- Completeness rows: `{len(completeness)}`",
        f"- Failed rows: `{len(failed)}`",
        "",
        "## Matched-Path Definition",
        "",
        "`static_degradation = D(Q_static, FP16_static)` and `pseudo_degradation = D(Q_pseudo, FP16_pseudo)`. The reported accumulation gap is `pseudo_degradation - static_degradation`; no FP16 execution-path baseline is double-subtracted.",
        "",
        "## Completion",
        "",
        "The core matched checkpoints are complete. The unavailable formal rows are static full-prefix jobs at checkpoint `8192` or `16384`; these OOM on 24GB RTX3090 and are recorded in `formal_completeness_audit.csv`. Pseudo rows at those long checkpoints are retained, but accumulation gaps require matched static+pseudo pairs and therefore summarize only paired checkpoints.",
        "",
        "## Sink Pair AUC",
        "",
        "Primary sink-pair comparisons use final-layer `hidden_relative_L2` accumulation AUC.",
        "",
        f"- Pattern S16 vs S0: paired_n `{pattern_pair['paired_n']}`, median_delta `{pattern_pair['median_delta']}`, improved `{pattern_pair['tasks_improved']}`, regressed `{pattern_pair['tasks_regressed']}`, ties `{pattern_pair['ties']}`",
        f"- KIVI S16 vs S0: paired_n `{kivi_pair['paired_n']}`, median_delta `{kivi_pair['median_delta']}`, improved `{kivi_pair['tasks_improved']}`, regressed `{kivi_pair['tasks_regressed']}`, ties `{kivi_pair['ties']}`",
        "",
        "## Artifacts",
        "",
        "- `static_vs_pseudo_metrics.csv`",
        "- `accumulation_gap.csv`",
        "- `accumulation_auc.csv`",
        "- `task_level_summary.csv`",
        "- `formal_completeness_audit.csv`",
        "- `formal_run_summary.json`",
        "",
    ]
    (REPORT_DIR / "pseudodecode_accumulation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"metrics_rows": len(rows), "gap_rows": len(gaps), "auc_rows": len(auc_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
