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


REPORT_DIR = ROOT / "reports/aime24_pseudodecode_3090_8gpu"


def coerce(row: dict, key: str):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return value


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

    pattern_pair = paired_delta(auc_rows, left_config="pattern_rolling_k2v2_s0_r128", right_config="pattern_rolling_k2v2_s16_r128", value_key="acc_auc")
    kivi_pair = paired_delta(auc_rows, left_config="kivi_rolling_k2v2_s0_r128", right_config="kivi_rolling_k2v2_s16_r128", value_key="acc_auc")
    summary_path = REPORT_DIR / "pseudodecode_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "formal_run_complete": bool(rows),
            "pattern_sink_pair_auc": pattern_pair,
            "kivi_sink_pair_auc": kivi_pair,
            "pattern_sink_reduces_accumulation": None if pattern_pair["paired_n"] == 0 else pattern_pair["tasks_improved"] > pattern_pair["tasks_regressed"],
            "kivi_sink_reduces_accumulation": None if kivi_pair["paired_n"] == 0 else kivi_pair["tasks_improved"] > kivi_pair["tasks_regressed"],
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"metrics_rows": len(rows), "gap_rows": len(gaps), "auc_rows": len(auc_rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
