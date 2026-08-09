from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports/aime24_pseudodecode_3090_8gpu"


def read_json(name: str):
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def read_csv(name: str) -> list[dict[str, str]]:
    with (REPORT_DIR / name).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_paper_s0_equality_audit():
    summary = read_json("paper_vs_s0_equality_summary.json")
    assert summary["pattern"]["total_compared"] == 1592
    assert summary["pattern"]["exact_equal_fraction"] == 1.0
    assert summary["kivi"]["total_compared"] == 1592
    assert summary["kivi"]["exact_equal_fraction"] == 1.0


def test_sink_pair_provenance_distinct():
    audit = read_json("formal_result_audit.json")
    assert audit["sink_pair_result_provenance_valid"] is True
    assert audit["sink_pair_details"]["pattern"]["distinct_sink"] is True
    assert audit["sink_pair_details"]["kivi"]["distinct_sink"] is True
    assert audit["sink_pair_details"]["pattern"]["distinct_result_provenance"] is True
    assert audit["sink_pair_details"]["kivi"]["distinct_result_provenance"] is True


def test_core_auc_uses_only_128_to_4096():
    audit = read_json("formal_result_audit.json")
    auc_rows = read_csv("accumulation_auc.csv")
    assert audit["core_auc_definition_valid"] is True
    assert {row["n_available"] for row in auc_rows} == {"5"}


def test_extended_pseudo_only_excluded_from_accumulation():
    audit = read_json("formal_result_audit.json")
    assert audit["extended_pseudo_only_diagnostic"]["pseudo_only_extended_rows"] > 0
    assert audit["extended_pseudo_only_diagnostic"]["matched_static_extended_rows"] == 0
    assert audit["extended_pseudo_only_diagnostic"]["used_for_matched_accumulation_decision"] is False


def test_accumulation_decision_multimetric():
    decisions = read_json("hypothesis_decisions.json")
    rows = read_csv("formal_accumulation_decision_table.csv")
    assert decisions["pseudodecode_accumulation_supported"] is True
    for config in ("pattern_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s0_r128"):
        supported = [row for row in rows if row["config"] == config and row["metric"] in {"hidden_relative_L2", "attention_output_relative_L2", "next_token_KL"}]
        assert len(supported) == 3
        assert sum(row["accumulation_supported"] == "True" for row in supported) >= 2


def test_remaining_error_classification():
    decisions = read_json("hypothesis_decisions.json")
    anatomy = read_csv("pattern_s16_residual_error_anatomy.csv")
    assert decisions["remaining_error_classification"] == "ACCUMULATION_DOMINATED"
    hidden_4096 = next(row for row in anatomy if row["metric"] == "hidden_relative_L2" and row["checkpoint"] == "4096")
    assert float(hidden_4096["median_accumulation_fraction"]) > 0.9


def test_null_decisions_resolved_or_explicitly_inconclusive():
    decisions = read_json("hypothesis_decisions.json")
    for key, value in decisions.items():
        assert value is not None, key
    assert decisions["token_norm_accumulation_supported"] == "insufficient_data"
    assert decisions["varn_next_priority"] == "not_yet_justified"


def test_failed_rows_are_hardware_limited_extended_static():
    audit = read_json("formal_result_audit.json")
    rows = read_csv("formal_completeness_audit.csv")
    failed = [row for row in rows if row["status"] != "ok"]
    assert audit["failed_rows_verified_as_hardware_limited_static"] is True
    assert len(failed) == 42
    assert {row["mode"] for row in failed} == {"static"}
    assert {row["checkpoint"] for row in failed} <= {"8192", "16384"}


def test_paper_alias_does_not_invalidate_sink_pair():
    audit = read_json("formal_result_audit.json")
    assert audit["pattern_paper_s0_runtime_equivalence"] is True
    assert audit["kivi_paper_s0_runtime_equivalence"] is True
    assert audit["paper_vs_s0_comparison_informative"] is False
    assert audit["formal_sink_conclusion_valid"] is True


def test_decision_summary_reproducible():
    decisions = read_json("hypothesis_decisions.json")
    summary = read_json("pseudodecode_summary.json")
    for key in (
        "pseudodecode_accumulation_supported",
        "pattern_sink_reduces_accumulation",
        "kivi_sink_reduces_accumulation",
        "cross_method_sink_reduces_accumulation",
        "remaining_error_classification",
        "token_norm_accumulation_supported",
        "next_priority",
    ):
        assert summary[key] == decisions[key]
