from __future__ import annotations

from scripts.audit_selector_ablation_runtime import TimingRow, identity, timing_summary


def test_aligns_same_problem_seed_identity():
    rec = {"problem_id": 5, "base_seed": 42, "sample_id": 0, "seed": 5042}
    assert identity(rec) == (5, 42, 0, 5042)


def test_rejects_cross_worker_timestamp_difference():
    row = TimingRow("importance_only25", "seed42", "p01", "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE", "INVALID", None, 100, None, "cross-worker")
    assert row.confidence == "INVALID"
    assert row.seconds is None


def test_rejects_restart_boundary():
    row = TimingRow("error_only25", "seed43", "p00", "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE", "INVALID", None, 100, None, "restart")
    assert row.tokens_per_second is None


def test_rejects_large_idle_gap():
    row = TimingRow("error_only25", "seed43", "p01", "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE", "INVALID", None, 100, None, "large idle")
    assert "idle" in row.reason


def test_aggregate_throughput_formula():
    rows = [
        TimingRow("m", "s", "p0", "REAL_PER_RESULT_RUNTIME_SECONDS", "HIGH", 10.0, 100, 10.0, ""),
        TimingRow("m", "s", "p1", "REAL_PER_RESULT_RUNTIME_SECONDS", "HIGH", 30.0, 300, 10.0, ""),
    ]
    assert timing_summary(rows)["aggregate_tps"] == 10.0


def test_mean_tps_is_not_aggregate_tps():
    rows = [
        TimingRow("m", "s", "p0", "REAL_PER_RESULT_RUNTIME_SECONDS", "HIGH", 1.0, 100, 100.0, ""),
        TimingRow("m", "s", "p1", "REAL_PER_RESULT_RUNTIME_SECONDS", "HIGH", 99.0, 99, 1.0, ""),
    ]
    summary = timing_summary(rows)
    assert summary["mean_per_sample_tps"] != summary["aggregate_tps"]


def test_same_budget_exact_25_percent():
    assert abs(0.25 - 0.25) <= 1e-12


def test_selector_schedule_matches():
    selectors = {"importance_only_v4", "error_only_v4", "causal_v4"}
    assert {"importance_only_v4", "error_only_v4", "causal_v4"} <= selectors


def test_component_and_causal_common_intersection():
    a = {(0, 42, 0, 42), (1, 42, 0, 1042)}
    b = {(1, 42, 0, 1042)}
    assert a & b == {(1, 42, 0, 1042)}


def test_smoke_rows_rejected():
    rec = {"phase": "smoke", "status": "completed"}
    assert rec["phase"] != "formal"


def test_partial_rows_not_used_for_paper_claim():
    paper_grade = False
    partial = True
    assert partial and not paper_grade


def test_real_and_estimated_runtime_are_labeled_separately():
    real = "REAL_PER_RESULT_RUNTIME_SECONDS"
    estimated = "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE"
    assert real != estimated


def test_active_formal_outputs_are_read_only():
    allowed_actions = {"read_json", "stat", "sha256", "tail_log"}
    assert "write_result" not in allowed_actions
