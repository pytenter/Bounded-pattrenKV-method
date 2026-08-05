from scripts.analyze_v_gate_layer_head_opportunity import (
    aggregate_rows,
    bootstrap_ci,
    duplicate_primary_key_count,
    metric_from_counts,
    paired_bootstrap_ci,
    parse_int,
    safe_ordinal_label,
    spearman,
    task_group_stats,
)


def test_metric_from_counts_micro_math():
    stats = metric_from_counts(10, 90, 5, 15)
    assert stats["micro_fpr"] == 5 / 95
    assert stats["micro_fnr"] == 15 / 25
    assert stats["precision_micro"] == 10 / 15
    assert stats["recall_micro"] == 10 / 25
    assert stats["acceptance_micro"] == 15 / 120


def test_metric_from_counts_zero_denominator():
    stats = metric_from_counts(0, 0, 0, 0)
    assert stats["micro_fpr"] is None
    assert stats["micro_fnr"] is None
    assert stats["precision_micro"] is None
    assert stats["recall_micro"] is None
    assert stats["acceptance_micro"] is None


def test_duplicate_primary_key_count_detects_repeated_rows():
    rows = [
        {"task": "a", "layer": "0", "kv_head": "0"},
        {"task": "a", "layer": "0", "kv_head": "0"},
        {"task": "a", "layer": "1", "kv_head": "0"},
    ]
    assert duplicate_primary_key_count(rows, ["task", "layer", "kv_head"]) == 1


def test_aggregate_rows_is_order_invariant():
    rows1 = [
        {"task": "a", "phase": "prefill", "kv_type": "v", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "gate_vs_mse_oracle", "true_positive": "1", "true_negative": "2", "false_positive": "3", "false_negative": "4"},
        {"task": "a", "phase": "prefill", "kv_type": "v", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "gate_vs_mse_oracle", "true_positive": "2", "true_negative": "3", "false_positive": "4", "false_negative": "5"},
    ]
    rows2 = list(reversed(rows1))
    agg1 = aggregate_rows(rows1, ["task", "layer", "kv_head"])
    agg2 = aggregate_rows(rows2, ["task", "layer", "kv_head"])
    assert agg1[("a", "0", "0")]["true_positive"] == 3
    assert agg1 == agg2


def test_spearman_handles_constant_inputs():
    assert spearman([1, 1, 1], [1, 2, 3]) is None
    assert round(spearman([1, 2, 3], [3, 2, 1]), 6) == -1.0


def test_bootstrap_ci_is_reproducible():
    units = [
        {"true_positive": 10, "true_negative": 90, "false_positive": 5, "false_negative": 15},
        {"true_positive": 20, "true_negative": 80, "false_positive": 2, "false_negative": 8},
    ]
    fn = lambda sample: metric_from_counts(
        sum(u["true_positive"] for u in sample),
        sum(u["true_negative"] for u in sample),
        sum(u["false_positive"] for u in sample),
        sum(u["false_negative"] for u in sample),
    )["micro_fnr"]
    a = bootstrap_ci(units, repetitions=200, seed=0, metric_fn=fn)
    b = bootstrap_ci(units, repetitions=200, seed=0, metric_fn=fn)
    assert a == b


def test_paired_bootstrap_ci_is_reproducible():
    left = [
        {"true_positive": 10, "true_negative": 90, "false_positive": 5, "false_negative": 15},
        {"true_positive": 20, "true_negative": 80, "false_positive": 2, "false_negative": 8},
    ]
    right = [
        {"true_positive": 12, "true_negative": 88, "false_positive": 4, "false_negative": 14},
        {"true_positive": 18, "true_negative": 82, "false_positive": 3, "false_negative": 7},
    ]
    fn = lambda sample: metric_from_counts(
        sum(u["true_positive"] for u in sample),
        sum(u["true_negative"] for u in sample),
        sum(u["false_positive"] for u in sample),
        sum(u["false_negative"] for u in sample),
    )["micro_fnr"]
    a = paired_bootstrap_ci(left, right, repetitions=200, seed=0, metric_fn=fn)
    b = paired_bootstrap_ci(left, right, repetitions=200, seed=0, metric_fn=fn)
    assert a == b


def test_safe_ordinal_label_respects_support_and_thresholds():
    assert safe_ordinal_label(0.31, 0.33, 0.01, 0.02, 150, 200) == "stable_high_fnr"
    assert safe_ordinal_label(0.31, 0.33, 0.01, 0.02, 50, 60) == "stable_high_fnr_low_support"
    assert safe_ordinal_label(0.10, 0.11, 0.01, 0.02, 200, 200) == "stable_low_fpr"


def test_task_group_stats_computes_micro_and_macro():
    rows = [
        {"task": "hotpotqa", "phase": "prefill", "kv_type": "v", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "gate_vs_mse_oracle", "true_positive": "90", "true_negative": "900", "false_positive": "100", "false_negative": "10"},
        {"task": "samsum", "phase": "prefill", "kv_type": "v", "layer": "0", "kv_head": "0", "bucket": "bucketfirst", "metric": "gate_vs_mse_oracle", "true_positive": "80", "true_negative": "920", "false_positive": "80", "false_negative": "20"},
    ]
    stats = task_group_stats(rows, {"hotpotqa", "samsum"})
    assert stats["micro_fpr"] == 180 / 2000
    assert stats["micro_fnr"] == 30 / 200
    assert stats["task_macro_fpr"] == ((100 / 1000) + (80 / 1000)) / 2
    assert stats["task_macro_fnr"] == ((10 / 100) + (20 / 100)) / 2


def test_parse_int_handles_blank():
    assert parse_int("") is None
    assert parse_int("7") == 7
