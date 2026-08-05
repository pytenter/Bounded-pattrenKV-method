from scripts.compare_insight_wave_a_hardware import compare_row_sets, compute_v_gate_stats, metric_summary


def test_compare_row_sets_finds_4090_only_rows():
    v100_rows = [
        {"task": "a", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "0", "bucket": "b0", "metric": "m"},
    ]
    gpu4090_rows = v100_rows + [
        {"task": "a", "phase": "decode", "kv_type": "k", "layer": "0", "kv_head": "1", "bucket": "b0", "metric": "m"},
    ]
    summary = compare_row_sets(v100_rows, gpu4090_rows, ["task", "phase", "kv_type", "layer", "kv_head", "bucket", "metric"])
    assert summary["common_rows"] == 1
    assert summary["only_v100_rows"] == 0
    assert summary["only_4090_rows"] == 1


def test_compute_v_gate_stats_distinguishes_micro_and_macro():
    rows = [
        {
            "true_positive": "90",
            "true_negative": "900",
            "false_positive": "100",
            "false_negative": "10",
            "total": "1100",
            "false_positive_rate": str(100 / 1000),
            "false_negative_rate": str(10 / 100),
        },
        {
            "true_positive": "1",
            "true_negative": "0",
            "false_positive": "1",
            "false_negative": "9",
            "total": "11",
            "false_positive_rate": "1.0",
            "false_negative_rate": str(9 / 10),
        },
    ]
    stats = compute_v_gate_stats(rows)
    assert round(stats["micro_fpr"], 6) == round(101 / 1001, 6)
    assert round(stats["micro_fnr"], 6) == round(19 / 110, 6)
    assert round(stats["macro_fpr_mean"], 6) == round(((100 / 1000) + 1.0) / 2, 6)
    assert round(stats["macro_fnr_mean"], 6) == round(((10 / 100) + (9 / 10)) / 2, 6)


def test_compute_v_gate_stats_handles_zero_denominator():
    rows = [
        {
            "true_positive": "0",
            "true_negative": "0",
            "false_positive": "0",
            "false_negative": "0",
            "total": "0",
            "false_positive_rate": "",
            "false_negative_rate": "",
        }
    ]
    stats = compute_v_gate_stats(rows)
    assert stats["micro_fpr"] is None
    assert stats["micro_fnr"] is None
    assert stats["macro_fpr_mean"] is None
    assert stats["macro_fnr_mean"] is None


def test_metric_summary_weighted_mean_uses_counts():
    rows = [
        {"count": 1, "mean": 10.0},
        {"count": 9, "mean": 0.0},
    ]
    summary = metric_summary(rows)
    assert summary["macro_mean"] == 5.0
    assert summary["weighted_mean"] == 1.0
