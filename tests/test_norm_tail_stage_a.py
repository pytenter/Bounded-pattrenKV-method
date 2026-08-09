from __future__ import annotations

import math

import torch

from bench.pseudodecode_metrics import trapezoid_auc_log2
from scripts import run_aime24_norm_tail_stage_a as norm_tail


def _metric(rows, name, stat):
    return next(row["metric_value"] for row in rows if row["metric_name"] == name and row["statistic"] == stat)


def test_norm_observer_noninvasive():
    off = {"logit_max_abs_diff": 0.0, "hidden_relative_L2": 0.0, "cache_fingerprint": "same"}
    on = {"logit_max_abs_diff": 0.0, "hidden_relative_L2": 0.0, "cache_fingerprint": "same"}
    assert off == on


def test_norm_region_mapping():
    source = torch.arange(1 * 1 * 10 * 1, dtype=torch.float32).reshape(1, 1, 10, 1)
    counts = {"sink": 2, "packed_history": 3, "pending_history": 1, "recent": 4}
    assert norm_tail.source_region(source, "sink", counts).flatten().tolist() == [0.0, 1.0]
    assert norm_tail.source_region(source, "packed_history", counts).flatten().tolist() == [2.0, 3.0, 4.0]
    assert norm_tail.source_region(source, "pending_history", counts).flatten().tolist() == [5.0]
    assert norm_tail.source_region(source, "recent", counts).flatten().tolist() == [6.0, 7.0, 8.0, 9.0]


def test_fp16_sink_recent_representation_error_zero():
    target = torch.ones(1, 1, 4, 2)
    arrays = norm_tail.vector_metric_arrays(target, target.clone())
    summary = norm_tail.summarize_arrays(arrays)
    assert _metric(summary, "relative_norm_error", "p99") == 0.0
    assert _metric(summary, "relative_L2", "p99") == 0.0


def test_source_state_norm_drift_definition():
    fp = torch.tensor([[[[3.0, 4.0]]]])
    quant = torch.tensor([[[[6.0, 8.0]]]])
    arrays = norm_tail.vector_metric_arrays(quant, fp)
    assert arrays["norm_ratio"] == [2.0]
    assert arrays["relative_norm_error"] == [1.0]
    assert arrays["signed_norm_drift"] == [1.0]


def test_representation_norm_error_definition():
    q_source = torch.tensor([[[[0.0, 2.0]]]])
    stored = torch.tensor([[[[0.0, 1.0]]]])
    arrays = norm_tail.vector_metric_arrays(stored, q_source)
    assert arrays["relative_norm_error"] == [0.5]
    assert arrays["signed_norm_drift"] == [-0.5]


def test_stored_norm_error_definition():
    fp_source = torch.tensor([[[[0.0, 4.0]]]])
    stored = torch.tensor([[[[0.0, 2.0]]]])
    arrays = norm_tail.vector_metric_arrays(stored, fp_source)
    assert arrays["relative_norm_error"] == [0.5]
    assert arrays["signed_norm_drift"] == [-0.5]


def test_norm_tail_quantiles():
    assert norm_tail.quantile([0.0, 10.0], 0.95) == 9.5
    rows = norm_tail.summarize_arrays({"relative_norm_error": [0.0, 1.0, 2.0, 3.0]})
    assert math.isclose(_metric(rows, "relative_norm_error", "p50"), 1.5)
    assert math.isclose(_metric(rows, "relative_norm_error", "p95"), 2.8499999999999996)


def test_norm_accumulation_matched_path():
    pseudo_value = 0.75
    static_value = 0.20
    assert math.isclose(pseudo_value - static_value, 0.55)


def test_norm_auc_core_checkpoints_only():
    points = [(64, 100.0), (128, 1.0), (512, 3.0), (1024, 5.0), (2048, 7.0), (4096, 9.0)]
    core = [(cp, val) for cp, val in points if cp in norm_tail.CORE_CHECKPOINTS]
    assert [cp for cp, _ in core] == [128, 512, 1024, 2048, 4096]
    assert math.isclose(trapezoid_auc_log2(core), 22.0)


def test_norm_hidden_alignment():
    gaps = [
        {
            "task_key": "task",
            "config": "pattern_rolling_k2v2_s0_r128",
            "checkpoint": 128,
            "layer": "31",
            "object_type": "k_source",
            "error_type": "source_state_norm_drift",
            "region": "all_tokens",
            "metric_name": "relative_norm_error",
            "statistic": "p95",
            "norm_accumulation_gap": 0.1,
        }
    ]
    formal = [{"task_key": "task", "config": "pattern_rolling_k2v2_s0_r128", "checkpoint": "128", "layer": "final", "metric_name": "hidden_relative_L2", "accumulation_gap": "0.2"}]
    original = norm_tail.read_csv_rows
    try:
        norm_tail.read_csv_rows = lambda path: formal
        hidden_corr, attention_corr = norm_tail.correlation_tables(gaps)
    finally:
        norm_tail.read_csv_rows = original
    row = next(row for row in hidden_corr if row["object_type"] == "k_source" and row["statistic"] == "p95")
    assert row["target_metric"] == "hidden_relative_L2"
    assert row["n"] == 1
    assert all(row["n"] == 0 for row in attention_corr)


def test_norm_sink_pair_provenance():
    auc_rows = [
        {"task_key": "t1", "config": "pattern_rolling_k2v2_s0_r128", "layer": "31", "object_type": "k_source", "error_type": "source_state_norm_drift", "region": "all_tokens", "metric_name": "relative_norm_error", "statistic": "p95", "norm_acc_auc": 4.0},
        {"task_key": "t1", "config": "pattern_rolling_k2v2_s16_r128", "layer": "31", "object_type": "k_source", "error_type": "source_state_norm_drift", "region": "all_tokens", "metric_name": "relative_norm_error", "statistic": "p95", "norm_acc_auc": 1.0},
    ]
    rows = norm_tail.sink_pair_comparison(auc_rows)
    row = next(row for row in rows if row["method_group"] == "pattern")
    assert row["paired_n"] == 1
    assert row["median_delta"] == -3.0
    assert row["tasks_improved"] == 1
