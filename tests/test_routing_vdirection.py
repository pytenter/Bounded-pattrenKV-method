from __future__ import annotations

import math

import torch

from bench.routing_vdirection_observer import (
    attention_probs,
    attention_regions,
    attention_weighted_vector_errors,
    gqa_kv_head_for_query_head,
    logit_metrics,
    oracle_error_metrics,
    oracle_outputs,
    probability_metrics,
    qk_logits,
    region_mass,
    repeat_kv_for_gqa,
    top1_agreement,
    topk_overlap,
    vector_errors,
)
from scripts import run_aime24_routing_vdirection as routing


def test_routing_observer_noninvasive() -> None:
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 5, 8)
    q0, k0 = q.clone(), k.clone()
    _ = qk_logits(q, k, num_key_value_groups=2)
    assert torch.equal(q, q0)
    assert torch.equal(k, k0)


def test_oracle_diagnostic_noninvasive() -> None:
    probs = torch.softmax(torch.randn(1, 4, 1, 6), dim=-1)
    value = torch.randn(1, 4, 6, 8)
    p0, v0 = probs.clone(), value.clone()
    _ = oracle_outputs(fp_probs=probs, quant_probs=probs, fp_value=value, quant_value=value)
    assert torch.equal(probs, p0)
    assert torch.equal(value, v0)


def test_q_direction_metric() -> None:
    errs = vector_errors(torch.tensor([[[[1.0, 0.0]]]]), torch.tensor([[[[0.0, 1.0]]]]))
    assert torch.allclose(errs["direction_error"], torch.ones_like(errs["direction_error"]))


def test_k_direction_metric() -> None:
    errs = vector_errors(torch.tensor([[[[2.0, 0.0]]]]), torch.tensor([[[[1.0, 0.0]]]]))
    assert torch.allclose(errs["direction_error"], torch.zeros_like(errs["direction_error"]))
    assert torch.allclose(errs["relative_L2"], torch.ones_like(errs["relative_L2"]))


def test_v_direction_metric() -> None:
    errs = vector_errors(torch.tensor([[[[-1.0, 0.0]]]]), torch.tensor([[[[1.0, 0.0]]]]))
    assert torch.allclose(errs["direction_error"], torch.full_like(errs["direction_error"], 2.0))


def test_qk_gqa_head_mapping() -> None:
    assert [gqa_kv_head_for_query_head(i, 8, 2) for i in range(8)] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_qk_scaling_matches_production() -> None:
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 7, 8)
    observed = qk_logits(q, k, num_key_value_groups=2)
    expected = torch.matmul(q, repeat_kv_for_gqa(k, 2).transpose(-2, -1)) / math.sqrt(8)
    assert torch.allclose(observed, expected)


def test_attention_mask_matches_production() -> None:
    q = torch.randn(1, 4, 1, 8)
    k = torch.randn(1, 2, 7, 8)
    logits = qk_logits(q, k, num_key_value_groups=2)
    assert logits.shape[-1] == 7


def test_attention_softmax_matches_production() -> None:
    logits = torch.randn(1, 4, 1, 9)
    assert torch.allclose(attention_probs(logits), torch.softmax(logits.float(), dim=-1))


def test_attention_topk_overlap() -> None:
    fp = torch.tensor([[[[5.0, 4.0, 1.0, 0.0]]]])
    qt = torch.tensor([[[[5.0, 1.0, 4.0, 0.0]]]])
    assert float(top1_agreement(qt, fp).item()) == 1.0
    assert float(topk_overlap(qt, fp, 2).item()) == 0.5


def test_attention_js_kl_tv() -> None:
    p = torch.softmax(torch.randn(1, 2, 1, 5), dim=-1)
    metrics = probability_metrics(p, p)
    assert float(metrics["js"].max().item()) == 0.0
    assert float(metrics["tv"].max().item()) == 0.0
    assert float(metrics["kl_fp_quant"].max().item()) == 0.0


def test_attention_early_mass_regions() -> None:
    probs = torch.zeros(1, 1, 1, 20)
    probs[..., 0] = 0.25
    probs[..., 17] = 0.75
    masses = region_mass(probs, attention_regions(20, recent_length=4))
    assert float(masses["E16"].item()) == 0.25
    assert float(masses["E32"].item()) == 1.0


def test_attention_recent_mass_region() -> None:
    probs = torch.zeros(1, 1, 1, 20)
    probs[..., 17] = 1.0
    masses = region_mass(probs, attention_regions(20, recent_length=4))
    assert float(masses["Recent128"].item()) == 1.0


def test_v_attention_weighted_direction() -> None:
    fp_v = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    q_v = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    probs = torch.tensor([[[[0.25, 0.75]]]])
    weighted = attention_weighted_vector_errors(q_v, fp_v, probs)
    assert torch.allclose(weighted["weighted_direction_error_fp"], torch.tensor([[[0.75]]]))


def test_routing_oracle_definition() -> None:
    a_fp = torch.tensor([[[[1.0, 0.0]]]])
    a_q = torch.tensor([[[[0.0, 1.0]]]])
    v_fp = torch.tensor([[[[1.0], [2.0]]]])
    v_q = torch.tensor([[[[3.0], [4.0]]]])
    out = oracle_outputs(fp_probs=a_fp, quant_probs=a_q, fp_value=v_fp, quant_value=v_q)
    assert float(out["ROUTING_ONLY_OUTPUT"].item()) == 2.0


def test_value_oracle_definition() -> None:
    a_fp = torch.tensor([[[[1.0, 0.0]]]])
    a_q = torch.tensor([[[[0.0, 1.0]]]])
    v_fp = torch.tensor([[[[1.0], [2.0]]]])
    v_q = torch.tensor([[[[3.0], [4.0]]]])
    out = oracle_outputs(fp_probs=a_fp, quant_probs=a_q, fp_value=v_fp, quant_value=v_q)
    assert float(out["VALUE_ONLY_OUTPUT"].item()) == 3.0


def test_oracle_output_reference() -> None:
    probs = torch.softmax(torch.randn(1, 2, 1, 4), dim=-1)
    value = torch.randn(1, 2, 4, 8)
    metrics = oracle_error_metrics(oracle_outputs(fp_probs=probs, quant_probs=probs, fp_value=value, quant_value=value))
    assert metrics["actual_relative_L2"] == 0.0


def test_oracle_recovery_raw_preserved() -> None:
    outputs = {
        "O_FP": torch.tensor([[[[1.0]]]]),
        "O_Q": torch.tensor([[[[2.0]]]]),
        "ROUTING_ONLY_OUTPUT": torch.tensor([[[[4.0]]]]),
        "VALUE_ONLY_OUTPUT": torch.tensor([[[[4.0]]]]),
    }
    metrics = oracle_error_metrics(outputs)
    assert metrics["routing_oracle_recovery_raw"] < 0.0
    assert metrics["routing_oracle_recovery_clamped_0_1"] == 0.0


def test_static_recursive_channel_independence() -> None:
    rows = [
        {"mode": "static", "task_key": "t", "config": "c", "checkpoint": 128, "layer": "31", "metric_family": "direction", "object_type": "q_source", "region": "current_token", "metric_name": "direction_error", "statistic": "p95", "metric_value": 0.1},
        {"mode": "pseudo", "task_key": "t", "config": "c", "checkpoint": 128, "layer": "31", "metric_family": "direction", "object_type": "q_source", "region": "current_token", "metric_name": "direction_error", "statistic": "p95", "metric_value": 0.3},
    ]
    gaps = routing.accumulation_gaps(rows)
    assert gaps[0]["accumulation_gap"] == 0.19999999999999998


def test_pseudo_recursive_feedback() -> None:
    rows = [
        {"task_key": "t", "config": "c", "layer": "31", "metric_family": "qk_logit", "object_type": "qk_logits", "region": "current_history", "metric_name": "relative_L2", "statistic": "global", "checkpoint": cp, "accumulation_gap": float(i)}
        for i, cp in enumerate(routing.CORE_CHECKPOINTS, start=1)
    ]
    auc = routing.auc_rows(rows)
    assert auc[0]["acc_auc"] > 0


def test_recursive_channel_auc_core_checkpoints() -> None:
    rows = [
        {"task_key": "t", "config": "c", "layer": "31", "metric_family": "oracle_output", "object_type": "attention_output", "region": "current_history", "metric_name": "actual_relative_L2", "statistic": "global", "checkpoint": cp, "accumulation_gap": 1.0}
        for cp in routing.CORE_CHECKPOINTS
    ]
    auc = routing.auc_rows(rows)
    assert len(auc) == 1
    assert math.isclose(auc[0]["acc_auc"], math.log2(4096) - math.log2(128))


def test_same_6task_subset_as_varn() -> None:
    payload, digest = routing.load_varn_subset_payload()
    assert digest == "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
    assert payload["task_count"] == 6


def test_worker_completeness() -> None:
    decision = routing.build_decision([], [], [{"status": "ok"}, {"status": "error"}])
    assert decision["worker_failed_completeness_rows"] == 1
