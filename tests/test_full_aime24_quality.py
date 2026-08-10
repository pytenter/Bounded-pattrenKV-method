from __future__ import annotations

import math

import torch

from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from bench.aime_utils import effective_seed, load_aime24
from models.segmented_cache import PatternQuantizedKVCache, build_cache_from_prefill, select_value_precision_mask
from scripts.run_aime24_full_causal25_quality import (
    BASE_SEEDS,
    METHOD_CONFIGS,
    METHOD_ORDER,
    block_bootstrap,
    classify,
    compact_record,
    effective_bits_for_method,
    ensure_complete_records,
    experiment_hash,
    frozen_generation_config,
    is_current_record,
    method_generation_hash,
    paired_counts,
    question_level_rows,
    set_selector_task_context,
    transition_rows,
    v4_realized_by_layer_rows,
)


def test_full_aime24_has_30_unique_questions() -> None:
    rows = load_aime24()
    assert len(rows) == 30
    assert {int(r["problem_id"]) for r in rows} == set(range(30))
    assert len({r["problem"] for r in rows}) == 30


def test_full_aime24_gold_answers() -> None:
    rows = load_aime24()
    assert all(normalize_aime_answer(row["answer"]) is not None for row in rows)


def test_aime_answer_parser() -> None:
    assert parse_aime_answer(r"After checking, \boxed{321}.")["parsed_answer"] == "321"
    assert parse_aime_answer(r"Final answer is \boxed{\frac{246}{2}}")["parsed_answer"] == "123"
    assert parse_aime_answer("no numeric answer")["parser_strategy"] == "failure"


def test_generation_seed_same_across_methods() -> None:
    seeds = {method: effective_seed(43, 17, 0) for method in METHOD_ORDER}
    assert len(set(seeds.values())) == 1
    assert seeds["FP16"] == 43 + 17 * 1000


def test_method_config_frozen() -> None:
    gen = frozen_generation_config()
    assert gen["do_sample"] is True
    assert gen["temperature"] == 0.6
    assert gen["top_p"] == 0.95
    assert gen["max_new_tokens"] == 32768
    assert METHOD_CONFIGS["PATTERN_BASE"]["config_name"] == "pattern_rolling_k2v2_s16_r128"
    assert METHOD_CONFIGS["CAUSAL_V4_25"]["config_name"] == "pattern_rolling_k2v2_s16_r128_causal_v4_b025"
    assert gen["patternkv_selector_task_key"] == "task_key3(problem_id, sample_id=0, effective_seed)"
    assert experiment_hash()


def test_random25_budget() -> None:
    cfg = METHOD_CONFIGS["RANDOM_V4_25"]
    assert cfg["selector"] == "random_v4"
    assert cfg["v4_budget_fraction"] == 0.25


def test_causal25_budget() -> None:
    cfg = METHOD_CONFIGS["CAUSAL_V4_25"]
    assert cfg["selector"] == "causal_v4"
    assert cfg["v4_budget_fraction"] == 0.25


def test_random_causal_same_bit() -> None:
    keys = ("k_bits", "v_bits", "sink_length", "recent_length", "residual_length", "group_size", "v4_budget_fraction")
    assert {k: METHOD_CONFIGS["RANDOM_V4_25"][k] for k in keys} == {k: METHOD_CONFIGS["CAUSAL_V4_25"][k] for k in keys}
    assert effective_bits_for_method("RANDOM_V4_25", 0.25) == effective_bits_for_method("CAUSAL_V4_25", 0.25)


def test_causal25_no_future_leakage() -> None:
    cache = PatternQuantizedKVCache(group_size=16, v_precision_selector="causal_v4", v4_budget_fraction=0.25)
    cache.v_causal_importance = torch.tensor([[0.0, 2.0, 0.0, 1.0]])
    v = torch.randn(1, 1, 4, 16)
    m = torch.zeros(1, 1, 4, dtype=torch.bool)
    first = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    cache.v_oracle_importance = torch.tensor([[999.0, 0.0, 999.0, 0.0]])
    second = select_value_precision_mask(cache, v, torch.zeros_like(v), m, absolute_start=0)
    assert torch.equal(first, second)


def test_worker_gpu_isolation() -> None:
    physical = ["1", "2", "3", "4"]
    mapping = dict(zip(("GPU_A", "GPU_B", "GPU_C", "GPU_D"), physical))
    assert len(set(mapping.values())) == 4
    assert all(logical == "cuda:0" for logical in ["cuda:0"] * 4)


def test_selector_task_context_is_per_generation() -> None:
    class Attn:
        selector_task_key = "old"
        v_causal_importance = object()
        v_oracle_importance = object()

    class Layer:
        self_attn = Attn()

    class Inner:
        layers = [Layer()]

    class Model:
        model = Inner()

        class Config:
            patternkv_selector_task_key = "old"

        config = Config()

    model = Model()
    set_selector_task_context(model, "aime24:p1:s0:seed1042")
    assert model.config.patternkv_selector_task_key == "aime24:p1:s0:seed1042"
    assert model.model.layers[0].self_attn.selector_task_key == "aime24:p1:s0:seed1042"
    assert model.model.layers[0].self_attn.v_causal_importance is None
    assert model.model.layers[0].self_attn.v_oracle_importance is None


def test_current_record_requires_matching_provenance() -> None:
    rec = {"method": "RANDOM_V4_25", "formal_config_hash": experiment_hash(), "generation_config_hash": method_generation_hash("RANDOM_V4_25")}
    assert is_current_record(rec)
    stale = {**rec, "formal_config_hash": "old"}
    assert not is_current_record(stale)


def test_compact_result_schema(tmp_path) -> None:
    raw = tmp_path / "raw.txt"
    raw.write_text("reasoning", encoding="utf-8")
    rec = {
        "experiment_id": "x",
        "method": "patternkv",
        "config_name": METHOD_CONFIGS["CAUSAL_V4_25"]["config_name"],
        "problem_id": 0,
        "sample_id": 0,
        "seed": effective_seed(42, 0, 0),
        "base_seed": 42,
        "problem": "p",
        "reference_answer": "1",
        "parsed_answer": "1",
        "is_correct": True,
        "parser_strategy": "boxed",
        "stop_reason": "eos",
        "input_tokens": 10,
        "generated_tokens": 20,
        "length_truncated": False,
        "wall_time_seconds": 1.0,
        "gpu_name": "NVIDIA GeForce RTX 3090",
        "patternkv_dynamic_stats": {"v_precision_v4_tokens_per_layer": [4], "v_precision_total_tokens_per_layer": [16]},
        "git_commit": "abc",
        "timestamp": "now",
        "model_path": "/models/DeepSeek-R1-Distill-Llama-8B",
    }
    out = compact_record(rec, phase="formal", method_id="CAUSAL_V4_25", physical_gpu="3", raw_path=raw, raw_sha256="sha", generation_hash=method_generation_hash("CAUSAL_V4_25"))
    for key in ("method", "base_seed", "effective_seed", "problem_id", "gold_answer", "parsed_answer", "correct", "generated_tokens", "raw_generation_sha256"):
        assert key in out
    assert "generated_text" not in out
    assert out["v4_realized_fraction"] == 0.25


def test_resume_key_unique() -> None:
    rows = [{"status": "completed", "method": m, "base_seed": s, "problem_id": p} for m in METHOD_ORDER for s in BASE_SEEDS for p in range(30)]
    check = ensure_complete_records(rows)
    assert check["expected_generations"] == 360
    assert check["completed_generations"] == 360
    assert check["missing"] == []


def _sample_quality_rows() -> list[dict]:
    rows = []
    for method in METHOD_ORDER:
        for seed in BASE_SEEDS:
            for pid in range(30):
                correct = method == "CAUSAL_V4_25" and pid < 15
                if method == "RANDOM_V4_25":
                    correct = pid < 12
                if method == "PATTERN_BASE":
                    correct = pid < 10
                if method == "FP16":
                    correct = pid < 18
                rows.append({"status": "completed", "method": method, "base_seed": seed, "problem_id": pid, "correct": correct, "generated_tokens": 100 + pid, "length_truncated": False, "stop_reason": "eos"})
    return rows


def test_question_level_pairing() -> None:
    qrows = question_level_rows(_sample_quality_rows())
    assert len(qrows) == 30
    assert qrows[0]["causal_minus_random"] == 0.0
    assert qrows[13]["causal_minus_random"] == 1.0


def test_block_bootstrap_by_question() -> None:
    boot = block_bootstrap(question_level_rows(_sample_quality_rows()))
    assert boot["bootstrap_unit"] == "question"
    assert boot["resamples"] == 10000
    assert boot["causal_minus_random"]["ci95_low"] <= boot["causal_minus_random"]["ci95_high"]


def test_accuracy_summary() -> None:
    rows = _sample_quality_rows()
    transitions = transition_rows(rows)
    qrows = question_level_rows(rows)
    summary = [
        {"method": "FP16", "mean_accuracy": 18 / 30},
        {"method": "PATTERN_BASE", "mean_accuracy": 10 / 30},
        {"method": "RANDOM_V4_25", "mean_accuracy": 12 / 30},
        {"method": "CAUSAL_V4_25", "mean_accuracy": 15 / 30},
    ]
    seed_rows = [{"method": m, "base_seed": s, "accuracy": next(r["mean_accuracy"] for r in summary if r["method"] == m)} for m in METHOD_ORDER for s in BASE_SEEDS]
    decision = classify(summary, transitions, qrows, seed_rows)
    assert decision["FULL_AIME24_METHOD_CLASSIFICATION"] == "SUPPORTED"
    assert decision["CAUSAL25_beats_RANDOM25_on_ge_2_of_3_seeds"] is True


def test_harmful_classification_takes_precedence() -> None:
    summary = [
        {"method": "FP16", "mean_accuracy": 0.5},
        {"method": "PATTERN_BASE", "mean_accuracy": 0.4},
        {"method": "RANDOM_V4_25", "mean_accuracy": 0.4},
        {"method": "CAUSAL_V4_25", "mean_accuracy": 0.3},
    ]
    seed_rows = [
        {"method": "PATTERN_BASE", "base_seed": seed, "accuracy": 0.4}
        for seed in BASE_SEEDS
    ] + [
        {"method": "RANDOM_V4_25", "base_seed": seed, "accuracy": 0.4}
        for seed in BASE_SEEDS
    ] + [
        {"method": "CAUSAL_V4_25", "base_seed": seed, "accuracy": 0.3}
        for seed in BASE_SEEDS
    ]
    decision = classify(summary, [], [{"causal_minus_random": -0.1}], seed_rows)
    assert decision["FULL_AIME24_METHOD_CLASSIFICATION"] == "HARMFUL"


def test_v4_realized_by_layer_export_rows() -> None:
    rows = [
        {
            "method": "RANDOM_V4_25",
            "base_seed": 42,
            "effective_seed": 42,
            "problem_id": 0,
            "selector_task_key": "aime24:p0:s0:seed42",
            "generation_config_hash": "g",
            "formal_config_hash": experiment_hash(),
            "v4_realized_by_layer": [{"layer": 0, "v4_tokens": 4, "total_tokens": 16, "fraction": 0.25}],
        },
        {"method": "FP16", "v4_realized_by_layer": [{"layer": 0, "v4_tokens": 16, "total_tokens": 16, "fraction": 1.0}]},
    ]
    out = v4_realized_by_layer_rows(rows)
    assert out == [
        {
            "method": "RANDOM_V4_25",
            "base_seed": 42,
            "effective_seed": 42,
            "problem_id": 0,
            "selector_task_key": "aime24:p0:s0:seed42",
            "layer": 0,
            "v4_tokens": 4,
            "total_tokens": 16,
            "fraction": 0.25,
            "generation_config_hash": "g",
            "formal_config_hash": experiment_hash(),
        }
    ]


def test_transition_matrix() -> None:
    trans = transition_rows(_sample_quality_rows())
    counts = paired_counts(trans)
    assert counts["causal_win_random_loss"] == 9
    assert counts["random_win_causal_loss"] == 0
    assert counts["tie_random_causal"] == 81


def test_budget_mask_counts_exact_for_25_percent() -> None:
    torch.manual_seed(2026)
    k = torch.randn(1, 2, 16, 16)
    v = torch.randn(1, 2, 16, 16)
    c = torch.randn(2, 5, 16)
    random_cache = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="random_v4", v4_budget_fraction=0.25)
    causal_cache = build_cache_from_prefill(k, v, sink_length=0, recent_length=0, group_size=16, k_bits=2, v_bits=2, pattern=True, k_centroids=c, v_centroids=c, v_precision_selector="causal_v4", v4_budget_fraction=0.25)
    assert int(random_cache.v_precision_mask.sum()) == 4
    assert int(causal_cache.v_precision_mask.sum()) == 4
    assert math.isclose(float(random_cache.v_precision_mask.float().mean()), 0.25)
