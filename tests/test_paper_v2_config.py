from argparse import Namespace

from bench.longbench_config import MAX_NEW_TOKENS, METRIC_NAMES, PROMPT_TEMPLATES, SUBTASKS, expected_samples
from bench.paper_config import apply_method_defaults, kivi_quantized_region_bits, method_config_dict, pattern_boundary_events
from scripts.run_gsm8k_selector_components_pilot import METHODS as GSM8K_COMPONENT_METHODS, SELECTORS as GSM8K_COMPONENT_SELECTORS, shared_patternkv_config
from scripts.run_longbench_paper_8k_single4090 import BASELINE_METHODS


def test_longbench_paper_v2_panel_is_complete():
    assert len(SUBTASKS) == 21
    assert set(SUBTASKS).issubset(PROMPT_TEMPLATES)
    assert set(SUBTASKS).issubset(MAX_NEW_TOKENS)
    assert set(SUBTASKS).issubset(METRIC_NAMES)
    assert expected_samples("multifieldqa_en") == 150
    assert expected_samples("lcc") == 500
    assert expected_samples("repobench-p") == 500
    assert expected_samples("qasper") == 200


def test_kivi_paper_g128_forces_paper_bits_and_group():
    args = Namespace(method="kivi_paper_g128", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    cfg = apply_method_defaults(args)
    assert cfg.backend_method == "kivi_official"
    assert args.k_bits == 2
    assert args.v_bits == 2
    assert args.group_size == 128
    assert args.residual_length == 128
    assert abs(kivi_quantized_region_bits(args.group_size, args.k_bits) - 2.25) < 1e-9
    assert method_config_dict(args)["kivi_quantized_region_theoretical_bits"] == 2.25


def test_patternkv_paper_forces_patterns_after_rope():
    args = Namespace(method="patternkv_paper", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    cfg = apply_method_defaults(args)
    assert cfg.backend_method == "patternkv"
    assert args.k_bits == 2
    assert args.v_bits == 2
    assert args.group_size == 128
    assert args.residual_length == 128
    assert args.num_k_base == 32
    assert args.num_v_base == 32
    assert cfg.pattern_selection_position == "post-RoPE key/value states"
    assert pattern_boundary_events(127, 128) == []
    assert pattern_boundary_events(128, 128) == [128]
    assert pattern_boundary_events(129, 128) == [128]


def test_causal_v4_25_uses_frozen_aime24_config():
    args = Namespace(method="causal_v4_25", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    cfg = apply_method_defaults(args)
    assert cfg.backend_method == "patternkv"
    assert args.k_bits == 2
    assert args.v_bits == 2
    assert args.group_size == 128
    assert args.residual_length == 128
    assert args.sink_length == 16
    assert args.recent_length == 128
    assert args.num_k_base == 32
    assert args.num_v_base == 32
    assert args.patternkv_v_precision_selector == "causal_v4"
    assert args.patternkv_v4_budget_fraction == 0.25
    assert cfg.value_precision_selector == "causal_v4"
    assert cfg.v4_budget_fraction == 0.25
    assert method_config_dict(args)["cache_mode"] == "segmented_rolling"


def test_gsm8k_component_methods_use_matched_selector_only_config():
    shared = shared_patternkv_config()
    assert shared["patternkv_v4_budget_fraction"] == 0.25
    assert set(GSM8K_COMPONENT_METHODS) == {"importance_only_v4_25", "error_only_v4_25", "causal_v4_25"}
    for method, selector in GSM8K_COMPONENT_SELECTORS.items():
        args = Namespace(method=method, k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
        cfg = apply_method_defaults(args)
        cfg_dict = method_config_dict(args)
        assert cfg.backend_method == "patternkv"
        assert cfg.value_precision_selector == selector
        assert args.patternkv_v_precision_selector == selector
        assert args.patternkv_v4_budget_fraction == 0.25
        assert args.k_bits == 2
        assert args.v_bits == 2
        assert args.group_size == 128
        assert args.sink_length == 16
        assert args.recent_length == 128
        assert args.residual_length == 128
        assert args.num_k_base == 32
        assert args.num_v_base == 32
        assert cfg_dict["patternkv_cache_path"] == "segmented"
        assert cfg_dict["cache_mode"] == "segmented_rolling"


def test_longbench_protocol_hash_keeps_baseline_method_set():
    assert BASELINE_METHODS == ("fp16", "kivi_paper_g128", "patternkv_paper")
