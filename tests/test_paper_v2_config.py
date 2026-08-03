from argparse import Namespace

from bench.longbench_config import MAX_NEW_TOKENS, METRIC_NAMES, PROMPT_TEMPLATES, SUBTASKS, expected_samples
from bench.paper_config import apply_method_defaults, kivi_quantized_region_bits, method_config_dict, pattern_boundary_events


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
