from argparse import Namespace
from pathlib import Path

import yaml

from bench.longbench_config import SUBTASKS
from bench.paper_config import apply_method_defaults, method_config_dict
from scripts.run_longbench_paper_8k_single4090 import config_hash


def test_8k_config_is_dedicated_and_complete():
    path = Path("configs/longbench_paper_v2_8k_single4090.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["experiment_name"] == "longbench_paper_v2_8k_single4090"
    assert data["max_input_length"] == 8192
    assert data["paper_alignment"]["max_input_length"] is False
    assert data["methods"] == ["fp16", "kivi_paper_g128", "patternkv_paper"]
    assert data["tasks"] == list(SUBTASKS)
    assert "31.5K" in data["description"]


def test_strict_config_still_uses_31500():
    strict = yaml.safe_load(Path("configs/longbench_paper_v2.yaml").read_text(encoding="utf-8"))
    assert strict["max_input_length"] == 31500


def test_patternkv_and_kivi_affine_bits_are_explicit():
    kivi_args = Namespace(method="kivi_paper_g128", k_bits=4, v_bits=4, group_size=32, residual_length=32, num_k_base=1, num_v_base=1)
    apply_method_defaults(kivi_args)
    assert method_config_dict(kivi_args)["quantized_region_affine_bits"] == 2.25
    pattern_args = Namespace(method="patternkv_paper", k_bits=4, v_bits=4, group_size=32, residual_length=32, num_k_base=1, num_v_base=1)
    apply_method_defaults(pattern_args)
    cfg = method_config_dict(pattern_args)
    assert cfg["quantized_region_affine_bits"] == 2.25
    assert cfg["initial_pattern_count"] == 32
    assert cfg["pattern_group"] == 128


def test_config_hash_depends_on_8192_cap():
    assert config_hash(8192) != config_hash(31500)
