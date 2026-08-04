from pathlib import Path

from insight.config import CANONICAL_METHODS, load_standard_baselines


def test_standard_baseline_config_is_source_of_truth():
    cfg = load_standard_baselines(Path("configs/standard_baselines.paper_v2.yaml"))
    assert cfg.canonical_methods == CANONICAL_METHODS
    assert cfg.methods["kivi_paper_g128"]["group_size"] == 128
    assert cfg.methods["patternkv_paper"]["initial_pattern_count"] == 32


def test_record_validation_rejects_legacy_method():
    cfg = load_standard_baselines(Path("configs/standard_baselines.paper_v2.yaml"))
    assert cfg.validate_record({"method": "kivi", "quantization_config": {}})
