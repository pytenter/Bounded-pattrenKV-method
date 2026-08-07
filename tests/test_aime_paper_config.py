from argparse import Namespace

from bench.aime_utils import DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_P
from bench.paper_config import apply_method_defaults


def test_defaults():
    assert DEFAULT_MAX_NEW_TOKENS == 32768
    assert DEFAULT_TEMPERATURE == 0.6
    assert DEFAULT_TOP_P == 0.95


def test_kivi_and_patternkv_config():
    k = Namespace(method="kivi_paper_g128", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    pk = Namespace(method="patternkv_paper", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    apply_method_defaults(k)
    apply_method_defaults(pk)
    assert (k.k_bits, k.v_bits, k.group_size, k.residual_length) == (2, 2, 128, 128)
    assert (pk.k_bits, pk.v_bits, pk.group_size, pk.residual_length, pk.num_k_base, pk.num_v_base) == (2, 2, 128, 128, 32, 32)


def test_legacy_recent_zero_keeps_chunk_residual_length():
    args = Namespace(
        method="patternkv",
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        sink_length=0,
        recent_length=0,
        num_k_base=32,
        num_v_base=32,
    )
    cfg = apply_method_defaults(args)
    assert args.recent_length == 0
    assert args.residual_length == 128
    assert cfg.recent_length == 0
    assert cfg.residual_length == 128


def test_schema_fields_minimal():
    required = {"experiment_id", "dataset", "model_path", "method", "problem_id", "sample_id", "task_key", "seed", "problem", "reference_answer", "rendered_prompt", "input_tokens", "max_new_tokens", "generated_text", "generated_tokens", "parsed_answer", "stop_reason", "cache_bitwidth_stats", "git_commit", "timestamp"}
    assert "cache_bitwidth_stats" in required
    assert "max_new_tokens" in required
