from __future__ import annotations

from copy import deepcopy

from bench.aime_generation_provenance import (
    build_portable_reference_generation_semantics,
    experiment_config_set_hash,
    portable_reference_generation_hash,
    recompute_legacy_generation_hash,
)


LEGACY_GENERATION_CONFIG = {
    "batch_size": 1,
    "configs": [
        "pattern_legacy_chunked_k2v2_r128",
        "pattern_rolling_k2v2_s0_r128",
        "pattern_rolling_k2v2_s64_r256",
        "pattern_rolling_k4v2_s0_r128",
        "pattern_rolling_k2v4_s0_r128",
        "kivi_legacy_chunked_k2v2_r128",
        "kivi_rolling_k2v2_s0_r128",
        "kivi_rolling_k2v2_s64_r256",
    ],
    "do_sample": True,
    "dtype": "float16",
    "manifest_hash": "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e",
    "max_model_len": 131072,
    "max_new_tokens": 32768,
    "model": "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B",
    "seed": 42,
    "task_count": 12,
    "temperature": 0.6,
    "top_p": 0.95,
}


TASKS = [
    {"task_key": "aime24:p12:s0:seed12042", "problem_id": 12, "sample_id": 0, "seed": 12042},
    {"task_key": "aime24:p14:s0:seed14042", "problem_id": 14, "sample_id": 0, "seed": 14042},
]


def portable_payload(**overrides):
    payload = build_portable_reference_generation_semantics(
        task_manifest_sha256="ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e",
        tasks=TASKS,
        model_name="DeepSeek-R1-Distill-Llama-8B",
        model_identity_hash="modelhash",
        tokenizer_identity_hash="tokhash",
        model_dtype="float16",
        context_limit=131072,
        resolved_pad_token_id=128001,
        resolved_eos_token_ids=[128001, 128009],
    )
    payload.update(overrides)
    return payload


def test_legacy_generation_hash_reconstruction():
    assert recompute_legacy_generation_hash(LEGACY_GENERATION_CONFIG) == "a7d6b2f8bab37893b6331c66b3e5eb6a"


def test_portable_hash_ignores_absolute_model_path():
    left = portable_payload()
    right = portable_payload()
    left_metadata_path = "/home/qinch2023/model/DeepSeek-R1-Distill-Llama-8B"
    right_metadata_path = "/data/zypan/model/DeepSeek-R1-Distill-Llama-8B"
    assert left_metadata_path != right_metadata_path
    assert portable_reference_generation_hash(left) == portable_reference_generation_hash(right)


def test_portable_hash_changes_on_true_semantic_changes():
    base = portable_payload()
    base_hash = portable_reference_generation_hash(base)
    for change in (
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"force_think_prefix": False, "think_prefix": ""},
        {"repetition_penalty": 1.1},
        {"task_seed_map": [["aime24:p12:s0:seed12043", 12043]]},
        {"tokenizer_identity_hash": "different-tokenizer"},
    ):
        changed = deepcopy(base)
        changed.update(change)
        assert portable_reference_generation_hash(changed) != base_hash


def test_path_only_changes_do_not_change_portable_hash():
    base = portable_payload()
    path_changed = portable_payload()
    metadata = {
        "model_local_path": "/tmp/not-in-payload",
        "result_dir": "/tmp/results",
        "report_dir": "/tmp/reports",
        "server_hostname": "host-a",
        "gpu_id": 7,
    }
    assert metadata
    assert portable_reference_generation_hash(base) == portable_reference_generation_hash(path_changed)


def test_config_set_independent_from_reference_semantics():
    reference_hash = portable_reference_generation_hash(portable_payload())
    configs_a = [{"config": "pattern_s0", "method": "patternkv", "sink_length": 0, "recent_length": 128}]
    configs_b = [{"config": "pattern_s16", "method": "patternkv", "sink_length": 16, "recent_length": 128}]
    assert portable_reference_generation_hash(portable_payload()) == reference_hash
    assert experiment_config_set_hash(configs_a) != experiment_config_set_hash(configs_b)
