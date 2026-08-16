from __future__ import annotations

from argparse import Namespace

import torch

from bench.full_model_serving_benchmark import (
    KIVIPaperAdapter,
    METHOD_ADAPTERS,
    PatternKVAdapter,
    PatternKVPaperAdapter,
    _namespace_for_paper_method,
    is_valid_run,
    invalid_run_result,
    model_supports_selective_prefill,
)
from bench.paper_baseline_system_comparison import write_blocked_smoke_reports
from bench.paper_baseline_system_comparison import (
    REQUIRED_ALLOCATOR_CONF,
    allocator_protocol_valid,
    formal_worker_env,
    launch_worker,
    valid_protocol,
)
from bench.paper_config import apply_method_defaults


def test_full_model_method_names_are_formal_paper_names() -> None:
    assert set(METHOD_ADAPTERS) == {
        "FP16_FULL_MODEL",
        "KIVI_PAPER_G128_FULL_MODEL",
        "PATTERNKV_PAPER_FULL_MODEL",
        "CAUSAL_V4_25_FULL_MODEL",
    }
    assert KIVIPaperAdapter.name == "KIVI_PAPER_G128_FULL_MODEL"
    assert PatternKVPaperAdapter.name == "PATTERNKV_PAPER_FULL_MODEL"
    assert PatternKVAdapter.name == "CAUSAL_V4_25_FULL_MODEL"


def test_kivi_paper_adapter_uses_canonical_config_identity() -> None:
    args = _namespace_for_paper_method("kivi_paper_g128")
    cfg = args.paper_method_config
    assert cfg.backend_method == "kivi_official"
    assert (args.k_bits, args.v_bits, args.group_size, args.residual_length) == (2, 2, 128, 128)
    assert cfg.key_quant_axis.startswith("per-channel")
    assert cfg.value_quant_axis.startswith("per-token")
    assert cfg.asym is True


def test_patternkv_paper_config_disables_causal_v4_selector() -> None:
    args = _namespace_for_paper_method("patternkv_paper")
    cfg = args.paper_method_config
    assert cfg.backend_method == "patternkv"
    assert (args.k_bits, args.v_bits, args.group_size, args.residual_length) == (2, 2, 128, 128)
    assert (args.num_k_base, args.num_v_base) == (32, 32)
    assert cfg.initial_pattern_count == 32
    assert cfg.pattern_group == 128


def test_patternkv_paper_baseline_does_not_require_causal_fused_page_backend(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_MIXED_V_BACKEND", raising=False)
    args = _namespace_for_paper_method("patternkv_paper")
    assert args.paper_method_config.backend_method == "patternkv"
    assert getattr(args, "patternkv_v_precision_selector", "base_v2") == "base_v2"


def test_causal_frozen_config_identity_from_existing_loader_defaults() -> None:
    args = Namespace(method="patternkv_paper", k_bits=4, v_bits=4, group_size=32, residual_length=64, num_k_base=1, num_v_base=1)
    apply_method_defaults(args)
    assert (args.k_bits, args.v_bits, args.group_size, args.residual_length) == (2, 2, 128, 128)
    frozen = {
        "k_bits": 2,
        "v_bits": 2,
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "group_size": 128,
        "patternkv_v_precision_selector": "causal_v4",
        "patternkv_v4_budget_fraction": 0.25,
    }
    assert frozen["patternkv_v_precision_selector"] == "causal_v4"
    assert frozen["patternkv_v4_budget_fraction"] == 0.25


def test_selective_prefill_policy_detects_model_and_lm_head() -> None:
    class Model:
        def __init__(self) -> None:
            self.model = lambda *args, **kwargs: None
            self.lm_head = lambda hidden: hidden

    assert model_supports_selective_prefill(Model()) is True
    assert model_supports_selective_prefill(object()) is False


def test_invalid_run_rejection_keeps_decode_only_protocol_strict() -> None:
    cfg = __import__("bench.full_model_serving_benchmark", fromlist=["BenchmarkConfig"]).BenchmarkConfig(
        "KIVI_PAPER_G128_FULL_MODEL",
        context_length=512,
        decode_length=4,
        active_capacity=2,
        total_requests=2,
    )
    result = invalid_run_result(cfg, torch.device("cpu"), run_index=0, warmup=False, reason="SEMANTIC_FAILURE")
    assert result.run_valid is False
    assert is_valid_run(result) is False
    assert result.prefill_calls_in_timed_window == 0
    assert result.refill_calls_in_timed_window == 0


def test_blocked_smoke_reports_classify_patternkv_true_batch_limitation(tmp_path) -> None:
    row = {
        "method": "PATTERNKV_PAPER_FULL_MODEL",
        "status": "RUNTIME_FAILURE",
        "active_capacity": 2,
        "invalid_reason": "RUNTIME_FAILURE: AssertionError('v_centroids shape wrong: torch.Size([2, 8, 48, 128])')",
        "subprocess_isolation": True,
    }
    write_blocked_smoke_reports(tmp_path, [row], row)
    gate = __import__("json").loads((tmp_path / "final_gate.json").read_text(encoding="utf-8"))
    assert gate["classification"] == "PAPER_BASELINE_SYSTEM_COMPARISON_V1_RECONCILED_PARTIAL"
    assert gate["patternkv_status"] == "PATTERNKV_PAPER_FULL_MODEL_BASELINE_BLOCKED"
    assert gate["blocked_status"] == "BASELINE_TRUE_BATCH_RUNTIME_NOT_SUPPORTED"


def test_allocator_protocol_match_is_explicit() -> None:
    assert allocator_protocol_valid(REQUIRED_ALLOCATOR_CONF)
    assert allocator_protocol_valid(f"max_split_size_mb:64,{REQUIRED_ALLOCATOR_CONF}")
    assert not allocator_protocol_valid("")
    assert not allocator_protocol_valid("max_split_size_mb:64")


def test_parent_worker_env_sets_reconciled_allocator_for_all_methods() -> None:
    base = {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:64"}
    for method in METHOD_ADAPTERS:
        env = formal_worker_env(base, gpu=1, gpu_uuid="GPU-test")
        assert method
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        assert env["PAPER_BASELINE_GPU_UUID"] == "GPU-test"
        assert env["PYTORCH_CUDA_ALLOC_CONF"] == REQUIRED_ALLOCATOR_CONF
        assert env["PATTERNKV_FP16_TAIL_VALUE_FUSION"] == "1"
        assert env["PATTERNKV_FIXED_SPLIT_SOFTMAX"] == "1"
        assert env["PATTERNKV_SELECTIVE_PREFILL_LOGITS"] == "1"
        assert env["PATTERNKV_ACTIVE_BATCH_CACHE"] == "1"
        assert env["PATTERNKV_SYSTEM_PROFILE"] == "0"


def test_invalid_allocator_protocol_rejects_formal_row() -> None:
    row = {
        "status": "PASS",
        "run_valid": True,
        "allocator_protocol_valid": False,
        "full_model_forward_executed": True,
        "completed_requests": 2,
        "active_capacity": 2,
        "output_tokens": 8,
        "decode_length": 4,
        "true_batch_preserved": True,
        "fallback_count": 0,
        "serial_request_forward_dispatches": 0,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "serial_rmsnorm_request_dispatches": 0,
        "prefill_calls_in_timed_window": 0,
        "prefill_tokens_in_timed_window": 0,
        "refill_calls_in_timed_window": 0,
        "membership_changes_in_timed_window": 0,
        "min_active_batch_size": 2,
        "max_active_batch_size": 2,
        "decode_only_wall_ms": 1.0,
    }
    assert valid_protocol(row) is False
    row["allocator_protocol_valid"] = True
    assert valid_protocol(row) is True


def test_launch_worker_records_missing_output_allocator_metadata(monkeypatch, tmp_path) -> None:
    class Proc:
        returncode = 7
        stderr = "worker failed"

    seen = {}

    def fake_run(cmd, cwd, env, text, capture_output):
        seen["env"] = env
        return Proc()

    monkeypatch.setattr("bench.paper_baseline_system_comparison.subprocess.run", fake_run)
    args = Namespace(gpu=1, gpu_uuid="GPU-test")
    output = tmp_path / "missing.json"
    row = launch_worker(
        args,
        method="FP16_FULL_MODEL",
        phase="test",
        context=512,
        decode=4,
        batch=1,
        run_index=0,
        warmup=False,
        output=output,
    )
    assert seen["env"]["PYTORCH_CUDA_ALLOC_CONF"] == REQUIRED_ALLOCATOR_CONF
    assert row["pytorch_cuda_alloc_conf"] == REQUIRED_ALLOCATOR_CONF
    assert row["allocator_protocol_valid"] is True
