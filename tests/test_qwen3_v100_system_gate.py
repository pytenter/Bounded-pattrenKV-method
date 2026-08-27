from bench.qwen3_full_model_serving_benchmark import current_gate


def test_qwen3_v100_formal_timing_is_blocked_until_gpu_gates_close():
    gate = current_gate()
    assert gate.classification == "QWEN_COMPRESSED_TRUE_BATCH_B2_FAIL"
    assert gate.formal_timing_allowed is False
    assert gate.compressed_domain_runtime_preserved is True
    assert gate.historical_fp16_k_materialization == 0
    assert gate.historical_fp16_v_materialization == 0


def test_qwen3_v100_gate_keeps_memory_out_of_scientific_outputs():
    payload = current_gate().__dict__
    assert "peak_memory" not in payload
    assert "capacity" not in payload
