from bench.analyze_existing_pattern_results import make_pair_row


def test_gsm8k_pair_row_uses_correctness_scores():
    records = {
        "fp16": {"method": "fp16", "problem_id": 1, "is_correct": True, "generated_tokens": 10, "stop_reason": "eos"},
        "kivi_paper_g128": {"method": "kivi_paper_g128", "problem_id": 1, "is_correct": False, "generated_tokens": 11, "stop_reason": "length"},
        "patternkv_paper": {
            "method": "patternkv_paper",
            "problem_id": 1,
            "is_correct": True,
            "generated_tokens": 9,
            "stop_reason": "eos",
            "cache_bitwidth_stats": {"quantized_tokens": 128, "fp16_residual_tokens": 128},
        },
    }
    row = make_pair_row("gsm8k", "gsm8k", records)
    assert row["pattern_minus_kivi"] == 1.0
    assert row["problem_id"] == 1
