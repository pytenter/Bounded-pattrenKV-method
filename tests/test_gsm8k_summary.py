from scripts.summarize_gsm8k_paper_results import summarize


def test_strict_accuracy_counts_failures():
    rows = [{"is_correct": True, "parsed_answer": "1", "stop_reason": "eos", "generated_tokens": 1, "wall_time_seconds": 1, "peak_memory_reserved_bytes": 1}]
    s = summarize(rows)
    assert s["completed"] == 1
    assert s["accuracy_completed"] == 100.0
    assert s["strict_accuracy"] < 1.0
