from bench.aime_utils import majority_vote, paired_stats
from scripts.summarize_aime24_results import summarize_method


def test_avg_and_strict_avg():
    rows = [
        {"parsed_answer": "1", "is_correct": True, "stop_reason": "eos", "generated_tokens": 10, "wall_time_seconds": 1, "tokens_per_second": 10, "sample_id": 0, "problem_id": 0, "reference_answer": "1"},
        {"parsed_answer": "2", "is_correct": False, "stop_reason": "length", "generated_tokens": 20, "wall_time_seconds": 2, "tokens_per_second": 10, "sample_id": 1, "problem_id": 0, "reference_answer": "1"},
    ]
    s = summarize_method(rows, planned=4, num_samples=2)
    assert s["avg_at_n"] == 50.0
    assert s["strict_avg"] == 25.0
    assert s["length_stop"] == 1


def test_majority_tie():
    vote = majority_vote(["1", "2"])
    assert vote["tie"] is True
    assert vote["answer"] is None


def test_paired_stats():
    a = [{"task_key": "x", "is_correct": True}]
    b = [{"task_key": "x", "is_correct": False}]
    s = paired_stats(a, b, "a", "b")
    assert s["paired_accuracy_difference"] == 1.0
    assert s["a_correct_b_wrong"] == 1
