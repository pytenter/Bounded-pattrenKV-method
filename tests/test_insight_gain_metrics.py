from insight.pattern_metrics import normalized_entropy, relative_benefit


def test_relative_benefit_positive_when_pattern_reduces_mse():
    assert relative_benefit(10.0, 5.0) > 0
    assert relative_benefit(10.0, 12.0) < 0


def test_entropy_range_and_dead_patterns():
    stats = normalized_entropy([0, 0, 1, 1], pattern_count=4)
    assert 0 <= stats["normalized_entropy"] <= 1
    assert stats["dead_pattern_count"] == 2
