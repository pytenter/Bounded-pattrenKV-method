from insight.dynamic_metrics import relative_gain, selected_fraction


def test_dynamic_relative_gain_and_selection_fraction():
    assert relative_gain(10.0, 9.0) > 0
    assert selected_fraction(1, 4) == 0.25
    assert selected_fraction(1, 0) == 0.0
