from insight.gate_metrics import confusion_from_decisions


def test_v_confusion_matrix_counts_match_total():
    conf = confusion_from_decisions([True, True, False, False], [True, False, True, False])
    d = conf.to_dict()
    assert d["true_positive"] == 1
    assert d["false_positive"] == 1
    assert d["false_negative"] == 1
    assert d["true_negative"] == 1
    assert d["total"] == 4
