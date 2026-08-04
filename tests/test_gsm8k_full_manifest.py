from bench.gsm8k_paper_utils import METHODS, load_gsm8k, manifest


def test_gsm8k_1319_and_manifest_3957():
    rows = load_gsm8k()
    assert len(rows) == 1319
    m = manifest(rows, METHODS, "abc")
    assert len(m) == 3957
    for method in METHODS:
        assert sum(1 for x in m if x["method"] == method) == 1319
