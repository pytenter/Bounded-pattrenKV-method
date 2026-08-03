from bench.aime_utils import METHODS, build_manifest, load_aime24


def test_aime24_has_30_problems():
    rows = load_aime24()
    assert len(rows) == 30


def test_manifest_n2_has_180_tasks_and_60_per_method():
    rows = load_aime24()
    manifest = build_manifest(rows, METHODS, num_samples=2, base_seed=42, cfg_hash="abc")
    assert len(manifest) == 180
    for method in METHODS:
        assert sum(1 for t in manifest if t["method"] == method) == 60
