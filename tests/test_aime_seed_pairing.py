from bench.aime_utils import METHODS, build_manifest, effective_seed, load_aime24


def test_seed_formula():
    assert effective_seed(42, 0, 0) == 42
    assert effective_seed(42, 0, 1) == 43
    assert effective_seed(42, 1, 0) == 1042


def test_same_task_key_same_seed_across_methods():
    manifest = build_manifest(load_aime24(), METHODS, num_samples=2, base_seed=42, cfg_hash="abc")
    grouped = {}
    for t in manifest:
        grouped.setdefault(t["task_key"], set()).add(t["seed"])
    assert all(len(seeds) == 1 for seeds in grouped.values())
    assert manifest[0]["seed"] != manifest[1]["seed"]
