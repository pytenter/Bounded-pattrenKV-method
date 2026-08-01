import json
from pathlib import Path


def test_patternkv_smoke_invariants_file():
    path = Path("results/smoke_patternkv.json")
    assert path.exists(), "run scripts/run_smoke.py --method patternkv first"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert not data.get("oom", False)
    assert data.get("error") in (None, "")
    assert data.get("output_tokens", 0) > 0
    layers = data.get("patternkv_layers", {}).get("layers", [])
    assert layers
    assert any("k_base" in layer for layer in layers)
    assert any("v_centroids" in layer for layer in layers)
