from pathlib import Path

from insight.io import atomic_write_json


def test_atomic_json_replaces_file(tmp_path: Path):
    path = tmp_path / "x.json"
    atomic_write_json(path, {"a": 1})
    atomic_write_json(path, {"a": 2})
    assert '"a": 2' in path.read_text()
