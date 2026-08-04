import json
import subprocess
import sys
from pathlib import Path


def test_insight_runner_dry_run_writes_manifest_only(tmp_path: Path):
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "selected": [
                    {
                        "dataset": "gsm8k",
                        "task": "gsm8k",
                        "sample_id": "gsm8k:0",
                        "problem_id": 0,
                        "sample_index": 0,
                        "selection_reason": "unit",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    generation = tmp_path / "generation"
    observer = tmp_path / "observer"
    reports = tmp_path / "reports"
    cmd = [
        sys.executable,
        "bench/bench_pattern_insight.py",
        "--dataset",
        "gsm8k",
        "--selected-samples-json",
        str(selected),
        "--model-path",
        "/tmp/model",
        "--output-dir",
        str(generation),
        "--observer-output-root",
        str(observer),
        "--insight-output-dir",
        str(reports),
        "--dry-run",
        "--limit",
        "1",
    ]
    subprocess.run(cmd, check=True)
    manifest = json.loads((reports / "runner_manifest.json").read_text())
    assert manifest["samples"][0]["sample"]["problem_id"] == 0
    assert not list(generation.rglob("*.json"))
    assert not list(observer.rglob("*.json"))


def test_insight_runner_source_has_no_manifest_only_blockers():
    text = Path("bench/bench_pattern_insight.py").read_text()
    assert "generation_not_connected" not in text
    assert "pending_generation_not_implemented" not in text
    assert "real generation is not connected yet" not in text
