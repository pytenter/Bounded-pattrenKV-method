from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from scripts.select_range_aware_targeted_samples_4090 import GSM_POSITIONS, LONG_POSITIONS, select_samples
from scripts.summarize_range_aware_targeted_4090 import compute_macro_metrics, decide_status


ROOT = Path(__file__).resolve().parents[1]


def build_reference_manifest() -> dict:
    longbench = {}
    for task in ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader"):
        rows = []
        for i in range(12):
            rows.append(
                {
                    "dataset": "longbench",
                    "task": task,
                    "sample_id": f"{task}:{i}",
                    "sample_index": i,
                    "problem_id": "",
                    "selection_reason": "reference_manifest_order",
                }
            )
        longbench[task] = rows
    return {
        "longbench_samples": longbench,
        "gsm8k_problem_ids": list(range(80)),
    }


def test_selection_is_deterministic_and_uses_required_positions():
    payload = select_samples(build_reference_manifest(), "abc")
    assert len(payload["selected"]) == 25
    hotpot = [row for row in payload["selected"] if row["task"] == "hotpotqa"]
    assert [row["manifest_position"] for row in hotpot] == list(LONG_POSITIONS)
    gsm = [row for row in payload["selected"] if row["task"] == "gsm8k"]
    assert [row["manifest_position"] for row in gsm] == list(GSM_POSITIONS)


def test_dry_run_reports_gpu_will_not_start(tmp_path: Path):
    report_root = tmp_path / "reports"
    manifest = tmp_path / "reference_manifest.json"
    manifest.write_text(json.dumps(build_reference_manifest()), encoding="utf-8")
    env = dict(os.environ)
    env["REPORT_ROOT"] = str(report_root)
    env["RUN_ROOT"] = str(tmp_path / "run")
    env["RESULT_ROOT"] = str(tmp_path / "results")
    env["LOG_ROOT"] = str(tmp_path / "logs")
    env["REFERENCE_MANIFEST"] = str(manifest)
    out = subprocess.check_output(
        ["bash", str(ROOT / "scripts" / "run_range_aware_targeted_4090.sh"), "--dry-run"],
        text=True,
        env=env,
    )
    assert "GPU will start: no" in out
    status = json.loads((report_root / "current_status.json").read_text(encoding="utf-8"))
    assert status["gpu_started"] is False
    assert status["parity_status"] == "not_run"


def test_run_targeted_refuses_without_passed_parity(tmp_path: Path):
    report_root = tmp_path / "reports"
    report_root.mkdir(parents=True)
    (report_root / "parity_status.json").write_text('{"parity_status":"failed"}\n', encoding="utf-8")
    env = dict(os.environ)
    env["REPORT_ROOT"] = str(report_root)
    env["RUN_ROOT"] = str(tmp_path / "run")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "run_range_aware_targeted_4090.sh"), "--run-targeted"],
        text=True,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 3
    assert "parity_status=failed" in result.stdout


def test_status_and_stop_scripts_are_targeted_and_non_destructive():
    status_text = (ROOT / "scripts" / "status_range_aware_targeted_4090.sh").read_text(encoding="utf-8")
    stop_text = (ROOT / "scripts" / "stop_range_aware_targeted_4090.sh").read_text(encoding="utf-8")
    assert "range_aware_targeted_4090" in status_text
    assert "range_aware_targeted_4090" in stop_text
    assert "pkill" not in stop_text
    assert "killall" not in stop_text


def test_micro_and_task_macro_do_not_let_gsm8k_dominate():
    rows = []
    for task in ("hotpotqa", "passage_retrieval_en", "passage_retrieval_zh", "samsum", "dureader"):
        rows.append({"task": task, "kv_type": "k", "assignment_mismatch_count": 1, "assignment_total_count": 2, "range_regret_sum": 0.2, "range_regret_count": 2})
    rows.append({"task": "gsm8k", "kv_type": "k", "assignment_mismatch_count": 80, "assignment_total_count": 80, "range_regret_sum": 16.0, "range_regret_count": 80})
    metrics = compute_macro_metrics(rows, "k")
    assert metrics["mismatch_micro"] > metrics["mismatch_task_macro"]
    assert metrics["tasks_complete"] == 6


def test_decision_thresholds_for_supported_and_insufficient():
    metrics = [
        {
            "kv_type": "k",
            "tasks_complete": 6,
            "mismatch_micro": 0.3,
            "mismatch_task_macro": 0.3,
            "range_regret_micro": 0.1,
            "range_regret_task_macro": 0.1,
        },
        {
            "kv_type": "v",
            "tasks_complete": 6,
            "mismatch_micro": 0.0,
            "mismatch_task_macro": 0.0,
            "range_regret_micro": 0.0,
            "range_regret_task_macro": 0.0,
        },
    ]
    assert decide_status(metrics, parity_status="passed", data_complete=True)["status"] == "targeted_supported"
    assert decide_status(metrics, parity_status="not_run", data_complete=True)["status"] == "targeted_data_insufficient"
