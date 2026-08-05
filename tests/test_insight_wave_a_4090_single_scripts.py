from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_plan_and_isolation():
    launcher = read("run_insight_wave_a_4090_single.sh")
    assert "hotpotqa" in launcher
    assert "passage_retrieval_en" in launcher
    assert "passage_retrieval_zh" in launcher
    assert "samsum" in launcher
    assert "dureader" in launcher
    assert "gsm8k" in launcher
    assert "total=140" in launcher
    assert "wave_a_4090_single" in launcher
    assert 'RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_4090_single}"' in launcher
    assert 'REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_4090_single}"' in launcher
    assert 'LOG_ROOT="${LOG_ROOT:-$ROOT/logs/insight_v2/wave_a_4090_single}"' in launcher
    assert 'RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_4090_single}"' in launcher
    assert "CUDA_VISIBLE_DEVICES" in launcher
    assert "--gpu-id 0" in launcher


def test_launcher_uses_gate_and_exact_gsm_ids():
    launcher = read("run_insight_wave_a_4090_single.sh")
    assert "check_insight_wave_a_4090_gate.py" in launcher
    assert "gsm8k_problem_ids_ordered_sha256" in launcher
    assert "--problem-ids" in launcher
    assert "--limit" not in launcher


def test_stop_is_pid_scoped():
    stop = read("stop_insight_wave_a_4090_single.sh")
    assert "pkill" not in stop
    assert "killall" not in stop
    assert "launcher.pid" in stop
    assert "worker.pid" in stop


def test_status_reports_required_fields():
    status = read("status_insight_wave_a_4090_single.sh")
    for token in ("launcher_alive", "worker_alive", "current_task", "current_sample", "completed", "failed", "missing", "oom", "hook_errors", "observer_completed", "observer_missing", "tail -n 30"):
        assert token in status


def test_reference_manifest_has_fixed_contract():
    text = read("prepare_insight_wave_a_4090.py")
    for token in ("V100_RUNTIME_COMMIT", "gsm8k_problem_ids", "longbench_total", "gsm8k_total", "total", "selected_samples_sha256"):
        assert token in text


def test_bash_and_python_compile():
    for name in ("run_insight_wave_a_4090_single.sh", "status_insight_wave_a_4090_single.sh", "stop_insight_wave_a_4090_single.sh"):
        assert (ROOT / "scripts" / name).exists()
