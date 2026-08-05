from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_wave_a_8gpu_launcher_uses_all_cards_and_isolated_outputs():
    text = read_script("run_insight_wave_a_8gpu.sh")
    assert 'GPU_IDS="${GPU_IDS:-0 1 2 3 4 5 6 7}"' in text
    assert 'RESULT_ROOT="${RESULT_ROOT:-$ROOT/results/insight_v2/wave_a_8gpu}"' in text
    assert 'REPORT_ROOT="${REPORT_ROOT:-$ROOT/reports/insight_v2/wave_a_8gpu}"' in text
    assert 'RUN_ROOT="${RUN_ROOT:-$ROOT/run/insight_v2/wave_a_8gpu}"' in text
    assert "Need exactly 8 GPU ids" in text


def test_wave_a_8gpu_launcher_splits_gsm8k_across_three_gpus():
    text = read_script("run_insight_wave_a_8gpu.sh")
    assert "gsm8k_shards = [gsm8k_ids[i::3] for i in range(3)]" in text
    assert 'args+=(--problem-ids "${shard_ids[@]}")' in text
    assert "--max-new-tokens 2048" in text


def test_wave_a_8gpu_launcher_has_dry_run_and_readiness_guard():
    text = read_script("run_insight_wave_a_8gpu.sh")
    assert 'if [[ "${1:-}" == "--dry-run" ]]' in text
    assert "dry_run=true; manifest generated and no model loaded." in text
    assert '"ready_to_launch"' in text
    assert "Wave A 8GPU not launched; readiness failed." in text


def test_wave_a_8gpu_status_and_stop_use_8gpu_paths():
    status = read_script("status_insight_wave_a_8gpu.sh")
    stop = read_script("stop_insight_wave_a_8gpu.sh")
    assert "wave_a_8gpu" in status
    assert "wave_a_8gpu" in stop
    assert "pkill" not in stop
    assert "killall" not in stop
