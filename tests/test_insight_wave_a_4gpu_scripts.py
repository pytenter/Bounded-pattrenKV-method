from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_wave_a_4gpu_launcher_has_fixed_gpu_queues():
    text = read_script("run_insight_wave_a_4gpu.sh")
    assert 'WAVE_A_GPU_IDS="${WAVE_A_GPU_IDS:-4 5 6 7}"' in text
    assert 'run_queue "${gpu_ids[0]}" longbench hotpotqa 12 longbench samsum 12' in text
    assert 'run_queue "${gpu_ids[1]}" longbench passage_retrieval_en 12 longbench passage_retrieval_zh 12' in text
    assert 'run_queue "${gpu_ids[2]}" longbench dureader 12' in text
    assert 'run_queue "${gpu_ids[3]}" gsm8k gsm8k 50' in text
    assert "gpu < 4 || gpu > 7" in text


def test_wave_a_4gpu_launcher_uses_isolated_outputs_and_guard():
    text = read_script("run_insight_wave_a_4gpu.sh")
    assert 'RESULT_ROOT="${RESULT_ROOT:-results/insight_v2/wave_a}"' in text
    assert 'REPORT_ROOT="${REPORT_ROOT:-reports/insight_v2/wave_a}"' in text
    assert 'LOG_ROOT="${LOG_ROOT:-logs/insight_v2/wave_a}"' in text
    assert 'RUN_ROOT="${RUN_ROOT:-run/insight_v2/wave_a}"' in text
    assert "GPU_MEMORY_THRESHOLD_MIB" in text
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_memory" in text
    assert "Wave A not launched; GPU4-7 occupancy guard failed." in text


def test_wave_a_4gpu_launcher_preserves_paper_protocol():
    text = read_script("run_insight_wave_a_4gpu.sh")
    assert "PATTERNKV_INSIGHT_LEVEL=oracle" in text
    assert "PATTERNKV_INSIGHT_ORACLE_LAYERS=0,7,15,23,31" in text
    assert "PATTERNKV_INSIGHT_SAMPLE_TOKENS=8" in text
    assert "--max-input-length 8192" in text
    assert "--max-new-tokens 2048" in text
    assert '"longbench_max_gen": "task-specific MAX_NEW_TOKENS from LongBench runner"' in text


def test_wave_a_4gpu_stop_script_only_uses_pid_files():
    text = read_script("stop_insight_wave_a_4gpu.sh")
    assert "pkill" not in text
    assert "killall" not in text
    assert "-name 'launcher.pid'" in text
    assert "-name 'gpu*.queue.pid'" in text
    assert "-name 'worker.pid'" in text
