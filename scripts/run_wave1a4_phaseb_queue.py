#!/usr/bin/env python
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_wave1a4_attention_observer import FREE_RUNNING_CONFIGS, FREE_RUNNING_DIR, safe_key, write_phaseb_selected_tasks


ROOT = Path(__file__).resolve().parent.parent
PYTHON_BIN = os.environ.get("PYTHON_BIN", "/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python")
RUN_DIR = ROOT / "run/aime24_int2_wave1_v100_8gpu_wave1a4"


def valid_record(config: str, task_key: str) -> bool:
    path = FREE_RUNNING_DIR / config / f"{safe_key(task_key)}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("trace_valid")) and not data.get("runtime_error")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tasks = write_phaseb_selected_tasks()
    jobs = [(config, task["task_key"]) for config in FREE_RUNNING_CONFIGS for task in tasks if not valid_record(config, task["task_key"])]
    gpus = [int(item) for item in os.environ.get("WAVE1A4_PHASEB_GPUS", "0,1,2,3,4,5,6,7").replace(",", " ").split()]
    print(json.dumps({"missing_jobs": len(jobs), "gpus": gpus}, sort_keys=True), flush=True)
    running: dict[subprocess.Popen, tuple[int, str, str, object]] = {}
    idle_gpus = list(gpus)
    failed = 0
    index = 0
    while index < len(jobs) or running:
        while index < len(jobs) and idle_gpus:
            config, task_key = jobs[index]
            gpu = idle_gpus.pop(0)
            index += 1
            log_path = RUN_DIR / f"wave1a4_phaseb_queue_{config}_{safe_key(task_key)}_gpu{gpu}.log"
            handle = log_path.open("w", encoding="utf-8")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [
                PYTHON_BIN,
                "scripts/run_wave1a4_attention_observer.py",
                "--mode",
                "free-run-config",
                "--config-name",
                config,
                "--task-key",
                task_key,
                "--checkpoints",
                "512,1024,2048,4096,8192,16384",
                "--layers",
                "0,7,15,23,31",
            ]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running[proc] = (gpu, config, task_key, handle)
            print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] start gpu={gpu} config={config} task={task_key} pid={proc.pid}", flush=True)
        time.sleep(5)
        for proc, (gpu, config, task_key, handle) in list(running.items()):
            ret = proc.poll()
            if ret is None:
                continue
            handle.close()
            running.pop(proc)
            idle_gpus.append(gpu)
            if ret != 0 or not valid_record(config, task_key):
                failed += 1
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] failed gpu={gpu} config={config} task={task_key} rc={ret}", flush=True)
            else:
                print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] done gpu={gpu} config={config} task={task_key}", flush=True)
    if failed:
        raise SystemExit(f"{failed} Phase B jobs failed")
    subprocess.check_call([PYTHON_BIN, "scripts/run_wave1a4_attention_observer.py", "--mode", "aggregate-free-running"], cwd=ROOT)
    subprocess.check_call([PYTHON_BIN, "scripts/summarize_wave1a4_attention_mechanism.py"], cwd=ROOT)


if __name__ == "__main__":
    main()
