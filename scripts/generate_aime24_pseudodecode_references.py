from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.aime_utils import config_hash, generation_config_dict  # noqa: E402
from bench.bench_aime24_patternkv import parse_args as bench_parse_args  # noqa: E402
from bench.bench_aime24_patternkv import main as bench_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--selected-tasks", type=Path, default=Path("configs/aime24_wave1_selected_tasks.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/aime24_pseudodecode_3090_8gpu/reference"))
    parser.add_argument("--status-dir", type=Path, default=Path("run/aime24_pseudodecode_3090_8gpu/reference"))
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    sys.argv = [
        "bench/bench_aime24_patternkv.py",
        "--method",
        "fp16",
        "--config-name",
        "fp16_reference",
        "--model-path",
        args.model_path,
        "--selected-tasks",
        str(args.selected_tasks),
        "--output-dir",
        str(args.output_dir),
        "--status-dir",
        str(args.status_dir),
        "--experiment-id",
        "aime24_pseudodecode_3090_8gpu_reference",
        "--worker-index",
        str(args.worker_index),
        "--num-workers",
        str(args.num_workers),
        "--model-dtype",
        "float16",
        "--temperature",
        "0.6",
        "--top-p",
        "0.95",
        "--repetition-penalty",
        "1.0",
        "--force-think-prefix",
        "--overwrite-invalid",
        "--retry-failed",
    ]
    bench_main()


if __name__ == "__main__":
    main()
