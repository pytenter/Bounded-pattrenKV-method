"""Shared planning and integrity helpers for the single-4090 Wave A run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/insight_v2/wave_a_4090_single"
VALIDATION_ROOT = REPORT_ROOT / "validation"
RESULT_ROOT = ROOT / "results/insight_v2/wave_a_4090_single"
RUN_ROOT = ROOT / "run/insight_v2/wave_a_4090_single"
LOG_ROOT = ROOT / "logs/insight_v2/wave_a_4090_single"
V100_MANIFEST = ROOT / "reports/insight_v2/wave_a_8gpu/manifest.json"
SELECTED_SAMPLES = ROOT / "reports/insight_v1/v0/selected_samples.json"

LONG_BENCH_TASKS = (
    "hotpotqa",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "samsum",
    "dureader",
)
PLAN = (
    ("longbench", "hotpotqa", 12),
    ("longbench", "passage_retrieval_en", 12),
    ("longbench", "passage_retrieval_zh", 12),
    ("longbench", "samsum", 12),
    ("longbench", "dureader", 12),
    ("gsm8k", "gsm8k", 80),
)
RUNTIME_SENSITIVE_PREFIXES = (
    "bench/bench_pattern_insight.py",
    "bench/bench_longbench_patternkv.py",
    "bench/bench_gsm8k_paper.py",
    "bench/gsm8k_paper_utils.py",
    "bench/paper_config.py",
    "insight/",
    "models/",
    "quant/",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_sha256(values: list[Any]) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def current_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def selected_rows() -> list[dict[str, Any]]:
    return json.loads(SELECTED_SAMPLES.read_text(encoding="utf-8")).get("selected", [])


def load_reference() -> dict[str, Any]:
    return json.loads((REPORT_ROOT / "reference_manifest.json").read_text(encoding="utf-8"))


def sample_key(sample: dict[str, Any]) -> str:
    if sample.get("dataset") == "gsm8k":
        return f"p{int(sample['problem_id']):04d}"
    if sample.get("sample_id"):
        return str(sample["sample_id"])
    return f"p{int(sample['problem_id']):04d}"


def result_path(root: Path, sample: dict[str, Any], level: str, seed: int = 0) -> Path:
    return root / str(sample["dataset"]) / str(sample["task"]) / f"{sample_key(sample)}_{level}_seed{seed}.json"


def is_completed_generation(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        not payload.get("error")
        and payload.get("stop_reason") not in {"error", "oom"}
        and payload.get("observer_status") == "completed"
    )


def is_completed_observer(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("status") == "completed" and not payload.get("error")


def plan_samples(reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in LONG_BENCH_TASKS:
        rows.extend(reference["longbench_samples"][task])
    for problem_id in reference["gsm8k_problem_ids"]:
        rows.append(
            {
                "dataset": "gsm8k",
                "task": "gsm8k",
                "problem_id": int(problem_id),
                "sample_id": f"gsm8k:{int(problem_id)}",
                "sample_index": int(problem_id),
                "selection_reason": "v100_manifest",
            }
        )
    return rows


def pending_samples(reference: dict[str, Any], result_root: Path = RESULT_ROOT) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for sample in plan_samples(reference):
        gen = result_path(result_root / "generation", sample, "oracle")
        obs = result_path(result_root / "observer", sample, "oracle")
        if not (is_completed_generation(gen) and is_completed_observer(obs)):
            pending.append(sample)
    return pending
