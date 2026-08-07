from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


METHODS = ("fp16", "kivi_paper_g128", "patternkv_paper", "kivi_official", "patternkv")
DEFAULT_MAX_NEW_TOKENS = 32768
DEFAULT_BASE_SEED = 42
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_aime24(path: Path = Path("datasets/aime/aime24.jsonl")) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != 30:
        raise ValueError(f"AIME24 must contain exactly 30 rows, found {len(rows)}")
    for i, row in enumerate(rows):
        if int(row.get("problem_id", -1)) != i:
            raise ValueError(f"problem_id mismatch at row {i}: {row.get('problem_id')}")
        if not str(row.get("problem") or "").strip():
            raise ValueError(f"empty problem at row {i}")
        if not str(row.get("answer") or "").strip():
            raise ValueError(f"empty answer at row {i}")
    return rows


def effective_seed(base_seed: int, problem_id: int, sample_id: int) -> int:
    return int(base_seed) + int(problem_id) * 1000 + int(sample_id)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_eos_token_ids(*values: Any) -> list[int]:
    ids: list[int] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, int):
            ids.append(value)
        elif isinstance(value, (list, tuple, set)):
            ids.extend(int(x) for x in value if x is not None)
    return sorted(set(ids))


def compute_stop_state(generated_token_ids: list[int], max_new_tokens: int, eos_token_ids: list[int]) -> dict[str, Any]:
    output_tokens = len(generated_token_ids)
    last_generated_token_id = generated_token_ids[-1] if generated_token_ids else None
    eos_set = set(eos_token_ids)
    ended_with_eos = bool(generated_token_ids) and last_generated_token_id in eos_set
    hit_max_new_tokens = output_tokens >= max_new_tokens
    length_truncated = hit_max_new_tokens and not ended_with_eos
    if ended_with_eos:
        stop_reason = "eos"
    elif length_truncated:
        stop_reason = "length"
    elif output_tokens == 0:
        stop_reason = "unknown"
    else:
        stop_reason = "unknown"
    return {
        "last_generated_token_id": last_generated_token_id,
        "eos_token_ids": sorted(eos_set),
        "ended_with_eos": ended_with_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "length_truncated": length_truncated,
        "stop_reason": stop_reason,
    }


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generation_config_dict(args) -> dict[str, Any]:
    return {
        "dataset": "aime24",
        "model_path": str(args.model_path),
        "methods": list(getattr(args, "manifest_methods", METHODS)),
        "num_samples_per_problem": int(args.num_samples),
        "batch_size": 1,
        "num_return_sequences": 1,
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
        "repetition_penalty": float(args.repetition_penalty),
        "base_seed": int(args.base_seed),
        "prompt_protocol": "deepseek_r1_recommended",
        "force_think_prefix": bool(args.force_think_prefix),
        "quantization": {"k_bits": 2, "v_bits": 2, "group_size": 128, "residual_length": 128},
        "patternkv": {"initial_k_patterns": 32, "initial_v_patterns": 32, "pattern_group_size": 128},
    }


def task_key(problem_id: int, sample_id: int) -> str:
    return f"aime24:p{problem_id}:s{sample_id}"


def build_manifest(rows: list[dict[str, Any]], methods=METHODS, num_samples: int = 2, base_seed: int = 42, cfg_hash: str = "") -> list[dict[str, Any]]:
    manifest = []
    for method in methods:
        for row in rows:
            pid = int(row["problem_id"])
            for sid in range(num_samples):
                manifest.append(
                    {
                        "dataset": "aime24",
                        "method": method,
                        "problem_id": pid,
                        "sample_id": sid,
                        "task_key": task_key(pid, sid),
                        "seed": effective_seed(base_seed, pid, sid),
                        "config_hash": cfg_hash,
                    }
                )
    return manifest


def result_path(output_dir: Path, method: str, problem_id: int, sample_id: int, cfg_hash: str) -> Path:
    return output_dir / method / f"p{problem_id:02d}_s{sample_id}_{cfg_hash}.json"


def is_complete_result(path: Path, cfg_hash: str, retry_failed: bool = False, retry_oom: bool = False) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required = ["task_key", "method", "problem_id", "sample_id", "config_hash", "generated_text", "stop_reason", "parsed_answer"]
    if any(key not in data for key in required):
        return False
    if data.get("config_hash") != cfg_hash:
        return False
    if data.get("stop_reason") == "oom":
        return not retry_oom
    if data.get("stop_reason") == "error" or data.get("error"):
        return not retry_failed
    return True


def shard_tasks(tasks: list[dict[str, Any]], worker_index: int, num_workers: int) -> list[dict[str, Any]]:
    return [task for i, task in enumerate(tasks) if i % num_workers == worker_index]


def majority_vote(values: list[str | None]) -> dict[str, Any]:
    valid = [v for v in values if v is not None]
    if not valid:
        return {"answer": None, "tie": False, "votes": {}}
    counts = Counter(valid)
    top = max(counts.values())
    winners = sorted([k for k, v in counts.items() if v == top])
    return {"answer": winners[0] if len(winners) == 1 else None, "tie": len(winners) > 1, "votes": dict(counts)}


def length_bucket(tokens: int) -> str:
    if tokens < 4000:
        return "<4K"
    if tokens < 8000:
        return "4K-8K"
    if tokens < 16000:
        return "8K-16K"
    if tokens < 24000:
        return "16K-24K"
    return ">=24K"


def paired_stats(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], name_a: str, name_b: str) -> dict[str, Any]:
    a = {r["task_key"]: r for r in rows_a}
    b = {r["task_key"]: r for r in rows_b}
    keys = sorted(set(a) & set(b))
    both_correct = both_wrong = a_only = b_only = 0
    diffs = []
    for key in keys:
        ca = bool(a[key].get("is_correct"))
        cb = bool(b[key].get("is_correct"))
        diffs.append((1 if ca else 0) - (1 if cb else 0))
        if ca and cb:
            both_correct += 1
        elif not ca and not cb:
            both_wrong += 1
        elif ca:
            a_only += 1
        else:
            b_only += 1
    return {
        "comparison": f"{name_a}-{name_b}",
        "paired_n": len(keys),
        "paired_accuracy_difference": sum(diffs) / len(diffs) if diffs else None,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        f"{name_a}_correct_{name_b}_wrong": a_only,
        f"{name_a}_wrong_{name_b}_correct": b_only,
    }


def search_model_candidates() -> list[str]:
    roots = [Path("/data/zypan"), Path("/data/zypan/models"), Path("/data/zypan/blockgtq-repro/models"), Path("/models")]
    out = []
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            path = str(config.parent)
            low = path.lower()
            if "deepseek" in low and "r1" in low and "distill" in low and "llama" in low and "8b" in low:
                out.append(path)
    return sorted(set(out))
