#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime24_int2_wave1 import BitwidthConfig, effective_bitwidth, stable_hash
from bench.aime_utils import load_aime24, sha256_file
from bench.paper_config import apply_method_defaults, method_config_dict


PYTHON_BIN = "/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python"
MODEL_PATH = "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B"
DATASET_PATH = Path("datasets/aime/aime24.jsonl")
CONFIG_MANIFEST_PATH = Path("configs/aime24_full30_formal_validation.json")
RESULT_DIR = Path("results/aime24_full30_3seed")
RUN_DIR = Path("run/aime24_full30_3seed")
REPORT_DIR = Path("reports/aime24_full30_3seed")
SEEDS = (42, 43, 44)
EXPECTED_RECORDS = 630
GENERATION_CONFIG_HASH = "a7d6b2f8bab37893b6331c66b3e5eb6a"


CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "gpu": 0,
        "name": "fp16",
        "display_name": "FP16",
        "method": "fp16",
        "method_group": "FP16",
        "cache_mode": "fp16",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 0,
        "recent_length": 0,
        "residual_length": 0,
        "k_bits": 16,
        "v_bits": 16,
        "group_size": 0,
    },
    {
        "gpu": 1,
        "name": "patternkv_paper",
        "display_name": "PatternKV_paper",
        "method": "patternkv_paper",
        "method_group": "PatternKV",
        "cache_mode": "legacy_tuple_chunked",
        "patternkv_cache_path": "legacy",
        "patternkv_cache_mode": "legacy_tuple_chunked",
        "sink_length": 0,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
    {
        "gpu": 2,
        "name": "pattern_rolling_s0_r128",
        "display_name": "Pattern S0/R128",
        "method": "patternkv",
        "method_group": "PatternKV",
        "cache_mode": "segmented_rolling",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 0,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
    {
        "gpu": 3,
        "name": "pattern_rolling_s16_r128",
        "display_name": "Pattern S16/R128",
        "method": "patternkv",
        "method_group": "PatternKV",
        "cache_mode": "segmented_rolling",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
    {
        "gpu": 4,
        "name": "kivi_paper",
        "display_name": "KIVI_paper",
        "method": "kivi_paper_g128",
        "method_group": "KIVI",
        "cache_mode": "legacy_tuple_chunked",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 0,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
    {
        "gpu": 5,
        "name": "kivi_rolling_s0_r128",
        "display_name": "KIVI S0/R128",
        "method": "kivi_official",
        "method_group": "KIVI",
        "cache_mode": "segmented_rolling",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 0,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
    {
        "gpu": 6,
        "name": "kivi_rolling_s16_r128",
        "display_name": "KIVI S16/R128",
        "method": "kivi_official",
        "method_group": "KIVI",
        "cache_mode": "segmented_rolling",
        "patternkv_cache_path": "segmented",
        "patternkv_cache_mode": "segmented_rolling",
        "sink_length": 16,
        "recent_length": 128,
        "residual_length": 128,
        "k_bits": 2,
        "v_bits": 2,
        "group_size": 128,
    },
)

COMPARISONS: tuple[dict[str, str], ...] = (
    {"name": "pattern_paper_to_s16", "baseline": "patternkv_paper", "candidate": "pattern_rolling_s16_r128", "label": "PatternKV_paper -> Pattern S16"},
    {"name": "pattern_paper_to_s0", "baseline": "patternkv_paper", "candidate": "pattern_rolling_s0_r128", "label": "PatternKV_paper -> Pattern S0"},
    {"name": "pattern_s0_to_s16", "baseline": "pattern_rolling_s0_r128", "candidate": "pattern_rolling_s16_r128", "label": "Pattern S0 -> Pattern S16"},
    {"name": "kivi_s0_to_s16", "baseline": "kivi_rolling_s0_r128", "candidate": "kivi_rolling_s16_r128", "label": "KIVI S0 -> KIVI S16"},
    {"name": "kivi_paper_to_s0", "baseline": "kivi_paper", "candidate": "kivi_rolling_s0_r128", "label": "KIVI_paper -> KIVI S0"},
    {"name": "kivi_paper_to_s16", "baseline": "kivi_paper", "candidate": "kivi_rolling_s16_r128", "label": "KIVI_paper -> KIVI S16"},
)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def generation_config() -> dict[str, Any]:
    return {
        "dataset": "aime24",
        "model_path": MODEL_PATH,
        "seeds": list(SEEDS),
        "batch_size": 1,
        "num_return_sequences": 1,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 32768,
        "max_model_len": 32768,
        "repetition_penalty": 1.0,
        "prompt_protocol": "deepseek_r1_recommended",
        "force_think_prefix": True,
        "answer_parser": "bench.aime_answer_parser.parse_aime_answer",
        "dtype": "float16",
    }


def full30_tasks() -> list[dict[str, Any]]:
    rows = load_aime24(DATASET_PATH)
    return [
        {
            "dataset": "aime24",
            "problem_id": int(row["problem_id"]),
            "problem": row["problem"],
            "answer": str(row["answer"]),
            "task_id": f"aime24:p{int(row['problem_id'])}",
        }
        for row in rows
    ]


def formal_tasks() -> list[dict[str, Any]]:
    tasks = []
    for seed_index, seed in enumerate(SEEDS):
        for row in load_aime24(DATASET_PATH):
            problem_id = int(row["problem_id"])
            tasks.append(
                {
                    "dataset": "aime24",
                    "problem_id": problem_id,
                    "sample_id": seed_index,
                    "seed": seed,
                    "task_key": f"aime24:p{problem_id}:seed{seed}",
                }
            )
    return tasks


def selected_tasks_path(kind: str) -> Path:
    return RUN_DIR / f"{kind}_selected_tasks.json"


def config_hash_for(cfg: dict[str, Any], task_count: int = 90) -> str:
    payload = generation_config()
    payload.update(
        {
            "config_name": cfg["name"],
            "method": cfg["method"],
            "task_count": task_count,
            "k_bits": cfg["k_bits"],
            "v_bits": cfg["v_bits"],
            "group_size": cfg["group_size"],
            "sink_length": cfg["sink_length"],
            "recent_length": cfg["recent_length"],
            "residual_length": cfg["residual_length"],
            "cache_mode": cfg["cache_mode"],
            "patternkv_cache_path": cfg["patternkv_cache_path"],
            "patternkv_cache_mode": cfg["patternkv_cache_mode"],
        }
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def prepare() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = full30_tasks()
    problem_ids = [task["problem_id"] for task in tasks]
    if len(tasks) != 30 or sorted(problem_ids) != list(range(30)) or len(set(problem_ids)) != 30:
        raise SystemExit("AIME24 full30 manifest must contain exactly problem_id 0..29")
    task_hash = sha256_file(DATASET_PATH)
    manifest = {
        "dataset": "aime24",
        "source": str(DATASET_PATH),
        "source_metadata": "datasets/aime/aime24_metadata.json",
        "task_count": len(tasks),
        "problem_ids": problem_ids,
        "answer_field": "answer",
        "task_sha256": task_hash,
        "generation_config": generation_config(),
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "seeds": list(SEEDS),
        "configs": CONFIGS,
        "expected_records": EXPECTED_RECORDS,
        "formal_tasks": formal_tasks(),
    }
    write_json(CONFIG_MANIFEST_PATH, tasks)
    write_json(REPORT_DIR / "aime24_full30_manifest.json", manifest)
    write_json(selected_tasks_path("formal"), formal_tasks())
    write_json(selected_tasks_path("preflight"), [formal_tasks()[0]])
    lines = [
        "# AIME24 Full30 Manifest",
        "",
        f"- Source: `{DATASET_PATH}`",
        f"- Task count: `{len(tasks)}`",
        f"- Problem IDs: `{problem_ids}`",
        "- Answer field: `answer`",
        f"- Task SHA256: `{task_hash}`",
        f"- Generation config hash: `{GENERATION_CONFIG_HASH}`",
        f"- Seeds: `{list(SEEDS)}`",
        "",
        "| problem_id | answer | problem preview |",
        "| ---: | --- | --- |",
    ]
    for task in tasks:
        preview = " ".join(task["problem"].split())[:120].replace("|", "\\|")
        lines.append(f"| {task['problem_id']} | {task['answer']} | {preview} |")
    (REPORT_DIR / "aime24_full30_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_audits()
    print(json.dumps({"task_count": len(tasks), "task_sha256": task_hash, "generation_config_hash": GENERATION_CONFIG_HASH}, indent=2, sort_keys=True))


class Args:
    pass


def audit_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    args = Args()
    args.method = cfg["method"]
    args.k_bits = cfg["k_bits"]
    args.v_bits = cfg["v_bits"]
    args.group_size = cfg["group_size"]
    args.residual_length = cfg["residual_length"]
    args.sink_length = cfg["sink_length"]
    args.recent_length = cfg["recent_length"]
    args.num_k_base = 32
    args.num_v_base = 32
    args.mixed_key_mask_path = None
    args.mixed_key_int4_ratio = 0.0
    args.patternkv_cache_path = cfg["patternkv_cache_path"]
    args.patternkv_cache_mode = cfg["patternkv_cache_mode"]
    args.paper_method_config = apply_method_defaults(args)
    return method_config_dict(args)


def write_audits() -> None:
    for config_name, filename, title in (
        ("patternkv_paper", "patternkv_paper_config_audit.md", "PatternKV Paper Configuration Audit"),
        ("kivi_paper", "kivi_paper_config_audit.md", "KIVI Paper Configuration Audit"),
    ):
        cfg = next(item for item in CONFIGS if item["name"] == config_name)
        payload = audit_payload(cfg)
        lines = [
            f"# {title}",
            "",
            f"- Config name: `{cfg['name']}`",
            f"- Backend method: `{payload['backend_method']}`",
            f"- Cache mode: `{cfg['cache_mode']}`",
            f"- PatternKV cache path: `{cfg['patternkv_cache_path']}`",
            f"- PatternKV cache mode: `{cfg['patternkv_cache_mode']}`",
            f"- Residual length: `{payload['residual_length']}`",
            f"- Sink length: `{payload['sink_length']}`",
            f"- Recent length: `{payload['recent_length']}`",
            f"- K bits: `{payload['k_bits']}`",
            f"- V bits: `{payload['v_bits']}`",
            f"- Group size: `{payload['group_size']}`",
            f"- K quantization granularity: `{payload['key_quant_axis']}`",
            f"- V quantization granularity: `{payload['value_quant_axis']}`",
            f"- Asymmetric quantization: `{payload['asym']}`",
            f"- Quantized-region affine bits: `{payload['quantized_region_affine_bits']}`",
        ]
        if config_name == "patternkv_paper":
            lines += [
                f"- Initial K centroid count: `{payload['initial_pattern_count']}`",
                f"- Initial V centroid count: `{payload['initial_pattern_count']}`",
                f"- Pattern group: `{payload['pattern_group']}`",
                f"- Pattern selection position: `{payload['pattern_selection_position']}`",
                "- Assignment behavior: paper PatternKV K/V assignment path with residual chunk semantics.",
                "- Centroid behavior: 32 initial K and V bases with PatternKV runtime dynamic state reset per sample.",
                "- V gate behavior: production PatternKV V residual/centroid gate semantics.",
            ]
        else:
            lines += [
                "- Metadata: packed K/V payload plus FP16 scale/min metadata.",
                "- Cache semantics: KIVI official chunk residual semantics with residual_length=128 and no Sink protection.",
            ]
        (REPORT_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def result_files(config_name: str) -> list[Path]:
    return sorted((RESULT_DIR / config_name).glob("*.json"))


def load_records() -> list[dict[str, Any]]:
    rows = []
    for cfg in CONFIGS:
        for path in result_files(cfg["name"]):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                row["_path"] = str(path)
            except Exception as exc:
                row = {"config_name": cfg["name"], "_path": str(path), "runtime_error": True, "runtime_error_message": repr(exc)}
            rows.append(row)
    return rows


def valid_record(row: dict[str, Any]) -> bool:
    required = ("config_name", "problem_id", "seed", "task_key", "generated_text", "stop_reason", "parsed_answer", "config_hash")
    return all(key in row for key in required) and not row.get("error") and row.get("stop_reason") != "error"


def record_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("config_name")), int(row.get("problem_id")), int(row.get("seed")))


def flatten_result(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    stats = row.get("cache_segment_stats") or {}
    bit_stats = row.get("cache_bitwidth_stats") or {}
    quant_cfg = row.get("quantization_config") or {}
    parser_failure = bool(row.get("parser_error")) or row.get("parsed_answer") is None
    runtime_error = bool(row.get("error")) or row.get("stop_reason") == "error"
    return {
        "task_id": f"aime24:p{row.get('problem_id')}",
        "task_key": row.get("task_key"),
        "problem_id": row.get("problem_id"),
        "problem": row.get("problem"),
        "gold_answer": row.get("reference_answer"),
        "config": cfg["name"],
        "method": cfg["method_group"],
        "cache_mode": cfg["cache_mode"],
        "seed": row.get("seed"),
        "sink_length": cfg["sink_length"],
        "recent_length": cfg["recent_length"],
        "K_bits": cfg["k_bits"],
        "V_bits": cfg["v_bits"],
        "group_size": cfg["group_size"],
        "prompt_tokens": row.get("input_tokens"),
        "generated_tokens": row.get("generated_tokens"),
        "total_tokens": row.get("total_sequence_tokens"),
        "generated_text": row.get("generated_text"),
        "parsed_answer": row.get("parsed_answer"),
        "strict_correct": bool(row.get("is_correct")) and not runtime_error,
        "stop_reason": row.get("stop_reason"),
        "length_truncated": bool(row.get("length_truncated")),
        "parser_failure": parser_failure,
        "runtime_error": runtime_error,
        "runtime_error_message": row.get("error"),
        "wall_time_sec": row.get("wall_time_seconds"),
        "theoretical_effective_bits": theoretical_bits(row, cfg),
        "actual_effective_bits": actual_bits(row, cfg),
        "sink_tokens_final": stats.get("sink_tokens"),
        "recent_tokens_final": stats.get("recent_tokens"),
        "packed_history_tokens_final": stats.get("packed_history_tokens"),
        "pending_tokens_final": stats.get("pending_history_tokens"),
        "K_centroid_count": bit_stats.get("initial_pattern_count"),
        "V_centroid_count": bit_stats.get("initial_pattern_count"),
        "assignment_storage": bit_stats.get("assignment_bytes"),
        "V_gate_storage": bit_stats.get("mask_bytes"),
        "metadata_bits": (bit_stats.get("scale_min_bytes") or 0) * 8,
        "python_tensor_storage_bytes": bit_stats.get("python_tensor_storage_bytes"),
        "config_hash": row.get("config_hash"),
        "path": row.get("_path"),
    }


def theoretical_bits(row: dict[str, Any], cfg: dict[str, Any]) -> float:
    if cfg["name"] == "fp16":
        return 16.0
    total = int(row.get("total_sequence_tokens") or 0)
    stats = effective_bitwidth(
        BitwidthConfig(
            method=cfg["name"],
            total_tokens=total,
            sink_length=int(cfg["sink_length"]),
            recent_length=int(cfg["recent_length"]),
            k_bits=float(cfg["k_bits"]),
            v_bits=float(cfg["v_bits"]),
            group_size=max(int(cfg["group_size"]), 1),
        )
    )
    return float(stats["total_effective_bits_per_scalar"])


def actual_bits(row: dict[str, Any], cfg: dict[str, Any]) -> float:
    if cfg["name"] == "fp16":
        return 16.0
    bit_stats = row.get("cache_bitwidth_stats") or {}
    bytes_used = bit_stats.get("python_tensor_storage_bytes")
    total = int(row.get("total_sequence_tokens") or 0)
    key_heads = bit_stats.get("persistent_key_heads") or 8
    value_heads = bit_stats.get("persistent_value_heads") or 8
    layers = 32
    head_dim = 128
    denom = total * layers * (int(key_heads) + int(value_heads)) * head_dim
    return (float(bytes_used) * 8.0 / denom) if bytes_used and denom > 0 else theoretical_bits(row, cfg)


def summarize_seed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for cfg in CONFIGS:
        for seed in SEEDS:
            subset = [r for r in rows if r["config"] == cfg["name"] and int(r["seed"]) == seed]
            correct = sum(1 for r in subset if r["strict_correct"])
            out.append(
                {
                    "config": cfg["name"],
                    "seed": seed,
                    "records": len(subset),
                    "correct": correct,
                    "accuracy": correct / 30 if subset else None,
                    "parser_failures": sum(1 for r in subset if r["parser_failure"]),
                    "length_truncations": sum(1 for r in subset if r["length_truncated"]),
                    "runtime_errors": sum(1 for r in subset if r["runtime_error"]),
                    "mean_generated_tokens": statistics.mean([int(r["generated_tokens"] or 0) for r in subset]) if subset else None,
                }
            )
    return out


def summarize_config(rows: list[dict[str, Any]], seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    seed_by_cfg = defaultdict(list)
    for row in seed_rows:
        seed_by_cfg[row["config"]].append(float(row["accuracy"] or 0.0))
    for cfg in CONFIGS:
        subset = [r for r in rows if r["config"] == cfg["name"]]
        generated = [int(r["generated_tokens"] or 0) for r in subset]
        theo = [float(r["theoretical_effective_bits"]) for r in subset if r["theoretical_effective_bits"] is not None]
        actual = [float(r["actual_effective_bits"]) for r in subset if r["actual_effective_bits"] is not None]
        accs = seed_by_cfg[cfg["name"]]
        out.append(
            {
                "config": cfg["name"],
                "display_name": cfg["display_name"],
                "records": len(subset),
                "correct_90_task_seed_pairs": sum(1 for r in subset if r["strict_correct"]),
                "task_seed_accuracy": sum(1 for r in subset if r["strict_correct"]) / 90 if subset else None,
                "seed42_accuracy": accs[0] if len(accs) > 0 else None,
                "seed43_accuracy": accs[1] if len(accs) > 1 else None,
                "seed44_accuracy": accs[2] if len(accs) > 2 else None,
                "mean_accuracy": statistics.mean(accs) if accs else None,
                "std_accuracy": statistics.pstdev(accs) if len(accs) > 1 else 0.0,
                "min_accuracy": min(accs) if accs else None,
                "max_accuracy": max(accs) if accs else None,
                "mean_generated_tokens": statistics.mean(generated) if generated else None,
                "median_generated_tokens": statistics.median(generated) if generated else None,
                "p90_generated_tokens": percentile(generated, 0.90),
                "p95_generated_tokens": percentile(generated, 0.95),
                "max_generated_tokens": max(generated) if generated else None,
                "length_truncations": sum(1 for r in subset if r["length_truncated"]),
                "normal_stop_count": sum(1 for r in subset if r["stop_reason"] == "eos"),
                "parser_failures": sum(1 for r in subset if r["parser_failure"]),
                "runtime_errors": sum(1 for r in subset if r["runtime_error"]),
                "theoretical_effective_bits_mean": statistics.mean(theo) if theo else None,
                "theoretical_effective_bits_median": statistics.median(theo) if theo else None,
                "theoretical_effective_bits_p90": percentile(theo, 0.90),
                "actual_effective_bits_mean": statistics.mean(actual) if actual else None,
                "actual_effective_bits_median": statistics.median(actual) if actual else None,
                "actual_effective_bits_p90": percentile(actual, 0.90),
            }
        )
    return out


def paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cfg_key = {(r["config"], int(r["problem_id"]), int(r["seed"])): r for r in rows}
    out = []
    for comp in COMPARISONS:
        for seed in list(SEEDS) + ["aggregate"]:
            keys = [(pid, s) for pid in range(30) for s in SEEDS if seed == "aggregate" or s == seed]
            rescues = regressions = both_correct = both_wrong = missing = 0
            for pid, actual_seed in keys:
                base = by_cfg_key.get((comp["baseline"], pid, actual_seed))
                cand = by_cfg_key.get((comp["candidate"], pid, actual_seed))
                if base is None or cand is None:
                    missing += 1
                    continue
                b = bool(base["strict_correct"])
                c = bool(cand["strict_correct"])
                if (not b) and c:
                    rescues += 1
                elif b and (not c):
                    regressions += 1
                elif b and c:
                    both_correct += 1
                else:
                    both_wrong += 1
            out.append(
                {
                    "comparison_name": comp["name"],
                    "comparison": comp["label"],
                    "baseline": comp["baseline"],
                    "candidate": comp["candidate"],
                    "seed": seed,
                    "paired_n": len(keys) - missing,
                    "rescues": rescues,
                    "regressions": regressions,
                    "ties_both_correct": both_correct,
                    "ties_both_wrong": both_wrong,
                    "ties": both_correct + both_wrong,
                    "net_gain": rescues - regressions,
                    "missing": missing,
                }
            )
    return out


def consistency_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cfg_problem = defaultdict(list)
    for row in rows:
        by_cfg_problem[(row["config"], int(row["problem_id"]))].append(row)
    paired_by_name = {item["name"]: item for item in COMPARISONS}
    out = []
    for comp_name in ("pattern_paper_to_s16", "pattern_paper_to_s0", "pattern_s0_to_s16", "kivi_s0_to_s16", "kivi_paper_to_s16"):
        comp = paired_by_name[comp_name]
        for pid in range(30):
            base_rows = by_cfg_problem[(comp["baseline"], pid)]
            cand_rows = by_cfg_problem[(comp["candidate"], pid)]
            base_correct = sum(1 for r in base_rows if r["strict_correct"])
            cand_correct = sum(1 for r in cand_rows if r["strict_correct"])
            delta = cand_correct - base_correct
            if delta >= 2:
                cls = "STABLE_RESCUE"
            elif delta > 0:
                cls = "PARTIAL_RESCUE"
            elif delta <= -2:
                cls = "STABLE_REGRESSION"
            elif delta < 0:
                cls = "SEED_SENSITIVE"
            else:
                cls = "NO_CHANGE" if len({r["strict_correct"] for r in base_rows + cand_rows}) <= 1 else "SEED_SENSITIVE"
            out.append(
                {
                    "comparison_name": comp_name,
                    "problem_id": pid,
                    "baseline": comp["baseline"],
                    "candidate": comp["candidate"],
                    "baseline_correct_count": base_correct,
                    "candidate_correct_count": cand_correct,
                    "delta_correct_count": delta,
                    "classification": cls,
                }
            )
    return out


def failure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row["runtime_error"] or row["parser_failure"] or row["length_truncated"]:
            out.append(
                {
                    "config": row["config"],
                    "problem_id": row["problem_id"],
                    "seed": row["seed"],
                    "runtime_error": row["runtime_error"],
                    "runtime_error_message": row["runtime_error_message"],
                    "parser_failure": row["parser_failure"],
                    "length_truncated": row["length_truncated"],
                    "stop_reason": row["stop_reason"],
                    "parsed_answer": row["parsed_answer"],
                    "path": row["path"],
                }
            )
    return out


def completeness(records: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [record_key(r) for r in records if "config_name" in r and "problem_id" in r and "seed" in r]
    counts = Counter(keys)
    expected = {(cfg["name"], pid, seed) for cfg in CONFIGS for pid in range(30) for seed in SEEDS}
    actual = set(keys)
    return {
        "expected_records": EXPECTED_RECORDS,
        "actual_records": len(keys),
        "missing": len(expected - actual),
        "duplicates": sum(count - 1 for count in counts.values() if count > 1),
        "missing_keys": sorted([f"{cfg}:p{pid}:seed{seed}" for cfg, pid, seed in expected - actual])[:100],
    }


def decisions(config_summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> dict[str, Any]:
    acc = {r["config"]: r for r in config_summary}
    pair = {(r["comparison_name"], r["seed"]): r for r in paired}

    def seed_deltas(base: str, cand: str) -> list[float]:
        return [
            float(acc[cand][f"seed{seed}_accuracy"]) - float(acc[base][f"seed{seed}_accuracy"])
            for seed in SEEDS
        ]

    pattern_deltas = seed_deltas("pattern_rolling_s0_r128", "pattern_rolling_s16_r128")
    pattern_agg = pair[("pattern_s0_to_s16", "aggregate")]
    if all(delta > 0 for delta in pattern_deltas) and pattern_agg["rescues"] > pattern_agg["regressions"]:
        pattern_supported: Any = True
        pattern_seed_consistent: Any = True
    elif sum(delta > 0 for delta in pattern_deltas) >= 2 and sum(delta < 0 for delta in pattern_deltas) == 0 and pattern_agg["rescues"] > pattern_agg["regressions"]:
        pattern_supported = True
        pattern_seed_consistent = "mostly"
    else:
        pattern_supported = "weak_or_inconclusive"
        pattern_seed_consistent = False

    kivi_deltas = seed_deltas("kivi_rolling_s0_r128", "kivi_rolling_s16_r128")
    kivi_agg = pair[("kivi_s0_to_s16", "aggregate")]
    kivi_supported: Any = bool(statistics.mean(kivi_deltas) > 0 and kivi_agg["rescues"] > kivi_agg["regressions"])
    pattern_final = bool(
        float(acc["pattern_rolling_s16_r128"]["mean_accuracy"]) > float(acc["patternkv_paper"]["mean_accuracy"])
        and pair[("pattern_paper_to_s16", "aggregate")]["rescues"] > pair[("pattern_paper_to_s16", "aggregate")]["regressions"]
    )
    cross = bool(pattern_supported is True and kivi_supported)
    pseudo = bool(float(acc["pattern_rolling_s16_r128"]["mean_accuracy"]) < float(acc["fp16"]["mean_accuracy"]))
    aime25 = bool(pattern_supported is True or pattern_supported == "mostly")
    return {
        "pattern_sink_supported": pattern_supported,
        "pattern_sink_seed_consistent": pattern_seed_consistent,
        "kivi_sink_supported": kivi_supported,
        "pattern_final_method_beats_paper_baseline": pattern_final,
        "cross_method_sink_effect_supported": cross,
        "aime25_validation_recommended": aime25,
        "pseudo_decode_recommended": pseudo,
        "next_priority": "AIME25 full30 validation" if aime25 else "diagnose seed-sensitive AIME24 failures",
    }


def aggregate() -> None:
    prepare()
    raw = load_records()
    cfg_by_name = {cfg["name"]: cfg for cfg in CONFIGS}
    flat = [flatten_result(row, cfg_by_name[str(row.get("config_name"))]) for row in raw if str(row.get("config_name")) in cfg_by_name]
    comp = completeness(raw)
    seed_summary = summarize_seed(flat)
    config_summary = summarize_config(flat, seed_summary)
    paired = paired_rows(flat)
    consistency = consistency_rows(flat)
    failures = failure_rows(flat)
    generation_length = [
        {
            "config": row["config"],
            "problem_id": row["problem_id"],
            "seed": row["seed"],
            "generated_tokens": row["generated_tokens"],
            "stop_reason": row["stop_reason"],
            "length_truncated": row["length_truncated"],
        }
        for row in flat
    ]
    bitwidth = [
        {
            "config": row["config"],
            "problem_id": row["problem_id"],
            "seed": row["seed"],
            "theoretical_effective_bits": row["theoretical_effective_bits"],
            "actual_effective_bits": row["actual_effective_bits"],
            "python_tensor_storage_bytes": row["python_tensor_storage_bytes"],
            "sink_tokens_final": row["sink_tokens_final"],
            "recent_tokens_final": row["recent_tokens_final"],
            "packed_history_tokens_final": row["packed_history_tokens_final"],
            "pending_tokens_final": row["pending_tokens_final"],
            "assignment_storage": row["assignment_storage"],
            "V_gate_storage": row["V_gate_storage"],
            "metadata_bits": row["metadata_bits"],
        }
        for row in flat
    ]
    write_csv(REPORT_DIR / "aime24_full30_all_results.csv", flat)
    write_csv(REPORT_DIR / "aime24_full30_seed_summary.csv", seed_summary)
    write_csv(REPORT_DIR / "aime24_full30_config_summary.csv", config_summary)
    write_csv(REPORT_DIR / "aime24_full30_paired_comparisons.csv", paired)
    write_csv(REPORT_DIR / "aime24_full30_task_seed_consistency.csv", consistency)
    write_csv(REPORT_DIR / "aime24_full30_generation_length.csv", generation_length)
    write_csv(REPORT_DIR / "aime24_full30_bitwidth_summary.csv", bitwidth)
    write_csv(REPORT_DIR / "aime24_full30_failures.csv", failures)
    decision = decisions(config_summary, paired) if len(flat) == EXPECTED_RECORDS else {}
    summary = {
        "full_run_complete": comp["actual_records"] == EXPECTED_RECORDS and comp["missing"] == 0 and comp["duplicates"] == 0 and not any(r["runtime_error"] for r in flat),
        **comp,
        "runtime_errors": sum(1 for r in flat if r["runtime_error"]),
        "parser_failures": sum(1 for r in flat if r["parser_failure"]),
        "length_truncations": sum(1 for r in flat if r["length_truncated"]),
        "seeds": list(SEEDS),
        "configs": [cfg["name"] for cfg in CONFIGS],
        "task_manifest_hash": sha256_file(DATASET_PATH),
        "generation_config_hash": GENERATION_CONFIG_HASH,
        "config_summary": config_summary,
        "paired_comparisons": paired,
        **decision,
    }
    write_json(REPORT_DIR / "aime24_full30_3seed_summary.json", summary)
    write_report(summary, config_summary, paired, consistency)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))


def write_report(summary: dict[str, Any], config_summary: list[dict[str, Any]], paired: list[dict[str, Any]], consistency: list[dict[str, Any]]) -> None:
    cfg = {row["config"]: row for row in config_summary}
    pair = {(row["comparison_name"], row["seed"]): row for row in paired}
    lines = [
        "# AIME24 Full30 Three-Seed Formal Validation",
        "",
        "## 1. Executive Summary",
        "",
        f"- Full run complete: `{summary['full_run_complete']}`",
        f"- Expected records: `{summary['expected_records']}`",
        f"- Actual records: `{summary['actual_records']}`",
        f"- Runtime errors: `{summary['runtime_errors']}`",
        f"- Parser failures: `{summary['parser_failures']}`",
        f"- Length truncations: `{summary['length_truncations']}`",
        f"- Pattern Sink16 supported: `{summary.get('pattern_sink_supported')}`",
        f"- KIVI Sink16 supported: `{summary.get('kivi_sink_supported')}`",
        f"- Cross-method Sink effect supported: `{summary.get('cross_method_sink_effect_supported')}`",
        "",
        "## 2. Experimental Question",
        "",
        "This formal validation tests whether rolling+S16 improves AIME24 long-CoT quality over paper and rolling-only baselines across all 30 AIME24 problems and three fixed seeds.",
        "",
        "## 3. Dataset",
        "",
        f"- Full AIME24 source: `{DATASET_PATH}`",
        f"- Full30 task hash: `{summary['task_manifest_hash']}`",
        "- Problems: `30` unique problem_id values `0..29`.",
        "",
        "## 4. Generation Configuration",
        "",
        f"- Generation config hash: `{GENERATION_CONFIG_HASH}`",
        "- Seeds: `42`, `43`, `44`",
        "- `temperature=0.6`, `top_p=0.95`, `do_sample=true`, `max_new_tokens=32768`, `dtype=float16`.",
        "",
        "## 5. Seven Configurations",
        "",
        "| config | method | cache | sink | recent | K/V bits |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for item in CONFIGS:
        lines.append(f"| {item['name']} | {item['method']} | {item['cache_mode']} | {item['sink_length']} | {item['recent_length']} | {item['k_bits']}/{item['v_bits']} |")
    lines += [
        "",
        "## 6. Paper Configuration Audit",
        "",
        "- PatternKV paper audit: `reports/aime24_full30_3seed/patternkv_paper_config_audit.md`",
        "- KIVI paper audit: `reports/aime24_full30_3seed/kivi_paper_config_audit.md`",
        "",
        "## 7. Runtime Completeness",
        "",
        f"- Missing records: `{summary['missing']}`",
        f"- Duplicate records: `{summary['duplicates']}`",
        "",
        "## 8. Main Three-Seed Accuracy Results",
        "",
        "| config | seed42 | seed43 | seed44 | mean | std | length truncations | actual bits mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in CONFIGS:
        row = cfg[item["name"]]
        lines.append(
            f"| {item['name']} | {row['seed42_accuracy']:.4f} | {row['seed43_accuracy']:.4f} | {row['seed44_accuracy']:.4f} | {row['mean_accuracy']:.4f} | {row['std_accuracy']:.4f} | {row['length_truncations']} | {row['actual_effective_bits_mean']:.4f} |"
        )
    lines += [
        "",
        "## 9. PatternKV Paper vs Rolling vs Sink16",
        "",
        comparison_block(pair, "pattern_paper_to_s0"),
        "",
        comparison_block(pair, "pattern_paper_to_s16"),
        "",
        "## 10. Pattern S0 vs S16",
        "",
        comparison_block(pair, "pattern_s0_to_s16"),
        "",
        "## 11. KIVI S0 vs S16",
        "",
        comparison_block(pair, "kivi_s0_to_s16"),
        "",
        "## 12. Seed Stability",
        "",
    ]
    for item in CONFIGS:
        row = cfg[item["name"]]
        lines.append(f"- `{item['name']}` seed std: `{row['std_accuracy']:.4f}`; min/max: `{row['min_accuracy']:.4f}` / `{row['max_accuracy']:.4f}`.")
    class_counts = Counter(row["classification"] for row in consistency)
    lines += [
        "",
        "## 13. Task-Level Stability",
        "",
        f"- Task-level classifications across pre-registered comparisons: `{dict(class_counts)}`",
        "- Full table: `aime24_full30_task_seed_consistency.csv`.",
        "",
        "## 14. Generation-Length Behavior",
        "",
    ]
    for item in CONFIGS:
        row = cfg[item["name"]]
        lines.append(f"- `{item['name']}` mean/median/P90/P95/max generated tokens: `{row['mean_generated_tokens']:.1f}` / `{row['median_generated_tokens']}` / `{row['p90_generated_tokens']}` / `{row['p95_generated_tokens']}` / `{row['max_generated_tokens']}`.")
    lines += [
        "",
        "## 15. Effective Bitwidth",
        "",
    ]
    for item in CONFIGS:
        row = cfg[item["name"]]
        lines.append(f"- `{item['name']}` theoretical bits mean `{row['theoretical_effective_bits_mean']:.4f}`, actual implementation bits mean `{row['actual_effective_bits_mean']:.4f}`.")
    lines += [
        "",
        "## 16. Quality-vs-Bitwidth Tradeoff",
        "",
        "Quality/bitwidth tradeoffs should be read from `aime24_full30_config_summary.csv` and `aime24_full30_bitwidth_summary.csv`; actual storage includes Python tensor metadata and PatternKV assignment/gate/centroid storage.",
        "",
        "## 17. Relation to Wave 1A Mechanism Findings",
        "",
        "Wave 1A.4 found early-token attention present/enriched and classified both PatternKV and KIVI mechanisms as mixed routing plus value-content protection. This run is benchmark validation only; no observer traces were collected.",
        "",
        "## 18. Hypothesis Decisions",
        "",
        f"- `FULL_AIME24_PATTERN_SINK_SUPPORTED={summary.get('pattern_sink_supported')}`",
        f"- `PATTERN_SINK_SEED_CONSISTENT={summary.get('pattern_sink_seed_consistent')}`",
        f"- `FULL_AIME24_KIVI_SINK_SUPPORTED={summary.get('kivi_sink_supported')}`",
        f"- `CROSS_METHOD_SINK_EFFECT_SUPPORTED={summary.get('cross_method_sink_effect_supported')}`",
        f"- `PATTERN_FINAL_METHOD_BEATS_PAPER_BASELINE={summary.get('pattern_final_method_beats_paper_baseline')}`",
        "",
        "## 19. Limitations",
        "",
        "- Three seeds provide stronger validation than Wave 1A diagnostics but still represent task-seed samples, not 90 independent problems.",
        "- Paper baselines use the repository's available paper-aligned reproduction path for DeepSeek-R1-Distill-Llama-8B.",
        "",
        "## 20. Next Experiment",
        "",
        f"- `AIME25_VALIDATION_RECOMMENDED={summary.get('aime25_validation_recommended')}`",
        f"- `PSEUDO_DECODE_RECOMMENDED={summary.get('pseudo_decode_recommended')}`",
        f"- `NEXT_PRIORITY={summary.get('next_priority')}`",
        "",
        "## 21. Reproducibility",
        "",
        f"- Result dir: `{RESULT_DIR}`",
        f"- Report dir: `{REPORT_DIR}`",
        f"- Config manifest: `{CONFIG_MANIFEST_PATH}`",
    ]
    (REPORT_DIR / "aime24_full30_3seed_formal_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def comparison_block(pair: dict[tuple[str, Any], dict[str, Any]], name: str) -> str:
    rows = [pair[(name, seed)] for seed in list(SEEDS) + ["aggregate"]]
    lines = ["| seed | rescues | regressions | ties | net |", "| --- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['seed']} | {row['rescues']} | {row['regressions']} | {row['ties']} | {row['net_gain']} |")
    return "\n".join(lines)


def preflight_check() -> None:
    rows = load_records()
    by_config = {str(row.get("config_name")): row for row in rows if int(row.get("problem_id", -1)) == 0 and int(row.get("seed", -1)) == 42}
    errors: list[str] = []
    for cfg in CONFIGS:
        row = by_config.get(cfg["name"])
        if row is None:
            errors.append(f"missing preflight record for {cfg['name']}")
            continue
        if not valid_record(row):
            errors.append(f"invalid preflight record for {cfg['name']}")
        if row.get("config_name") != cfg["name"]:
            errors.append(f"{cfg['name']}: config_name mismatch")
        if int(row.get("seed", -1)) != 42:
            errors.append(f"{cfg['name']}: seed mismatch")
        if row.get("stop_reason") == "error" or row.get("error"):
            errors.append(f"{cfg['name']}: runtime error {row.get('error')}")
    pattern_paper = by_config.get("patternkv_paper", {}).get("quantization_config") or {}
    pattern_s0 = by_config.get("pattern_rolling_s0_r128", {}).get("quantization_config") or {}
    if pattern_paper.get("patternkv_cache_path") == pattern_s0.get("patternkv_cache_path") and pattern_paper.get("cache_mode") == pattern_s0.get("cache_mode"):
        errors.append("PatternKV_paper is not distinct from Pattern rolling S0")
    kivi_paper = by_config.get("kivi_paper", {}).get("quantization_config") or {}
    kivi_s0 = by_config.get("kivi_rolling_s0_r128", {}).get("quantization_config") or {}
    if kivi_paper.get("method") == kivi_s0.get("method"):
        errors.append("KIVI_paper is not distinct from KIVI rolling S0")
    payload = {"precheck_pass": not errors, "records": len(by_config), "errors": errors}
    write_json(REPORT_DIR / "aime24_full30_preflight_status.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if errors:
        raise SystemExit("FORMAL_RUN_APPROVED=false")


def status() -> None:
    rows = load_records()
    comp = completeness(rows)
    by_cfg = Counter(str(row.get("config_name")) for row in rows)
    print(json.dumps({"completeness": comp, "by_config": dict(by_cfg)}, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight-check", "aggregate", "status"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "preflight-check":
        preflight_check()
    elif args.mode == "aggregate":
        aggregate()
    else:
        status()


if __name__ == "__main__":
    main()
