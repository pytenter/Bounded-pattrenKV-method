#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TASK_HASH = "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
EXPECTED_GENERATION_HASH = "a7d6b2f8bab37893b6331c66b3e5eb6a"
PYTHON_BIN = "/home/qinch2023/miniconda3/envs/patternkv-v100/bin/python"
MODEL_PATH = "/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B"
RESULT_DIR_NEW = "results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3"
RUN_DIR = "run/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep"
REPORT_DIR = "reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_sweep"

LOGICAL_CONFIGS: tuple[dict[str, Any], ...] = (
    {"gpu": 0, "config_name": "pattern_rolling_k2v2_s0_r128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "reused", "source_result_path": "results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a/pattern_rolling_k2v2_s0_r128"},
    {"gpu": 1, "config_name": "pattern_rolling_k2v2_s16_r128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 16, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/pattern_rolling_k2v2_s16_r128"},
    {"gpu": 2, "config_name": "pattern_rolling_k2v2_s32_r128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 32, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/pattern_rolling_k2v2_s32_r128"},
    {"gpu": 3, "config_name": "pattern_rolling_k2v2_s64_r128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 64, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "reused", "source_result_path": "results/aime24_int2_wave1_v100_8gpu_wave1a2_sink_recent/wave1a2/pattern_rolling_k2v2_s64_r128"},
    {"gpu": 4, "config_name": "pattern_rolling_k2v2_s128_r128", "method_group": "PatternKV", "method": "patternkv", "cache_mode": "segmented_rolling", "sink_length": 128, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/pattern_rolling_k2v2_s128_r128"},
    {"gpu": 5, "config_name": "kivi_rolling_k2v2_s0_r128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 0, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "reused", "source_result_path": "results/aime24_int2_wave1_v100_8gpu_revised_full/wave1a/kivi_rolling_k2v2_s0_r128"},
    {"gpu": 6, "config_name": "kivi_rolling_k2v2_s16_r128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 16, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/kivi_rolling_k2v2_s16_r128"},
    {"gpu": 7, "config_name": "kivi_rolling_k2v2_s32_r128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 32, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/kivi_rolling_k2v2_s32_r128"},
    {"gpu": 8, "config_name": "kivi_rolling_k2v2_s64_r128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 64, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "reused", "source_result_path": "results/aime24_int2_wave1_v100_8gpu_wave1a2_sink_recent/wave1a2/kivi_rolling_k2v2_s64_r128"},
    {"gpu": 9, "config_name": "kivi_rolling_k2v2_s128_r128", "method_group": "KIVI", "method": "kivi_official", "cache_mode": "segmented_rolling", "sink_length": 128, "recent_length": 128, "residual_length": 128, "k_bits": 2, "v_bits": 2, "group_size": 128, "result_source": "newly_run", "source_result_path": f"{RESULT_DIR_NEW}/kivi_rolling_k2v2_s128_r128"},
)

NEW_GPU_MAPPING: tuple[dict[str, Any], ...] = (
    {"gpu": 0, "config_name": "pattern_rolling_k2v2_s16_r128"},
    {"gpu": 1, "config_name": "pattern_rolling_k2v2_s32_r128"},
    {"gpu": 2, "config_name": "pattern_rolling_k2v2_s128_r128"},
    {"gpu": 3, "config_name": "kivi_rolling_k2v2_s16_r128"},
    {"gpu": 4, "config_name": "kivi_rolling_k2v2_s32_r128"},
    {"gpu": 5, "config_name": "kivi_rolling_k2v2_s128_r128"},
)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(path.glob("*.json"))]


def config_rows_match(rows: list[dict[str, Any]], cfg: dict[str, Any], task_keys: list[str]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if len(rows) != len(task_keys):
        errors.append(f"{cfg['config_name']}: expected {len(task_keys)} rows, got {len(rows)}")
    keys = [row.get("task_key") for row in rows]
    if set(keys) != set(task_keys):
        errors.append(f"{cfg['config_name']}: task key set mismatch")
    if len(keys) != len(set(keys)):
        errors.append(f"{cfg['config_name']}: duplicate task keys")
    for row in rows:
        checks = {
            "method": cfg["method"],
            "config_name": cfg["config_name"],
            "sink_length": cfg["sink_length"],
            "recent_length": cfg["recent_length"],
            "k_bits": cfg["k_bits"],
            "v_bits": cfg["v_bits"],
            "model_path": MODEL_PATH,
            "temperature": 0.6,
            "top_p": 0.95,
            "do_sample": True,
            "max_new_tokens": 32768,
        }
        if cfg["method"] == "patternkv":
            checks["cache_mode"] = cfg["cache_mode"]
        for key, expected in checks.items():
            if row.get(key) != expected:
                errors.append(f"{cfg['config_name']} {row.get('task_key')}: {key}={row.get(key)!r}, expected {expected!r}")
        quant_cfg = row.get("quantization_config") or {}
        for key in ("sink_length", "recent_length", "k_bits", "v_bits"):
            if quant_cfg.get(key) != cfg[key]:
                errors.append(f"{cfg['config_name']} {row.get('task_key')}: quantization_config.{key}={quant_cfg.get(key)!r}, expected {cfg[key]!r}")
        segment_stats = row.get("cache_segment_stats") or {}
        if segment_stats:
            if segment_stats.get("sink_tokens") != min(int(row.get("total_sequence_tokens") or 0), cfg["sink_length"]):
                errors.append(f"{cfg['config_name']} {row.get('task_key')}: sink token accounting mismatch")
            if segment_stats.get("recent_tokens") != min(max(int(row.get("total_sequence_tokens") or 0) - cfg["sink_length"], 0), cfg["recent_length"]):
                errors.append(f"{cfg['config_name']} {row.get('task_key')}: recent token accounting mismatch")
        if row.get("error"):
            errors.append(f"{cfg['config_name']} {row.get('task_key')}: runtime error {row.get('error')}")
    return not errors, errors


def validate_reuse(task_keys: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    per_config: list[dict[str, Any]] = []
    for cfg in LOGICAL_CONFIGS:
        if cfg["result_source"] != "reused":
            continue
        config_rows = load_rows(Path(cfg["source_result_path"]))
        ok, cfg_errors = config_rows_match(config_rows, cfg, task_keys)
        per_config.append(
            {
                "config_name": cfg["config_name"],
                "status": "passed" if ok else "failed",
                "record_count": len(config_rows),
                "source_result_path": cfg["source_result_path"],
                "source_commit": config_rows[0].get("git_commit") if config_rows else None,
                "source_config_hash": config_rows[0].get("config_hash") if config_rows else None,
                "errors": cfg_errors,
            }
        )
        rows.extend(config_rows)
        errors.extend(cfg_errors)
    return {
        "status": "passed" if not errors else "failed",
        "reused_config_count": len(per_config),
        "newly_run_config_count": sum(1 for cfg in LOGICAL_CONFIGS if cfg["result_source"] == "newly_run"),
        "reused_record_count": len(rows),
        "planned_new_record_count": sum(1 for cfg in LOGICAL_CONFIGS if cfg["result_source"] == "newly_run") * len(task_keys),
        "per_config": per_config,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-manifest", type=Path, default=Path("configs/aime24_wave1_selected_tasks.json"))
    parser.add_argument("--previous-manifest", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/wave1a2_sink_recent_manifest.json"))
    parser.add_argument("--output-json", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_length_sweep_manifest.json"))
    parser.add_argument("--output-md", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_length_sweep_manifest.md"))
    args = parser.parse_args()
    task_hash = sha256_file(args.task_manifest)
    if task_hash != EXPECTED_TASK_HASH:
        raise SystemExit(f"ABORT_SINK_SWEEP=true task manifest hash mismatch: {task_hash}")
    previous = json.loads(args.previous_manifest.read_text(encoding="utf-8"))
    generation_hash = previous["generation_config_hash"]
    if generation_hash != EXPECTED_GENERATION_HASH:
        raise SystemExit(f"ABORT_SINK_SWEEP=true generation config hash mismatch: {generation_hash}")
    task_keys = list(previous["task_keys"])
    reuse_validation = validate_reuse(task_keys)
    if reuse_validation["status"] != "passed":
        raise SystemExit("ABORT_SINK_SWEEP=true reuse validation failed\n" + "\n".join(reuse_validation["errors"][:20]))
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "HEAD"])
    manifest = {
        "branch": branch,
        "starting_head": head,
        "task_manifest_path": str(args.task_manifest),
        "task_manifest_hash": task_hash,
        "task_keys": task_keys,
        "generation_config": previous["generation_config"],
        "generation_config_hash": generation_hash,
        "model_path": MODEL_PATH,
        "python": PYTHON_BIN,
        "torch": previous["torch"],
        "cuda_runtime": previous["cuda_runtime"],
        "logical_configs": list(LOGICAL_CONFIGS),
        "new_gpu_mapping": list(NEW_GPU_MAPPING),
        "reuse_validation": reuse_validation,
        "result_dir_new": RESULT_DIR_NEW,
        "run_dir": RUN_DIR,
        "report_dir": REPORT_DIR,
        "launch_timestamp": datetime.now(timezone.utc).isoformat(),
        "effective_bitwidth_accounting_method": "compact theoretical bits and actual implementation storage bits from cache_bitwidth_stats; includes packed payload, scale/min, assignments, V gate, centroids, FP16 sink/recent/pending when present.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Wave 1A.3 Sink Length Sweep Manifest",
        "",
        f"- Branch: `{branch}`",
        f"- Starting HEAD: `{head}`",
        f"- Task manifest: `{args.task_manifest}`",
        f"- Task manifest hash: `{task_hash}`",
        f"- Generation config hash: `{generation_hash}`",
        f"- Model: `{MODEL_PATH}`",
        f"- Reuse validation: `{reuse_validation['status']}`",
        f"- Reused configs: `{reuse_validation['reused_config_count']}`",
        f"- Newly-run configs: `{reuse_validation['newly_run_config_count']}`",
        "",
        "## Logical Configs",
        "",
        "| config | method | sink | recent | source | source path |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for cfg in LOGICAL_CONFIGS:
        lines.append(f"| `{cfg['config_name']}` | {cfg['method_group']} | {cfg['sink_length']} | {cfg['recent_length']} | {cfg['result_source']} | `{cfg['source_result_path']}` |")
    lines += ["", "## New Run GPU Mapping", "", "| GPU | config |", "| ---: | --- |"]
    for row in NEW_GPU_MAPPING:
        lines.append(f"| {row['gpu']} | `{row['config_name']}` |")
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output_json), "reuse_validation": reuse_validation["status"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
