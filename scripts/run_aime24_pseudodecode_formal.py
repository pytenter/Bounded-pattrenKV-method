#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.pseudodecode_controls import ACCUMULATION_METRIC_SCHEMA_VERSION, MATCHED_PATH_CONTROL_VERSION, compute_accumulation_gap  # noqa: E402
from bench.pseudodecode_metrics import CHECKPOINTS, write_csv_rows  # noqa: E402
from scripts.run_aime24_pseudodecode_preflight import (  # noqa: E402
    REPORT_DIR,
    SOURCE_COMMIT,
    compare_replays,
    load_model,
    make_args,
    replay_prefix,
    reset_method_state,
    segment_counts,
    write_json,
)
from scripts.run_aime24_execution_path_resolution import replay_pseudo_checkpoints  # noqa: E402


RESULT_DIR = ROOT / "results/aime24_pseudodecode_3090_8gpu"
FORMAL_DIR = RESULT_DIR / "formal"
FP16_REPLAY_DIR = FORMAL_DIR / "fp16_replays"
SHARD_DIR = FORMAL_DIR / "shards"
LOG_DIR = ROOT / "run/aime24_pseudodecode_3090_8gpu/formal_logs"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_reference_artifact(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def load_reference_records() -> list[dict[str, Any]]:
    manifest = read_json(REPORT_DIR / "reference_trajectories_manifest.json")
    out = []
    for row in manifest["rows"]:
        artifact = read_reference_artifact(ROOT / row["artifact_path"])
        out.append({**row, **artifact})
    return out


def quant_configs() -> list[dict[str, Any]]:
    manifest = read_json(REPORT_DIR / "pseudodecode_manifest.json")
    return [cfg for cfg in manifest["conceptual_configs"] if cfg["config"] != "fp16"]


def available_checkpoints(record: dict[str, Any]) -> list[int]:
    return [int(cp) for cp in CHECKPOINTS if int(record["generated_token_count"]) >= int(cp)]


def replay_path(task_key: str, checkpoint: int, mode: str) -> Path:
    safe = task_key.replace(":", "_")
    return FP16_REPLAY_DIR / f"{safe}.cp{int(checkpoint)}.{mode}.pt"


def compact_output(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "logits": output["logits"].detach().cpu(),
        "hidden": output["hidden"].detach().cpu(),
        "attentions": None,
        "past_key_values": None,
    }


def metric_subset(comp: dict[str, Any]) -> dict[str, float]:
    return {
        "hidden_cosine_loss": max(0.0, 1.0 - float(comp["hidden_cosine"])),
        "hidden_relative_L2": float(comp["hidden_relative_L2"]),
        "attention_output_relative_L2": float(comp["attention_output_relative_L2"]),
        "next_token_KL": max(float(comp["next_token_KL"]), 0.0),
        "next_token_JS": max(float(comp["next_token_JS"]), 0.0),
        "target_token_NLL_delta": float(comp["target_token_NLL_delta"]),
        "top1_disagreement": float(comp["top1_disagreement"]),
        "logit_max_abs_diff": float(comp["logit_max_abs_diff"]),
    }


def metric_rows(
    *,
    task: dict[str, Any],
    config: dict[str, Any],
    mode: str,
    checkpoint: int,
    fp16: dict[str, Any],
    quant: dict[str, Any],
) -> list[dict[str, Any]]:
    target = task["generated_token_ids"][checkpoint] if len(task["generated_token_ids"]) > checkpoint else None
    comp = metric_subset(compare_replays(fp16, quant, target))
    return [
        {
            "task_key": task["task_key"],
            "trajectory_sha256": task["full_trajectory_sha256"],
            "config": config["config"],
            "method": config["method"],
            "mode": mode,
            "matched_fp16_mode": mode,
            "checkpoint": checkpoint,
            "generated_checkpoint": checkpoint,
            "absolute_sequence_position": int(task["prompt_token_count"]) + checkpoint,
            "reference_next_token_id": target,
            "layer": "final",
            "metric_name": metric_name,
            "metric_value": value,
            "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
            "accumulation_metric_schema_version": ACCUMULATION_METRIC_SCHEMA_VERSION,
            "source_commit": SOURCE_COMMIT,
        }
        for metric_name, value in comp.items()
    ]


@torch.no_grad()
def precompute_fp16(model_path: Path, *, force: bool = False) -> dict[str, Any]:
    records = load_reference_records()
    FP16_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    args = make_args(model_path, "fp16", 0, 0, config_name="fp16")
    model, _ = load_model(args)
    written = 0
    skipped = 0
    errors: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in records:
            checkpoints = available_checkpoints(task)
            missing_pseudo = [cp for cp in checkpoints if force or not replay_path(task["task_key"], cp, "pseudo").exists()]
            if missing_pseudo:
                pseudo_by_cp = replay_pseudo_checkpoints(
                    model,
                    prompt_ids=task["prompt_token_ids"],
                    generated_ids=task["generated_token_ids"],
                    checkpoints=missing_pseudo,
                )
                for cp, output in pseudo_by_cp.items():
                    torch.save(compact_output(output), replay_path(task["task_key"], cp, "pseudo"))
                    written += 1
            for cp in checkpoints:
                path = replay_path(task["task_key"], cp, "static")
                if path.exists() and not force:
                    skipped += 1
                    continue
                try:
                    output = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                    torch.save(compact_output(output), path)
                    written += 1
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    errors.append({"task_key": task["task_key"], "checkpoint": cp, "mode": "static", "error": f"CUDA OOM: {exc}"})
    finally:
        del model
        torch.cuda.empty_cache()
    manifest = {
        "status": "complete" if not errors else "partial",
        "written": written,
        "skipped_existing": skipped,
        "errors": errors,
        "elapsed_seconds": time.time() - started,
        "fp16_replay_dir": str(FP16_REPLAY_DIR.relative_to(ROOT)),
    }
    write_json(FORMAL_DIR / "fp16_replay_manifest.json", manifest)
    return manifest


def load_fp16(task_key: str, checkpoint: int, mode: str) -> dict[str, Any]:
    return torch.load(replay_path(task_key, checkpoint, mode), map_location="cpu", weights_only=True)


@torch.no_grad()
def run_config_worker(model_path: Path, config_name: str, gpu_id: int) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    configs = {cfg["config"]: cfg for cfg in quant_configs()}
    config = configs[config_name]
    records = load_reference_records()
    args = make_args(model_path, config["method"], config["sink_length"], config["recent_length"], config_name=config["config"])
    model, _ = load_model(args)
    rows: list[dict[str, Any]] = []
    completeness: list[dict[str, Any]] = []
    started = time.time()
    try:
        for task in records:
            checkpoints = available_checkpoints(task)
            reset_method_state(model, config["method"])
            pseudo_by_cp: dict[int, dict[str, Any]] = {}
            try:
                pseudo_by_cp = replay_pseudo_checkpoints(
                    model,
                    prompt_ids=task["prompt_token_ids"],
                    generated_ids=task["generated_token_ids"],
                    checkpoints=checkpoints,
                    method=config["method"],
                )
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                completeness.append({"task_key": task["task_key"], "config": config["config"], "mode": "pseudo", "checkpoint": "all", "status": "oom", "error": str(exc)})
            for cp in checkpoints:
                target = task["generated_token_ids"][cp] if len(task["generated_token_ids"]) > cp else None
                if cp in pseudo_by_cp:
                    fp16_pseudo = load_fp16(task["task_key"], cp, "pseudo")
                    rows.extend(metric_rows(task=task, config=config, mode="pseudo", checkpoint=cp, fp16=fp16_pseudo, quant=pseudo_by_cp[cp]))
                    counts = pseudo_by_cp[cp].get("segment_counts", {})
                    completeness.append(
                        {
                            "task_key": task["task_key"],
                            "config": config["config"],
                            "mode": "pseudo",
                            "checkpoint": cp,
                            "status": "ok",
                            "reference_next_token_id": target,
                            "total_tokens": counts.get("total_tokens"),
                        }
                    )
                reset_method_state(model, config["method"])
                try:
                    q_static = replay_prefix(model, prompt_ids=task["prompt_token_ids"], generated_ids=task["generated_token_ids"], checkpoint=cp, mode="static")
                    counts = segment_counts(q_static["past_key_values"], config["method"])
                    q_static["past_key_values"] = None
                    fp16_static = load_fp16(task["task_key"], cp, "static")
                    rows.extend(metric_rows(task=task, config=config, mode="static", checkpoint=cp, fp16=fp16_static, quant=q_static))
                    completeness.append(
                        {
                            "task_key": task["task_key"],
                            "config": config["config"],
                            "mode": "static",
                            "checkpoint": cp,
                            "status": "ok",
                            "reference_next_token_id": target,
                            "total_tokens": counts.get("total_tokens"),
                        }
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    torch.cuda.empty_cache()
                    completeness.append({"task_key": task["task_key"], "config": config["config"], "mode": "static", "checkpoint": cp, "status": "oom", "error": str(exc)})
                except FileNotFoundError as exc:
                    completeness.append({"task_key": task["task_key"], "config": config["config"], "mode": "static", "checkpoint": cp, "status": "missing_fp16", "error": str(exc)})
    finally:
        del model
        torch.cuda.empty_cache()

    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = SHARD_DIR / f"{config_name}.metrics.csv"
    completeness_path = SHARD_DIR / f"{config_name}.completeness.csv"
    write_csv_rows(metrics_path, rows)
    write_csv_rows(completeness_path, completeness)
    summary = {
        "config": config_name,
        "gpu_id": gpu_id,
        "metric_rows": len(rows),
        "completeness_rows": len(completeness),
        "ok_rows": sum(1 for row in completeness if row.get("status") == "ok"),
        "failed_rows": sum(1 for row in completeness if row.get("status") != "ok"),
        "elapsed_seconds": time.time() - started,
        "metrics_path": str(metrics_path.relative_to(ROOT)),
        "completeness_path": str(completeness_path.relative_to(ROOT)),
    }
    write_json(SHARD_DIR / f"{config_name}.summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def aggregate() -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    completeness_rows: list[dict[str, Any]] = []
    for path in sorted(SHARD_DIR.glob("*.metrics.csv")):
        metric_rows.extend(read_csv_rows(path))
    for path in sorted(SHARD_DIR.glob("*.completeness.csv")):
        completeness_rows.extend(read_csv_rows(path))

    write_csv_rows(REPORT_DIR / "static_vs_pseudo_metrics.csv", metric_rows)
    write_csv_rows(REPORT_DIR / "formal_completeness_audit.csv", completeness_rows)

    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for row in metric_rows:
        key = (row["task_key"], row["config"], row["checkpoint"], row["layer"], row["metric_name"])
        by_key[(row["mode"], *key)] = row
    gaps: list[dict[str, Any]] = []
    for mode, task_key, config, checkpoint, layer, metric_name in list(by_key):
        if mode != "pseudo":
            continue
        pseudo = by_key[(mode, task_key, config, checkpoint, layer, metric_name)]
        static = by_key.get(("static", task_key, config, checkpoint, layer, metric_name))
        if not static:
            continue
        pv = float(pseudo["metric_value"])
        sv = float(static["metric_value"])
        if math.isfinite(pv) and math.isfinite(sv):
            gaps.append(
                {
                    "task_key": task_key,
                    "config": config,
                    "checkpoint": int(float(checkpoint)),
                    "layer": layer,
                    "metric_name": metric_name,
                    "accumulation_gap": compute_accumulation_gap(pseudo_degradation=pv, static_degradation=sv),
                    "pseudo_value": pv,
                    "static_value": sv,
                    "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
                    "accumulation_metric_schema_version": ACCUMULATION_METRIC_SCHEMA_VERSION,
                    "source_commit": pseudo.get("source_commit"),
                    "trajectory_sha256": pseudo.get("trajectory_sha256"),
                }
            )
    write_csv_rows(REPORT_DIR / "accumulation_gap.csv", gaps)

    ok = [row for row in completeness_rows if row.get("status") == "ok"]
    failed = [row for row in completeness_rows if row.get("status") != "ok"]
    core_checkpoints = {"128", "512", "1024", "2048", "4096"}
    core_failed = [row for row in completeness_rows if row.get("checkpoint") in core_checkpoints and row.get("status") != "ok"]
    core_gaps = [row for row in gaps if str(row.get("checkpoint")) in core_checkpoints]
    summary = {
        "metric_rows": len(metric_rows),
        "gap_rows": len(gaps),
        "completeness_rows": len(completeness_rows),
        "ok_completeness_rows": len(ok),
        "failed_completeness_rows": len(failed),
        "formal_core_matched_checkpoints_complete": bool(metric_rows) and not core_failed and len(core_gaps) == 2880,
        "formal_core_checkpoints": [128, 512, 1024, 2048, 4096],
        "formal_failed_reason": None if not failed else "static_full_prefix_oom_at_8192_or_16384_on_24gb_rtx3090",
        "configs": [cfg["config"] for cfg in quant_configs()],
        "formal_run_complete": bool(metric_rows) and not failed,
        "matched_path_control_version": MATCHED_PATH_CONTROL_VERSION,
        "accumulation_metric_schema_version": ACCUMULATION_METRIC_SCHEMA_VERSION,
    }
    write_json(REPORT_DIR / "formal_run_summary.json", summary)
    return summary


def launch(model_path: Path, gpus: list[int]) -> dict[str, Any]:
    preflight = read_json(REPORT_DIR / "preflight_gate_summary.json")
    if not preflight.get("formal_run_approved"):
        raise SystemExit("FORMAL_RUN_APPROVED is not true; refusing to launch formal run.")
    configs = quant_configs()
    if len(gpus) < len(configs):
        raise SystemExit(f"Need at least {len(configs)} GPUs for one-worker-per-config launch, got {len(gpus)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    procs = []
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = f"{ROOT / 'quant'}:{ROOT}:{env_base.get('PYTHONPATH', '')}"
    for gpu, cfg in zip(gpus, configs):
        log_path = LOG_DIR / f"{cfg['config']}.log"
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "worker",
            "--model-path",
            str(model_path),
            "--config",
            cfg["config"],
            "--gpu-id",
            str(gpu),
        ]
        log_f = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log_f, stderr=subprocess.STDOUT, env=env)
        procs.append((cfg["config"], proc, log_f, log_path))
    failures = []
    for config, proc, log_f, log_path in procs:
        code = proc.wait()
        log_f.close()
        if code != 0:
            failures.append({"config": config, "returncode": code, "log_path": str(log_path.relative_to(ROOT))})
    if failures:
        write_json(FORMAL_DIR / "launch_failures.json", failures)
        raise SystemExit(json.dumps({"worker_failures": failures}, indent=2, sort_keys=True))
    return aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("precompute-fp16", "launch", "worker", "aggregate"):
        p = sub.add_parser(name)
        p.add_argument("--model-path", type=Path, default=Path("/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"))
        if name == "precompute-fp16":
            p.add_argument("--force", action="store_true")
        if name == "launch":
            p.add_argument("--gpus", default="0,1,2,3,4,5")
        if name == "worker":
            p.add_argument("--config", required=True)
            p.add_argument("--gpu-id", type=int, required=True)
    args = parser.parse_args()
    if args.cmd == "precompute-fp16":
        print(json.dumps(precompute_fp16(args.model_path, force=args.force), indent=2, sort_keys=True))
    elif args.cmd == "worker":
        run_config_worker(args.model_path, args.config, args.gpu_id)
    elif args.cmd == "launch":
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
        print(json.dumps(launch(args.model_path, gpus), indent=2, sort_keys=True))
    elif args.cmd == "aggregate":
        print(json.dumps(aggregate(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
