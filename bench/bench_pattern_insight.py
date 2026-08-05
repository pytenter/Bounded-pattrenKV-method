#!/usr/bin/env python
"""Real PatternKV Insight runner with observer lifecycle."""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.bench_longbench_patternkv import load_task, sample_id
from bench.gsm8k_paper_utils import config_hash as gsm_config_hash
from bench.gsm8k_paper_utils import extract_reference, load_gsm8k
from bench.paper_config import apply_method_defaults, method_config_dict
from insight.config import InsightRuntimeConfig, load_standard_baselines
from insight.io import atomic_write_json
from insight.runtime import abort_sample, begin_sample, end_sample, get_active_observer
from scripts import run_longbench_paper_8k_single4090 as lb_runner
from bench import bench_gsm8k_paper as gsm_runner


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_selected(path: Path, dataset: str, tasks: list[str]) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [x for x in payload.get("selected", []) if x.get("dataset") == dataset]
    if tasks:
        rows = [x for x in rows if x.get("task") in set(tasks)]
    return rows


def result_path(root: Path, sample: dict[str, Any], level: str, seed: int) -> Path:
    sample_key = sample.get("sample_id") or f"p{int(sample.get('problem_id')):04d}"
    return root / str(sample.get("dataset")) / str(sample.get("task")) / f"{sample_key}_{level}_seed{seed}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["longbench", "gsm8k"], required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--selected-samples-json", type=Path, default=Path("reports/insight_v1/v0/selected_samples.json"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("LONGBENCH_DATA_DIR", "")) if os.environ.get("LONGBENCH_DATA_DIR") else None)
    parser.add_argument("--gsm8k-data-path", type=Path, default=Path("datasets/gsm8k/gsm8k_test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/insight_v2/generation"))
    parser.add_argument("--observer-output-root", type=Path, default=Path("results/insight_v2/observer"))
    parser.add_argument("--insight-output-dir", type=Path, default=Path("reports/insight_v2"))
    parser.add_argument("--gpu-id", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--insight-level", choices=["basic", "oracle"], default="basic")
    parser.add_argument("--oracle-samples-per-head", type=int, default=8)
    parser.add_argument("--oracle-layers", nargs="*", type=int, default=[0, 7, 15, 23, 31])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--method", default="patternkv_paper")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--problem-ids", nargs="*", type=int, default=[])
    parser.add_argument("--max-input-length", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def filter_samples(samples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.sample_ids:
        wanted = set(args.sample_ids)
        samples = [s for s in samples if str(s.get("sample_id")) in wanted]
    if args.problem_ids:
        wanted_ids = set(args.problem_ids)
        samples = [s for s in samples if s.get("problem_id") in wanted_ids]
    if args.limit > 0:
        samples = samples[: args.limit]
    return samples


def _base_metadata(args: argparse.Namespace, sample: dict[str, Any], baselines_hash: str) -> dict[str, Any]:
    return {
        "dataset": args.dataset,
        "task": sample.get("task"),
        "sample_id": sample.get("sample_id") or f"gsm8k:{sample.get('problem_id')}",
        "problem_id": sample.get("problem_id"),
        "sample_index": sample.get("sample_index"),
        "selection_reason": sample.get("selection_reason"),
        "model_path": str(args.model_path),
        "method": "patternkv_paper",
        "seed": args.seed,
        "git_commit": git_commit(),
        "config_hash": baselines_hash,
    }


def _normalize_generation_record(rec: dict[str, Any], args: argparse.Namespace, sample: dict[str, Any], observer_path: Path, observer_status: str) -> dict[str, Any]:
    error = rec.get("error") or rec.get("exception_message")
    return {
        "schema_version": "insight_v2.generation",
        "dataset": args.dataset,
        "task": sample.get("task"),
        "sample_id": sample.get("sample_id") or rec.get("sample_id"),
        "problem_id": sample.get("problem_id") if sample.get("problem_id") != "" else rec.get("problem_id"),
        "method": "patternkv_paper",
        "git_commit": rec.get("git_commit") or git_commit(),
        "config_hash": rec.get("config_hash"),
        "input_token_ids_sha256": rec.get("input_token_ids_sha256"),
        "generated_token_ids": rec.get("generated_token_ids", []),
        "generated_token_ids_sha256": rec.get("generated_token_ids_sha256"),
        "generated_text": rec.get("generated_text") or rec.get("prediction") or "",
        "score": rec.get("score"),
        "is_correct": rec.get("is_correct"),
        "stop_reason": rec.get("stop_reason"),
        "hit_max_new_tokens": rec.get("hit_max_new_tokens"),
        "input_tokens": rec.get("input_tokens") or rec.get("input_tokens_after_special_tokens"),
        "generated_tokens": rec.get("generated_tokens"),
        "wall_time_seconds": rec.get("wall_time_seconds"),
        "peak_memory_allocated": rec.get("peak_memory_allocated_bytes"),
        "peak_memory_reserved": rec.get("peak_memory_reserved_bytes"),
        "observer_output_path": str(observer_path),
        "observer_status": observer_status,
        "error": error,
        "source_record": rec,
    }


def _error_generation_record(args: argparse.Namespace, sample: dict[str, Any], observer_path: Path, exc: BaseException, cfg_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "insight_v2.generation",
        "dataset": args.dataset,
        "task": sample.get("task"),
        "sample_id": sample.get("sample_id") or f"gsm8k:{sample.get('problem_id')}",
        "problem_id": sample.get("problem_id"),
        "method": "patternkv_paper",
        "git_commit": git_commit(),
        "config_hash": cfg_hash,
        "input_token_ids_sha256": None,
        "generated_token_ids": [],
        "generated_token_ids_sha256": None,
        "generated_text": "",
        "score": 0.0 if args.dataset == "longbench" else None,
        "is_correct": False if args.dataset == "gsm8k" else None,
        "stop_reason": "error",
        "hit_max_new_tokens": False,
        "input_tokens": None,
        "generated_tokens": 0,
        "wall_time_seconds": None,
        "peak_memory_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_memory_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "observer_output_path": str(observer_path),
        "observer_status": "aborted",
        "error": repr(exc),
    }


def _is_completed_generation(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        not payload.get("error")
        and payload.get("stop_reason") not in {"error", "oom"}
        and payload.get("observer_status") == "completed"
    )


def build_longbench_context(args: argparse.Namespace) -> tuple[Any, Any, Namespace, dict[str, Any], str, dict[str, Any]]:
    lb_args = lb_runner.method_args("patternkv_paper", str(args.model_path), args.max_input_length, args.output_dir, Path("run/insight_v2/longbench_adapter"), args.data_dir)
    lb_args.gpu_id = str(args.gpu_id)
    lb_args.paper_method_config = apply_method_defaults(lb_args)
    hashes = {
        "git_commit": git_commit(),
        "model_config_sha256": lb_runner.sha256_file(args.model_path / "config.json"),
        "tokenizer_config_sha256": lb_runner.sha256_file(args.model_path / "tokenizer_config.json"),
    }
    cfg_hash = lb_runner.config_hash(args.max_input_length)
    gpu = lb_runner.gpu_info()
    model, tokenizer = lb_runner.load_model(lb_args)
    return model, tokenizer, lb_args, hashes, cfg_hash, gpu


def run_longbench_sample(model: Any, tokenizer: Any, lb_args: Namespace, hashes: dict[str, Any], cfg_hash: str, gpu: dict[str, Any], args: argparse.Namespace, sample: dict[str, Any]) -> dict[str, Any]:
    task = str(sample["task"])
    index = int(sample["sample_index"])
    data = load_task(task, 0, args.data_dir)
    if index < 0 or index >= len(data):
        raise ValueError(f"sample_identity_error: index {index} outside task {task} size {len(data)}")
    ex = data[index]
    sid = sample_id(task, index, ex)
    if sid != sample.get("sample_id"):
        raise ValueError(f"sample_identity_error: selected sample_id={sample.get('sample_id')} actual={sid} index={index}")
    return lb_runner.run_one(model, tokenizer, lb_args, task, index, ex, len(data), hashes, cfg_hash, gpu)


def build_gsm8k_context(args: argparse.Namespace) -> tuple[Any, Any, Namespace, str, str, list[dict[str, Any]]]:
    gsm_args = Namespace(
        method="patternkv_paper",
        model_path=str(args.model_path),
        data_path=args.gsm8k_data_path,
        output_dir=args.output_dir,
        status_dir=Path("run/insight_v2/gsm8k_adapter"),
        experiment_id="insight_v2_gsm8k",
        max_new_tokens=args.max_new_tokens,
        worker_index=0,
        num_workers=1,
        problem_ids=None,
        gpu_id=str(args.gpu_id),
        retry_failed=True,
        retry_oom=True,
        dry_run=False,
        dtype="float16",
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        num_k_base=32,
        num_v_base=32,
    )
    gsm_args.paper_method_config = apply_method_defaults(gsm_args)
    cfg = {"dataset": "gsm8k", "split": "test", "model_path": str(args.model_path), "method": "patternkv_paper", "max_new_tokens": args.max_new_tokens, "do_sample": False, "batch_size": 1, "num_return_sequences": 1, "paper_method_config": method_config_dict(gsm_args)}
    cfg_hash = gsm_config_hash(cfg)
    rows = load_gsm8k(args.gsm8k_data_path)
    model, tokenizer = gsm_runner.load_model(gsm_args)
    return model, tokenizer, gsm_args, cfg_hash, git_commit(), rows


def run_gsm8k_sample(model: Any, tokenizer: Any, gsm_args: Namespace, cfg_hash: str, commit: str, rows: list[dict[str, Any]], sample: dict[str, Any]) -> dict[str, Any]:
    pid = int(sample["problem_id"])
    if pid < 0 or pid >= len(rows):
        raise ValueError(f"sample_identity_error: problem_id {pid} outside GSM8K size {len(rows)}")
    row = rows[pid]
    if int(row["problem_id"]) != pid:
        raise ValueError(f"sample_identity_error: selected problem_id={pid} actual={row.get('problem_id')}")
    if extract_reference(str(row.get("answer") or "")) is None:
        raise ValueError(f"sample_identity_error: problem_id={pid} has unparseable reference answer")
    rec = gsm_runner.run_one(gsm_args, model, tokenizer, row, cfg_hash, commit)
    rec["sample_id"] = f"gsm8k:{pid}"
    return rec


def main() -> None:
    args = parse_args()
    if args.method != "patternkv_paper":
        raise SystemExit("bench_pattern_insight only runs patternkv_paper")
    baselines = load_standard_baselines()
    samples = filter_samples(load_selected(args.selected_samples_json, args.dataset, args.tasks), args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.observer_output_root.mkdir(parents=True, exist_ok=True)
    args.insight_output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sample in samples:
        manifest.append({"sample": sample, "generation_output": str(result_path(args.output_dir, sample, args.insight_level, args.seed)), "observer_output": str(result_path(args.observer_output_root, sample, args.insight_level, args.seed))})
    atomic_write_json(args.insight_output_dir / "runner_manifest.json", {"schema_version": "insight_v2.runner_manifest", "samples": manifest})
    if args.dry_run:
        print(json.dumps({"selected": len(samples), "dry_run": True}, sort_keys=True))
        return

    os.environ.setdefault("PATTERNKV_INSIGHT", "1")
    os.environ.setdefault("PATTERNKV_INSIGHT_LEVEL", args.insight_level)
    os.environ.setdefault("PATTERNKV_INSIGHT_SAMPLE_TOKENS", str(args.oracle_samples_per_head))
    os.environ.setdefault("PATTERNKV_INSIGHT_ORACLE_LAYERS", ",".join(str(x) for x in args.oracle_layers))
    os.environ.setdefault("PATTERNKV_INSIGHT_OUTPUT", str(args.observer_output_root))
    runtime_config = InsightRuntimeConfig.from_env()

    if args.dataset == "longbench":
        model, tokenizer, eval_args, hashes, eval_cfg_hash, gpu = build_longbench_context(args)
        run_one = lambda sample: run_longbench_sample(model, tokenizer, eval_args, hashes, eval_cfg_hash, gpu, args, sample)
    else:
        model, tokenizer, eval_args, eval_cfg_hash, commit, rows = build_gsm8k_context(args)
        run_one = lambda sample: run_gsm8k_sample(model, tokenizer, eval_args, eval_cfg_hash, commit, rows, sample)

    written = skipped = failed = 0
    try:
        for sample in samples:
            gen_path = result_path(args.output_dir, sample, args.insight_level, args.seed)
            obs_path = result_path(args.observer_output_root, sample, args.insight_level, args.seed)
            if args.skip_existing and gen_path.exists() and obs_path.exists() and _is_completed_generation(gen_path):
                skipped += 1
                continue
            metadata = _base_metadata(args, sample, baselines.config_hash)
            begin_sample(metadata, runtime_config)
            try:
                rec = run_one(sample)
                if rec.get("error") or rec.get("stop_reason") in {"error", "oom"}:
                    raise RuntimeError(str(rec.get("error") or rec.get("exception_message") or rec.get("stop_reason")))
                if runtime_config.enabled:
                    end_sample(obs_path)
                    observer_status = "completed"
                else:
                    observer_status = "disabled"
                atomic_write_json(gen_path, _normalize_generation_record(rec, args, sample, obs_path, observer_status))
                written += 1
            except Exception as exc:
                abort_sample(exc, obs_path)
                atomic_write_json(gen_path, _error_generation_record(args, sample, obs_path, exc, eval_cfg_hash))
                failed += 1
            finally:
                if get_active_observer() is not None:
                    abort_sample("active observer leaked after sample", obs_path)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        atomic_write_json(
            Path("run/insight_v2") / args.dataset / f"{args.insight_level}_seed{args.seed}_status.json",
            {
                "schema_version": "insight_v2.status",
                "dataset": args.dataset,
                "method": "patternkv_paper",
                "insight_level": args.insight_level,
                "selected": len(samples),
                "written": written,
                "skipped": skipped,
                "failed": failed,
                "created_at": utc_now(),
            },
        )
        print(json.dumps({"selected": len(samples), "written": written, "skipped": skipped, "failed": failed}, sort_keys=True))
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
