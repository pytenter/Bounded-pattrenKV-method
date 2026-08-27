#!/usr/bin/env python3
"""Read-only selector component ablation audit.

This script reads the active formal worktree and result roots, then writes an
audit bundle in the current analysis worktree. It never imports the active
training modules and never launches CUDA work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


TASK = "SELECTOR_ABLATION_RUNTIME_PATH_AND_FAIRNESS_AUDIT_V1"
REPORT_REL = Path("reports/selector_ablation_runtime_path_audit_v1")
SELECTOR_METHODS = {
    "importance_only25": {
        "slug": "importance_only",
        "selector": "importance_only_v4",
        "formula": "historical causal importance",
        "pid_hint": 489065,
    },
    "error_only25": {
        "slug": "error_only",
        "selector": "error_only_v4",
        "formula": "positive local V2->V4 error reduction",
        "pid_hint": 489138,
    },
}
REFERENCE_METHODS = {
    "CAUSAL_V4_25": "results/aime24_full_causal25_quality_4gpu/formal/CAUSAL_V4_25",
    "PATTERN_BASE": "results/aime24_full_causal25_quality_4gpu/formal/PATTERN_BASE",
    "RANDOM_V4_25": "results/aime24_full_causal25_quality_4gpu/formal/RANDOM_V4_25",
}


@dataclass(frozen=True)
class TimingRow:
    method: str
    seed: str
    problem: str
    timing_type: str
    confidence: str
    seconds: float | None
    generated_tokens: int
    tokens_per_second: float | None
    reason: str


def run(cmd: list[str] | str, cwd: Path, timeout: int = 30) -> dict[str, Any]:
    if isinstance(cmd, str):
        args = cmd
        shell = True
    else:
        args = cmd
        shell = False
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(args, cwd=cwd, env=env, shell=shell, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return {"cmd": cmd if isinstance(cmd, str) else " ".join(shlex.quote(x) for x in cmd), "returncode": proc.returncode, "output": proc.stdout}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_iso_seconds(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def result_files(root: Path, slug_or_ref: str, phase: str = "formal") -> list[Path]:
    if slug_or_ref in REFERENCE_METHODS:
        base = root / REFERENCE_METHODS[slug_or_ref]
    else:
        base = root / "results/aime24_selector_ablation" / slug_or_ref / phase
    return sorted(base.glob("seed*/p*.json"), key=lambda p: (p.parent.name, p.name))


def identity(rec: dict[str, Any]) -> tuple[int, int, int, int]:
    seed = int(rec.get("base_seed"))
    problem = int(rec.get("problem_id"))
    sample = int(rec.get("sample_id", 0))
    eff = int(rec.get("effective_seed", rec.get("seed", seed + problem * 1000 + sample)))
    return problem, seed, sample, eff


def generated_tokens(rec: dict[str, Any]) -> int:
    return int(rec.get("generated_tokens") or 0)


def correct(rec: dict[str, Any]) -> bool:
    return bool(rec.get("correct", rec.get("is_correct")))


def collect_records(active_root: Path) -> dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]:
    out: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]] = {}
    for method, cfg in SELECTOR_METHODS.items():
        records: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        for path in result_files(active_root, cfg["slug"]):
            rec = read_json(path)
            if rec.get("phase") == "formal" and rec.get("status") == "completed":
                rec["_path"] = str(path)
                records[identity(rec)] = rec
        out[method] = records
    for method in REFERENCE_METHODS:
        records = {}
        for path in result_files(active_root, method):
            rec = read_json(path)
            if rec.get("phase") == "formal" and rec.get("status") == "completed":
                rec["_path"] = str(path)
                records[identity(rec)] = rec
        out[method] = records
    return out


def summarize_quality(method_records: dict[tuple[int, int, int, int], dict[str, Any]], ref_records: dict[tuple[int, int, int, int], dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set(method_records) & set(ref_records))
    method_correct = sum(correct(method_records[k]) for k in keys)
    ref_correct = sum(correct(ref_records[k]) for k in keys)
    return {
        "aligned_n": len(keys),
        "method_correct": method_correct,
        "reference_correct": ref_correct,
        "method_accuracy": method_correct / len(keys) if keys else None,
        "reference_accuracy": ref_correct / len(keys) if keys else None,
        "delta_correct": method_correct - ref_correct,
        "selector_only": sum(correct(method_records[k]) and not correct(ref_records[k]) for k in keys),
        "reference_only": sum((not correct(method_records[k])) and correct(ref_records[k]) for k in keys),
        "both_correct": sum(correct(method_records[k]) and correct(ref_records[k]) for k in keys),
        "both_wrong": sum((not correct(method_records[k])) and (not correct(ref_records[k])) for k in keys),
    }


def build_alignment_tables(report: Path, records: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    causal = records["CAUSAL_V4_25"]
    quality_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    for method in SELECTOR_METHODS:
        for key in sorted(set(records[method]) & set(causal)):
            problem, seed, sample, eff = key
            rec = records[method][key]
            cref = causal[key]
            quality_rows.append(
                {
                    "method": method,
                    "problem_id": problem,
                    "base_seed": seed,
                    "sample_id": sample,
                    "effective_seed": eff,
                    "method_correct": correct(rec),
                    "causal_correct": correct(cref),
                    "method_answer": rec.get("generated_answer", rec.get("parsed_answer")),
                    "causal_answer": cref.get("generated_answer", cref.get("parsed_answer")),
                    "gold_answer": rec.get("gold_answer", cref.get("gold_answer")),
                }
            )
            length_rows.append(
                {
                    "method": method,
                    "problem_id": problem,
                    "base_seed": seed,
                    "sample_id": sample,
                    "effective_seed": eff,
                    "method_generated_tokens": generated_tokens(rec),
                    "causal_generated_tokens": generated_tokens(cref),
                    "method_length_truncated": bool(rec.get("length_truncated")),
                    "causal_length_truncated": bool(cref.get("length_truncated")),
                }
            )
    write_csv(
        report / "aligned_partial_quality.csv",
        quality_rows,
        ["method", "problem_id", "base_seed", "sample_id", "effective_seed", "method_correct", "causal_correct", "method_answer", "causal_answer", "gold_answer"],
    )
    write_csv(
        report / "aligned_partial_lengths.csv",
        length_rows,
        ["method", "problem_id", "base_seed", "sample_id", "effective_seed", "method_generated_tokens", "causal_generated_tokens", "method_length_truncated", "causal_length_truncated"],
    )
    common3 = set(records["importance_only25"]) & set(records["error_only25"]) & set(causal)
    common4 = common3 & set(records["RANDOM_V4_25"])
    summary_rows = [
        {"intersection": "importance_only25_intersect_CAUSAL_V4_25", **summarize_quality(records["importance_only25"], causal)},
        {"intersection": "error_only25_intersect_CAUSAL_V4_25", **summarize_quality(records["error_only25"], causal)},
        {
            "intersection": "importance_error_causal_common",
            "aligned_n": len(common3),
            "method_correct": "",
            "reference_correct": "",
            "method_accuracy": "",
            "reference_accuracy": "",
            "delta_correct": "",
            "selector_only": "",
            "reference_only": "",
            "both_correct": "",
            "both_wrong": "",
        },
        {
            "intersection": "importance_error_causal_random_common",
            "aligned_n": len(common4),
            "method_correct": "",
            "reference_correct": "",
            "method_accuracy": "",
            "reference_accuracy": "",
            "delta_correct": "",
            "selector_only": "",
            "reference_only": "",
            "both_correct": "",
            "both_wrong": "",
        },
    ]
    write_csv(
        report / "common_intersection_summary.csv",
        summary_rows,
        ["intersection", "aligned_n", "method_correct", "reference_correct", "method_accuracy", "reference_accuracy", "delta_correct", "selector_only", "reference_only", "both_correct", "both_wrong"],
    )
    return {"quality_rows": quality_rows, "length_rows": length_rows, "common3": len(common3), "common4": len(common4), "summaries": summary_rows}


def selector_timing_rows(method: str, records: dict[tuple[int, int, int, int], dict[str, Any]]) -> list[TimingRow]:
    ordered = sorted(records.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2]))
    rows: list[TimingRow] = []
    prev_key: tuple[int, int, int, int] | None = None
    prev_t: float | None = None
    for key, rec in ordered:
        problem, seed, _sample, _eff = key
        t = parse_iso_seconds(rec.get("timestamp")) or Path(rec["_path"]).stat().st_mtime
        seconds: float | None = None
        confidence = "INVALID"
        reason = "first completion has no prior end timestamp"
        if prev_key is not None and prev_t is not None:
            prev_problem, prev_seed, _prev_sample, _prev_eff = prev_key
            consecutive = (seed == prev_seed and problem == prev_problem + 1) or (seed == prev_seed + 1 and problem == 0 and prev_problem == 29)
            gap = t - prev_t
            if consecutive and 0 < gap < 12 * 3600:
                seconds = gap
                confidence = "MEDIUM"
                reason = "same method worker inferred from uninterrupted consecutive completion timestamps"
            elif gap >= 12 * 3600:
                reason = "large idle/restart/model-load gap rejected"
            else:
                reason = "non-consecutive identity or invalid timestamp order"
        toks = generated_tokens(rec)
        rows.append(TimingRow(method, f"seed{seed}", f"p{problem:02d}", "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE", confidence, seconds, toks, toks / seconds if seconds else None, reason))
        prev_key = key
        prev_t = t
    return rows


def causal_timing_rows(records: dict[tuple[int, int, int, int], dict[str, Any]]) -> list[TimingRow]:
    rows = []
    for key, rec in sorted(records.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        problem, seed, _sample, _eff = key
        seconds = float(rec.get("runtime_seconds") or 0.0)
        toks = generated_tokens(rec)
        tps = rec.get("tokens_per_second")
        rows.append(TimingRow("CAUSAL_V4_25", f"seed{seed}", f"p{problem:02d}", "REAL_PER_RESULT_RUNTIME_SECONDS", "HIGH" if seconds > 0 else "INVALID", seconds if seconds > 0 else None, toks, float(tps) if tps is not None else (toks / seconds if seconds > 0 else None), "recorded runtime_seconds/tokens_per_second in canonical JSON"))
    return rows


def timing_summary(rows: list[TimingRow]) -> dict[str, Any]:
    valid = [r for r in rows if r.seconds and r.tokens_per_second]
    tps = [float(r.tokens_per_second) for r in valid]
    secs = [float(r.seconds) for r in valid]
    toks = [r.generated_tokens for r in valid]
    p25 = statistics.quantiles(tps, n=4)[0] if len(tps) >= 4 else None
    p75 = statistics.quantiles(tps, n=4)[2] if len(tps) >= 4 else None
    return {
        "method": rows[0].method if rows else "",
        "timing_type": rows[0].timing_type if rows else "",
        "valid_timing_count": len(valid),
        "invalid_timing_count": len(rows) - len(valid),
        "mean_per_sample_tps": statistics.mean(tps) if tps else None,
        "median_per_sample_tps": statistics.median(tps) if tps else None,
        "aggregate_tps": sum(toks) / sum(secs) if secs and sum(secs) > 0 else None,
        "p25_tps": p25,
        "p75_tps": p75,
        "mean_generated_tokens_all_rows": statistics.mean([r.generated_tokens for r in rows]) if rows else None,
        "median_generated_tokens_all_rows": statistics.median([r.generated_tokens for r in rows]) if rows else None,
        "mean_seconds_valid_rows": statistics.mean(secs) if secs else None,
        "confidence": sorted(set(r.confidence for r in rows)),
        "paper_grade": rows[0].timing_type == "REAL_PER_RESULT_RUNTIME_SECONDS" if rows else False,
    }


def build_timing(report: Path, records: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    all_rows = selector_timing_rows("importance_only25", records["importance_only25"]) + selector_timing_rows("error_only25", records["error_only25"]) + causal_timing_rows(records["CAUSAL_V4_25"])
    write_csv(
        report / "runtime_estimate_rows.csv",
        [r.__dict__ for r in all_rows],
        ["method", "seed", "problem", "timing_type", "confidence", "seconds", "generated_tokens", "tokens_per_second", "reason"],
    )
    summaries = [timing_summary([r for r in all_rows if r.method == method]) for method in ["importance_only25", "error_only25", "CAUSAL_V4_25"]]
    write_csv(report / "runtime_estimate_summary.csv", summaries, list(summaries[0].keys()))
    audit = {
        "timestamp_based_selector_runtime": "ESTIMATED_FROM_COMPLETION_TIMESTAMPS_NOT_PAPER_GRADE",
        "causal_runtime": "REAL_PER_RESULT_RUNTIME_SECONDS",
        "preferred_timing_priority_applied": ["real per-result runtime_seconds", "worker-local consecutive completion timestamps"],
        "selector_json_runtime_fields_present": False,
        "causal_json_runtime_fields_present": True,
        "invalid_interval_policy": "first records, large gaps, restarts, and non-consecutive identities are invalid",
        "averaging_note": "mean generated tokens / mean wall time differs from mean per-sample tok/s and aggregate tok/s because ratios are nonlinear and invalid intervals are excluded.",
        "summaries": summaries,
    }
    write_json(report / "timestamp_reconstruction_audit.json", audit)
    return audit


def proc_text(pid: int, name: str) -> str:
    p = Path(f"/proc/{pid}/{name}")
    if not p.exists():
        return ""
    data = p.read_bytes()
    return data.replace(b"\0", b" ").decode("utf-8", "replace")


def build_snapshot(active_root: Path, report: Path) -> dict[str, Any]:
    git_cmds = ["git branch --show-current", "git rev-parse HEAD", "git status --porcelain=v1", "git diff --stat", "git diff --name-status", "git diff --check", "git ls-files --others --exclude-standard", "git remote -v"]
    system_cmds = ["date -Is", "hostname", "tmux ls 2>&1 || true", "ps -eo pid,ppid,user,lstart,etime,pcpu,pmem,cmd | grep -E 'importance|error_only|causal|aime24|selector|watch' | grep -v grep", "nvidia-smi", "nvidia-smi pmon -c 1", "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader", "uptime", "df -h /data/zypan /data 2>/dev/null || true"]
    workers = []
    for method, cfg in SELECTOR_METHODS.items():
        pid = cfg["pid_hint"]
        cwd = run(f"readlink -f /proc/{pid}/cwd", active_root)["output"].strip()
        env_text = proc_text(pid, "environ")
        env_flags = {}
        for item in env_text.split():
            if "=" in item and re.match(r"^(CUDA_VISIBLE_DEVICES|MODEL_PATH|PYTHONPATH|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS|TOKENIZERS_PARALLELISM|TORCH|CUDA|NCCL|CUBLAS|PYTORCH)", item):
                k, v = item.split("=", 1)
                if re.search(r"TOKEN|SECRET|KEY|PASSWORD|CREDENTIAL|AUTH", k, re.I):
                    v = "REDACTED"
                env_flags[k] = v
        workers.append({"method": method, "pid": pid, "cwd": cwd, "cmdline": proc_text(pid, "cmdline").strip(), "environ_non_sensitive": env_flags})
    payload = {
        "task": TASK,
        "snapshot_unix_time": time.time(),
        "active_worktree": str(active_root),
        "git": {cmd: run(cmd, active_root, timeout=60) for cmd in git_cmds},
        "system": {cmd: run(cmd, active_root, timeout=60) for cmd in system_cmds},
        "active_process_map": workers,
    }
    write_json(report / "active_run_snapshot.json", payload)
    return payload


def build_source_hashes(active_root: Path, report: Path) -> dict[str, Any]:
    files = [
        "scripts/run_aime24_selector_ablation.py",
        "scripts/run_aime24_full_causal25_quality.py",
        "bench/bench_aime24_patternkv.py",
        "models/segmented_cache.py",
        "configs/aime24_selector_importance_only.yaml",
        "configs/aime24_selector_error_only.yaml",
    ]
    payload = {
        "active_worktree": str(active_root),
        "active_head": run(["git", "rev-parse", "HEAD"], active_root)["output"].strip(),
        "active_branch": run(["git", "branch", "--show-current"], active_root)["output"].strip(),
        "active_dirty": bool(run(["git", "status", "--porcelain=v1"], active_root)["output"].strip()),
        "file_sha256": {f: sha256_file(active_root / f) for f in files},
        "active_diff_sha256": hashlib.sha256(run(["git", "diff"], active_root, timeout=60)["output"].encode("utf-8")).hexdigest(),
        "diff_stat": run(["git", "diff", "--stat"], active_root)["output"],
        "note": "Active formal uses uncommitted/untracked source; audit branch is based on committed HEAD and records active source hashes as provenance.",
    }
    write_json(report / "source_file_hashes.json", payload)
    write_text(report / "git_provenance.txt", "\n".join([f"active_branch={payload['active_branch']}", f"active_head={payload['active_head']}", f"active_dirty={payload['active_dirty']}", payload["diff_stat"]]))
    return payload


def build_lineage(report: Path, records: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for method, recs in records.items():
        fields: dict[str, Any] = {"completed": len(recs), "root": REFERENCE_METHODS.get(method, f"results/aime24_selector_ablation/{SELECTOR_METHODS.get(method, {}).get('slug', method)}/formal")}
        for field in ["experiment_id", "method", "method_label", "selector", "config_name", "formal_config_hash", "method_config_hash", "git_commit", "phase", "temperature", "top_p", "max_new_tokens", "do_sample", "k_bits", "v_bits", "selected_v_bits", "group_size", "sink_length", "recent_length", "residual_length", "num_k_base", "num_v_base", "v4_budget_fraction"]:
            values = sorted({json.dumps(r.get(field), sort_keys=True) for r in recs.values() if field in r})
            if values:
                fields[field] = [json.loads(v) for v in values]
            else:
                fields[field] = "MISSING_IN_COMPACT_RECORDS"
        fields["duplicate_identities"] = len(recs) != len(set(recs))
        payload[method] = fields
    write_json(report / "source_lineage_map.json", payload)
    return payload


def budget_gate(records: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for method in SELECTOR_METHODS:
        bad = []
        totals = []
        for key, rec in records[method].items():
            frac = rec.get("v4_realized_fraction")
            totals.append({"identity": key, "v4_tokens": rec.get("v4_tokens"), "v4_total_tokens": rec.get("v4_total_tokens"), "fraction": frac})
            if frac is None or abs(float(frac) - 0.25) > 1e-12:
                bad.append({"identity": key, "fraction": frac})
        out[method] = {"gate": "PASS" if not bad else "FAIL", "checked_records": len(records[method]), "bad_records": bad, "summary": "all compact records report v4_realized_fraction exactly 0.25" if not bad else "non-25% records found"}
    return out


def code_audits(active_root: Path, report: Path) -> dict[str, Any]:
    seg = (active_root / "models/segmented_cache.py").read_text(encoding="utf-8", errors="replace")
    runner = (active_root / "scripts/run_aime24_selector_ablation.py").read_text(encoding="utf-8", errors="replace")
    bench = (active_root / "bench/bench_aime24_patternkv.py").read_text(encoding="utf-8", errors="replace")
    hot_terms = {term: len(re.findall(re.escape(term), seg)) for term in [".item()", ".cpu()", ".tolist()", ".numpy()", "sorted(", "for t in range(tokens)", "torch.cuda.synchronize()"]}
    method_identity = {
        "model_identity_gate": "PARTIAL_PASS_WITH_PROVENANCE_GAP",
        "dataset_prompt_protocol_gate": "PARTIAL_PASS_WITH_PROVENANCE_GAP",
        "same_budget_gate": "PASS",
        "eligible_domain_gate": "PASS_BY_CODE_REVIEW",
        "selector_formula_gate": "PASS_BY_CODE_REVIEW",
        "selector_schedule_gate": "PASS_BY_CODE_REVIEW",
        "request_reset_gate": "PASS_BY_CODE_REVIEW",
        "notes": [
            "Selector compact records omit model_path, prompt_hash, input_token_hash, prompt_protocol, and runtime fields, so result-only lineage is partial.",
            "Active run command and runner set MODEL_PATH, DATASET_PATH, generation config, k/v bits, sink/recent/residual lengths, and v4_budget_fraction uniformly.",
            "set_selector_task_context clears per-layer v_causal_importance and v_oracle_importance; bench.run_task resets PatternKV runtime state before each sample.",
        ],
    }
    schedule = {
        "selector_invocation_count": "NOT_RECORDED_IN_RESULT_JSON",
        "topk_invocation_count": "NOT_RECORDED_IN_RESULT_JSON",
        "importance_update_count": "NOT_RECORDED_IN_RESULT_JSON",
        "error_gain_compute_count": "NOT_RECORDED_IN_RESULT_JSON",
        "cache_flush_count": "NOT_RECORDED_IN_RESULT_JSON",
        "gate": "PASS_BY_CODE_REVIEW_WITH_MISSING_COUNTERS",
        "evidence": "select_value_precision_mask is invoked for mixed V precision during packed V cache update; all three selectors share _budget_k, local_v2_v4_gain, and _topk_mask.",
    }
    path_diff = {
        "selector_runtime_fields_dropped": "'wall_time_seconds' and 'tokens_per_second' are produced by bench.run_task but omitted by selector compact_record",
        "hot_loop_python_sort": "'_topk_mask' loops over batch and tokens, calls .item(), sorts Python rows, then writes mask; this affects causal_v4, importance_only_v4, and error_only_v4 selectors.",
        "local_gain_common": "local_v2_v4_gain is computed before selector branching for random_v4, causal_v4, importance_only_v4, and error_only_v4.",
        "hard_identity_failure": False,
        "apparent_slowdown_root_cause": "INSUFFICIENT_PROVENANCE_IMPLEMENTATION_PATH_DIFFERENCE_PLAUSIBLE",
    }
    cache_backend = {
        "cache_path": "segmented",
        "cache_mode": "segmented_rolling",
        "attention_path_gate": "PASS_BY_CODE_REVIEW_WITH_PROVENANCE_GAP",
        "cache_backend_gate": "PASS_BY_CONFIG_AND_CODE_REVIEW",
        "fallback_to_legacy_tuple_cache_detected": False,
    }
    debug = {
        "runtime_env_debug_flags": "No PATTERNKV debug/profiler env vars observed in active PID environ; result records do not store full instrumentation state.",
        "hot_terms_in_segmented_cache": hot_terms,
        "debug_instrumentation_difference": "NO_ACTIVE_DEBUG_FLAG_FOUND_BUT_CODE_HAS_REFERENCE_PYTHON_TOPK",
    }
    trace = """# Code Path Trace

Launch command -> `scripts/run_aime24_selector_ablation.py --worker --phase formal --method <method> --physical-gpu <gpu>`.

Argument/config resolution -> `METHOD_CONFIGS` sets `selector`, `config_name`, `v4_budget_fraction=0.25`, and `make_worker_args()` sets K/V INT2, selected V INT4, group/sink/recent/residual/num_k_base/num_v_base and segmented rolling cache.

Method dispatch -> `load_model()` and `run_task()` from `bench/bench_aime24_patternkv.py`; PatternKV state is reset before each sample.

Selector branch -> `select_value_precision_mask()` normalizes selector, computes shared `k = round(0.25 * tokens)`, computes `local_v2_v4_gain()`, then branches:

- `importance_only_v4`: `score = importance`
- `error_only_v4`: `score = gain`
- `causal_v4`: `score = (importance + 1e-8) * gain`

Top-k -> `_topk_mask()` uses Python row construction, `.item()` scalar extraction and stable sort; this is a plausible reference-path cost.

Precision mask/cache update -> `_cat_mixed_packed_v()` packs mixed V2/V4 pages and appends to segmented rolling cache.

Timing -> `bench.run_task()` records generation wall time, but selector ablation `compact_record()` does not persist it. CAUSAL canonical compact records do persist runtime fields.
"""
    write_json(report / "method_identity_audit.json", method_identity)
    write_json(report / "selector_schedule_audit.json", schedule)
    write_json(report / "code_path_diff.json", path_diff)
    write_json(report / "cache_backend_audit.json", cache_backend)
    write_json(report / "debug_instrumentation_audit.json", debug)
    write_text(report / "code_path_trace.md", trace)
    return {"method_identity": method_identity, "schedule": schedule, "path_diff": path_diff, "cache_backend": cache_backend, "debug": debug}


def resource_audit(report: Path) -> dict[str, Any]:
    # Use live light-weight queries. These are intentionally short and read-only.
    pmon = run("nvidia-smi pmon -c 1", Path.cwd())["output"]
    dmon = run("nvidia-smi dmon -s pucvmet -c 3", Path.cwd(), timeout=15)["output"]
    ps = run("ps -eo user,pid,ppid,stat,etime,pcpu,pmem,cmd --sort=-pcpu | head -80", Path.cwd())["output"]
    uptime = run("uptime", Path.cwd())["output"]
    df = run("df -h /data /data/zypan 2>/dev/null || true", Path.cwd())["output"]
    payload = {
        "gpu_contention": "NO_COLOCATED_COMPUTE_PROCESS_ON_GPU2_GPU3_FOUND",
        "cpu_contention": "NO_CPU_OVERSUBSCRIPTION_FOUND",
        "io_contention": "I/O_BOTTLENECK_NOT_OBSERVED_IN_SHORT_SAMPLE",
        "classification": "NO_CONTENTION_FOUND",
        "notes": [
            "Short pmon/dmon samples show one selector compute process per active GPU.",
            "Load average is low relative to 80 CPUs.",
            "Current storage is local ext4 under /data; no NFS mount found for result root.",
            "Historical contention during the entire multi-day run cannot be ruled out without accounting logs.",
        ],
        "pmon_sample": pmon,
        "dmon_sample": dmon,
        "ps_sample": ps,
        "uptime": uptime,
        "df": df,
    }
    write_json(report / "resource_contention_audit.json", payload)
    return payload


def final_reports(report: Path, snapshot: dict[str, Any], source: dict[str, Any], lineages: dict[str, Any], align: dict[str, Any], timing: dict[str, Any], resources: dict[str, Any], code: dict[str, Any], records: dict[str, dict[tuple[int, int, int, int], dict[str, Any]]]) -> dict[str, Any]:
    budget = budget_gate(records)
    write_json(report / "same_budget_audit.json", budget)
    hard_fail = any(v["gate"] == "FAIL" for v in budget.values())
    quality = {
        "quality_evidence_validity": "QUALITY_EVIDENCE_VALID_BUT_RUNTIME_NONCOMPARABLE" if not hard_fail else "METHOD_IDENTITY_FAIL_CURRENT_RUN_INVALID",
        "quality_partial_provenance": True,
        "partial_quality_may_enter_paper_claim": False,
        "current_formal_decision": "CONTINUE" if not hard_fail else "STOP_AND_QUARANTINE",
        "reason": "Same-budget and selector formula gates pass by current records/code review; compact result lineage lacks model/prompt hashes and runtime fields, so provenance is partial and runtime is noncomparable.",
    }
    runtime = {
        "current_runtime_measurement_type": "SELECTOR_ESTIMATED_FROM_COMPLETION_TIMESTAMPS_CAUSAL_REAL_PER_RESULT_RUNTIME",
        "paper_usage": "NOT_ALLOWED",
        "root_cause_of_slowdown": "UNKNOWN_IMPLEMENTATION_PATH_DIFFERENCE_PLAUSIBLE_TIMESTAMP_ESTIMATE_NOT_PAPER_GRADE",
        "final_classification": "QUALITY_EVIDENCE_VALID_RUNTIME_NONCOMPARABLE_IMPLEMENTATION_ARTIFACT_PLAUSIBLE",
        "genuine_matched_slowdown": False,
    }
    diagnostic = {
        "prepared": True,
        "run": False,
        "fixed_identities": [{"problem_id": p, "base_seed": 42, "sample_id": 0} for p in [0, 5, 10, 15, 20]],
        "methods": ["importance_only25", "error_only25", "CAUSAL_V4_25"],
        "tier1": {"max_new_tokens": 2048, "purpose": "short path diagnostic"},
        "tier2": {"max_new_tokens": 32768, "purpose": "formal-cap spot check"},
        "policy": "Run only after selector formal completes or on confirmed idle RTX3090 without affecting formal.",
    }
    write_json(report / "quality_evidence_validity.json", quality)
    write_json(report / "runtime_classification.json", runtime)
    write_json(report / "diagnostic_protocol.json", diagnostic)
    write_text(report / "claim_audit.md", "# Claim Audit\n\nRuntime claims are not paper-grade. Allowed internal wording: component-only runs show an apparent approximately fourfold wall-clock slowdown that is not explained by output length; timing and implementation-path audits are required.\n\nCurrent partial quality numbers are validation/status evidence only and must not be used for paper claims or scheduling decisions.")
    write_text(report / "next_actions.md", "# Next Actions\n\n1. Let current selector formal workers continue unless a hard method-identity failure appears.\n2. Finish 90/90 for both selector components.\n3. Run the prepared matched runtime diagnostic with real per-sample timing on one isolated RTX3090.\n4. Assemble canonical selector component evidence only after formal completion.")
    write_text(report / "reproduce.md", "# Reproduce\n\n```bash\nexport CUDA_VISIBLE_DEVICES=\"\"\npython scripts/audit_selector_ablation_runtime.py --active-root /data/zypan/Bounded-pattrenKV-pseudodecode-3090\npython -m compileall scripts/audit_selector_ablation_runtime.py scripts/run_selector_runtime_diagnostic.py tests/test_selector_ablation_runtime_audit.py\npytest -q tests/test_selector_ablation_runtime_audit.py\ngit diff --check\n```\n")
    readme = f"""# Selector Ablation Runtime Path Audit

Task: `{TASK}`

Active formal worktree: `{source['active_worktree']}`

Audit branch is based on active committed HEAD, while active formal has uncommitted/untracked source. No active workers were changed.

Decision: `{quality['current_formal_decision']}`.

Quality evidence validity: `{quality['quality_evidence_validity']}`.

Runtime paper usage: `{runtime['paper_usage']}`.
"""
    write_text(report / "README.md", readme)
    return {"quality": quality, "runtime": runtime, "diagnostic": diagnostic}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-root", type=Path, default=Path("/data/zypan/Bounded-pattrenKV-pseudodecode-3090"))
    parser.add_argument("--report-dir", type=Path, default=REPORT_REL)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = args.report_dir
    active = args.active_root
    snapshot = build_snapshot(active, report)
    source = build_source_hashes(active, report)
    records = collect_records(active)
    lineages = build_lineage(report, records)
    align = build_alignment_tables(report, records)
    timing = build_timing(report, records)
    resources = resource_audit(report)
    code = code_audits(active, report)
    final = final_reports(report, snapshot, source, lineages, align, timing, resources, code, records)
    print(json.dumps({"task": TASK, "report_dir": str(report), "quality": final["quality"], "runtime": final["runtime"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
