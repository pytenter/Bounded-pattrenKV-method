#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "system_asymmetric_kv_serving_v1"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "quant"))

import patternkv_gemv


START_HEAD = "f27e4e04e6cfd8c9cd7be71ae726380b6fb5d1c7"
BRANCH = "sys/causal-v4-25-kernel-v1"
ARCHITECTURE = "ASYMMETRIC_KV_RUNTIME"
MODEL_PATH = "/data/zypan/blockgtq-repro/models/DeepSeek-R1-Distill-Llama-8B"


def run_text(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return proc.stdout.strip()


def git_text(*args: str) -> str:
    return run_text(["git", *args])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fieldnames or []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def env_snapshot() -> dict[str, Any]:
    ext = Path(patternkv_gemv.__file__).resolve()
    try:
        smi = run_text(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        ).splitlines()
    except Exception as exc:
        smi = [f"nvidia-smi failed: {exc}"]
    return {
        "repo_root": git_text("rev-parse", "--show-toplevel"),
        "current_branch": git_text("branch", "--show-current"),
        "start_head_expected": START_HEAD,
        "head_at_start": git_text("rev-parse", "HEAD"),
        "worktree_clean_at_start": git_text("status", "--short") == "",
        "bounded_remote": next((line for line in git_text("remote", "-v").splitlines() if line.startswith("bounded") and "(push)" in line), ""),
        "origin_remote": next((line for line in git_text("remote", "-v").splitlines() if line.startswith("origin") and "(push)" in line), ""),
        "git_log_10": git_text("log", "-10", "--oneline").splitlines(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": smi,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "python_executable": sys.executable,
        "model_path": MODEL_PATH,
        "dtype": "float16",
        "extension_path": str(ext),
        "extension_sha256": sha256(ext),
        "backend_env": {
            "PATTERNKV_MIXED_V_BACKEND": "fused",
            "PATTERNKV_GQA_V_BACKEND": "baseline",
            "PATTERNKV_PAGE_V_READER": "contiguous",
            "PATTERNKV_CACHE_GROWTH_BACKEND_MAIN": "baseline,chunked_capacity",
        },
    }


def source_hits() -> list[dict[str, Any]]:
    patterns = [
        ("models/segmented_cache.py", "mixed Value precision currently requires batch size 1"),
        ("quant/matmul.py", "currently supports B=1"),
        ("models/segmented_cache.py", "mask = precision_mask[0].bool()"),
        ("quant/matmul.py", "mask = precision_mask[0].bool()"),
        ("models/segmented_cache.py", "low = v_adjusted[:, :, ~mask, :]"),
        ("quant/matmul.py", "attn2 = attn_q[..., low_mask].contiguous()"),
    ]
    hits: list[dict[str, Any]] = []
    for rel, needle in patterns:
        path = ROOT / rel
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if needle in line:
                hits.append({"file": rel, "line": lineno, "evidence": line.strip()})
    return hits


def run_batch2_probes() -> dict[str, Any]:
    probes: dict[str, Any] = {"cuda_available": bool(torch.cuda.is_available()), "cases": []}
    if not torch.cuda.is_available():
        probes["cases"].append({"name": "cuda_probe", "status": "SKIPPED", "reason": "CUDA unavailable"})
        return probes
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused"
    os.environ["PATTERNKV_CACHE_GROWTH_BACKEND"] = "chunked_capacity"
    os.environ["PATTERNKV_GQA_V_BACKEND"] = "baseline"
    os.environ["PATTERNKV_PAGE_V_READER"] = "contiguous"

    try:
        from models.segmented_cache import build_cache_from_prefill

        key = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.float16)
        value = torch.randn(2, 8, 512, 128, device="cuda", dtype=torch.float16)
        centroids = torch.randn(8, 16, 128, device="cuda", dtype=torch.float16)
        build_cache_from_prefill(
            key,
            value,
            sink_length=16,
            recent_length=128,
            group_size=128,
            k_bits=2,
            v_bits=2,
            pattern=True,
            k_centroids=centroids,
            v_centroids=centroids,
            cache_mode="segmented_rolling",
            chunk_length=128,
            value_objective="base",
            v_precision_selector="causal_v4",
            v4_budget_fraction=0.25,
            selector_layer_idx=0,
        )
        probes["cases"].append({"name": "build_cache_from_prefill_batch2_mixed", "status": "UNEXPECTED_PASS"})
    except Exception as exc:
        probes["cases"].append(
            {
                "name": "build_cache_from_prefill_batch2_mixed",
                "status": "EXPECTED_BLOCKED",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )

    try:
        from quant.matmul import cuda_attn_v_mixed_fused_with_base

        total = 128
        v2 = 96
        v4 = 32
        attn = torch.softmax(torch.randn(2, 32, 1, total, device="cuda", dtype=torch.float16), dim=-1).contiguous()
        precision = torch.zeros(2, total, device="cuda", dtype=torch.uint8)
        precision[:, ::4] = 1
        cuda_attn_v_mixed_fused_with_base(
            128,
            attn,
            torch.randint(-(2**30), 2**30 - 1, (2, 8, v2, 8), device="cuda", dtype=torch.int32),
            torch.ones(2, 8, v2, 1, device="cuda", dtype=torch.float16),
            torch.zeros(2, 8, v2, 1, device="cuda", dtype=torch.float16),
            torch.randint(-(2**30), 2**30 - 1, (2, 8, v4, 16), device="cuda", dtype=torch.int32),
            torch.ones(2, 8, v4, 1, device="cuda", dtype=torch.float16),
            torch.zeros(2, 8, v4, 1, device="cuda", dtype=torch.float16),
            precision,
            torch.randn(8, 16, 128, device="cuda", dtype=torch.float16),
            torch.ones(2, 8, total, device="cuda", dtype=torch.uint8),
            torch.zeros(2, 8, total, device="cuda", dtype=torch.int32),
            32,
            8,
        )
        probes["cases"].append({"name": "mixed_v_fused_kernel_entry_batch2", "status": "UNEXPECTED_PASS"})
    except Exception as exc:
        probes["cases"].append(
            {
                "name": "mixed_v_fused_kernel_entry_batch2",
                "status": "EXPECTED_BLOCKED",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
        )
    return probes


def build_gate(validation: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "algorithm_changed": False,
        "architecture": ARCHITECTURE,
        "k_layout": "tight",
        "v_layout": "chunked_capacity",
        "model_copies_per_run": 1,
        "true_shared_model_concurrency": False,
        "serial_pseudo_concurrency_used": False,
        "strided_k_used": False,
        "page_native_used": False,
        "experimental_gqa_used": False,
        "cuda_vmm_used": False,
        "correctness_passed": None,
        "cache_isolation_passed": None,
        "selector_isolation_passed": None,
        "v_historical_materialized_bytes": None,
        "baseline_max_concurrency_8k": None,
        "chunked_max_concurrency_8k": None,
        "baseline_max_concurrency_16k": None,
        "chunked_max_concurrency_16k": None,
        "baseline_max_concurrency_32k": None,
        "chunked_max_concurrency_32k": None,
        "throughput_gain_16k_high_common": None,
        "throughput_gain_32k_high_common": None,
        "tpot_gain_16k_high_common": None,
        "tpot_gain_32k_high_common": None,
        "chunked_capacity_utilization_16k": None,
        "chunked_capacity_utilization_32k": None,
        "vmm_priority": "MEDIUM",
        "serving_throughput_gain": "BLOCKED",
        "classification": "CONCURRENCY_RUNTIME_BLOCKED",
        "recommended_next_phase": "BATCH_SAFE_RUNTIME_FEASIBILITY",
        "blocker": "Mixed V2/V4 cache packing and fused mixed-V attention are B=1-only under the frozen causal_v4 25% split ABI.",
        "validation": validation or {},
    }


def write_blocked_csvs() -> None:
    write_csv(
        OUT_DIR / "capacity_sweep.csv",
        [
            {
                "backend": "baseline",
                "context": 8192,
                "concurrency": 2,
                "decode_probe": 32,
                "status": "BLOCKED_NOT_RUN",
                "prefill_success": False,
                "decode_success": False,
                "oom_phase": "",
                "steady_allocated_bytes": "",
                "steady_reserved_bytes": "",
                "peak_allocated_bytes": "",
                "peak_reserved_bytes": "",
                "logical_kv_tokens": "",
                "capacity_utilization": "",
            }
        ],
        [
            "backend",
            "context",
            "concurrency",
            "decode_probe",
            "status",
            "prefill_success",
            "decode_success",
            "oom_phase",
            "steady_allocated_bytes",
            "steady_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "logical_kv_tokens",
            "capacity_utilization",
        ],
    )
    write_csv(
        OUT_DIR / "oom_events.csv",
        [],
        ["backend", "context", "concurrency", "phase", "allocated_bytes", "reserved_bytes", "free_memory_bytes", "status"],
    )
    write_csv(
        OUT_DIR / "max_concurrency_summary.csv",
        [{"context": c, "baseline_max": "", "chunked_max": "", "delta": "", "limiting_phase": "runtime_batch_blocked"} for c in (8192, 16384, 32768)],
        ["context", "baseline_max", "chunked_max", "delta", "limiting_phase"],
    )
    common_fields = [
        "context",
        "decode_tokens",
        "concurrency",
        "backend",
        "status",
        "decode_wall_ms",
        "tpot_ms",
        "aggregate_decode_tokens_per_second",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ]
    write_csv(OUT_DIR / "profile_off_runs.csv", [], common_fields + ["round"])
    write_csv(OUT_DIR / "profile_off_summary.csv", [], common_fields + ["rounds", "cv"])
    write_csv(
        OUT_DIR / "throughput_summary.csv",
        [],
        [
            "context",
            "decode_tokens",
            "concurrency",
            "baseline_tok_s",
            "chunked_tok_s",
            "throughput_speedup",
            "baseline_tpot_ms",
            "chunked_tpot_ms",
            "tpot_improvement",
            "baseline_peak_allocated",
            "chunked_peak_allocated",
        ],
    )
    write_csv(OUT_DIR / "latency_summary.csv", [], common_fields + ["p50_request_latency_ms", "p90_request_latency_ms", "p99_request_latency_ms"])
    write_csv(OUT_DIR / "memory_summary.csv", [], common_fields + ["steady_allocated_bytes", "steady_reserved_bytes", "peak_decode_allocated_bytes"])
    write_csv(
        OUT_DIR / "kv_memory_summary.csv",
        [],
        ["context", "decode_tokens", "concurrency", "backend", "logical_resident_kv_tokens", "logical_v_bytes", "reserved_v_bytes", "unused_v_bytes"],
    )
    write_csv(
        OUT_DIR / "mutation_summary.csv",
        [],
        ["context", "decode_tokens", "concurrency", "backend", "mutation_us_per_token", "growth_events", "growth_copied_bytes", "historical_materialized_bytes"],
    )
    write_csv(
        OUT_DIR / "copy_breakdown.csv",
        [],
        ["context", "decode_tokens", "concurrency", "backend", "k_copied_bytes_per_token", "v_copied_bytes_per_token", "total_copied_bytes_per_token"],
    )
    write_csv(
        OUT_DIR / "capacity_utilization.csv",
        [],
        ["context", "decode_tokens", "concurrency", "backend", "valid_v_bytes", "reserved_v_bytes", "unused_v_bytes", "capacity_utilization"],
    )
    write_csv(
        OUT_DIR / "long_decode_summary.csv",
        [],
        ["context", "decode_tokens", "concurrency", "baseline_tok_s", "chunked_tok_s", "throughput_speedup", "status"],
    )
    write_csv(
        OUT_DIR / "profile_on_components.csv",
        [],
        ["context", "decode_tokens", "concurrency", "backend", "component", "total_us", "percent_decode_time", "profile_on_shares_approximate"],
    )


def write_markdown_reports(probes: dict[str, Any], hits: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    audit = [
        "# Concurrency Capability Audit",
        "",
        "## Verdict",
        "",
        "- `CONCURRENCY_RUNTIME_BLOCKED`.",
        "- The current frozen mixed V2/V4 runtime is single-request only for true batched serving.",
        "- S6-B performance stages were not run because serial pseudo-concurrency is explicitly disallowed.",
        "",
        "## Questions",
        "",
        "| Question | Answer | Evidence |",
        "|---|---|---|",
        "| cache object supports B>1? | Partial | Base tensor fields carry batch dim, but mixed-V packing rejects B>1. |",
        "| packed K/V carries real B dim? | Partial | K/V pack tensors include B dim; mixed split assumes request 0 mask. |",
        "| assignments/masks support B>1? | Partial | Shapes include B, but compact V2/V4 split is generated from `precision_mask[0]`. |",
        "| capacity buffer allows batch>1? | Likely yes for raw tensor storage | It stores shape_except_token including B, but cannot solve variable per-request mixed split. |",
        "| mixed-V kernel batch-aware? | No | `cuda_attn_v_mixed_fused_with_base` raises for `B != 1`. |",
        "| QK kernel batch-aware? | Likely yes | QK path consumes tensors with B dim and has no B=1 guard in the production tight reader. |",
        "| selector request-isolated? | Scoring is batch-shaped; downstream storage is not | Selector can return `[B,T]`, but packer consumes `precision_mask[0]`. |",
        "| append_decode handles batch? | Partial | Generic append can carry B tensors, but mixed packing raises before/at flush. |",
        "| hard-coded B=1 exists? | Yes | See source evidence below. |",
        "| model forward can advance multiple requests? | No for frozen causal_v4 mixed-V path | Batch prefill/decode hits B=1 mixed-V limitations. |",
        "",
        "## Source Evidence",
        "",
        "| File | Line | Evidence |",
        "|---|---:|---|",
    ]
    for hit in hits:
        audit.append(f"| `{hit['file']}` | {hit['line']} | `{hit['evidence']}` |")
    audit += [
        "",
        "## Runtime Probes",
        "",
        "| Probe | Status | Exception | Message |",
        "|---|---|---|---|",
    ]
    for case in probes.get("cases", []):
        audit.append(f"| {case['name']} | {case['status']} | {case.get('exception_type', '')} | {case.get('message', '')} |")
    (OUT_DIR / "concurrency_capability_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")

    (OUT_DIR / "concurrency_semantics.md").write_text(
        "\n".join(
            [
                "# Concurrency Semantics",
                "",
                "- Required S6-B serving semantics: one model copy, N independent resident request states, and one batched decode step advances all active requests.",
                "- Serial pseudo-concurrency was not used.",
                "- N-process model-copy concurrency was not used.",
                "- The current runtime cannot satisfy true shared-model concurrency for the frozen mixed V2/V4 causal_v4 path.",
                "- Therefore `MODEL_COPIES=1` remains a design requirement, but no throughput claim is made.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        OUT_DIR / "correctness_summary.json",
        {"passed": None, "status": "BLOCKED_NOT_RUN", "reason": gate["blocker"], "nan": None, "inf": None},
    )
    write_json(
        OUT_DIR / "cache_isolation.json",
        {"passed": None, "status": "BLOCKED_NOT_RUN", "reason": "True B>1 mixed-V cache construction is blocked before cache isolation can be benchmarked."},
    )
    write_json(
        OUT_DIR / "selector_request_isolation.json",
        {
            "passed": None,
            "status": "BLOCKED_NOT_RUN",
            "reason": "Selector scoring is batch-shaped, but mixed-V compact packing uses request-0 precision mask, so isolation cannot be claimed.",
        },
    )
    (OUT_DIR / "serving_scaling_analysis.md").write_text(
        "# Serving Scaling Analysis\n\nNo throughput-vs-concurrency scaling curve was produced because true shared-model B>1 serving is blocked in the frozen mixed-V runtime.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "memory_scaling_analysis.md").write_text(
        "# Memory Scaling Analysis\n\nNo memory-vs-concurrency curve was produced. Capacity/OOM sweep was not valid to run because batch-safe cache/runtime semantics are blocked before serving decode.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "vmm_decision.md").write_text(
        "# VMM Decision\n\nVMM priority remains `MEDIUM` from S6-A evidence, but S6-B cannot upgrade or downgrade it without a valid true-concurrency run. Do VMM next: `NO`; first complete `BATCH_SAFE_RUNTIME_FEASIBILITY`.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "final_report.md").write_text(
        "\n".join(
            [
                "# Final Report",
                "",
                "## Result",
                "",
                "- Final classification: `CONCURRENCY_RUNTIME_BLOCKED`.",
                "- Recommended next phase: `BATCH_SAFE_RUNTIME_FEASIBILITY`.",
                "- Serving throughput gain: `BLOCKED`.",
                "",
                "## Why",
                "",
                "The S6-B definition requires true shared-model batched serving. The frozen causal_v4 mixed-V runtime stores selected V4 tokens in compact V2/V4 streams selected by `precision_mask[0]`, and the fused mixed-V attention wrapper raises on `B != 1`. Supporting B>1 while preserving independent per-request V4 identities requires a batch-safe mixed-V cache ABI and runtime plumbing, not a benchmark-only change.",
                "",
                "## What Was Not Run",
                "",
                "- No capacity sweep.",
                "- No throughput sweep.",
                "- No long-decode serving stress.",
                "- No vLLM/SGLang/AIME24/AIME25/GPQA/CUDA VMM.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run_validation() -> dict[str, Any]:
    commands = {
        "compileall": [sys.executable, "-m", "compileall", "bench", "models", "quant", "scripts", "tests"],
        "pytest": [sys.executable, "-m", "pytest", "-q"],
        "diff_check": ["git", "diff", "--check"],
    }
    validation: dict[str, Any] = {}
    clean_env = os.environ.copy()
    clean_env["PATTERNKV_CACHE_GROWTH_BACKEND"] = "baseline"
    clean_env["PATTERNKV_PAGE_V_READER"] = "contiguous"
    clean_env["PATTERNKV_GQA_V_BACKEND"] = "baseline"
    clean_env.pop("PATTERNKV_SYSTEM_PROFILE", None)
    clean_env.pop("PATTERNKV_PROFILE", None)
    for name, cmd in commands.items():
        proc = subprocess.run(cmd, cwd=ROOT, env=clean_env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        validation[name] = {
            "returncode": proc.returncode,
            "passed": proc.returncode == 0,
            "tail": "\n".join(proc.stdout.splitlines()[-20:]),
        }
    return validation


def print_terminal_summary(gate: dict[str, Any]) -> None:
    print("=" * 60)
    print("PHASE S6-B -- ASYMMETRIC KV SERVING / CONCURRENCY")
    print("=" * 60)
    print("Repository:\npytenter/Bounded-pattrenKV-method")
    print(f"Branch:\n{BRANCH}")
    print(f"Start HEAD:\n{START_HEAD}")
    print(f"End HEAD:\n{git_text('rev-parse', 'HEAD')}")
    print("\nARCHITECTURE")
    print("Algorithm changed:\nNO")
    print(f"Architecture:\n{ARCHITECTURE}")
    print("K:\nTIGHT")
    print("V:\nCHUNKED_CAPACITY")
    print("Strided K:\nNO\nPage-native:\nNO\nExperimental GQA:\nNO\nCUDA VMM:\nNO")
    print("\nCONCURRENCY SEMANTICS")
    print("One model copy:\nYES")
    print("Shared model concurrency:\nNO")
    print("Serial pseudo-concurrency:\nNO")
    print("Batch-safe runtime:\nNO")
    print("\nCORRECTNESS")
    print("Correctness:\nBLOCKED_NOT_RUN")
    print("Cache isolation:\nBLOCKED_NOT_RUN")
    print("Selector isolation:\nBLOCKED_NOT_RUN")
    print("V historical materialization:\nNOT_MEASURED")
    print("\nMAX CONCURRENCY")
    for ctx in ("8K", "16K", "32K"):
        print(f"{ctx}:\nBaseline max: BLOCKED\nChunked max: BLOCKED")
    print("\nVMM DECISION")
    print("VMM priority:\nMEDIUM")
    print("Reason:\nS6-A indicated medium slack; S6-B true-concurrency data is blocked.")
    print("Do VMM next:\nNO")
    print("\nSERVING RESULT")
    print("Serving throughput gain:\nBLOCKED")
    print("\nFINAL CLASSIFICATION")
    print("Decision:\nCONCURRENCY_RUNTIME_BLOCKED")
    print(f"Reason:\n{gate['blocker']}")
    print("\nNEXT TASK")
    print("BATCH_SAFE_RUNTIME_FEASIBILITY")
    print("\nVALIDATION")
    for name in ("compileall", "pytest", "diff_check"):
        val = gate.get("validation", {}).get(name, {})
        print(f"{name}: {val.get('passed')}")
    print("\nGIT")
    print("Commit:\nperf: evaluate asymmetric PatternKV concurrency")
    print("Pushed to bounded:\nPENDING")
    print("Pushed to origin:\nNO")
    print(f"Worktree clean:\n{git_text('status', '--short') == ''}")
    print("=" * 60)
    print("ALGORITHM_CHANGED=NO")
    print(f"ARCHITECTURE={ARCHITECTURE}")
    print("K_LAYOUT=TIGHT")
    print("V_LAYOUT=CHUNKED_CAPACITY")
    print("TRUE_CONCURRENCY=NO")
    print("SERVING_GAIN=BLOCKED")
    print("VMM_PRIORITY=MEDIUM")
    print("NEXT_TASK=BATCH_SAFE_RUNTIME_FEASIBILITY")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "environment.json", env_snapshot())
    if git_text("branch", "--show-current") != BRANCH:
        raise RuntimeError(f"wrong branch: expected {BRANCH}")
    if git_text("rev-parse", "HEAD") != START_HEAD:
        raise RuntimeError(f"wrong start HEAD: expected {START_HEAD}")
    hits = source_hits()
    probes = run_batch2_probes()
    write_json(OUT_DIR / "batch2_probe.json", probes)
    write_blocked_csvs()
    validation = {} if args.skip_validation else run_validation()
    gate = build_gate(validation)
    write_json(OUT_DIR / "final_gate.json", gate)
    write_markdown_reports(probes, hits, gate)
    print_terminal_summary(gate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
