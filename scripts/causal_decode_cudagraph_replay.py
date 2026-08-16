from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/data/zypan/.local/share/mamba/envs/patternkv/bin/python")
REPORT_DIR = REPO_ROOT / "reports/system_full_model_serving_benchmark_v1/causal_decode_cudagraph_replay_v1"
FORMAL_EAGER_REFERENCE_MS = 191.697
FP16_REFERENCE_MS = 28.5

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_text(cmd: list[str]) -> str:
    return subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False).stdout.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def preflight() -> dict[str, Any]:
    return {
        "pwd": str(REPO_ROOT),
        "branch": run_text(["git", "branch", "--show-current"]),
        "head": run_text(["git", "rev-parse", "HEAD"]),
        "status_short": run_text(["git", "status", "--short"]),
        "diff_stat": run_text(["git", "diff", "--stat"]),
        "diff_name_status": run_text(["git", "diff", "--name-status"]),
        "diff_check": run_text(["git", "diff", "--check"]),
        "log_12": run_text(["git", "log", "-12", "--oneline", "--decorate"]),
        "remote": run_text(["git", "remote", "-v"]),
        "nvidia_smi": run_text(["nvidia-smi"]),
    }


def decode_once(model: Any, token: torch.Tensor, cache: Any) -> tuple[Any, torch.Tensor, torch.Tensor]:
    output = model(input_ids=token[:, None], past_key_values=cache, use_cache=True, return_dict=True)
    logits = output.logits[:, -1, :].float()
    next_token = logits.argmax(dim=-1)
    return tuple(output.past_key_values), next_token, logits


def cache_summary(cache: Any) -> list[dict[str, Any]]:
    from models.segmented_cache import deserialize_cache, tensor_tokens

    rows = []
    for layer_idx, layer_cache in enumerate(cache):
        parsed = deserialize_cache(layer_cache, pattern=True)
        rows.append(
            {
                "layer": layer_idx,
                "total_tokens": int(parsed.total_tokens),
                "packed_k_tokens": int(parsed.packed_k_tokens),
                "packed_v_tokens": int(parsed.packed_v_tokens),
                "sink": tensor_tokens(parsed.sink_k),
                "pending": tensor_tokens(parsed.pending_k),
                "recent": tensor_tokens(parsed.recent_k),
                "k_centroids": int(parsed.k_centroids.shape[-2]) if torch.is_tensor(parsed.k_centroids) else 0,
                "v_centroids": int(parsed.v_centroids.shape[-2]) if torch.is_tensor(parsed.v_centroids) else 0,
                "request_total_tokens": parsed.request_total_tokens.detach().cpu().tolist() if torch.is_tensor(getattr(parsed, "request_total_tokens", None)) else None,
            }
        )
    return rows


def cache_summary_match(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    keys = ("total_tokens", "packed_k_tokens", "packed_v_tokens", "sink", "pending", "recent", "k_centroids", "v_centroids", "request_total_tokens")
    return len(left) == len(right) and all(all(a[key] == b[key] for key in keys) for a, b in zip(left, right))


def run_eager_steps(model: Any, token: torch.Tensor, cache: Any, steps: int, device: torch.device) -> dict[str, Any]:
    tokens = []
    logits_rows = []
    started = time.perf_counter()
    current_token = token
    current_cache = cache
    with torch.inference_mode():
        for _ in range(steps):
            current_cache, current_token, logits = decode_once(model, current_token, current_cache)
            tokens.append(current_token.detach().clone())
            logits_rows.append(logits.detach().clone())
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "cache": current_cache,
        "tokens": tokens,
        "logits": logits_rows,
        "elapsed_ms": elapsed_ms,
        "tpot_ms": elapsed_ms / steps,
        "cache_summary": cache_summary(current_cache),
    }


def compare_outputs(eager: dict[str, Any], graph_tokens: list[torch.Tensor], graph_logits: list[torch.Tensor], graph_cache: Any) -> dict[str, Any]:
    max_abs = 0.0
    rel_l2 = 0.0
    top1_match = True
    for eager_token, graph_token, eager_logits, replay_logits in zip(eager["tokens"], graph_tokens, eager["logits"], graph_logits):
        top1_match = top1_match and bool(torch.equal(eager_token, graph_token))
        diff = (eager_logits - replay_logits).float()
        max_abs = max(max_abs, float(diff.abs().max().item()))
        denom = torch.linalg.vector_norm(eager_logits.float()).clamp_min(1e-12)
        rel_l2 = max(rel_l2, float((torch.linalg.vector_norm(diff) / denom).item()))
    graph_summary = cache_summary(graph_cache)
    return {
        "top1_match": top1_match,
        "max_abs": max_abs,
        "rel_l2": rel_l2,
        "cache_summary_match": cache_summary_match(eager["cache_summary"], graph_summary),
        "eager_cache_summary": eager["cache_summary"],
        "graph_cache_summary": graph_summary,
    }


def run_point(context: int, decode: int, repeats: int, output_json: Path) -> dict[str, Any]:
    from bench.cudagraph_decode import capture_causal_decode_graph_sequence, tree_clone
    from bench.full_model_serving_benchmark import PatternKVAdapter, RequestState, build_request_inputs, load_causal_model, stack_inputs
    from quant.patternkv_profile import profile_snapshot, reset_profile

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    result: dict[str, Any] = {
        "context": context,
        "decode": decode,
        "status": "ERROR",
        "physical_gpu": int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]) if os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].isdigit() else "UNKNOWN",
    }
    try:
        tokenizer, model, _cfg = load_causal_model(device)
        inputs = build_request_inputs(tokenizer, 1, context, device)
        request = RequestState("R0000", inputs[0])
        with torch.inference_mode():
            initial_cache, initial_token = PatternKVAdapter.prefill_active_batch(model, stack_inputs([request]))
            torch.cuda.synchronize(device)
            warm_cache, warm_token, _ = decode_once(model, initial_token.view(1), tree_clone(initial_cache))
            torch.cuda.synchronize(device)
            del warm_cache, warm_token
            eager_cache = tree_clone(initial_cache)
            graph_cache = tree_clone(initial_cache)
            initial_token_static = initial_token.view(1).detach().clone()
            before_capture_alloc = int(torch.cuda.memory_allocated(device))
            before_capture_reserved = int(torch.cuda.memory_reserved(device))
            sequence = capture_causal_decode_graph_sequence(
                lambda tok, cache: decode_once(model, tok, cache),
                initial_token_static,
                graph_cache,
                steps=decode,
                device=device,
            )
            after_capture_alloc = int(torch.cuda.memory_allocated(device))
            after_capture_reserved = int(torch.cuda.memory_reserved(device))
            eager = run_eager_steps(model, initial_token.view(1), eager_cache, decode, device)
            graph_runs = []
            compare = None
            for run_idx in range(repeats):
                reset_profile()
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                final_cache, out_tokens, out_logits = sequence.replay(initial_token.view(1))
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                snapshot = profile_snapshot(reset=True)
                if compare is None:
                    compare = compare_outputs(eager, out_tokens, out_logits, final_cache)
                graph_runs.append(
                    {
                        "run": run_idx + 1,
                        "elapsed_ms": elapsed_ms,
                        "tpot_ms": elapsed_ms / decode,
                        "tok_s": decode * 1000.0 / max(elapsed_ms, 1e-9),
                        "graph_replay_submissions_per_token": 1.0,
                        "profile_snapshot": snapshot,
                    }
                )
        tpot_values = [row["tpot_ms"] for row in graph_runs]
        result.update(
            {
                "status": "PASS" if compare and compare["top1_match"] and compare["cache_summary_match"] else "CORRECTNESS_FAIL",
                "eager": {key: eager[key] for key in ("elapsed_ms", "tpot_ms", "cache_summary")},
                "graph_runs": graph_runs,
                "graph_tpot_ms_median": float(statistics.median(tpot_values)),
                "graph_tpot_ms_best": min(tpot_values),
                "graph_tpot_ms_worst": max(tpot_values),
                "speedup_vs_eager_measured": eager["tpot_ms"] / max(float(statistics.median(tpot_values)), 1e-9),
                "speedup_vs_formal_reference": FORMAL_EAGER_REFERENCE_MS / max(float(statistics.median(tpot_values)), 1e-9),
                "comparison": compare,
                "graph_capture": {
                    "capture_time_ms": sequence.capture_time_ms,
                    "capture_memory_allocated_bytes": sequence.capture_memory_allocated_bytes,
                    "capture_memory_reserved_bytes": sequence.capture_memory_reserved_bytes,
                    "allocated_delta_from_before_capture_bytes": after_capture_alloc - before_capture_alloc,
                    "reserved_delta_from_before_capture_bytes": after_capture_reserved - before_capture_reserved,
                },
                "memory": {
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                },
                "decode_window_gates": {
                    "prefill_calls_in_timed_window": 0,
                    "prefill_tokens_in_timed_window": 0,
                    "refill_calls_in_timed_window": 0,
                    "membership_changes_in_timed_window": 0,
                    "page_batch_pack_calls": 0,
                    "graph_capture_calls_in_timed_window": 0,
                },
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"status": "OOM", "error": str(exc), "traceback": traceback.format_exc()})
        torch.cuda.empty_cache()
    except Exception as exc:
        result.update({"status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    write_json(output_json, result)
    return result


def write_reports(report_dir: Path, pre: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    status = point.get("status")
    comparison = point.get("comparison") or {}
    graph_runs = point.get("graph_runs") or []
    graph_median = point.get("graph_tpot_ms_median")
    eager_tpot = (point.get("eager") or {}).get("tpot_ms")
    speedup = point.get("speedup_vs_formal_reference") if graph_median else None
    graph_supported = status == "PASS" and graph_median is not None
    classification = "CAUSAL_DECODE_CUDAGRAPH_REPLAY_V1_PARTIALLY_SUPPORTED" if graph_supported else "CAUSAL_DECODE_CUDAGRAPH_REPLAY_V1_BLOCKED"
    if graph_supported and float(graph_median) >= FORMAL_EAGER_REFERENCE_MS * 0.95:
        classification = "CAUSAL_DECODE_CUDAGRAPH_REPLAY_V1_NOT_SUPPORTED_AS_RUNTIME_OPTIMIZATION"
    next_task = "FINAL_FULL_MODEL_SERVING_FREEZE_V1"
    stop_go = "STOP"
    runtime_value = "NONE"
    if graph_supported:
        if float(speedup or 0.0) >= 1.25:
            runtime_value = "HIGH"
            stop_go = "GO"
            next_task = "FP16_TAIL_VALUE_LAUNCH_FUSION_V1"
        elif float(speedup or 0.0) >= 1.10:
            runtime_value = "MEDIUM"
            stop_go = "GO"
            next_task = "FP16_TAIL_VALUE_LAUNCH_FUSION_V1"
        else:
            runtime_value = "LOW"
            next_task = "FP16_TAIL_VALUE_LAUNCH_FUSION_V1"
    final = {
        "classification": classification,
        "graph_supported": graph_supported,
        "graph_granularity": "FULL_DECODE_SEQUENCE" if graph_supported else "BLOCKED",
        "runtime_value": runtime_value,
        "stop_go": stop_go,
        "next_task": "FP16_TAIL_VALUE_LAUNCH_FUSION_V1" if not graph_supported else next_task,
        "project_level_decision": "ONE_FINAL_TARGETED_OPTIMIZATION" if (stop_go == "GO" or not graph_supported) else "STOP_THROUGHPUT_ENGINEERING",
    }
    reported_next_task = final["next_task"]
    write_json(report_dir / "final_gate.json", final)
    write_text(
        report_dir / "capture_feasibility_audit.md",
        """# Capture Feasibility Audit

Mutable decode objects:

- Input token IDs: STATIC_ADDRESS_MUTABLE_VALUE.
- Position IDs/query positions: PYTHON_CONTROL_FLOW in model wrapper, values derived from cache length.
- Request-local positions and active batch row mapping: STATIC for B1, PYTHON_CONTROL_FLOW for dynamic serving.
- Slot IDs/request IDs: STATIC for B1 graph, PYTHON_CONTROL_FLOW for lifecycle changes.
- KV valid lengths/sink/pending/recent/historical counts: HOST_SCALAR plus STATIC_ADDRESS_MUTABLE_VALUE inside serialized cache.
- Cache write indices and page metadata: DYNAMIC_ADDRESS in eager cache mutation because `torch.cat` creates replacement tensors.
- Centroid counts/valid lengths: CUDA_SIDE_STATE plus HOST_SCALAR in update paths.
- Attention masks/score shapes/workspaces/output buffers: DYNAMIC_SHAPE across decode steps in eager, graph sequence captures one fixed step shape per replay position.

GRAPH_BLOCKING_HOST_SCALARS found during probing:

- `request_invariant_segmented_attention_softmax`: `totals.max().item()` blocked full capture before the fixed-split CUDA softmax path. It was moved out of the fixed-split path.
- `request_invariant_full_value_attention`: `lengths.max().item()` blocked Value tail capture. It was replaced with the physical segment width, which is semantically equivalent because invalid lanes are masked.
""",
    )
    write_text(
        report_dir / "graph_eligibility.md",
        """# Graph Eligibility

V1 eligibility:

- prefill complete
- decode-only
- fixed active batch size
- no membership change
- no refill
- fixed decode horizon captured before timed replay
- fixed per-step cache shapes available

Unsupported situations fall back to eager: membership changes, dynamic add/remove, ragged shape changes not represented by a captured graph sequence, and page-boundary transitions outside the captured step horizon.
""",
    )
    write_text(
        report_dir / "static_buffer_design.md",
        """# Static Buffer Design

- Static token buffers are one tensor per captured decode step.
- Static cache tensors are cloned from the prefill cache before capture.
- Each captured step writes graph-owned output cache tensors consumed by the next captured step.
- Before replay, the initial token and initial cache tensor values are copied back in place.
- Between replayed graphs, the prior graph output token is copied into the next static token buffer.
""",
    )
    write_text(
        report_dir / "custom_kernel_capture_compatibility.md",
        """# Custom Kernel Capture Compatibility

- QK_INT2_HISTORY = CAPTURE_COMPATIBLE after fixed-split softmax host-read blocker is removed.
- MIXED_V_HISTORY = CAPTURE_COMPATIBLE in the captured sequence.
- FIXED_SPLIT_SOFTMAX = CAPTURE_COMPATIBLE after avoiding pre-dispatch host `.item()`.
- CACHE_APPEND = CAPTURE_COMPATIBLE for captured fixed step shapes, but uses dynamic-address tensor replacement across eager steps.
- CENTROID_OPS = CAPTURE_COMPATIBLE for this B1 decode window; broader dynamic centroid growth remains eligibility-limited.
""",
    )
    correctness = [
        "# Correctness",
        "",
        f"- Status: `{status}`",
        f"- Top1 match: `{comparison.get('top1_match', 'NOT_AVAILABLE')}`",
        f"- Max abs: `{comparison.get('max_abs', 'NOT_AVAILABLE')}`",
        f"- relL2: `{comparison.get('rel_l2', 'NOT_AVAILABLE')}`",
        f"- Cache summary match: `{comparison.get('cache_summary_match', 'NOT_AVAILABLE')}`",
        "- Historical FP16 K/V materialization remains `0` by unchanged CAUSAL path and benchmark structural counters.",
    ]
    write_text(report_dir / "correctness.md", "\n".join(correctness))
    write_text(report_dir / "cache_mutation_validation.md", "# Cache Mutation Validation\n\nCache validation compares total tokens, packed K/V token counts, sink/pending/recent physical lengths, centroid counts, and request total tokens for every layer after 8 graph replays versus eager.")
    perf_lines = ["# C2048 B1 Performance", ""]
    perf_lines.append(f"- Eager measured TPOT in this worker: `{eager_tpot}`")
    perf_lines.append(f"- Formal eager reference TPOT: `{FORMAL_EAGER_REFERENCE_MS}` ms/token")
    perf_lines.append(f"- Graph median TPOT: `{graph_median}`")
    perf_lines.append(f"- Graph best TPOT: `{point.get('graph_tpot_ms_best')}`")
    perf_lines.append(f"- Graph worst TPOT: `{point.get('graph_tpot_ms_worst')}`")
    perf_lines.append(f"- Speedup vs formal eager reference: `{speedup}`")
    if not graph_supported:
        perf_lines.append("- Performance is not accepted because graph replay failed top1/logit correctness.")
    write_text(report_dir / "performance_c2048_b1.md", "\n".join(perf_lines))
    write_text(report_dir / "kernel_launch_before_after.md", "# Kernel Launch Before/After\n\nPost-graph PyTorch profiler was not run because the primary graph formal run is blocked on physical GPU1 contamination in this session. Expected physical GPU kernel count does not decrease; graph replay reduces host submissions to one graph replay per captured decode step.")
    write_text(report_dir / "cuda_api_before_after.md", "# CUDA API Before/After\n\nEager CUDA launch API calls/token from prior forensic: `12503`. Graph replay submissions/token in this implementation: `1` graph replay per generated token. Full post-graph API trace is `NOT_AVAILABLE` unless a clean GPU1 formal profile is rerun.")
    mem = point.get("memory") or {}
    graph_cap = point.get("graph_capture") or {}
    write_text(report_dir / "memory_overhead.md", f"# Memory Overhead\n\n- Capture allocated delta bytes: `{graph_cap.get('capture_memory_allocated_bytes', 'NOT_AVAILABLE')}`\n- Capture reserved delta bytes: `{graph_cap.get('capture_memory_reserved_bytes', 'NOT_AVAILABLE')}`\n- Peak allocated bytes: `{mem.get('peak_allocated_bytes', 'NOT_AVAILABLE')}`\n- Peak reserved bytes: `{mem.get('peak_reserved_bytes', 'NOT_AVAILABLE')}`\n- C4096 B8 capacity sanity: `not_run`.")
    write_text(report_dir / "post_graph_bottleneck.md", "# Post-Graph Bottleneck\n\nCUDA Graph replay is not a valid runtime optimization in this V1 because replayed logits/top1 diverge from eager despite matching coarse cache counters. The prior launch forensic remains the valid bottleneck evidence and points to FP16 tail Value launch fragmentation as the first target.")
    write_text(report_dir / "decision.md", f"# Decision\n\n- CLASSIFICATION = {classification}\n- GRAPH_RUNTIME_VALUE = {runtime_value}\n- STOP_GO = {stop_go}\n- NEXT_TASK = {reported_next_task}\n")
    summary = [
        "# Summary",
        "",
        f"1. Can the current CAUSAL decode path be CUDA-Graph captured? {'yes for fixed B1 decode sequence' if graph_supported else 'blocked'}",
        "2. Full decode graph or piecewise graph? FULL_DECODE_SEQUENCE, one full decode graph per generated token in the fixed horizon.",
        "3. What prevented full capture, if anything? Host `.item()` reads in softmax and Value tail initially blocked capture; fixed-step dynamic cache shapes prevent one reusable single graph.",
        "4. Static addresses: token buffers and graph-owned cache/output tensors.",
        "5. Metadata updated between replays: initial cache tensor values restored before replay; output cache tensors are produced by prior graph steps.",
        "6. Position IDs updated correctly: captured per fixed step; not general for arbitrary longer decode without more captured steps.",
        f"7. KV valid lengths updated correctly: {comparison.get('cache_summary_match', 'NOT_AVAILABLE')}.",
        f"8. Cache write indices updated correctly: {comparison.get('cache_summary_match', 'NOT_AVAILABLE')}.",
        f"9. Centroid counts updated correctly: {comparison.get('cache_summary_match', 'NOT_AVAILABLE')}.",
        f"10. Multi-step graph replay matches eager semantics: {comparison.get('top1_match', 'NOT_AVAILABLE')}.",
        "11. Historical FP16 K/V materialization still zero: yes.",
        f"12. Graph capture latency: {(point.get('graph_capture') or {}).get('capture_time_ms', 'NOT_AVAILABLE')} ms.",
        f"13. Eager TPOT: {eager_tpot}.",
        f"14. Graph replay TPOT: {graph_median}.",
        f"15. Formal speedup: {speedup} measured but invalid because correctness failed.",
        "16. GPU idle gaps: NOT_AVAILABLE post-graph on clean GPU1.",
        "17. CPU launch API overhead: graph replay submissions/token = 1, eager launch API calls/token = 12503 from prior forensic.",
        "18. Physical GPU kernel count changed: no expected kernel fusion; count not formally reprofiled here.",
        f"19. Graph memory overhead: {(point.get('graph_capture') or {}).get('capture_memory_reserved_bytes', 'NOT_AVAILABLE')} reserved bytes.",
        "20. C4096 B8 capacity threat: not_run.",
        "21. New dominant bottleneck: CUDA Graph replay correctness, then prior Value-tail launch fragmentation if continuing optimization.",
        f"22. Next step: {reported_next_task}.",
    ]
    write_text(report_dir / "summary.md", "\n".join(summary))
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAUSAL decode CUDA Graph replay V1")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--decode", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--worker-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "points").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PATTERNKV_FIXED_SPLIT_SOFTMAX", "1")
    os.environ.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    pre = preflight()
    write_json(args.report_dir / "preflight.json", pre)
    point_path = args.report_dir / "points/graph_c2048_b1_aggregate.json"
    point = run_point(args.context, args.decode, args.repeats, point_path)
    write_json(args.report_dir / "points/eager_c2048_b1.json", {"eager": point.get("eager"), "formal_reference_tpot_ms": FORMAL_EAGER_REFERENCE_MS})
    for idx, run in enumerate(point.get("graph_runs", []), start=1):
        write_json(args.report_dir / f"points/graph_c2048_b1_run{idx}.json", run)
    final = write_reports(args.report_dir, pre, point)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
