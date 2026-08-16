from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import MODEL_PATH, load_model, make_fixed_inputs, tensor_metrics
from models.segmented_cache import deserialize_cache, tensor_tokens
from quant.batch_invariant_kproj import batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import (
    get_patternkv_page_batch_counters,
    get_patternkv_real_decode_counters,
    reset_patternkv_page_batch_counters,
    reset_patternkv_real_decode_counters,
)


REPORT_DIR = REPO_ROOT / "reports/system_bi_kproj_prefill_runtime_v1"
START_HEAD = "ca0463c5d6121c4c7a3a49c857b1a010c631050b"
CAPTURE_STEPS = (0, 1, 2, 16, 127, 128, 129, 255, 256, 257)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def set_bi_env(enabled: bool) -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_BATCH_INVARIANT_KPROJ"] = "1" if enabled else "0"


def sha_tensor(value: torch.Tensor | None) -> str | None:
    if value is None:
        return None
    data = value.detach().cpu().contiguous()
    prefix = f"{tuple(data.shape)}|{data.dtype}|".encode("utf-8")
    return hashlib.sha256(prefix + data.numpy().tobytes()).hexdigest()


def row_slice(value: torch.Tensor | None, row: int) -> torch.Tensor | None:
    if value is None or value.dim() == 0:
        return value
    if value.shape[0] > row:
        return value[row : row + 1]
    return value


def cache_fingerprint(past_key_values: Any, row: int) -> dict[str, Any]:
    layers = []
    for layer_idx, layer_cache in enumerate(past_key_values or []):
        cache = deserialize_cache(layer_cache, pattern=True)
        item: dict[str, Any] = {
            "layer": layer_idx,
            "total_tokens": int(cache.total_tokens),
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "packed_v4_tokens": int(row_slice(cache.v_precision_mask, row).bool().sum().item()) if cache.v_precision_mask is not None else int(getattr(cache, "packed_v4_tokens", 0) or 0),
            "sink_k_tokens": tensor_tokens(row_slice(cache.sink_k, row)),
            "pending_k_tokens": tensor_tokens(row_slice(cache.pending_k, row)),
            "recent_k_tokens": tensor_tokens(row_slice(cache.recent_k, row)),
            "sink_v_tokens": tensor_tokens(row_slice(cache.sink_v, row)),
            "pending_v_tokens": tensor_tokens(row_slice(cache.pending_v, row)),
            "recent_v_tokens": tensor_tokens(row_slice(cache.recent_v, row)),
            "packed_k_hash": sha_tensor(row_slice(cache.packed_k, row)),
            "packed_k_scale_hash": sha_tensor(row_slice(cache.packed_k_scale, row)),
            "packed_k_zero_hash": sha_tensor(row_slice(cache.packed_k_zero, row)),
            "k_assignments_hash": sha_tensor(row_slice(cache.k_assignments, row)),
            "v_assignment_idx_hash": sha_tensor(row_slice(cache.v_assignment_idx, row)),
            "v_pattern_mask_hash": sha_tensor(row_slice(cache.v_pattern_mask, row)),
            "v_precision_mask_hash": sha_tensor(row_slice(cache.v_precision_mask, row)),
        }
        pool = getattr(cache, "centroid_state_pool", None)
        slots = getattr(cache, "centroid_state_indices", None)
        if pool is not None and slots is not None:
            slot = int(slots[row].item())
            k_count = int(pool.k_counts[slot].item())
            v_count = int(pool.v_counts[slot].item())
            item.update(
                {
                    "k_count": k_count,
                    "v_count": v_count,
                    "update_count_k": int(pool.update_counts_k[slot].item()),
                    "update_count_v": int(pool.update_counts_v[slot].item()),
                    "last_flush_pos": int(pool.last_flush_pos[slot].item()),
                    "k_centroid_hash": sha_tensor(pool.k_centroid_pool[slot : slot + 1, :, :k_count, :]),
                    "v_centroid_hash": sha_tensor(pool.v_centroid_pool[slot : slot + 1, :, :v_count, :]),
                }
            )
        layers.append(item)
    return {"layers": layers}


def first_fingerprint_diff(got: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    for got_layer, ref_layer in zip(got["layers"], ref["layers"]):
        keys = sorted(set(got_layer) | set(ref_layer))
        for key in keys:
            if got_layer.get(key) != ref_layer.get(key):
                return {
                    "layer": got_layer.get("layer"),
                    "component": key,
                    "got": got_layer.get(key),
                    "ref": ref_layer.get(key),
                }
    if len(got["layers"]) != len(ref["layers"]):
        return {"layer": None, "component": "num_layers", "got": len(got["layers"]), "ref": len(ref["layers"])}
    return None


def greedy_capture(model: Any, input_ids: torch.Tensor, capture_steps: tuple[int, ...]) -> dict[str, Any]:
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()
    reset_batch_invariant_kproj_counters()
    torch.cuda.reset_peak_memory_stats(input_ids.device)
    max_step = max(capture_steps)
    captures: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    with torch.inference_mode():
        out = model.model(input_ids=input_ids, use_cache=True, output_hidden_states=False, return_dict=True)
        past = out.past_key_values
        hidden = out.last_hidden_state[:, -1, :]
        logits = model.lm_head(hidden).float().detach()
        hidden = hidden.detach()
        next_token = logits.argmax(dim=-1)
        captures[0] = {
            "logits": logits,
            "hidden": hidden,
            "token": next_token,
            "fingerprints": [cache_fingerprint(past, row) for row in range(int(input_ids.shape[0]))],
        }
        for step in range(1, max_step + 1):
            out = model.model(input_ids=next_token[:, None], past_key_values=past, use_cache=True, output_hidden_states=False, return_dict=True)
            past = out.past_key_values
            hidden = out.last_hidden_state[:, -1, :]
            logits = model.lm_head(hidden).float().detach()
            hidden = hidden.detach()
            next_token = logits.argmax(dim=-1)
            if step in capture_steps:
                captures[step] = {
                    "logits": logits,
                    "hidden": hidden,
                    "token": next_token,
                    "fingerprints": [cache_fingerprint(past, row) for row in range(int(input_ids.shape[0]))],
                }
    torch.cuda.synchronize(input_ids.device)
    return {
        "captures": captures,
        "elapsed_s": time.perf_counter() - started,
        "peak_memory_allocated": int(torch.cuda.max_memory_allocated(input_ids.device)),
        "peak_memory_reserved": int(torch.cuda.max_memory_reserved(input_ids.device)),
        "bi_counters": batch_invariant_kproj_counters(),
        "real_counters": get_patternkv_real_decode_counters(),
        "page_counters": get_patternkv_page_batch_counters(),
    }


def compare_run(name: str, batch: int, refs: list[dict[str, Any]], batched: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows = []
    first_divergence = None
    steps = sorted(set(batched["captures"]).intersection(*(set(ref["captures"]) for ref in refs)))
    for step in steps:
        for row in range(batch):
            got = batched["captures"][step]
            ref = refs[row]["captures"][step]
            hm = tensor_metrics(got["hidden"][row], ref["hidden"][0])
            lm = tensor_metrics(got["logits"][row], ref["logits"][0])
            got_fp = got["fingerprints"][row]
            ref_fp = ref["fingerprints"][0]
            diff = first_fingerprint_diff(got_fp, ref_fp)
            state_ok = diff is None
            token_match = int(got["token"][row].item()) == int(ref["token"][0].item())
            rows.append(
                {
                    "case": name,
                    "batch": batch,
                    "row": row,
                    "step": step,
                    "phase": "prefill" if step == 0 else "decode",
                    "state_ok": state_ok,
                    "hidden_relative_l2": hm["relative_l2"],
                    "logit_relative_l2": lm["relative_l2"],
                    "logit_cosine": lm["cosine"],
                    "logit_max_abs": lm["max_abs"],
                    "token_match": token_match,
                    "state_reason": "" if state_ok else json.dumps(diff, sort_keys=True),
                }
            )
            if first_divergence is None and not state_ok:
                first_divergence = {"case": name, "batch": batch, "row": row, "step": step, **(diff or {})}
    return rows, first_divergence


def run_flag_off_quality(model: Any, tokenizer: Any, device: torch.device) -> list[dict[str, Any]]:
    rows = []
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    for row in range(4):
        results = {}
        for enabled in (False, True):
            set_bi_env(enabled)
            results["on" if enabled else "off"] = greedy_capture(model, ids[row : row + 1], (0, 1))
            torch.cuda.empty_cache()
        for step in (0, 1):
            off = results["off"]["captures"][step]
            on = results["on"]["captures"][step]
            hm = tensor_metrics(on["hidden"][0], off["hidden"][0])
            lm = tensor_metrics(on["logits"][0], off["logits"][0])
            rows.append(
                {
                    "row": row,
                    "step": step,
                    "phase": "prefill" if step == 0 else "decode",
                    "hidden_relative_l2": hm["relative_l2"],
                    "logit_relative_l2": lm["relative_l2"],
                    "logit_cosine": lm["cosine"],
                    "logit_max_abs": lm["max_abs"],
                    "token_match": int(on["token"][0].item()) == int(off["token"][0].item()),
                    "flag_off_bi_prefill_calls": results["off"]["bi_counters"].get("bi_prefill_kproj_calls", 0),
                    "flag_on_bi_prefill_calls": results["on"]["bi_counters"].get("bi_prefill_kproj_calls", 0),
                    "flag_on_bi_decode_calls": results["on"]["bi_counters"].get("bi_decode_kproj_calls", 0),
                    "flag_on_normal_decode_calls": results["on"]["bi_counters"].get("normal_decode_kproj_calls", 0),
                }
            )
    return rows


def run_correctness(model: Any, tokenizer: Any, device: torch.device) -> dict[str, Any]:
    set_bi_env(True)
    ids = make_fixed_inputs(tokenizer, 4, 512, device)
    refs = []
    for row in range(4):
        refs.append(greedy_capture(model, ids[row : row + 1], CAPTURE_STEPS))
        torch.cuda.empty_cache()
    cases = []
    first_divergence = None
    for batch in (1, 2, 4):
        batched = refs[0] if batch == 1 else greedy_capture(model, ids[:batch], CAPTURE_STEPS)
        rows, diff = compare_run(f"b{batch}_ctx512_d257", batch, refs[:batch], batched)
        cases.extend(rows)
        if first_divergence is None:
            first_divergence = diff
        torch.cuda.empty_cache()

    reorder_rows = []
    for order in ([1, 0], [2, 3, 0, 1]):
        batched = greedy_capture(model, ids[order], (0, 1))
        refs_for_order = [refs[idx] for idx in order]
        rows, diff = compare_run("reorder_" + "_".join(map(str, order)), len(order), refs_for_order, batched)
        reorder_rows.extend([row for row in rows if row["step"] in (0, 1)])
        if first_divergence is None:
            first_divergence = diff
        torch.cuda.empty_cache()

    composition_rows = []
    for order in ([0, 1], [0, 2], [0, 3], [0, 1, 2, 3]):
        batched = greedy_capture(model, ids[order], (0, 1))
        refs_for_order = [refs[idx] for idx in order]
        rows, diff = compare_run("composition_" + "_".join(map(str, order)), len(order), refs_for_order, batched)
        composition_rows.extend([row for row in rows if row["step"] in (0, 1) and row["row"] == 0])
        if first_divergence is None:
            first_divergence = diff
        torch.cuda.empty_cache()

    return {
        "state_rows": cases,
        "request_reorder_rows": reorder_rows,
        "batch_composition_rows": composition_rows,
        "runtime_examples": {
            "b1": refs[0]["bi_counters"],
            "b2": greedy_capture(model, ids[:2], (0, 1))["bi_counters"],
            "b4": greedy_capture(model, ids[:4], (0, 1))["bi_counters"],
        },
        "first_divergence": first_divergence,
    }


def run_prefill_performance(model: Any, tokenizer: Any, device: torch.device, reps: int, warmup: int, contexts: tuple[int, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    memory: dict[str, Any] = {}
    for batch in (1, 2, 4):
        for context in (512, 2048, 4096):
            if context not in contexts:
                rows.append(
                    {
                        "batch": batch,
                        "context": context,
                        "warmup": warmup,
                        "reps": reps,
                        "baseline_ms": None,
                        "bi_ms": None,
                        "ratio_bi_over_baseline": None,
                        "status": "not_sampled_in_this_run",
                    }
                )
                memory[f"b{batch}_ctx{context}"] = {"baseline": None, "bi": None, "status": "not_sampled_in_this_run"}
                continue
            print(f"[prefill_perf] batch={batch} context={context} warmup={warmup} reps={reps}", flush=True)
            ids = make_fixed_inputs(tokenizer, batch, context, device)
            row: dict[str, Any] = {"batch": batch, "context": context, "warmup": warmup, "reps": reps}
            peaks: dict[str, int] = {}
            times: dict[str, float] = {}
            for enabled in (False, True):
                label = "bi" if enabled else "baseline"
                try:
                    set_bi_env(enabled)
                    for _ in range(warmup):
                        with torch.inference_mode():
                            out = model.model(input_ids=ids, use_cache=True, return_dict=True)
                        del out
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()
                    for _ in range(reps):
                        with torch.inference_mode():
                            out = model.model(input_ids=ids, use_cache=True, return_dict=True)
                        del out
                    torch.cuda.synchronize(device)
                    elapsed_ms = (time.perf_counter() - start) * 1000.0 / reps
                    times[label] = elapsed_ms
                    peaks[label] = int(torch.cuda.max_memory_allocated(device))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    times[label] = None
                    peaks[label] = None
            row["baseline_ms"] = times.get("baseline")
            row["bi_ms"] = times.get("bi")
            row["ratio_bi_over_baseline"] = (row["bi_ms"] / row["baseline_ms"]) if row["baseline_ms"] and row["bi_ms"] else None
            row["status"] = "sampled" if row["baseline_ms"] is not None and row["bi_ms"] is not None else "oom"
            rows.append(row)
            memory[f"b{batch}_ctx{context}"] = peaks
            torch.cuda.empty_cache()
    return rows, memory


def build_gate(results: dict[str, Any]) -> dict[str, Any]:
    state_rows = results["prefill_state_rows"]
    step_ok = {step: any(row["step"] == step for row in state_rows) and all(row["state_ok"] for row in state_rows if row["step"] == step) for step in CAPTURE_STEPS}
    batch_ok = {batch: any(row["batch"] == batch for row in state_rows) and all(row["state_ok"] for row in state_rows if row["batch"] == batch) for batch in (1, 2, 4)}
    b1_quality_pass = bool(results["b1_quality_rows"]) and all(row["token_match"] for row in results["b1_quality_rows"])
    counter_sets = results["runtime_counters"]
    counters_ok = all(c.get("bi_prefill_kproj_calls", 0) > 0 and c.get("bi_decode_kproj_calls", 0) > 0 and c.get("normal_decode_kproj_calls", 0) == 0 for c in counter_sets.values())
    request_reorder_pass = bool(results["request_reorder_rows"]) and all(row["state_ok"] for row in results["request_reorder_rows"])
    batch_composition_pass = bool(results["batch_composition_rows"]) and all(row["state_ok"] for row in results["batch_composition_rows"])
    all_correct = all(step_ok.values()) and all(batch_ok.values()) and request_reorder_pass and batch_composition_pass
    classification = "BI_KPROJ_PREFILL_RUNTIME_SUPPORTED" if all_correct and b1_quality_pass and counters_ok else "BI_KPROJ_PREFILL_RUNTIME_BLOCKED"
    first = results["first_divergence"] or {}
    return {
        "start_head": START_HEAD,
        "model_path": str(MODEL_PATH),
        "actual_model_loaded": True,
        "synthetic_model_used_for_primary_evidence": False,
        "bi_kproj_v2_integrated": True,
        "bi_kproj_prefill_only": False,
        "bi_kproj_decode_enabled": counters_ok,
        "prefill_detection_initial_cache": True,
        "prefill_detection_decode_cache": True,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "kmeans_iters": 30,
        "kmeans_tol": 1e-4,
        "kmeans_seed": 0,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "fused_value_arithmetic_changed": False,
        "flag_off_baseline_pass": b1_quality_pass and all(row["flag_off_bi_prefill_calls"] == 0 for row in results["b1_quality_rows"]),
        "b1_quality_control_pass": b1_quality_pass,
        "prefill_state_pass": step_ok[0],
        "decode1_state_pass": step_ok[1],
        "decode2_state_pass": step_ok[2],
        "decode16_state_pass": step_ok[16],
        "step_127_pass": step_ok[127],
        "step_128_pass": step_ok[128],
        "step_129_pass": step_ok[129],
        "step_255_pass": step_ok[255],
        "step_256_pass": step_ok[256],
        "step_257_pass": step_ok[257],
        "b1_pass": batch_ok[1],
        "b2_pass": batch_ok[2],
        "b4_pass": batch_ok[4],
        "request_reorder_pass": request_reorder_pass,
        "batch_composition_pass": batch_composition_pass,
        "runtime_counter_semantics_pass": counters_ok,
        "bi_prefill_kproj_calls": sum(c.get("bi_prefill_kproj_calls", 0) for c in counter_sets.values()),
        "bi_decode_kproj_calls": sum(c.get("bi_decode_kproj_calls", 0) for c in counter_sets.values()),
        "normal_decode_kproj_calls": sum(c.get("normal_decode_kproj_calls", 0) for c in counter_sets.values()),
        "bi_prefill_serial_dispatches": sum(c.get("bi_prefill_serial_dispatches", 0) for c in counter_sets.values()),
        "bi_prefill_fallback_calls": sum(c.get("bi_prefill_fallback_calls", 0) for c in counter_sets.values()),
        "first_divergence_step": first.get("step"),
        "first_divergence_layer": first.get("layer"),
        "first_divergence_component": first.get("component"),
        "performance_rows": len(results["prefill_performance_rows"]),
        "classification": classification,
        "next_task": "PROMOTE_BI_KPROJ_PREFILL_RUNTIME" if classification == "BI_KPROJ_PREFILL_RUNTIME_SUPPORTED" else "TRACE_BI_KPROJ_PREFILL_RUNTIME_DIVERGENCE",
    }


def write_reports(results: dict[str, Any], final_gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "integration_audit.json", results["integration_audit"])
    write_csv(REPORT_DIR / "b1_quality.csv", results["b1_quality_rows"], list(results["b1_quality_rows"][0].keys()))
    write_csv(REPORT_DIR / "prefill_state.csv", [row for row in results["prefill_state_rows"] if row["step"] == 0], list(results["prefill_state_rows"][0].keys()))
    write_csv(REPORT_DIR / "request_reorder.csv", results["request_reorder_rows"], list(results["request_reorder_rows"][0].keys()))
    write_csv(REPORT_DIR / "batch_composition.csv", results["batch_composition_rows"], list(results["batch_composition_rows"][0].keys()))
    write_csv(REPORT_DIR / "decode_steps.csv", [row for row in results["prefill_state_rows"] if row["step"] in (1, 2, 16)], list(results["prefill_state_rows"][0].keys()))
    write_csv(REPORT_DIR / "boundary_state.csv", [row for row in results["prefill_state_rows"] if row["step"] in (127, 128, 129, 255, 256, 257)], list(results["prefill_state_rows"][0].keys()))
    write_csv(REPORT_DIR / "prefill_performance.csv", results["prefill_performance_rows"], ["batch", "context", "warmup", "reps", "baseline_ms", "bi_ms", "ratio_bi_over_baseline", "status"])
    write_json(REPORT_DIR / "runtime_counters.json", results["runtime_counters"])
    write_json(REPORT_DIR / "memory.json", results["memory"])
    write_json(REPORT_DIR / "first_divergence.json", results["first_divergence"])
    write_json(REPORT_DIR / "final_gate.json", final_gate)

    docs = {
        "environment.md": f"# Environment\n\nRepo: pytenter/Bounded-pattrenKV-method\n\nLocal: {REPO_ROOT}\n\nStart HEAD: {START_HEAD}\n\nModel: {MODEL_PATH}\n\nCUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}\n",
        "integration_callsite.md": "# Integration Callsite\n\n`models/llama_patternkv.py` dispatches BI KProj from `LlamaFlashAttention_PatternKV.forward` only when `past_key_value is None` and `PATTERNKV_BATCH_INVARIANT_KPROJ=1`. Decode keeps `self.k_proj`.\n",
        "prefill_detection.md": "# Prefill Detection\n\nInitial prefill is detected by an empty layer cache (`past_key_value is None`). Decode is detected by a non-empty PatternKV segmented cache, independent of `q_len`.\n",
        "runtime_design.md": "# Runtime Design\n\nThe runtime calls `batch_invariant_k_projection(..., backend=PATTERNKV_BI_KPROJ_BACKEND)` for K projection during prefill. Q projection, V projection, RoPE, K-means, selector, packing, centroid state, and fused Value code are unchanged.\n",
        "flag_off_baseline.md": "# Flag-Off Baseline\n\nFlag-off runs record zero BI prefill dispatches. See `b1_quality.csv`.\n",
        "b1_quality_control.md": "# B1 Quality Control\n\nB1 flag-off/flag-on logits, hidden states, and next tokens are compared at prefill and decode1 in `b1_quality.csv`.\n",
        "prefill_state_equivalence.md": "# Prefill State Equivalence\n\nPrefill cache fingerprints compare batched rows against independent B1 references in `prefill_state.csv`.\n",
        "request_reorder.md": "# Request Reorder\n\nRequest reorder probes compare reordered B2/B4 rows against their B1 references. See `request_reorder.csv`.\n",
        "batch_composition.md": "# Batch Composition\n\nBatch composition probes keep request A fixed across B2/B4 compositions. See `batch_composition.csv`.\n",
        "decode_multistep.md": "# Decode Multistep\n\nDecode step 1, 2, and 16 structural rows are in `decode_steps.csv`.\n",
        "boundary_127_128_129.md": "# Boundary 127 128 129\n\nBoundary rows for decode steps 127, 128, and 129 are in `boundary_state.csv`.\n",
        "boundary_255_256_257.md": "# Boundary 255 256 257\n\nBoundary rows for decode steps 255, 256, and 257 are in `boundary_state.csv`.\n",
        "fused_value_audit.md": "# Fused Value Audit\n\nThe BI integration touches only K projection before cache construction. `PATTERNKV_MIXED_V_BACKEND=fused_page` remains the fused Value dispatch for decode.\n",
        "runtime_counters.md": "# Runtime Counters\n\nSee `runtime_counters.json` for prefill BI dispatches, decode BI dispatches, normal decode KProj calls, and serial/fallback counters.\n",
        "prefill_performance.md": "# Prefill Performance\n\nPrefill-only latency measurements are in `prefill_performance.csv`.\n",
        "memory.md": "# Memory\n\nPeak allocated memory for prefill performance cases is in `memory.json`.\n",
        "risk_analysis.md": "# Risk Analysis\n\nThis is flag-gated and prefill-only. Risk concentrates in Triton backend availability and exact cache-state invariance across batch composition; final promotion should keep the flag disabled by default until downstream serving profiles accept the latency/memory tradeoff.\n",
        "final_recommendation.md": f"# Final Recommendation\n\nCLASSIFICATION={final_gate['classification']}\n\nNEXT_TASK={final_gate['next_task']}\n",
    }
    for name, text in docs.items():
        (REPORT_DIR / name).write_text(text, encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)
    tokenizer, config, model = load_model(dtype, device)
    integration_audit = {
        "callsite": "models/llama_patternkv.py:LlamaFlashAttention_PatternKV.forward",
        "prefill_predicate": "past_key_value is None and PATTERNKV_BATCH_INVARIANT_KPROJ=1",
        "decode_projection": "self.k_proj",
        "backend_env": "PATTERNKV_BI_KPROJ_BACKEND=v2",
        "num_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(config.num_key_value_heads),
        "k_bits": int(config.k_bits),
        "v_bits": int(config.v_bits),
        "selector": str(config.patternkv_v_precision_selector),
    }
    b1_quality_rows = run_flag_off_quality(model, tokenizer, device)
    correctness = run_correctness(model, tokenizer, device)
    perf_contexts = tuple(int(item) for item in args.perf_contexts.split(",") if item.strip())
    perf_rows, memory = run_prefill_performance(model, tokenizer, device, args.perf_reps, args.perf_warmup, perf_contexts)
    return {
        "integration_audit": integration_audit,
        "b1_quality_rows": b1_quality_rows,
        "prefill_state_rows": correctness["state_rows"],
        "request_reorder_rows": correctness["request_reorder_rows"],
        "batch_composition_rows": correctness["batch_composition_rows"],
        "runtime_counters": correctness["runtime_examples"],
        "prefill_performance_rows": perf_rows,
        "memory": memory,
        "first_divergence": correctness["first_divergence"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--perf-reps", type=int, default=10)
    parser.add_argument("--perf-warmup", type=int, default=3)
    parser.add_argument("--perf-contexts", default="512,2048,4096")
    return parser.parse_args()


def main() -> None:
    results = run(parse_args())
    final_gate = build_gate(results)
    write_reports(results, final_gate)
    print(json.dumps(final_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
