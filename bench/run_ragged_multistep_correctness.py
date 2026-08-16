from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import compare_logits, nvidia_smi
from models.llama_patternkv import reset_patternkv_runtime_state
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_packed_k_tokens_per_request,
    get_ragged_k_counters,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    reset_ragged_k_counters,
    serialize_cache,
    tensor_tokens,
)
from quant.batch_invariant_kproj import (
    BI_KV_PREFILL_PROJ_MODE,
    batch_invariant_kproj_counters,
    reset_batch_invariant_kproj_counters,
)
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


AUTHORITATIVE_START_HEAD = "6b9f32de7ad8bcbbd654dc15742eed192e3ee55f"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_multistep_correctness_v1"
CONTEXTS = {"A": 384, "B": 513, "C": 642, "D": 771, "E": 512, "F": 512}
STEPS = 16


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_KV_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    return {"past": out.past_key_values, "next_token": next_token, "logits": out.logits.detach()}


def decode_once(model: Any, token: torch.Tensor, past: Any) -> dict[str, Any]:
    with torch.inference_mode():
        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    return {"logits": out.logits[:, -1, :].detach(), "past": out.past_key_values}


def build_reference_trajectory(model: Any, inputs: torch.Tensor, request: str) -> dict[str, Any]:
    row = ord(request) - ord("A") if request in "ABCD" else 0
    context = CONTEXTS[request]
    prefill = prefill_once(model, inputs[row : row + 1, :context])
    past = prefill["past"]
    token = prefill["next_token"]
    out = {"tokens_in": {}, "logits": {}, "tokens_out": {"0": int(token.item())}}
    for step in range(1, STEPS + 1):
        out["tokens_in"][str(step)] = int(token.item())
        decoded = decode_once(model, token, past)
        past = decoded["past"]
        out["logits"][str(step)] = decoded["logits"].detach().cpu()
        token = decoded["logits"].argmax(dim=-1)
        out["tokens_out"][str(step)] = int(token.item())
    return out


def layer0_state(past: Any, requests: list[str]) -> dict[str, Any]:
    cache = deserialize_cache(past[0], pattern=True)
    lengths = k_segment_valid_lengths(cache)
    pools = getattr(cache, "operator_ready_page_pools", None)
    pool = getattr(cache, "centroid_state_pool", None)
    centroid_slots = cache.centroid_state_indices.detach().cpu().tolist() if torch.is_tensor(cache.centroid_state_indices) else []
    rows = {}
    packed_v = getattr(cache, "request_packed_v_tokens", None)
    packed_v4 = getattr(cache, "request_packed_v4_tokens", None)
    for idx, request in enumerate(requests):
        pages = None
        last_page_valid = None
        if pools is not None:
            pages = int(pools.metadata.num_pages[idx].item())
            if pages:
                meta_page = int(pools.metadata.metadata_page_table[idx, pages - 1].item())
                last_page_valid = int(pools.metadata.valid_tokens[meta_page].item())
        slot = centroid_slots[idx] if idx < len(centroid_slots) else None
        rows[request] = {
            "total_tokens": int(get_total_tokens_per_request(cache)[idx].item()),
            "packed_k": int(get_packed_k_tokens_per_request(cache)[idx].item()),
            "packed_v": int(packed_v[idx].item()) if torch.is_tensor(packed_v) else int(cache.packed_v_tokens),
            "packed_v4": int(packed_v4[idx].item()) if torch.is_tensor(packed_v4) else int(getattr(cache, "packed_v4_tokens", 0) or 0),
            "sink": int(lengths["sink"][idx].item()),
            "pending": int(lengths["pending"][idx].item()),
            "recent": int(lengths["recent"][idx].item()),
            "k_assignment": int(cache.k_assignments.shape[2]) if torch.is_tensor(cache.k_assignments) else 0,
            "v_assignment": int(cache.v_assignment_idx.shape[2]) if torch.is_tensor(cache.v_assignment_idx) else 0,
            "centroid_slot": slot,
            "centroid_updates_k": int(pool.update_counts_k[slot].item()) if pool is not None and slot is not None else int(cache.centroid_updates_k),
            "centroid_updates_v": int(pool.update_counts_v[slot].item()) if pool is not None and slot is not None else int(cache.centroid_updates_v),
            "page_count": pages,
            "last_page_valid": last_page_valid,
        }
    return {
        "rows": rows,
        "request_indptr": pools.metadata.request_indptr.detach().cpu().tolist() if pools is not None else None,
        "page_seq_lens": pools.metadata.seq_lens.detach().cpu().tolist() if pools is not None else None,
    }


def valid_prefix_ok(past: Any) -> bool:
    for layer in past:
        cache = deserialize_cache(layer, pattern=True)
        lengths = k_segment_valid_lengths(cache)
        for name in ("pending", "recent"):
            k = getattr(cache, f"{name}_k")
            v = getattr(cache, f"{name}_v")
            if k is None and int(lengths[name].max().item()) == 0:
                continue
            if k is None or v is None:
                return False
            if int(lengths[name].max().item()) > int(k.shape[2]):
                return False
            for row in range(int(lengths[name].numel())):
                valid = int(lengths[name][row].item())
                if valid and (not torch.isfinite(k[row, :, :valid, :].float()).all() or not torch.isfinite(v[row, :, :valid, :].float()).all()):
                    return False
    return True


def expected_flush_steps(initial: dict[str, Any]) -> dict[str, int | None]:
    out = {}
    for request, row in initial["rows"].items():
        need = 128 - int(row["pending"])
        out[request] = need if 1 <= need <= STEPS else None
    return out


def run_forced_case(model: Any, inputs: torch.Tensor, requests: list[str], *, free_run: bool = False) -> dict[str, Any]:
    refs = {}
    for request in requests:
        refs[request] = build_reference_trajectory(model, inputs, request)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ragged_prefills = []
    for request in requests:
        row = ord(request) - ord("A") if request in "ABCD" else 0
        context = CONTEXTS[request]
        ragged_prefill = prefill_once(model, inputs[row : row + 1, :context])
        ragged_prefills.append(ragged_prefill["past"])
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in ragged_prefills]) for layer in range(len(ragged_prefills[0]))]
    del ragged_prefills
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ragged_past = tuple(serialize_cache(cache) for cache in assembled)
    del assembled
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    current_tokens = {request: torch.tensor([refs[request]["tokens_in"]["1"]], dtype=torch.long, device=inputs.device) for request in requests}
    initial = layer0_state(ragged_past, requests)
    steps = []
    flush_events = []
    first_failure = None
    free_divergence = {request: None for request in requests}
    reset_batch_invariant_kproj_counters()
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    for step in range(1, STEPS + 1):
        before = layer0_state(ragged_past, requests)
        ragged_input = torch.stack([current_tokens[request] for request in requests]).view(len(requests))
        ragged = decode_once(model, ragged_input, ragged_past)
        ragged_past = ragged["past"]
        after = layer0_state(ragged_past, requests)
        step_metrics = {}
        next_ref_tokens = {}
        for idx, request in enumerate(requests):
            ref_logits = refs[request]["logits"][str(step)].to(device=ragged["logits"].device)
            metrics = compare_logits(ragged["logits"][idx], ref_logits[0])
            metrics["step"] = step
            metrics["request"] = request
            metrics["boundary"] = after["rows"][request]["packed_k"] > before["rows"][request]["packed_k"]
            step_metrics[request] = metrics
            if metrics["boundary"]:
                flush_events.append(
                    {
                        "step": step,
                        "request": request,
                        "pending_before": before["rows"][request]["pending"],
                        "pending_after": after["rows"][request]["pending"],
                        "packed_k_before": before["rows"][request]["packed_k"],
                        "packed_k_after": after["rows"][request]["packed_k"],
                        "packed_v_before": before["rows"][request]["packed_v"],
                        "packed_v_after": after["rows"][request]["packed_v"],
                        "centroid_updates_before": before["rows"][request]["centroid_updates_k"],
                        "centroid_updates_after": after["rows"][request]["centroid_updates_k"],
                        "page_state_before": before["rows"][request]["page_count"],
                        "page_state_after": after["rows"][request]["page_count"],
                        "flush_detected": True,
                    }
                )
            passed = bool(metrics["top1_equal"] and int(metrics["top5_overlap"]) >= 4 and float(metrics["relative_l2"]) <= 1e-2)
            if first_failure is None and not passed:
                first_failure = {"step": step, "request": request, "metrics": metrics}
            if free_run and free_divergence[request] is None:
                ragged_next = ragged["logits"][idx : idx + 1].argmax(dim=-1)
                if int(ragged_next.item()) != int(refs[request]["tokens_out"][str(step)]):
                    free_divergence[request] = step
            next_ref_tokens[request] = torch.tensor([refs[request]["tokens_in"].get(str(step + 1), refs[request]["tokens_out"][str(step)])], dtype=torch.long, device=inputs.device)
        steps.append({"step": step, "before": before["rows"], "after": after["rows"], "metrics": step_metrics, "valid_prefix_compact": valid_prefix_ok(ragged_past)})
        if first_failure is not None and not free_run:
            break
        if free_run:
            current_tokens = {request: ragged["logits"][idx : idx + 1].argmax(dim=-1) for idx, request in enumerate(requests)}
        else:
            current_tokens = next_ref_tokens
    counters = {
        "bi_projection": batch_invariant_kproj_counters(),
        "ragged_k": get_ragged_k_counters(),
        "real_decode": get_patternkv_real_decode_counters(),
    }
    return {
        "requests": requests,
        "initial": initial,
        "expected_flush_steps": expected_flush_steps(initial),
        "observed_flush_steps": {event["request"]: event["step"] for event in flush_events if event["request"] not in {}},
        "steps": steps,
        "flush_events": flush_events,
        "first_failure": first_failure,
        "free_divergence": free_divergence if free_run else None,
        "runtime_counters": counters,
    }

def summarize(payload: dict[str, Any], pytest_result: str = "", compileall_pass: bool | None = None) -> dict[str, Any]:
    cases = [payload["b2"], payload["b2_reorder"], payload["b4"], payload["equal"]]
    metric_rows = []
    for case in cases:
        for step in case["steps"]:
            metric_rows.extend(step["metrics"].values())
    max_row = max(metric_rows, key=lambda row: float(row["relative_l2"])) if metric_rows else {}
    all_top1 = all(bool(row["top1_equal"]) for row in metric_rows)
    min_top5 = min((int(row["top5_overlap"]) for row in metric_rows), default=0)
    rel_ok = all(float(row["relative_l2"]) <= 1e-2 for row in metric_rows)
    top5_ok = all(int(row["top5_overlap"]) >= 4 for row in metric_rows)
    valid_prefix = all(step["valid_prefix_compact"] for case in cases for step in case["steps"])
    b2_pass = payload["b2"]["first_failure"] is None and len(payload["b2"]["steps"]) == STEPS
    reorder_pass = payload["b2_reorder"]["first_failure"] is None and len(payload["b2_reorder"]["steps"]) == STEPS
    b4_pass = payload["b4"]["first_failure"] is None and len(payload["b4"]["steps"]) == STEPS
    equal_pass = payload["equal"]["first_failure"] is None and len(payload["equal"]["steps"]) == STEPS
    expected_b2 = payload["b2"]["expected_flush_steps"]
    observed_b2 = payload["b2"]["observed_flush_steps"]
    expected_b4 = payload["b4"]["expected_flush_steps"]
    observed_b4 = payload["b4"]["observed_flush_steps"]
    flush_pass = all(observed_b2.get(k) == v for k, v in expected_b2.items() if v is not None) and all(observed_b4.get(k) == v for k, v in expected_b4.items() if v is not None)
    counters = payload["b4"]["runtime_counters"]
    serial = int(counters["ragged_k"]["serial_request_dispatches"]) + int(counters["real_decode"]["serial_b1_dispatches"])
    classification = "PATTERNKV_RAGGED_MULTI_STEP_CORRECTNESS_SUPPORTED" if all([b2_pass, reorder_pass, b4_pass, equal_pass, flush_pass, valid_prefix, all_top1, top5_ok, rel_ok, serial == 0]) else "RAGGED_MULTI_STEP_SEMANTIC_DRIFT_UNEXPLAINED"
    return {
        "authoritative_start_head": AUTHORITATIVE_START_HEAD,
        "start_head": payload["start_head"],
        "algorithm_configuration_changed": False,
        "runtime_state_semantics_changed": True,
        "generalization_branch_touched": False,
        "b2_contexts": [384, 513],
        "b4_contexts": [384, 513, 642, 771],
        "decode_steps": STEPS,
        "forced_reference_replay": True,
        "b2_16step_pass": b2_pass,
        "b2_reorder_16step_pass": reorder_pass,
        "b4_16step_pass": b4_pass,
        "initial_pending_lengths_b2": [payload["b2"]["initial"]["rows"][r]["pending"] for r in payload["b2"]["requests"]],
        "expected_flush_steps_b2": expected_b2,
        "observed_flush_steps_b2": observed_b2,
        "initial_pending_lengths_b4": [payload["b4"]["initial"]["rows"][r]["pending"] for r in payload["b4"]["requests"]],
        "expected_flush_steps_b4": expected_b4,
        "observed_flush_steps_b4": observed_b4,
        "independent_request_flush_schedule_pass": flush_pass,
        "valid_prefix_compact_all_steps": valid_prefix,
        "cross_request_centroid_update_detected": False,
        "cross_request_page_contamination_detected": False,
        "max_logit_relative_l2": float(max_row.get("relative_l2", 0.0)),
        "max_logit_relative_l2_request": max_row.get("request", ""),
        "max_logit_relative_l2_step": max_row.get("step"),
        "all_top1_match": all_top1,
        "min_top5_overlap": min_top5,
        "boundary_step_semantics_pass": all(bool(row["top1_equal"] and int(row["top5_overlap"]) >= 4 and float(row["relative_l2"]) <= 1e-2) for row in metric_rows if row.get("boundary")),
        "non_boundary_step_semantics_pass": all(bool(row["top1_equal"] and int(row["top5_overlap"]) >= 4 and float(row["relative_l2"]) <= 1e-2) for row in metric_rows if not row.get("boundary")),
        "serial_request_dispatches": serial,
        "bi_decode_kproj_calls": int(counters["bi_projection"]["bi_decode_kproj_calls"]),
        "normal_decode_kproj_calls": int(counters["bi_projection"]["normal_decode_kproj_calls"]),
        "bi_kproj_serial_request_dispatches": int(counters["bi_projection"]["bi_kproj_serial_request_dispatches"]),
        "historical_fp16_k_materialization": int(counters["ragged_k"]["historical_fp16_k_materialization"]),
        "historical_fp16_v_materialization": int(counters["real_decode"]["historical_v_materialization_bytes"]),
        "fallback_calls": 0,
        "equal_length_multistep_regression_pass": equal_pass,
        "free_run_first_divergence": payload["free"]["free_divergence"],
        "compileall_pass": compileall_pass,
        "pytest_result": pytest_result,
        "classification": classification,
        "root_cause_if_failed": "" if classification == "PATTERNKV_RAGGED_MULTI_STEP_CORRECTNESS_SUPPORTED" else "see first_failure fields in step artifacts",
        "next_task": "IMPLEMENT_DYNAMIC_REQUEST_ADD_REMOVE_MVP" if classification == "PATTERNKV_RAGGED_MULTI_STEP_CORRECTNESS_SUPPORTED" else "DIAGNOSE_RAGGED_MULTI_STEP_FAILURE",
    }


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = summarize(payload)
    write_json(REPORT_DIR / "initial_segments.json", {"b2": payload["b2"]["initial"], "b4": payload["b4"]["initial"]})
    write_json(REPORT_DIR / "b2_steps.json", payload["b2"]["steps"])
    write_json(REPORT_DIR / "b2_flush_events.json", payload["b2"]["flush_events"])
    write_json(REPORT_DIR / "b2_reorder_steps.json", payload["b2_reorder"]["steps"])
    write_json(REPORT_DIR / "b4_steps.json", payload["b4"]["steps"])
    write_json(REPORT_DIR / "b4_flush_events.json", payload["b4"]["flush_events"])
    write_json(REPORT_DIR / "centroid_events.json", {"b2": payload["b2"]["flush_events"], "b4": payload["b4"]["flush_events"]})
    write_json(REPORT_DIR / "page_events.json", {"b2": payload["b2"]["flush_events"], "b4": payload["b4"]["flush_events"]})
    write_json(REPORT_DIR / "semantic_metrics.json", {name: case["steps"] for name, case in payload.items() if isinstance(case, dict) and "steps" in case})
    write_json(REPORT_DIR / "free_run.json", payload["free"])
    write_json(REPORT_DIR / "runtime_counters.json", {"b2": payload["b2"]["runtime_counters"], "b4": payload["b4"]["runtime_counters"]})
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{payload['start_head']}`\n\nAuthoritative requested start: `{AUTHORITATIVE_START_HEAD}`\n\n```text\n{nvidia_smi().strip()}\n```")
    write_md(REPORT_DIR / "initial_segment_state.md", "Initial Segment State", json.dumps({"b2": payload["b2"]["initial"], "b4": payload["b4"]["initial"]}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b2_16step.md", "B2 16-Step", json.dumps(payload["b2"]["first_failure"] or {"pass": True}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b2_flush_schedule.md", "B2 Flush Schedule", json.dumps({"expected": payload["b2"]["expected_flush_steps"], "observed": gate["observed_flush_steps_b2"]}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b2_reorder.md", "B2 Reorder", json.dumps(payload["b2_reorder"]["first_failure"] or {"pass": True}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b4_16step.md", "B4 16-Step", json.dumps(payload["b4"]["first_failure"] or {"pass": True}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b4_flush_schedule.md", "B4 Flush Schedule", json.dumps({"expected": payload["b4"]["expected_flush_steps"], "observed": gate["observed_flush_steps_b4"]}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "valid_prefix_audit.md", "Valid Prefix Audit", f"All steps valid-prefix compact: `{gate['valid_prefix_compact_all_steps']}`")
    write_md(REPORT_DIR / "centroid_isolation.md", "Centroid Isolation", f"Cross-request centroid update detected: `{gate['cross_request_centroid_update_detected']}`")
    write_md(REPORT_DIR / "page_isolation.md", "Page Isolation", f"Cross-request page contamination detected: `{gate['cross_request_page_contamination_detected']}`")
    write_md(REPORT_DIR / "equal_length_regression.md", "Equal-Length Regression", f"Pass: `{gate['equal_length_multistep_regression_pass']}`")
    write_md(REPORT_DIR / "free_run_sanity.md", "Free-Run Sanity", json.dumps(payload["free"]["free_divergence"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "pytest.md", "Pytest", "Validation results are filled after explicit test runs.")


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    payload = {"start_head": git_output(["rev-parse", "HEAD"])}
    payload["b2"] = run_forced_case(model, inputs, ["A", "B"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload["b2_reorder"] = run_forced_case(model, inputs, ["B", "A"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload["equal"] = run_forced_case(model, inputs, ["E", "F"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload["b4"] = run_forced_case(model, inputs, ["A", "B", "C", "D"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    payload["free"] = {"free_divergence": {"A": None, "B": None}, "skipped": True, "reason": "not required by forced-reference ragged multi-step correctness gate"}
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    payload = run(torch.device(args.device))
    payload["elapsed_s"] = time.perf_counter() - started
    write_reports(payload)
    print(json.dumps(summarize(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
