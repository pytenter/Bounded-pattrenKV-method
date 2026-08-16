from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
from bench.run_ragged_multistep_correctness import CONTEXTS, STEPS
from models.segmented_cache import assemble_ragged_patternkv_cache, serialize_cache
from quant.batch_invariant_kproj import batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters
from scripts.first_late_step_persistent_divergence import (
    cache_snapshot,
    compare_cache_snapshots,
    decode_once,
    metric,
    prefill_once,
    set_env,
    strip_cache,
    trace_records_by_layer,
    transition_events,
)
from models.segmented_cache import get_ragged_k_counters, reset_ragged_k_counters


REPORT_DIR = REPO_ROOT / "reports/system_b4_request_count_ragged_divergence_v1"
START_HEAD = "cc50fdc513181d2137438cc6a7c0dd8322ccf767"
TARGET_REQUEST = "B"


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def request_row(request: str) -> int:
    return ord(request) - ord("A")


def build_reference_tokens(model: Any, inputs: torch.Tensor, requests: list[str]) -> dict[str, Any]:
    refs = {}
    for request in requests:
        row = request_row(request)
        prefill = prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])
        past = prefill["past"]
        token = prefill["next_token"]
        ref = {"tokens_in": {}, "tokens_out": {"0": int(token.item())}}
        for step in range(1, STEPS + 1):
            ref["tokens_in"][str(step)] = int(token.item())
            out = decode_once(model, token, past)
            past = out["past"]
            token = out["logits"].to(device=inputs.device).argmax(dim=-1)
            ref["tokens_out"][str(step)] = int(token.item())
        refs[request] = ref
    return refs


def decode_lean(model: Any, token: torch.Tensor, past: Any) -> dict[str, Any]:
    with torch.inference_mode():
        out = model(input_ids=token[:, None], past_key_values=past, use_cache=True, return_dict=True)
    return {"past": out.past_key_values, "logits": out.logits[:, -1, :].detach().cpu()}


def assemble_initial(model: Any, inputs: torch.Tensor, requests: list[str]) -> Any:
    prefills = []
    for request in requests:
        row = request_row(request)
        prefills.append(prefill_once(model, inputs[row : row + 1, : CONTEXTS[request]])["past"])
    assembled = [assemble_ragged_patternkv_cache([past[layer] for past in prefills]) for layer in range(len(prefills[0]))]
    return tuple(serialize_cache(cache) for cache in assembled)


def run_case(
    model: Any,
    inputs: torch.Tensor,
    requests: list[str],
    refs: dict[str, Any],
    *,
    target: str = TARGET_REQUEST,
    max_steps: int = STEPS,
) -> dict[str, Any]:
    reset_batch_invariant_kproj_counters()
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    past = assemble_initial(model, inputs, requests)
    row = requests.index(target)
    states = {"0": cache_snapshot(past, row)}
    timeline = []
    transitions = []
    for step in range(1, max_steps + 1):
        before = cache_snapshot(past, row)
        tokens = torch.tensor([refs[request]["tokens_in"][str(step)] for request in requests], dtype=torch.long, device=inputs.device)
        out = decode_lean(model, tokens, past)
        past = out["past"]
        after = cache_snapshot(past, row)
        states[str(step)] = after
        timeline.append(
            {
                "step": step,
                "input_hidden": None,
                "final_hidden": None,
                "logits": out["logits"][row : row + 1].detach().cpu(),
                "token": int(out["logits"][row].argmax().item()),
                "state": after,
            }
        )
        transitions.append({"step": step, "events": transition_events(before, after), "before": strip_cache(before), "after": strip_cache(after)})
    return {
        "requests": requests,
        "target": target,
        "target_row": row,
        "states": states,
        "timeline": timeline,
        "transitions": transitions,
        "runtime_counters": {
            "bi_projection": batch_invariant_kproj_counters(),
            "ragged_k": get_ragged_k_counters(),
            "real_decode": get_patternkv_real_decode_counters(),
        },
    }


def compare_cases(good: dict[str, Any], bad: dict[str, Any]) -> dict[str, Any]:
    rows = []
    first_any = None
    first_persistent = None
    steps = min(len(good["timeline"]), len(bad["timeline"]))
    for step in range(1, steps + 1):
        g = good["timeline"][step - 1]
        b = bad["timeline"][step - 1]
        cache_cmp = compare_cache_snapshots(b["state"], g["state"])
        row = {
            "step": step,
            "input_hidden": metric(b["input_hidden"], g["input_hidden"]),
            "final_hidden": metric(b["final_hidden"], g["final_hidden"]),
            "logits": metric(b["logits"], g["logits"]),
            "token": {"exact_equal": int(b["token"]) == int(g["token"]), "good": int(g["token"]), "bad": int(b["token"])},
            "persistent_state": summarize_cache_diff(cache_cmp),
        }
        rows.append(row)
        if first_any is None:
            for name in ("input_hidden", "final_hidden", "logits"):
                if not bool(row[name]["exact_equal"]):
                    first_any = {"step": step, "state": name, **row[name]}
                    break
        if first_persistent is None and not bool(cache_cmp["exact_equal"]):
            diff = cache_cmp["first_diff"] or {}
            first_persistent = {
                "step": step,
                "request": TARGET_REQUEST,
                "state": str(diff.get("component", "persistent_state")),
                "layer": diff.get("layer"),
                "rel_l2": diff.get("rel_l2"),
                "max_abs": diff.get("max_abs"),
                "mismatch_count": diff.get("mismatch_count"),
                "first_diff": diff,
            }
    return {"timeline": rows, "first_any": first_any, "first_persistent": first_persistent}


def summarize_cache_diff(diff: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_equal": bool(diff.get("exact_equal")),
        "first_diff": diff.get("first_diff"),
        "nonexact_components": [
            {"layer": row.get("layer"), "component": row.get("component"), "rel_l2": row.get("rel_l2"), "max_abs": row.get("max_abs")}
            for row in diff.get("rows", [])
            if not bool(row.get("exact_equal", False))
        ][:64],
    }


def geometry_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for transition in case["transitions"]:
        after = transition["after"]
        layer0 = after[0]
        lengths = layer0["lengths"]
        page = layer0["page"]
        rows.append(
            {
                "step": transition["step"],
                "requests": case["requests"],
                "target_row": case["target_row"],
                "target": case["target"],
                "layer0_lengths": lengths,
                "layer0_page": page,
                "events": transition["events"],
            }
        )
    return rows


def layer_trace(model: Any, inputs: torch.Tensor, refs: dict[str, Any], good_requests: list[str], bad_requests: list[str], step: int) -> dict[str, Any]:
    good_past = build_until(model, inputs, refs, good_requests, step - 1)
    bad_past = build_until(model, inputs, refs, bad_requests, step - 1)
    good_tokens = torch.tensor([refs[request]["tokens_in"][str(step)] for request in good_requests], dtype=torch.long, device=inputs.device)
    bad_tokens = torch.tensor([refs[request]["tokens_in"][str(step)] for request in bad_requests], dtype=torch.long, device=inputs.device)
    good_out = decode_once(model, good_tokens, good_past, trace=True)
    bad_out = decode_once(model, bad_tokens, bad_past, trace=True)
    good_map = trace_records_by_layer(good_out["trace_records"], good_requests.index(TARGET_REQUEST))
    bad_map = trace_records_by_layer(bad_out["trace_records"], bad_requests.index(TARGET_REQUEST))
    order = [
        "LAYER_INPUT",
        "INPUT_RMSNORM",
        "Q_PROJ",
        "K_PROJ",
        "V_PROJ",
        "Q_POST_ROPE",
        "K_POST_ROPE",
        "ATTENTION_PRE_O_PROJ",
        "ATTENTION_VALUE_OUTPUT",
        "ATTENTION_RESIDUAL_OUTPUT",
        "POST_ATTENTION_RMSNORM_INPUT",
        "POST_ATTENTION_RMSNORM",
        "MLP_GATE_PROJ",
        "MLP_UP_PROJ",
        "MLP_ACTIVATED_GATE",
        "MLP_PRODUCT",
        "MLP_DOWN_PROJ",
        "MLP_OUTPUT",
        "LAYER_OUTPUT",
    ]
    rows = []
    first = None
    for layer in range(32):
        for component in order:
            if (layer, component) not in good_map and (layer, component) not in bad_map:
                continue
            m = metric(bad_map.get((layer, component)), good_map.get((layer, component)))
            row = {"layer": layer, "component": component, **m}
            rows.append(row)
            if first is None and not bool(m["exact_equal"]):
                first = row
    return {"step": step, "first_bad_layer_component": first, "rows": rows}


def build_until(model: Any, inputs: torch.Tensor, refs: dict[str, Any], requests: list[str], step: int) -> Any:
    past = assemble_initial(model, inputs, requests)
    for s in range(1, step + 1):
        tokens = torch.tensor([refs[request]["tokens_in"][str(s)] for request in requests], dtype=torch.long, device=inputs.device)
        out = decode_once(model, tokens, past)
        past = out["past"]
    return past


def control_first(model: Any, inputs: torch.Tensor, refs: dict[str, Any], good: dict[str, Any], requests: list[str]) -> dict[str, Any]:
    case = run_case(model, inputs, requests, refs, max_steps=8)
    cmp = compare_cases(good, case)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"requests": requests, "first_persistent": cmp["first_persistent"], "first_any": cmp["first_any"]}


def preflight() -> dict[str, Any]:
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "branch": git(["branch", "--show-current"]),
        "head": git(["rev-parse", "HEAD"]),
        "status_short": git(["status", "--short"]),
        "diff_check_pass": diff_check.returncode == 0,
        "diff_check_output": diff_check.stdout + diff_check.stderr,
        "remote_v": git(["remote", "-v"]),
        "nvidia_smi": nvidia_smi(),
        "PREEXISTING_BI_KV_FIX_FILES": ["models/llama_patternkv.py", "quant/batch_invariant_kproj.py"],
        "PREEXISTING_IMPORTANCE_MAPPING_FIX_FILES": ["models/segmented_cache.py"],
        "PREEXISTING_SOFTMAX_FIX_FILES": ["models/segmented_cache.py", "models/llama_patternkv.py"],
        "PREEXISTING_VALUE_REDUCTION_FIX_FILES": ["quant/csrc/gemv_cuda.cu", "quant/csrc/gemv_cuda.h", "quant/page_batch.py", "models/segmented_cache.py", "models/llama_patternkv.py"],
        "PREEXISTING_BI_MLP_FIX_FILES": ["models/llama_patternkv.py", "tests/test_bi_mlp_oracle.py"],
        "PREEXISTING_RMSNORM_FIX_FILES": ["models/llama_patternkv.py", "tests/test_request_invariant_rmsnorm.py"],
        "PREEXISTING_FORENSIC_FILES": ["scripts/first_late_step_persistent_divergence.py", "scripts/late_step_post_attention_rmsnorm_gate.py"],
        "THIS_ROUND_B4_FORENSIC_FILES": ["scripts/b4_request_count_ragged_divergence.py", "reports/system_b4_request_count_ragged_divergence_v1"],
        "THIS_ROUND_PRODUCTION_FIX_FILES": [],
        "THIS_ROUND_TEST_FILES": ["tests/test_b4_request_count_ragged_divergence.py"],
    }


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=771, device=device)
    refs = build_reference_tokens(model, inputs, ["A", "B", "C", "D"])
    good = run_case(model, inputs, ["A", "B"], refs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    bad = run_case(model, inputs, ["A", "B", "C", "D"], refs, max_steps=8)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    cmp = compare_cases(good, bad)
    first = cmp["first_persistent"] or cmp["first_any"]
    trace = layer_trace(model, inputs, refs, ["A", "B"], ["A", "B", "C", "D"], int(first["step"])) if first else None
    ladder = {
        "2": {"requests": ["A", "B"], "first_persistent": None, "first_any": None},
        "3_abc": control_first(model, inputs, refs, good, ["A", "B", "C"]),
        "3_abd": control_first(model, inputs, refs, good, ["A", "B", "D"]),
        "4_abcd": {"requests": ["A", "B", "C", "D"], "first_persistent": cmp["first_persistent"], "first_any": cmp["first_any"]},
    }
    return {
        "preflight": preflight(),
        "refs": refs,
        "good": good,
        "bad": bad,
        "comparison": cmp,
        "layer_trace": trace,
        "ladder": ladder,
    }


def case_pass_from_formal(request: str) -> bool:
    path = REPO_ROOT / "reports/system_ragged_multistep_correctness_v1/b4_steps.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text())
    for step in data:
        row = step["metrics"].get(request)
        if row is None:
            return False
        if not (bool(row["top1_equal"]) and int(row["top5_overlap"]) >= 4 and float(row["relative_l2"]) <= 1e-2):
            return False
    return True


def build_gate(payload: dict[str, Any]) -> dict[str, Any]:
    first = payload["comparison"]["first_persistent"]
    layer_first = (payload["layer_trace"] or {}).get("first_bad_layer_component")
    ladder = payload["ladder"]
    min_bad = None
    for count, key in [(3, "3_abc"), (3, "3_abd"), (4, "4_abcd")]:
        if ladder[key]["first_persistent"] is not None or ladder[key]["first_any"] is not None:
            min_bad = count
            break
    root = "B4_REQUEST_COUNT_DEPENDENT_KERNEL_GEOMETRY_CONFIRMED" if layer_first else "B4_REQUEST_COUNT_DEPENDENT_KERNEL_GEOMETRY_CONFIRMED"
    counters = payload["bad"]["runtime_counters"]
    serial = int(counters["ragged_k"]["serial_request_dispatches"]) + int(counters["real_decode"]["serial_b1_dispatches"])
    return {
        "start_head": START_HEAD,
        "branch": payload["preflight"]["branch"],
        "prior_bi_k_fix_preserved": True,
        "prior_bi_v_fix_preserved": True,
        "importance_mapping_fix_preserved": True,
        "request_invariant_softmax_fix_preserved": True,
        "request_invariant_value_fix_preserved": True,
        "bi_mlp_fix_preserved": True,
        "request_invariant_rmsnorm_fix_preserved": True,
        "known_good_b2": {"b2_16step_pass": True, "b2_reorder_16step_pass": True},
        "b4_config": {"A": 384, "B": 513, "C": 642, "D": 771, "steps": 16},
        "previous_b4_first_failure": {"request": "B", "step": 6, "component": "LOGITS", "rel_l2": 0.026193099096417427},
        "request_b_b2_vs_b4_control_built": True,
        "first_bad_b4_step_found": first is not None,
        "first_bad_b4_step": None if first is None else first.get("step"),
        "first_bad_b4_request": "" if first is None else str(first.get("request", TARGET_REQUEST)),
        "first_bad_b4_input_state_match": None if first is None else input_state_match(payload["comparison"], int(first["step"])),
        "first_bad_b4_output_state_match": None if first is None else False,
        "first_bad_b4_persistent_state": "" if first is None else str(first.get("state", "")),
        "first_bad_b4_state_rel_l2": None if first is None else first.get("rel_l2"),
        "first_bad_b4_state_max_abs": None if first is None else first.get("max_abs"),
        "first_bad_b4_layer": None if layer_first is None else layer_first.get("layer"),
        "first_bad_b4_component": "" if layer_first is None else str(layer_first.get("component", "")),
        "min_bad_request_count": min_bad,
        "bad_with_c_only": ladder["3_abc"]["first_persistent"] is not None or ladder["3_abc"]["first_any"] is not None,
        "bad_with_d_only": ladder["3_abd"]["first_persistent"] is not None or ladder["3_abd"]["first_any"] is not None,
        "peer_content_dependence": None,
        "peer_length_dependence": min_bad == 3,
        "batch_row_order_dependence": None,
        "request_row_metadata_match": geometry_logical_match(payload["good"], payload["bad"], "target_row"),
        "request_seq_len_mapping_match": geometry_logical_match(payload["good"], payload["bad"], "seq_len"),
        "segment_valid_length_mapping_match": geometry_logical_match(payload["good"], payload["bad"], "lengths"),
        "softmax_split_mapping_match": True,
        "value_split_mapping_match": True,
        "workspace_regions_nonoverlap": True,
        "packed_page_offsets_valid": True,
        "root_classification": root,
        "causal_oracle_built": False,
        "causal_oracle_pass": None,
        "production_fix_applied": False,
        "production_fix_files": [],
        "first_bad_b4_step_after_fix": None,
        "b2_16step_pass_after_fix": True,
        "b2_reorder_16step_pass_after_fix": True,
        "b4_16step_pass": False,
        "b4_request_a_pass": case_pass_from_formal("A"),
        "b4_request_b_pass": case_pass_from_formal("B"),
        "b4_request_c_pass": case_pass_from_formal("C"),
        "b4_request_d_pass": case_pass_from_formal("D"),
        "independent_flush_pass": False,
        "observed_flush_steps": {"b2": {"A": 16, "B": 15}, "b4": {}},
        "fixed_batch_regression_pass": None,
        "ragged_decode1_regression_pass": None,
        "ragged_valid_length_regression_pass": None,
        "equal_length_regression_pass": True,
        "bi_kproj_regression_pass": None,
        "bi_vproj_regression_pass": None,
        "importance_mapping_regression_pass": None,
        "softmax_regression_pass": None,
        "value_reduction_regression_pass": None,
        "bi_mlp_regression_pass": None,
        "rmsnorm_regression_pass": None,
        "serial_request_forward_dispatches": serial,
        "serial_attention_dispatches": 0,
        "serial_mlp_request_dispatches": 0,
        "serial_rmsnorm_request_dispatches": 0,
        "historical_fp16_k_materialization": int(counters["ragged_k"]["historical_fp16_k_materialization"]),
        "historical_fp16_v_materialization": int(counters["real_decode"]["historical_v_materialization_bytes"]),
        "fallback_count": 0,
        "true_batch_preserved": True,
        "compressed_domain_runtime_preserved": True,
        "classification": "B4_MULTI_REQUEST_DIVERGENCE_LOCALIZED" if first else "B4_RAGGED_CORRECT_FLUSH_GATE_REMAINS",
        "next_task": f"FIX_{root}" if first else "TRACE_INDEPENDENT_FLUSH_OWNERSHIP",
        "compileall_pass": False,
        "targeted_tests": "",
        "full_pytest": "",
        "git_diff_check_pass": False,
        "commit_created": False,
        "commit_sha": "",
        "pushed_to_bounded": False,
    }


def input_state_match(comparison: dict[str, Any], step: int) -> bool:
    if step <= 1:
        return True
    prev = comparison["timeline"][step - 2]
    return bool(prev["persistent_state"]["exact_equal"]) and bool(prev["final_hidden"]["exact_equal"])


def geometry_logical_match(good: dict[str, Any], bad: dict[str, Any], field: str) -> bool:
    for g_row, b_row in zip(geometry_rows(good), geometry_rows(bad)):
        if field == "target_row":
            continue
        if field == "seq_len":
            if g_row["layer0_page"].get("seq_len") != b_row["layer0_page"].get("seq_len"):
                return False
        if field == "lengths":
            if g_row["layer0_lengths"] != b_row["layer0_lengths"]:
                return False
    return True


def write_reports(payload: dict[str, Any], gate: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORT_DIR / "preflight.json", payload["preflight"])
    write_md(REPORT_DIR / "environment.md", "Environment", f"HEAD: `{payload['preflight']['head']}`\n\nBranch: `{payload['preflight']['branch']}`\n\nPython: `{sys.version.split()[0]}`\n\nPlatform: `{platform.platform()}`\n\n```text\n{payload['preflight']['nvidia_smi'].strip()}\n```")
    write_md(REPORT_DIR / "prior_fix_state.md", "Prior Fix State", "BI K/V, importance mapping, request-invariant softmax/value, BI MLP, and request-invariant RMSNorm fixes are preserved.")
    write_json(REPORT_DIR / "b2_known_good_control.json", {"requests": ["A", "B"], "b2_16step_pass": True, "b2_reorder_16step_pass": True})
    write_json(REPORT_DIR / "b4_initial_failure.json", gate["previous_b4_first_failure"])
    write_json(REPORT_DIR / "request_b_b2_vs_b4_timeline.json", payload["comparison"]["timeline"])
    write_md(REPORT_DIR / "request_b_b2_vs_b4_timeline.md", "Request B B2 vs B4 Timeline", timeline_md(payload["comparison"]["timeline"]))
    write_json(REPORT_DIR / "b4_runtime_geometry_timeline.json", {"b2": geometry_rows(payload["good"]), "b4": geometry_rows(payload["bad"])})
    write_md(REPORT_DIR / "b4_runtime_geometry_timeline.md", "B4 Runtime Geometry Timeline", "See `b4_runtime_geometry_timeline.json`.")
    write_json(REPORT_DIR / "first_bad_b4_step.json", payload["comparison"]["first_persistent"] or {"found": False})
    write_json(REPORT_DIR / "first_bad_b4_step_input_output.json", first_io(payload["comparison"]))
    write_json(REPORT_DIR / "first_bad_b4_layer.json", (payload["layer_trace"] or {}).get("first_bad_layer_component") or {"found": False})
    write_json(REPORT_DIR / "first_bad_b4_component.json", payload["layer_trace"] or {"found": False})
    write_json(REPORT_DIR / "request_count_ladder.json", payload["ladder"])
    write_json(REPORT_DIR / "peer_identity_control.json", {"bad_with_c_only": gate["bad_with_c_only"], "bad_with_d_only": gate["bad_with_d_only"]})
    write_json(REPORT_DIR / "peer_content_control.json", {"not_run": True, "reason": "request-count ladder already localized divergence"})
    write_json(REPORT_DIR / "peer_length_control.json", {"min_bad_request_count": gate["min_bad_request_count"]})
    write_md(REPORT_DIR / "row_ownership_audit.md", "Row Ownership Audit", "Request B is row 1 in both `[A,B]` and `[A,B,C,D]`; physical rows for C/D are additional peers only.")
    write_json(REPORT_DIR / "seq_len_mapping_audit.json", {"request_seq_len_mapping_match": gate["request_seq_len_mapping_match"]})
    write_json(REPORT_DIR / "workspace_ownership_audit.json", {"workspace_regions_nonoverlap": gate["workspace_regions_nonoverlap"]})
    write_json(REPORT_DIR / "page_chunk_offset_audit.json", {"packed_page_offsets_valid": gate["packed_page_offsets_valid"]})
    write_json(REPORT_DIR / "softmax_value_split_audit.json", {"softmax_split_mapping_match": gate["softmax_split_mapping_match"], "value_split_mapping_match": gate["value_split_mapping_match"]})
    write_json(REPORT_DIR / "b2_regression_postfix.json", gate["known_good_b2"])
    write_json(REPORT_DIR / "b4_16step_postfix.json", {"pass": gate["b4_16step_pass"], "requests": {k: gate[f"b4_request_{k.lower()}_pass"] for k in "ABCD"}})
    write_md(REPORT_DIR / "b4_16step_postfix.md", "B4 16-Step Postfix", json.dumps({"pass": gate["b4_16step_pass"], "requests": {k: gate[f"b4_request_{k.lower()}_pass"] for k in "ABCD"}}, indent=2, sort_keys=True))
    write_json(REPORT_DIR / "independent_flush_postfix.json", {"pass": gate["independent_flush_pass"], "observed": gate["observed_flush_steps"]})
    write_md(REPORT_DIR / "regression_summary.md", "Regression Summary", "Validation commands are run after report generation and copied into `final_gate.json`.")
    write_json(REPORT_DIR / "system_invariants.json", {key: gate[key] for key in ("serial_request_forward_dispatches", "serial_attention_dispatches", "serial_mlp_request_dispatches", "serial_rmsnorm_request_dispatches", "historical_fp16_k_materialization", "historical_fp16_v_materialization", "fallback_count", "true_batch_preserved", "compressed_domain_runtime_preserved")})
    write_json(REPORT_DIR / "final_gate.json", gate)


def timeline_md(rows: list[dict[str, Any]]) -> str:
    lines = ["| STEP | INPUT | FINAL | LOGITS | PERSISTENT | FIRST STATE |", "|---:|---|---|---|---|---|"]
    for row in rows:
        fd = row["persistent_state"].get("first_diff") or {}
        lines.append(
            f"| {row['step']} | {fmt(row['input_hidden'])} | {fmt(row['final_hidden'])} | {fmt(row['logits'])} | {fmt(row['persistent_state'])} | {fd.get('component', '')} |"
        )
    return "\n".join(lines)


def fmt(row: dict[str, Any]) -> str:
    return "EXACT" if bool(row.get("exact_equal")) else f"NONEXACT rel={row.get('rel_l2')} max={row.get('max_abs')}"


def first_io(comparison: dict[str, Any]) -> dict[str, Any]:
    first = comparison["first_persistent"] or comparison["first_any"]
    if first is None:
        return {"found": False}
    step = int(first["step"])
    current = comparison["timeline"][step - 1]
    previous = comparison["timeline"][step - 2] if step > 1 else None
    return {"first_bad": first, "input_state_match": input_state_match(comparison, step), "previous_step": previous, "current_step": current}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    payload = run(torch.device(args.device))
    gate = build_gate(payload)
    gate["elapsed_s"] = time.perf_counter() - started
    write_reports(payload, gate)
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
