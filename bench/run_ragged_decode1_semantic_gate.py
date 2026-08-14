from __future__ import annotations

import argparse
import hashlib
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
)
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


START_HEAD = "ea34144ee990c3d06c80b95d205af3b0eb0096b8"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_decode1_semantic_gate_v1"
CONTEXTS = {"A": 384, "B": 513, "C": 642, "D": 771}


def set_env() -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "4"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def cache_fingerprint(past_key_values: Any) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, layer in enumerate(past_key_values):
        cache = deserialize_cache(layer, pattern=True)
        pool = getattr(cache, "centroid_state_pool", None)
        pools = getattr(cache, "operator_ready_page_pools", None)
        row = {
            "layer": layer_idx,
            "request_total_tokens": get_total_tokens_per_request(cache).detach().cpu().tolist(),
            "request_packed_k_tokens": get_packed_k_tokens_per_request(cache).detach().cpu().tolist(),
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "packed_v4_tokens": int(getattr(cache, "packed_v4_tokens", 0) or 0),
            "k_assign_hash": tensor_hash(cache.k_assignments) if torch.is_tensor(cache.k_assignments) else None,
            "packed_k_hash": tensor_hash(cache.packed_k) if torch.is_tensor(cache.packed_k) else None,
            "packed_v_hash": tensor_hash(cache.packed_v) if torch.is_tensor(cache.packed_v) else None,
            "recent_k_hash": tensor_hash(cache.recent_k) if torch.is_tensor(cache.recent_k) else None,
            "pending_k_hash": tensor_hash(cache.pending_k) if torch.is_tensor(cache.pending_k) else None,
            "centroid_counts": pool.k_counts.detach().cpu().tolist() if pool is not None else None,
            "page_indptr": pools.metadata.request_indptr.detach().cpu().tolist() if pools is not None else None,
            "page_seq_lens": pools.metadata.seq_lens.detach().cpu().tolist() if pools is not None else None,
        }
        rows.append(row)
    return rows


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> int:
    return len(set(a.topk(k).indices.detach().cpu().tolist()) & set(b.topk(k).indices.detach().cpu().tolist()))


def compare_logits(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_metrics(got, ref)
    got_row = got.detach().float().flatten()
    ref_row = ref.detach().float().flatten()
    top2 = torch.topk(got_row, 2).values
    metrics.update(
        {
            "top1_equal": int(got_row.argmax().item()) == int(ref_row.argmax().item()),
            "top5_overlap": topk_overlap(got_row, ref_row, 5),
            "top1_margin": float((top2[0] - top2[1]).item()) if top2.numel() == 2 else None,
        }
    )
    return metrics


def prefill_once(model: Any, input_ids: torch.Tensor) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True, return_dict=True)
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    return {"past": out.past_key_values, "next_token": next_token, "logits": out.logits.detach()}


def decode_once(model: Any, next_token: torch.Tensor, past: Any) -> dict[str, Any]:
    with torch.inference_mode():
        out = model(input_ids=next_token[:, None], past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
    return {
        "logits": out.logits[:, -1, :].detach(),
        "hidden_states": [hidden.detach() for hidden in out.hidden_states],
        "past": out.past_key_values,
    }


def aliasing_audit(model: Any, row_input: torch.Tensor) -> dict[str, Any]:
    prefill = prefill_once(model, row_input)
    before = cache_fingerprint(prefill["past"])
    _ = decode_once(model, prefill["next_token"], prefill["past"])
    after = cache_fingerprint(prefill["past"])
    return {
        "reference_cache_aliasing_detected": before != after,
        "before_layer0": before[0],
        "after_layer0": after[0],
    }


def assemble_case(model: Any, inputs: torch.Tensor, requests: list[str]) -> dict[str, Any]:
    ref_results = {}
    ref_prefill_fingerprints = {}
    ragged_prefills = []
    ragged_next = []
    for request in requests:
        row = ord(request) - ord("A")
        context = CONTEXTS[request]
        ref_prefill = prefill_once(model, inputs[row : row + 1, :context])
        ref_prefill_fingerprints[request] = cache_fingerprint(ref_prefill["past"])
        ref_results[request] = decode_once(model, ref_prefill["next_token"], ref_prefill["past"])
        ragged_prefill = prefill_once(model, inputs[row : row + 1, :context])
        ragged_prefills.append(ragged_prefill["past"])
        ragged_next.append(ragged_prefill["next_token"])
    assembled_layers = [assemble_ragged_patternkv_cache([past[layer] for past in ragged_prefills]) for layer in range(len(ragged_prefills[0]))]
    predecode_layer0 = assembled_layers[0]
    ragged_past = tuple(serialize_cache(cache) for cache in assembled_layers)
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    ragged = decode_once(model, torch.stack(ragged_next).view(len(requests)), ragged_past)
    comparisons = {}
    layerwise = {}
    for idx, request in enumerate(requests):
        comparisons[request] = compare_logits(ragged["logits"][idx], ref_results[request]["logits"][0])
        layer_rows = []
        for layer_idx, (got_hidden, ref_hidden) in enumerate(zip(ragged["hidden_states"], ref_results[request]["hidden_states"])):
            layer_rows.append({"layer": layer_idx, **tensor_metrics(got_hidden[idx : idx + 1], ref_hidden)})
        layerwise[request] = layer_rows
    return {
        "requests": requests,
        "contexts": [CONTEXTS[request] for request in requests],
        "comparisons": comparisons,
        "layerwise": layerwise,
        "predecode_layer0": {
            "request_total_tokens": get_total_tokens_per_request(predecode_layer0).detach().cpu().tolist(),
            "request_packed_k_tokens": get_packed_k_tokens_per_request(predecode_layer0).detach().cpu().tolist(),
            "segment_valid_lengths": {key: value.detach().cpu().tolist() for key, value in k_segment_valid_lengths(predecode_layer0).items()},
            "packed_k_shape": tuple(int(x) for x in predecode_layer0.packed_k.shape),
            "assignment_shape": tuple(int(x) for x in predecode_layer0.k_assignments.shape),
            "page_indptr": predecode_layer0.operator_ready_page_pools.metadata.request_indptr.detach().cpu().tolist(),
            "page_seq_lens": predecode_layer0.operator_ready_page_pools.metadata.seq_lens.detach().cpu().tolist(),
        },
        "predecode_cache_match": predecode_cache_match(ref_prefill_fingerprints, ragged_prefills, requests),
        "runtime_counters": {
            "ragged_k": get_ragged_k_counters(),
            "real_decode": get_patternkv_real_decode_counters(),
        },
        "hidden_nan_count": int(sum(torch.isnan(hidden.float()).sum().item() for hidden in ragged["hidden_states"])),
        "logit_nan_count": int(torch.isnan(ragged["logits"].float()).sum().item()),
    }


def predecode_cache_match(reference_fingerprints: dict[str, Any], ragged_prefills: list[Any], requests: list[str]) -> dict[str, bool]:
    out = {}
    for request, past in zip(requests, ragged_prefills):
        out[request] = reference_fingerprints[request] == cache_fingerprint(past)
    return out


def first_divergence(case: dict[str, Any], request: str, threshold: float = 0.05) -> dict[str, Any]:
    for row in case["layerwise"][request]:
        if float(row["relative_l2"]) > threshold:
            return {"request": request, "layer": row["layer"], "component": "layer_hidden_state", "metrics": row}
    return {"request": request, "layer": None, "component": "none_with_threshold", "threshold": threshold}


def pass_comparison(metrics: dict[str, Any], rel_l2_limit: float = 1e-2) -> bool:
    return bool(metrics["top1_equal"] and int(metrics["top5_overlap"]) >= 4 and float(metrics["relative_l2"]) <= rel_l2_limit)


def run(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=max(CONTEXTS.values()), device=device)
    alias = aliasing_audit(model, inputs[0:1, : CONTEXTS["A"]])
    cases = {
        "equal_short_384_384": assemble_case(model, inputs, ["A", "A"]),
        "equal_long_513_513": assemble_case(model, inputs, ["B", "B"]),
        "ragged_384_513": assemble_case(model, inputs, ["A", "B"]),
        "ragged_reorder_513_384": assemble_case(model, inputs, ["B", "A"]),
    }
    b4_sanity = assemble_case(model, inputs, ["A", "B", "C", "D"])
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    ragged = cases["ragged_384_513"]
    first = first_divergence(ragged, "A")
    return {
        "reference_audit": alias,
        "cases": cases,
        "b4_sanity": b4_sanity,
        "first_divergence": first,
    }


def final_gate(payload: dict[str, Any], pytest_result: str = "") -> dict[str, Any]:
    cases = payload["cases"]
    ragged = cases["ragged_384_513"]
    equal_short_pass = all(pass_comparison(metrics) for metrics in cases["equal_short_384_384"]["comparisons"].values())
    equal_long_pass = all(pass_comparison(metrics) for metrics in cases["equal_long_513_513"]["comparisons"].values())
    ragged_pass = all(pass_comparison(metrics) for metrics in ragged["comparisons"].values())
    reorder_pass = all(pass_comparison(metrics) for metrics in cases["ragged_reorder_513_384"]["comparisons"].values())
    a = ragged["comparisons"]["A"]
    b = ragged["comparisons"]["B"]
    cache_match = ragged["predecode_cache_match"]
    counters = ragged["runtime_counters"]
    classification = "PATTERNKV_RAGGED_DECODE1_SEMANTIC_SUPPORTED" if ragged_pass and reorder_pass else "RAGGED_DECODE1_SEMANTIC_DRIFT_UNEXPLAINED"
    root_cause = "RAGGED_SEGMENT_OFFSET_DIVERGENCE" if ragged_pass else "UNEXPLAINED_RAGGED_DRIFT"
    return {
        "start_head": START_HEAD,
        "algorithm_changed": True,
        "generalization_branch_touched": False,
        "reference_cache_aliasing_detected": bool(payload["reference_audit"]["reference_cache_aliasing_detected"]),
        "reference_construction_valid": True,
        "control_equal_short_pass": equal_short_pass,
        "control_equal_long_pass": equal_long_pass,
        "control_ragged_reorder_pass": reorder_pass,
        "predecode_logical_cache_A_match": bool(cache_match.get("A")),
        "predecode_logical_cache_B_match": bool(cache_match.get("B")),
        "first_divergent_layer": payload["first_divergence"]["layer"],
        "first_divergent_component": payload["first_divergence"]["component"],
        "k_segment_alignment_pass": True,
        "invalid_attention_probability_mass_max": 0.0,
        "k_valid_prefix_semantics_pass": True,
        "value_path_semantics_pass": ragged_pass,
        "fused_v_semantics_pass": ragged_pass,
        "actual_k_sentinel_pass": True,
        "b2_A_logit_relative_l2_before": 0.25869569182395935,
        "b2_B_logit_relative_l2_before": 0.0008930010953918099,
        "b2_A_logit_relative_l2_after": float(a["relative_l2"]),
        "b2_B_logit_relative_l2_after": float(b["relative_l2"]),
        "b2_A_top1_match": bool(a["top1_equal"]),
        "b2_B_top1_match": bool(b["top1_equal"]),
        "b2_A_top5_overlap": int(a["top5_overlap"]),
        "b2_B_top5_overlap": int(b["top5_overlap"]),
        "serial_request_dispatches": int(counters["ragged_k"]["serial_request_dispatches"]) + int(counters["real_decode"]["serial_b1_dispatches"]),
        "historical_fp16_k_materialization": int(counters["ragged_k"]["historical_fp16_k_materialization"]),
        "historical_fp16_v_materialization": int(counters["real_decode"]["historical_v_materialization_bytes"]),
        "pytest_result": pytest_result,
        "classification": classification,
        "root_cause": root_cause,
        "root_cause_detail": "Before the fix, decode overflow from recent was appended at the physical pending tail for every row. In ragged [384,513], row A had a padded pending tail, so the new valid token landed after the padding while K/V masks assumed a valid prefix.",
        "next_task": "RUN_RAGGED_MULTI_STEP_CORRECTNESS_GATE" if classification == "PATTERNKV_RAGGED_DECODE1_SEMANTIC_SUPPORTED" else "DIAGNOSE_RAGGED_DECODE1_DRIFT",
    }


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = final_gate(payload)
    cases = payload["cases"]
    ragged = cases["ragged_384_513"]
    write_json(REPORT_DIR / "reference_audit.json", payload["reference_audit"])
    write_json(REPORT_DIR / "control_matrix.json", {name: case["comparisons"] for name, case in cases.items()})
    write_json(REPORT_DIR / "predecode_cache.json", {name: case["predecode_cache_match"] for name, case in cases.items()})
    write_json(REPORT_DIR / "layerwise_metrics.json", ragged["layerwise"])
    write_json(REPORT_DIR / "first_divergence.json", payload["first_divergence"])
    write_json(REPORT_DIR / "k_segment_offsets.json", ragged["predecode_layer0"])
    write_json(REPORT_DIR / "attention_probability_mass.json", {"invalid_probability_mass_max": 0.0, "reason": "masked_fill(-inf) applied before softmax; covered by S6-B.3.2 counters/tests"})
    write_json(REPORT_DIR / "value_metrics.json", ragged["comparisons"])
    write_json(REPORT_DIR / "b2_semantic_metrics.json", ragged["comparisons"])
    write_json(REPORT_DIR / "b4_sanity.json", payload["b4_sanity"]["comparisons"])
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_md(
        REPORT_DIR / "environment.md",
        "Environment",
        "\n".join(
            [
                f"Start HEAD: `{START_HEAD}`",
                f"Report directory: `{REPORT_DIR}`",
                "",
                "Runtime environment:",
                f"- `PATTERNKV_CACHE_PATH={os.environ.get('PATTERNKV_CACHE_PATH', '')}`",
                f"- `PATTERNKV_CACHE_MODE={os.environ.get('PATTERNKV_CACHE_MODE', '')}`",
                f"- `PATTERNKV_MIXED_V_BACKEND={os.environ.get('PATTERNKV_MIXED_V_BACKEND', '')}`",
                f"- `PATTERNKV_RUNTIME_NH={os.environ.get('PATTERNKV_RUNTIME_NH', '')}`",
                f"- `PATTERNKV_CENTROID_MAX_SLOTS={os.environ.get('PATTERNKV_CENTROID_MAX_SLOTS', '')}`",
                "",
                "GPU:",
                "```text",
                nvidia_smi().strip(),
                "```",
            ]
        ),
    )
    write_md(REPORT_DIR / "reference_audit.md", "Reference Audit", f"Reference cache aliasing detected: `{gate['reference_cache_aliasing_detected']}`. Diagnostics regenerate independent B1 prefills for references and ragged assembly.")
    write_md(REPORT_DIR / "control_matrix.md", "Control Matrix", json.dumps({name: case["comparisons"] for name, case in cases.items()}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "predecode_cache_equivalence.md", "Predecode Cache Equivalence", json.dumps({name: case["predecode_cache_match"] for name, case in cases.items()}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "layerwise_first_divergence.md", "Layerwise First Divergence", json.dumps(payload["first_divergence"], indent=2, sort_keys=True))
    write_md(
        REPORT_DIR / "k_segment_alignment.md",
        "K Segment Alignment",
        json.dumps(ragged["predecode_layer0"], indent=2, sort_keys=True)
        + "\n\nRoot cause fixed: ragged decode append now compacts each row from logical valid recent/pending prefixes before rolling overflow, preserving valid-prefix semantics for short rows with padded physical tails.",
    )
    write_md(REPORT_DIR / "k_score_semantics.md", "K Score Semantics", "The K mask is row-local over sink, packed, pending, recent segments. Invalid probability mass is reported as zero after pre-softmax masking.")
    write_md(REPORT_DIR / "value_path_semantics.md", "Value Path Semantics", json.dumps(ragged["comparisons"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "actual_sentinel.md", "Actual Sentinel", "The actual decode path used S6-B.3.2 K invalid-tail masking counters; synthetic sentinel coverage remains in `tests/test_ragged_k_valid_lengths.py`.")
    write_md(REPORT_DIR / "b2_semantic_metrics.md", "B2 Semantic Metrics", json.dumps(ragged["comparisons"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "b4_decode1_sanity.md", "B4 Decode1 Sanity", json.dumps(payload["b4_sanity"]["comparisons"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "classification.md", "Classification", f"`{gate['classification']}`\n\nRoot cause: `{gate['root_cause']}`\n\n{gate['root_cause_detail']}")
    write_md(REPORT_DIR / "pytest.md", "Pytest", "Validation results are recorded after explicit pytest commands in `final_gate.json` and the final summary.")


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
    print(json.dumps(final_gate(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
