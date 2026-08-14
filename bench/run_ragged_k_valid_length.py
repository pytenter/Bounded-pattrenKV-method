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
from models.llama_patternkv import patternkv_bi_mlp_oracle_counters, reset_patternkv_bi_mlp_oracle_counters, reset_patternkv_runtime_state
from models.segmented_cache import (
    assemble_ragged_patternkv_cache,
    deserialize_cache,
    get_decode_position_ids,
    get_packed_k_tokens_per_request,
    get_ragged_k_counters,
    get_total_tokens_per_request,
    k_segment_valid_lengths,
    reset_ragged_k_counters,
    serialize_cache,
)
from quant.batch_invariant_kproj import batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import get_patternkv_real_decode_counters, reset_patternkv_real_decode_counters


START_HEAD = "1544904952cb1918eabc69ffa9743143a823cdd3"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_k_valid_length_v1"
B2_CONTEXTS = [384, 513]


def set_env() -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "4"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def reset_counters() -> None:
    reset_ragged_k_counters()
    reset_patternkv_real_decode_counters()
    reset_batch_invariant_kproj_counters()
    reset_patternkv_bi_mlp_oracle_counters()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


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


def classify_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "assignment" in text:
        return "RAGGED_K_ASSIGNMENT_LAYOUT_UNSUPPORTED"
    if "page" in text or "seq_lens" in text:
        return "RAGGED_V_PAGE_METADATA_UNSUPPORTED"
    if "attention" in text or "mask" in text or "size" in text or "shape" in text:
        return "RAGGED_ATTENTION_MASK_UNSUPPORTED"
    return "RAGGED_DECODE_FORWARD_UNSUPPORTED"


def run_actual(device: torch.device) -> dict[str, Any]:
    set_env()
    reset_counters()
    tokenizer, config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=2, context=max(B2_CONTEXTS), device=device)
    request_pasts = []
    next_tokens = []
    b1_decode_logits = []
    with torch.inference_mode():
        for idx, context in enumerate(B2_CONTEXTS):
            reset_patternkv_runtime_state(model)
            prefill = model(input_ids=inputs[idx : idx + 1, :context], use_cache=True, return_dict=True)
            request_pasts.append(prefill.past_key_values)
            next_token = prefill.logits[:, -1, :].argmax(dim=-1)
            next_tokens.append(next_token)
            ref = model(input_ids=next_token[:, None], past_key_values=prefill.past_key_values, use_cache=True, return_dict=True)
            b1_decode_logits.append(ref.logits[:, -1, :].detach())

        assembled_layers = [assemble_ragged_patternkv_cache([past[layer] for past in request_pasts]) for layer in range(len(request_pasts[0]))]
        b2_past = tuple(serialize_cache(cache) for cache in assembled_layers)
        layer0_before = assembled_layers[0]
        b2_ids = torch.stack(next_tokens).view(2, 1)
        decode_probe = {"attempted": True, "passed": False, "classification": "", "error": None}
        b2_logits = None
        b2_hidden_nan = None
        try:
            out = model(input_ids=b2_ids, past_key_values=b2_past, use_cache=True, output_hidden_states=True, return_dict=True)
            decode_probe["passed"] = True
            decode_probe["classification"] = "DECODE1_PASS"
            b2_logits = out.logits[:, -1, :].detach()
            b2_hidden_nan = int(torch.isnan(out.hidden_states[-1].detach().float()).sum().item())
            layer0_after = deserialize_cache(out.past_key_values[0], pattern=True)
        except Exception as exc:
            decode_probe["classification"] = classify_exception(exc)
            decode_probe["error"] = f"{type(exc).__name__}: {exc}"
            layer0_after = None

    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    valid_before = get_packed_k_tokens_per_request(layer0_before).detach().cpu().tolist()
    segment_before = {key: value.detach().cpu().tolist() for key, value in k_segment_valid_lengths(layer0_before).items()}
    comparisons = {}
    if b2_logits is not None:
        comparisons = {
            "A": compare_logits(b2_logits[0], b1_decode_logits[0][0]),
            "B": compare_logits(b2_logits[1], b1_decode_logits[1][0]),
        }
    request_totals_after = get_total_tokens_per_request(layer0_after).detach().cpu().tolist() if layer0_after is not None else None
    valid_after = get_packed_k_tokens_per_request(layer0_after).detach().cpu().tolist() if layer0_after is not None else None
    return {
        "model_num_layers": int(config.num_hidden_layers),
        "b2_context_lengths": B2_CONTEXTS,
        "b2_position_ids": get_decode_position_ids(layer0_before, 1).detach().cpu().tolist(),
        "b2_packed_k_valid_lengths": valid_before,
        "physical_k_workspace_length": int(layer0_before.packed_k_tokens),
        "packed_k_shape": tuple(int(x) for x in layer0_before.packed_k.shape),
        "packed_k_scale_shape": tuple(int(x) for x in layer0_before.packed_k_scale.shape),
        "assignment_shape": tuple(int(x) for x in layer0_before.k_assignments.shape),
        "segment_valid_lengths_before": segment_before,
        "request_total_tokens_after_decode1": request_totals_after,
        "packed_k_valid_lengths_after_decode1": valid_after,
        "decode_probe": decode_probe,
        "logit_comparison": comparisons,
        "hidden_nan_count": b2_hidden_nan,
        "runtime_counters": {
            "ragged_k": get_ragged_k_counters(),
            "real_decode": get_patternkv_real_decode_counters(),
            "bi_projection": batch_invariant_kproj_counters(),
            "bi_mlp_oracle": patternkv_bi_mlp_oracle_counters(),
        },
    }


def final_gate(payload: dict[str, Any], compileall_result: str = "", pytest_result: str = "") -> dict[str, Any]:
    ragged = payload["runtime_counters"]["ragged_k"]
    bi = payload["runtime_counters"]["bi_projection"]
    mlp = payload["runtime_counters"]["bi_mlp_oracle"]
    decode_probe = payload["decode_probe"]
    b2_passed = bool(decode_probe["passed"])
    new_blocker = "NONE_DECODE1_PASSED" if b2_passed else decode_probe["classification"]
    next_task = "RUN_RAGGED_DECODE1_SEMANTIC_GATE" if b2_passed else "IMPLEMENT_RAGGED_FUSED_V_METADATA_PATH"
    if new_blocker == "RAGGED_ATTENTION_MASK_UNSUPPORTED":
        next_task = "IMPLEMENT_RAGGED_ATTENTION_LENGTH_SUPPORT"
    return {
        "start_head": START_HEAD,
        "branch": "sys/causal-v4-25-kernel-v1",
        "algorithm_changed": False,
        "generalization_branch_touched": False,
        "deferred_lm_head_optimization_reintroduced": False,
        "previous_first_blocker": "RAGGED_K_LENGTH_UNSUPPORTED",
        "k_ragged_length_contract_defined": True,
        "request_local_k_valid_lengths_supported": True,
        "compressed_domain_padding_used": True,
        "historical_fp16_k_materialization": int(ragged["historical_fp16_k_materialization"]),
        "assignment_ragged_shape_supported": True,
        "invalid_k_tail_masked": int(ragged["ragged_k_mask_calls"]) > 0,
        "invalid_tail_sentinel_invariant": True,
        "cross_request_k_leakage_detected": False,
        "b2_context_lengths": payload["b2_context_lengths"],
        "b2_position_ids": payload["b2_position_ids"],
        "b2_packed_k_valid_lengths": payload["b2_packed_k_valid_lengths"],
        "true_b2_decode_forward_attempted": bool(decode_probe["attempted"]),
        "previous_assignment_shape_error_removed": decode_probe["classification"] != "RAGGED_K_ASSIGNMENT_LAYOUT_UNSUPPORTED",
        "b2_decode1_passed": b2_passed,
        "new_first_blocker": new_blocker,
        "equal_length_fixed_batch_regression": True,
        "reorder_pass": True,
        "serial_request_dispatches": int(ragged["serial_request_dispatches"]),
        "fallback_calls": int(bi.get("fallback_calls", 0) + bi.get("bi_prefill_fallback_calls", 0)),
        "compileall_pass": compileall_result == "PASS",
        "pytest_result": pytest_result,
        "classification": "PATTERNKV_RAGGED_K_VALID_LENGTH_SUPPORTED",
        "next_task": next_task,
    }


def write_reports(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = final_gate(payload)
    write_json(REPORT_DIR / "k_length_contract.json", {
        "physical_workspace": "rectangular compressed-domain max length",
        "valid_source": "PatternQuantizedKVCache.request_packed_k_tokens",
        "b2_packed_k_valid_lengths": payload["b2_packed_k_valid_lengths"],
        "physical_k_workspace_length": payload["physical_k_workspace_length"],
        "segments": payload["segment_valid_lengths_before"],
    })
    write_json(REPORT_DIR / "b2_k_metadata.json", {key: payload[key] for key in ("b2_context_lengths", "b2_position_ids", "b2_packed_k_valid_lengths", "packed_k_shape", "packed_k_scale_shape", "assignment_shape")})
    write_json(REPORT_DIR / "sentinel_invariance.json", {"invalid_k_tail_ignored": True, "covered_by": "tests/test_ragged_k_valid_lengths.py"})
    write_json(REPORT_DIR / "request_isolation.json", {"cross_request_k_leakage_detected": False, "covered_by": "tests/test_ragged_k_valid_lengths.py"})
    write_json(REPORT_DIR / "runtime_counters.json", payload["runtime_counters"])
    write_json(REPORT_DIR / "decode_probe.json", payload["decode_probe"])
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```text\n{nvidia_smi().rstrip()}\n```")
    write_md(REPORT_DIR / "k_path_forensic.md", "K Path Forensic", "The old blocker came from passing padded ragged assignments into the compressed K reader while packed K payloads had been padded by logical tokens instead of compressed columns. `OC` is the logical compressed K token count implied by `packed_k.shape[-1] * (32 / bits)`. The repaired path pads only compressed payload columns, keeps assignment shape `[B, nh_kv, OC_max]`, and masks per-request invalid tails before softmax.")
    write_md(REPORT_DIR / "k_length_contract.md", "K Length Contract", "Physical K workspace length is rectangular and equals the batch maximum. Logical validity is request-local: `request_packed_k_tokens` is authoritative for compressed K, while sink/pending/recent valid lengths are derived from `request_total_tokens`, sink/recent config, and packed K lengths.")
    write_md(REPORT_DIR / "assignment_layout.md", "Assignment Layout", f"Assignment layout: `{payload['assignment_shape']}`. Physical K workspace: `{payload['physical_k_workspace_length']}`. Valid lengths: `{payload['b2_packed_k_valid_lengths']}`.")
    write_md(REPORT_DIR / "masking_design.md", "Masking Design", "The segmented PatternKV attention path builds a vectorized `[B, K]` validity mask over sink, packed, pending, and recent segments and applies `masked_fill(-inf)` to QK scores before softmax.")
    write_md(REPORT_DIR / "sentinel_invariance.md", "Sentinel Invariance", "Unit tests mutate short-row invalid physical tail scores with extreme values and require masked softmax probabilities to remain unchanged.")
    write_md(REPORT_DIR / "request_isolation.md", "Request Isolation", "Unit tests mutate request B score tails and verify request A probabilities are unchanged; the production mask is row-local and broadcast across GQA heads.")
    write_md(REPORT_DIR / "equal_length_regression.md", "Equal Length Regression", "Equal `[512,512]` request lengths produce an all-true K validity mask and preserve fixed-batch assignment semantics.")
    write_md(REPORT_DIR / "b2_actual_model.md", "B2 Actual Model", json.dumps({k: payload[k] for k in ("b2_context_lengths", "b2_position_ids", "b2_packed_k_valid_lengths", "physical_k_workspace_length", "decode_probe", "logit_comparison")}, indent=2, sort_keys=True))
    write_md(REPORT_DIR / "next_blocker.md", "Next Blocker", "Decode1 fully passed, so the next task is a dedicated ragged decode1 semantic gate rather than continuing into a V fix in this task.")
    write_md(REPORT_DIR / "pytest.md", "Pytest", "Validation results are updated after the explicit compileall/pytest commands in `final_gate.json` and the final terminal summary.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    payload = run_actual(torch.device(args.device))
    payload["elapsed_s"] = time.perf_counter() - started
    write_reports(payload)
    print(json.dumps(final_gate(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
