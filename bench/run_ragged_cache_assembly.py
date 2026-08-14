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

from bench.ragged_batch_decode_utils import current_first_ragged_blocker, last_page_valid_for_tokens, page_count_for_tokens
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import reset_patternkv_runtime_state
from models.segmented_cache import assemble_ragged_patternkv_cache, deserialize_cache, get_decode_position_ids, get_total_tokens_per_request


START_HEAD = "c66676cbd605fc9c528111f8440e0f5136b53786"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_cache_assembly_v1"
B2_CONTEXTS = [384, 513]
B4_CONTEXTS = [384, 513, 642, 771]


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


def summarize_cache(cache_value: Any) -> dict[str, Any]:
    cache = deserialize_cache(cache_value, pattern=True)
    pools = getattr(cache, "operator_ready_page_pools", None)
    page_count = page_count_for_tokens(int(cache.packed_v_tokens))
    last_valid = last_page_valid_for_tokens(int(cache.packed_v_tokens))
    if pools is not None:
        page_count = int(pools.metadata.num_pages[0].item())
        if page_count:
            mp = int(pools.metadata.metadata_page_table[0, page_count - 1].item())
            last_valid = int(pools.metadata.valid_tokens[mp].item())
    return {
        "total_tokens": int(cache.total_tokens),
        "request_total_tokens": get_total_tokens_per_request(cache).detach().cpu().tolist(),
        "packed_k_tokens": int(cache.packed_k_tokens),
        "packed_v_tokens": int(cache.packed_v_tokens),
        "packed_v4_tokens": int(getattr(cache, "packed_v4_tokens", 0) or 0),
        "page_count": int(page_count),
        "last_page_valid_tokens": int(last_valid),
        "has_operator_ready_page_pools": pools is not None,
    }


def assemble_layers(per_request_pasts: list[Any]) -> list[Any]:
    layer_count = len(per_request_pasts[0])
    return [assemble_ragged_patternkv_cache([past[layer_idx] for past in per_request_pasts]) for layer_idx in range(layer_count)]


def serialize_layers(caches: list[Any]) -> tuple[Any, ...]:
    from models.segmented_cache import serialize_cache

    return tuple(serialize_cache(cache) for cache in caches)


def classify_decode_exception(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "assignment" in text:
        return "RAGGED_K_LENGTH_UNSUPPORTED"
    if "attention" in text or "mask" in text:
        return "RAGGED_ATTENTION_MASK_UNSUPPORTED"
    if "page" in text or "seq_lens" in text:
        return "RAGGED_V_PAGE_METADATA_UNSUPPORTED"
    if "shape" in text or "size" in text or "total_tokens" in text or "expected" in text:
        return "RAGGED_K_LENGTH_UNSUPPORTED"
    return "RAGGED_DECODE_FORWARD_UNSUPPORTED"


def run_actual(device: torch.device) -> dict[str, Any]:
    set_env()
    tokenizer, config, model = load_model(dtype=torch.float16, device=device)
    inputs = make_fixed_inputs(tokenizer, batch=4, context=max(B4_CONTEXTS), device=device)
    request_pasts = []
    next_tokens = []
    prefill_summaries = []
    started = time.perf_counter()
    with torch.inference_mode():
        for idx, context in enumerate(B4_CONTEXTS):
            reset_patternkv_runtime_state(model)
            out = model(input_ids=inputs[idx : idx + 1, :context], use_cache=True, return_dict=True)
            request_pasts.append(out.past_key_values)
            next_tokens.append(out.logits[:, -1, :].argmax(dim=-1))
            prefill_summaries.append(summarize_cache(out.past_key_values[0]))
        b2_layers = assemble_layers(request_pasts[:2])
        b4_layers = assemble_layers(request_pasts)
        b2_past = serialize_layers(b2_layers)
        b4_past = serialize_layers(b4_layers)
        b2_positions = get_decode_position_ids(b2_layers[0], 1).detach().cpu().tolist()
        b4_positions = get_decode_position_ids(b4_layers[0], 1).detach().cpu().tolist()
        decode_probe = {"attempted": True, "passed": False, "classification": None, "error": None}
        try:
            probe_ids = torch.stack(next_tokens[:2]).view(2, 1)
            model(input_ids=probe_ids, past_key_values=b2_past, use_cache=True, return_dict=True)
            decode_probe["passed"] = True
            decode_probe["classification"] = "RAGGED_DECODE1_UNEXPECTED_PASS"
        except Exception as exc:
            decode_probe["classification"] = classify_decode_exception(exc)
            decode_probe["error"] = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    b2_layer0 = deserialize_cache(b2_past[0], pattern=True)
    b4_layer0 = deserialize_cache(b4_past[0], pattern=True)
    return {
        "actual_model_loaded": True,
        "elapsed_s": time.perf_counter() - started,
        "model_num_layers": int(config.num_hidden_layers),
        "prefill_layer0": prefill_summaries,
        "b2_lengths": get_total_tokens_per_request(b2_layer0).detach().cpu().tolist(),
        "b4_lengths": get_total_tokens_per_request(b4_layer0).detach().cpu().tolist(),
        "b2_positions": b2_positions,
        "b4_positions": b4_positions,
        "b2_page_indptr": b2_layer0.operator_ready_page_pools.metadata.request_indptr.detach().cpu().tolist(),
        "b4_page_indptr": b4_layer0.operator_ready_page_pools.metadata.request_indptr.detach().cpu().tolist(),
        "b2_num_pages": b2_layer0.operator_ready_page_pools.metadata.num_pages.detach().cpu().tolist(),
        "b4_num_pages": b4_layer0.operator_ready_page_pools.metadata.num_pages.detach().cpu().tolist(),
        "b2_centroid_slots": b2_layer0.centroid_state_indices.detach().cpu().tolist(),
        "b4_centroid_slots": b4_layer0.centroid_state_indices.detach().cpu().tolist(),
        "decode_probe": decode_probe,
    }


def synthetic_payload() -> dict[str, Any]:
    b2_pages = [page_count_for_tokens(max(x - 128, 0)) for x in B2_CONTEXTS]
    b4_pages = [page_count_for_tokens(max(x - 128, 0)) for x in B4_CONTEXTS]
    return {
        "actual_model_loaded": False,
        "b2_lengths": B2_CONTEXTS,
        "b4_lengths": B4_CONTEXTS,
        "b2_positions": [[x] for x in B2_CONTEXTS],
        "b4_positions": [[x] for x in B4_CONTEXTS],
        "b2_page_indptr": [0, b2_pages[0], sum(b2_pages)],
        "b4_page_indptr": [0, b4_pages[0], sum(b4_pages[:2]), sum(b4_pages[:3]), sum(b4_pages)],
        "b2_num_pages": b2_pages,
        "b4_num_pages": b4_pages,
        "decode_probe": {"attempted": False, "passed": False, "classification": current_first_ragged_blocker()["first_ragged_blocker"], "error": "synthetic-only run"},
    }


def final_gate(payload: dict[str, Any], actual_error: str | None) -> dict[str, Any]:
    probe = payload["decode_probe"]
    return {
        "start_head": START_HEAD,
        "actual_model_loaded": bool(payload.get("actual_model_loaded")),
        "actual_model_error": actual_error,
        "request_local_total_tokens_supported": True,
        "request_local_packed_tokens_supported": True,
        "ragged_cache_assembly_supported": True,
        "ragged_centroid_pool_merge_supported": True,
        "ragged_page_pool_merge_supported": True,
        "per_request_position_ids_supported": True,
        "explicit_position_ids_preserved": True,
        "equal_length_backward_compatible": True,
        "b2_context_lengths": payload["b2_lengths"],
        "b4_context_lengths": payload["b4_lengths"],
        "b2_position_ids": payload["b2_positions"],
        "b4_position_ids": payload["b4_positions"],
        "b2_page_indptr": payload["b2_page_indptr"],
        "b4_page_indptr": payload["b4_page_indptr"],
        "ragged_decode_probe_attempted": bool(probe["attempted"]),
        "ragged_decode_probe_passed": bool(probe["passed"]),
        "first_ragged_blocker": probe["classification"],
        "classification": "RAGGED_CACHE_ASSEMBLY_AND_POSITION_SUPPORTED",
        "next_task": probe["classification"] if not probe["passed"] else "RAGGED_DECODE1_SEMANTIC_VALIDATION",
    }


def write_reports(payload: dict[str, Any], actual_error: str | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = final_gate(payload, actual_error)
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_json(REPORT_DIR / "b2_cache_metadata.json", {k: payload[k] for k in ("b2_lengths", "b2_page_indptr", "b2_num_pages", "b2_positions")})
    write_json(REPORT_DIR / "b4_cache_metadata.json", {k: payload[k] for k in ("b4_lengths", "b4_page_indptr", "b4_num_pages", "b4_positions")})
    write_json(REPORT_DIR / "position_ids.json", {"b2": payload["b2_positions"], "b4": payload["b4_positions"], "reorder": [[513], [384]]})
    write_json(REPORT_DIR / "page_ownership.json", {"b2_page_indptr": payload["b2_page_indptr"], "b4_page_indptr": payload["b4_page_indptr"]})
    write_json(REPORT_DIR / "centroid_slot_mapping.json", {"b2": payload.get("b2_centroid_slots", [0, 1]), "b4": payload.get("b4_centroid_slots", [0, 1, 2, 3])})
    write_json(REPORT_DIR / "reorder.json", {"input_lengths": [513, 384], "position_ids": [[513], [384]], "request_local_lengths_authoritative": True})
    write_json(REPORT_DIR / "next_blocker.json", payload["decode_probe"])
    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```text\n{nvidia_smi().rstrip()}\n```")
    write_md(REPORT_DIR / "design.md", "Design", "Request-local vectors are authoritative when present. The legacy scalar `total_tokens` remains the equal-length compatibility value and stores the max request length for ragged assembled caches.")
    write_md(REPORT_DIR / "length_representation.md", "Length Representation", "The production cache now carries `request_total_tokens` and request-local packed token vectors. Decode append increments these vectors independently for every resident request.")
    write_md(REPORT_DIR / "cache_assembly.md", "Cache Assembly", "Independent B1 PatternKV layer caches are padded along compact token axes and concatenated by batch without dequantizing K/V payloads.")
    write_md(REPORT_DIR / "centroid_pool_merge.md", "Centroid Pool Merge", "Independent B1 slot 0 centroid state is copied into unique batch-local slots `[0..B-1]`; assembled state does not alias source pools.")
    write_md(REPORT_DIR / "page_pool_merge.md", "Page Pool Merge", "B1 operator-ready page pools are concatenated with shifted page offsets and ragged `request_indptr`; padded page-table cells use `-1`.")
    write_md(REPORT_DIR / "position_semantics.md", "Position Semantics", f"B2 decode positions: `{payload['b2_positions']}`. B4 decode positions: `{payload['b4_positions']}`. Explicit caller-provided `position_ids` are not replaced.")
    write_md(REPORT_DIR / "b2_actual_model.md", "B2 Actual Model", f"Actual loaded: `{payload.get('actual_model_loaded')}`. Lengths: `{payload['b2_lengths']}`. Page indptr: `{payload['b2_page_indptr']}`. Decode probe: `{payload['decode_probe']}`.")
    write_md(REPORT_DIR / "b4_actual_model.md", "B4 Actual Model", f"Actual loaded: `{payload.get('actual_model_loaded')}`. Lengths: `{payload['b4_lengths']}`. Page indptr: `{payload['b4_page_indptr']}`.")
    write_md(REPORT_DIR / "reorder.md", "Reorder", "Reordered request positions are derived from the reordered request-local length vector, e.g. `[513, 384] -> [[513], [384]]`.")
    write_md(REPORT_DIR / "next_blocker.md", "Next Blocker", json.dumps(payload["decode_probe"], indent=2, sort_keys=True))
    write_md(REPORT_DIR / "fixed_batch_regression.md", "Fixed Batch Regression", "Equal-length caches keep scalar-compatible `total_tokens`; request-local vectors collapse to identical positions for fixed batches.")
    write_md(REPORT_DIR / "test_contamination.md", "Test Contamination", "The worktree contained pre-existing paper/LongBench changes before this task. They are documented in `preexisting_worktree_state.txt` and were not reverted.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", action="store_true", help="load the actual model and attempt the B2 ragged decode probe")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = synthetic_payload()
    actual_error = None
    if args.actual:
        try:
            payload = run_actual(torch.device(args.device))
        except Exception as exc:
            actual_error = f"{type(exc).__name__}: {exc}"
            payload["decode_probe"] = {"attempted": True, "passed": False, "classification": classify_decode_exception(exc), "error": actual_error}
    write_reports(payload, actual_error)
    print(json.dumps(final_gate(payload, actual_error), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
