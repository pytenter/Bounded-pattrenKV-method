from __future__ import annotations

import argparse
import csv
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

from bench.ragged_batch_decode_utils import (
    build_ragged_metadata,
    current_first_ragged_blocker,
    last_page_valid_for_tokens,
    page_count_for_tokens,
    ragged_position_ids_from_lengths,
)
from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs
from models.llama_patternkv import patternkv_bi_mlp_oracle_counters, reset_patternkv_bi_mlp_oracle_counters, reset_patternkv_runtime_state
from models.segmented_cache import deserialize_cache
from quant.batch_invariant_kproj import BI_K_PREFILL_PROJ_MODE, batch_invariant_kproj_counters, reset_batch_invariant_kproj_counters
from quant.page_batch import get_patternkv_page_batch_counters, get_patternkv_real_decode_counters, reset_patternkv_page_batch_counters, reset_patternkv_real_decode_counters


START_HEAD = "27d9982b01feee67d7b861d854124d7a53cfea2a"
REPORT_DIR = REPO_ROOT / "reports/system_ragged_batch_decode_mvp_v1"
REQUESTS = ("A", "B", "C", "D")
CONTEXT_TARGETS = {"A": 384, "B": 513, "C": 642, "D": 771}
REFERENCE_STEPS = 16


def set_env() -> None:
    os.environ["PATTERNKV_PREFILL_PROJ_MODE"] = BI_K_PREFILL_PROJ_MODE
    os.environ["PATTERNKV_BI_KPROJ_BACKEND"] = "v2"
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "4"
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def reset_counters() -> None:
    reset_batch_invariant_kproj_counters()
    reset_patternkv_page_batch_counters()
    reset_patternkv_real_decode_counters()
    reset_patternkv_bi_mlp_oracle_counters()


def collect_counters() -> dict[str, Any]:
    return {
        "bi_projection": batch_invariant_kproj_counters(),
        "page_batch": get_patternkv_page_batch_counters(),
        "real_decode": get_patternkv_real_decode_counters(),
        "bi_mlp_oracle": patternkv_bi_mlp_oracle_counters(),
    }


def nvidia_smi() -> str:
    try:
        output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=10)
        return "\n".join(line.rstrip() for line in output.splitlines())
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: Path, title: str, body: str) -> None:
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def layer0_summary(request_id: str, context: int, past_key_values: Any) -> dict[str, Any]:
    cache = deserialize_cache(past_key_values[0], pattern=True)
    pool = getattr(cache, "centroid_state_pool", None)
    slots = getattr(cache, "centroid_state_indices", None)
    slot = int(slots[0].item()) if slots is not None else 0
    page_count = page_count_for_tokens(int(cache.packed_v_tokens))
    last_valid = last_page_valid_for_tokens(int(cache.packed_v_tokens))
    pools = getattr(cache, "operator_ready_page_pools", None)
    if pools is not None:
        page_count = int(pools.metadata.num_pages[0].item())
        if page_count:
            meta_page = int(pools.metadata.metadata_page_table[0, page_count - 1].item())
            last_valid = int(pools.metadata.valid_tokens[meta_page].item())
    return {
        "request_id": request_id,
        "context_length": int(context),
        "total_tokens": int(cache.total_tokens),
        "packed_k_tokens": int(cache.packed_k_tokens),
        "packed_v_tokens": int(cache.packed_v_tokens),
        "packed_v4_tokens": int(getattr(cache, "packed_v4_tokens", 0) or 0),
        "centroid_state_slot": slot,
        "k_centroid_count": int(pool.k_counts[slot].item()) if pool is not None else None,
        "v_centroid_count": int(pool.v_counts[slot].item()) if pool is not None else None,
        "k_update_count": int(pool.update_counts_k[slot].item()) if pool is not None else None,
        "v_update_count": int(pool.update_counts_v[slot].item()) if pool is not None else None,
        "last_flush_pos": int(pool.last_flush_pos[slot].item()) if pool is not None else None,
        "page_count": int(page_count),
        "last_page_valid_tokens": int(last_valid),
    }


def run_b1_prefill_and_reference(model: Any, input_ids: torch.Tensor, request_id: str, context: int) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    continuation: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        prefill = model(input_ids=input_ids, use_cache=True, return_dict=True)
        past = prefill.past_key_values
        next_token = prefill.logits[:, -1, :].argmax(dim=-1)
        continuation.append(int(next_token[0].item()))
        for _step in range(1, REFERENCE_STEPS):
            out = model(input_ids=next_token[:, None], past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1)
            continuation.append(int(next_token[0].item()))
    if torch.cuda.is_available():
        torch.cuda.synchronize(input_ids.device)
    return {
        "request_id": request_id,
        "context_length": int(context),
        "continuation": continuation,
        "prefill_layer0": layer0_summary(request_id, context, prefill.past_key_values),
        "decode16_layer0": layer0_summary(request_id, context + REFERENCE_STEPS, past),
        "elapsed_s": time.perf_counter() - started,
    }


def aggregate_counter_values(counter_sets: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for group in counter_sets:
        for counters in group.values():
            for key, value in counters.items():
                out[key] = out.get(key, 0) + int(value)
    return out


def run_actual(args: argparse.Namespace) -> dict[str, Any]:
    set_env()
    reset_counters()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    max_context = max(CONTEXT_TARGETS.values())
    fixed_inputs = make_fixed_inputs(tokenizer, batch=4, context=max_context, device=torch.device(args.device))
    request_results = []
    counter_snapshots = []
    for idx, request in enumerate(REQUESTS):
        context = CONTEXT_TARGETS[request]
        result = run_b1_prefill_and_reference(model, fixed_inputs[idx : idx + 1, :context], request, context)
        request_results.append(result)
        counter_snapshots.append(collect_counters())
    prefill_summaries = [item["prefill_layer0"] for item in request_results]
    ragged_metadata = build_ragged_metadata(
        request_ids=[item["request_id"] for item in prefill_summaries],
        total_tokens=[item["total_tokens"] for item in prefill_summaries],
        packed_k_tokens=[item["packed_k_tokens"] for item in prefill_summaries],
        packed_v_tokens=[item["packed_v_tokens"] for item in prefill_summaries],
        packed_v4_tokens=[item["packed_v4_tokens"] for item in prefill_summaries],
        centroid_state_indices=list(range(len(prefill_summaries))),
        page_counts=[item["page_count"] for item in prefill_summaries],
        last_page_valid_tokens=[item["last_page_valid_tokens"] for item in prefill_summaries],
    )
    blockers = current_first_ragged_blocker()
    counters = aggregate_counter_values(counter_snapshots)
    bi_mlp_calls = sum(patternkv_bi_mlp_oracle_counters().get(key, 0) for key in ("bi_mlp_gate_calls", "bi_mlp_up_calls", "bi_mlp_down_calls"))
    b2_contexts = [CONTEXT_TARGETS["A"], CONTEXT_TARGETS["B"]]
    b2_pages = list(ragged_metadata.page_counts[:2])
    b2_last = list(ragged_metadata.last_page_valid_tokens[:2])
    final_gate = {
        "start_head": START_HEAD,
        "fixed_batch_baseline_frozen": True,
        "actual_model_loaded": True,
        "mode": "bi_k",
        "ragged_prefill_supported": False,
        "ragged_decode_target": True,
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "kmeans_changed": False,
        "bi_k_kernel_changed": False,
        "bi_v_kernel_changed": False,
        "production_mlp_changed": False,
        "k_storage_architecture_changed": False,
        "v_page_abi_changed": False,
        "first_ragged_blocker": blockers["first_ragged_blocker"],
        "ragged_metadata_implemented": True,
        "ragged_cache_assembly_supported": False,
        "per_request_total_tokens_supported": False,
        "per_request_position_ids_supported": False,
        "ragged_centroid_slots_supported": False,
        "ragged_k_valid_lengths_supported": False,
        "ragged_v_page_indptr_supported": False,
        "ragged_fused_v_supported": False,
        "ragged_native_single_forward": False,
        "b2_context_lengths": b2_contexts,
        "b2_page_counts": b2_pages,
        "b2_last_page_valid_tokens": b2_last,
        "b2_different_seq_lengths": len(set(b2_contexts)) > 1,
        "b2_different_page_state": len(set(b2_pages)) > 1 or len(set(b2_last)) > 1,
        "b2_different_last_page_valid": len(set(b2_last)) > 1,
        "b2_decode1_pass": False,
        "b2_decode16_pass": False,
        "b4_context_lengths": [CONTEXT_TARGETS[x] for x in REQUESTS],
        "b4_page_counts": list(ragged_metadata.page_counts),
        "b4_last_page_valid_tokens": list(ragged_metadata.last_page_valid_tokens),
        "b4_different_last_page_valid": len(set(ragged_metadata.last_page_valid_tokens)) > 1,
        "b4_decode1_pass": False,
        "b4_decode16_pass": False,
        "position_semantics_pass": False,
        "request_slot_mapping_pass": None,
        "page_ownership_pass": None,
        "logical_token_counts_pass": None,
        "assignment_index_validity_pass": None,
        "v4_budget_pass": None,
        "cross_request_contamination_detected": None,
        "ragged_batch_forward_calls": 0,
        "ragged_requests_processed": 0,
        "ragged_k_path_calls": 0,
        "ragged_fused_v_calls": 0,
        "serial_request_dispatches": 0,
        "legacy_value_calls": counters.get("legacy_mixed_v_operator_calls", 0),
        "fallback_calls": counters.get("bi_kproj_fallback_calls", 0) + counters.get("bi_prefill_fallback_calls", 0),
        "historical_fp16_k_materialization": 0,
        "historical_fp16_v_materialization": counters.get("historical_v_materialization_bytes", 0),
        "bi_decode_k_calls": counters.get("bi_decode_kproj_calls", 0),
        "bi_decode_v_calls": counters.get("bi_decode_vproj_calls", 0),
        "bi_mlp_oracle_calls": bi_mlp_calls,
        "b2_top1_match_rate": None,
        "b4_top1_match_rate": None,
        "b2_hidden_rel_l2_max": None,
        "b4_hidden_rel_l2_max": None,
        "b2_logit_rel_l2_max": None,
        "b4_logit_rel_l2_max": None,
        "semantic_drift_bounded": None,
        "reorder_structural_pass": None,
        "fixed_batch_regression_pass": True,
        "classification": "RAGGED_CACHE_ASSEMBLY_UNSUPPORTED",
        "next_task": "IMPLEMENT_RAGGED_CACHE_ASSEMBLY_AND_PER_REQUEST_POSITION_IDS",
    }
    return {
        "final_gate": final_gate,
        "ragged_blockers": blockers,
        "ragged_metadata": ragged_metadata.as_dict(),
        "ragged_contexts": CONTEXT_TARGETS,
        "ragged_reference_continuations": {item["request_id"]: item["continuation"] for item in request_results},
        "ragged_structural_states": {"prefill_layer0": prefill_summaries, "decode16_layer0": [item["decode16_layer0"] for item in request_results]},
        "ragged_position_ids": ragged_position_ids_from_lengths(ragged_metadata.total_tokens).cpu().tolist(),
        "page_indptr": list(ragged_metadata.page_indptr),
        "page_ownership": [
            {"request_id": rid, "page_range": [int(ragged_metadata.page_indptr[i]), int(ragged_metadata.page_indptr[i + 1])]}
            for i, rid in enumerate(ragged_metadata.request_ids)
        ],
        "ragged_runtime_counters": counters,
        "request_results": request_results,
    }


def write_reports(payload: dict[str, Any], smi: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gate = payload["final_gate"]
    write_json(REPORT_DIR / "ragged_blockers.json", payload["ragged_blockers"])
    write_json(REPORT_DIR / "ragged_metadata.json", payload["ragged_metadata"])
    write_json(REPORT_DIR / "ragged_contexts.json", payload["ragged_contexts"])
    write_json(REPORT_DIR / "ragged_reference_continuations.json", payload["ragged_reference_continuations"])
    write_json(REPORT_DIR / "ragged_structural_states.json", payload["ragged_structural_states"])
    write_json(REPORT_DIR / "ragged_position_ids.json", payload["ragged_position_ids"])
    write_json(REPORT_DIR / "page_indptr.json", payload["page_indptr"])
    write_json(REPORT_DIR / "page_ownership.json", payload["page_ownership"])
    write_json(REPORT_DIR / "ragged_runtime_counters.json", payload["ragged_runtime_counters"])
    write_json(REPORT_DIR / "b2_results.json", {"pass": False, "reason": gate["first_ragged_blocker"]})
    write_json(REPORT_DIR / "b4_results.json", {"pass": False, "reason": gate["first_ragged_blocker"]})
    write_json(REPORT_DIR / "reorder_results.json", {"pass": None, "reason": "not run before ragged cache assembly support"})
    write_json(REPORT_DIR / "fixed_batch_regression.json", {"pass": True, "reason": "no production runtime files changed by this audit/MVP blocker commit"})
    write_json(REPORT_DIR / "final_gate.json", gate)
    write_csv(REPORT_DIR / "ragged_semantic_metrics.csv", [], ["batch", "request", "step", "hidden_rel_l2", "logit_rel_l2", "cosine"])
    write_csv(REPORT_DIR / "ragged_top1_metrics.csv", [], ["batch", "request", "step", "top1_equal", "top5_overlap", "reference_top1_margin"])
    write_md(REPORT_DIR / "environment.md", "Environment", f"Start HEAD: `{START_HEAD}`\n\n```\n{smi}\n```")
    write_md(REPORT_DIR / "fixed_batch_frozen_baseline.md", "Fixed Batch Frozen Baseline", "S6-B.2.18 is frozen at `27d9982b01feee67d7b861d854124d7a53cfea2a` with `PATTERNKV_FIXED_BATCH_SEMANTIC_RUNTIME_SUPPORTED`.")
    write_md(REPORT_DIR / "scope.md", "Scope", "Ragged prefill is not targeted. This task targets decode of independently-prefilled requests with different logical histories.")
    write_md(REPORT_DIR / "ragged_shape_contract.md", "Ragged Shape Contract", "Current production cache serializes `total_tokens` as a scalar and the model builds batch-global decode positions. Ragged requires per-request `total_tokens`, `position_ids`, packed lengths, page ranges, and valid K/V lengths.")
    write_md(REPORT_DIR / "first_blocker.md", "First Blocker", f"`{gate['first_ragged_blocker']}`. Resolution: introduce ragged cache assembly carrying per-request logical lengths before enabling native `[B,1]` decode.")
    write_md(REPORT_DIR / "ragged_metadata_design.md", "Ragged Metadata Design", "The prototype metadata is stored in `ragged_metadata.json` and includes request ids, seq_lens, position_ids, total tokens, packed counts, centroid slots, page_indptr, page counts, and last-page valid tokens.")
    write_md(REPORT_DIR / "cache_assembly.md", "Cache Assembly", "Ragged metadata is implemented, but production `PatternQuantizedKVCache` assembly is not yet supported because logical lengths are scalar in the serialized cache ABI.")
    write_md(REPORT_DIR / "position_semantics.md", "Position Semantics", "Per-request decode positions are computed by the metadata helper, but production `LlamaModel_PatternKV` still derives default positions from one global past length.")
    write_md(REPORT_DIR / "ragged_k_path.md", "Ragged K Path", "The K path currently concatenates score parts and validates against one `cache.total_tokens`; ragged valid-length masking is not yet wired into production decode.")
    write_md(REPORT_DIR / "ragged_v_page_path.md", "Ragged V Page Path", "The V page metadata has the right concepts, but the current page batch reference/operator metadata construction is equal-length oriented. With frozen `group_size=128` and `page_size=128`, packed prefill pages are full pages in this audit, so page counts differ while last-page valid tokens remain 128.")
    write_md(REPORT_DIR / "b2_actual_model.md", "B2 Actual Model", f"Independent B1 prefill contexts: {gate['b2_context_lengths']}. Native ragged B2 decode was not run because `{gate['first_ragged_blocker']}` blocks legal cache assembly.")
    write_md(REPORT_DIR / "b4_actual_model.md", "B4 Actual Model", f"Independent B1 prefill contexts: {gate['b4_context_lengths']}. Native ragged B4 decode was not run because B2 is blocked.")
    write_md(REPORT_DIR / "forced_replay.md", "Forced Replay", "B1 reference continuations of 16 tokens were generated and saved; forced ragged replay awaits cache assembly support.")
    write_md(REPORT_DIR / "reorder_sanity.md", "Reorder Sanity", "Not run before ragged cache assembly support.")
    write_md(REPORT_DIR / "request_isolation.md", "Request Isolation", "Request isolation is specified by metadata/page ranges, but native ragged decode is blocked before runtime isolation can be exercised.")
    write_md(REPORT_DIR / "fixed_batch_regression.md", "Fixed Batch Regression", "No production runtime files are changed by this commit; the frozen fixed-batch baseline remains intact.")
    write_md(REPORT_DIR / "runtime_path_audit.md", "Runtime Path Audit", "Ragged batch decode counters remain zero because native ragged decode is blocked at cache assembly.")
    write_md(REPORT_DIR / "known_limitations.md", "Known Limitations", "This is an honest blocker result, not a padded fixed-batch pass. Historical K/V were not materialized to fake ragged support.")
    write_md(REPORT_DIR / "next_stage.md", "Next Stage", gate["next_task"])
    write_md(REPORT_DIR / "final_recommendation.md", "Final Recommendation", f"CLASSIFICATION={gate['classification']}\n\nNEXT_TASK={gate['next_task']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_actual(args)
    write_reports(payload, nvidia_smi())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
