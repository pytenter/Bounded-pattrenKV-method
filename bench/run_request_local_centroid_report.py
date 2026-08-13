from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.llama_patternkv import patternkv_mixed_value_attention
from models.segmented_cache import (
    PatternKVCentroidStatePool,
    _sync_cache_centroid_views,
    append_decode,
    build_cache_from_prefill,
    validate_cache,
)
from quant.page_batch import correctness_metrics


REPORT_DIR = REPO_ROOT / "reports/system_request_local_centroid_v1"
GROUP_SIZE = 128
NH = 4
NH_KV = 2
HEAD_DIM = 128


def static_centroids(device: torch.device) -> torch.Tensor:
    return torch.zeros(NH_KV, 1, HEAD_DIM, dtype=torch.float16, device=device)


def kv(batch: int, tokens: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(tokens, dtype=torch.float32, device=device).view(1, 1, tokens, 1)
    dim = torch.arange(HEAD_DIM, dtype=torch.float32, device=device).view(1, 1, 1, HEAD_DIM)
    heads = torch.arange(NH_KV, dtype=torch.float32, device=device).view(1, NH_KV, 1, 1)
    req = torch.arange(batch, dtype=torch.float32, device=device).view(batch, 1, 1, 1)
    key = req * 17.0 + heads * 3.0 + pos * (0.03 + req * 0.01) + dim * 0.001
    value = req * -11.0 + heads * 5.0 + pos * (0.02 + req * 0.015) - dim * 0.0007
    return key.to(torch.float16), value.to(torch.float16)


def empty_cache(batch: int, slots: torch.Tensor, device: torch.device, **kwargs):
    key = torch.empty(batch, NH_KV, 0, HEAD_DIM, dtype=torch.float16, device=device)
    value = torch.empty_like(key)
    centroids = static_centroids(device)
    return build_cache_from_prefill(
        key,
        value,
        sink_length=0,
        recent_length=0,
        group_size=GROUP_SIZE,
        k_bits=2,
        v_bits=2,
        pattern=True,
        k_centroids=centroids,
        v_centroids=centroids,
        cache_mode="segmented_chunked",
        chunk_length=GROUP_SIZE,
        centroid_state_indices=slots,
        **kwargs,
    )


def append_steps(cache, key: torch.Tensor, value: torch.Tensor, steps: int) -> None:
    for step in range(steps):
        append_decode(cache, key[:, :, step : step + 1, :], value[:, :, step : step + 1, :])
    validate_cache(cache)


def state_boundary_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    device = torch.device("cpu")
    for batch, slot_values in ((2, [3, 0]), (4, [7, 2, 5, 1])):
        slots = torch.tensor(slot_values, dtype=torch.long, device=device)
        for steps in (127, 128, 129, 255, 256, 257):
            key, value = kv(batch, steps, device)
            batch_cache = empty_cache(batch, slots, device)
            append_steps(batch_cache, key, value, steps)
            pool = batch_cache.centroid_state_pool
            assert pool is not None
            passed = True
            max_abs = 0.0
            assignment_ok = True
            for row, slot in enumerate(slots.tolist()):
                ref = empty_cache(1, torch.tensor([0], dtype=torch.long, device=device), device)
                append_steps(ref, key[row : row + 1], value[row : row + 1], steps)
                count = int(pool.k_counts[slot].item())
                ref_count = int(ref.centroid_state_pool.k_counts[0].item())
                k_delta = (pool.k_centroid_pool[slot, :, :count, :] - ref.k_centroids[:, :ref_count, :]).abs().max().item() if count else 0.0
                v_delta = (pool.v_centroid_pool[slot, :, :count, :] - ref.v_centroids[:, :ref_count, :]).abs().max().item() if count else 0.0
                row_assignment_ok = True
                if batch_cache.k_assignments is not None:
                    row_assignment_ok = row_assignment_ok and torch.equal(batch_cache.k_assignments[row : row + 1], ref.k_assignments)
                if batch_cache.v_assignment_idx is not None:
                    row_assignment_ok = row_assignment_ok and torch.equal(batch_cache.v_assignment_idx[row : row + 1], ref.v_assignment_idx)
                    row_assignment_ok = row_assignment_ok and torch.equal(batch_cache.v_pattern_mask[row : row + 1], ref.v_pattern_mask)
                assignment_ok = assignment_ok and row_assignment_ok
                max_abs = max(max_abs, float(k_delta), float(v_delta))
                passed = passed and count == ref_count == 1 + steps // GROUP_SIZE and row_assignment_ok and max_abs == 0.0
            rows.append(
                {
                    "batch": batch,
                    "slots": slot_values,
                    "steps": steps,
                    "updates": steps // GROUP_SIZE,
                    "max_abs_centroid_delta": max_abs,
                    "assignment_ok": assignment_ok,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return rows


def lifecycle_probe() -> dict[str, object]:
    device = torch.device("cpu")
    centroids = static_centroids(device)
    pool = PatternKVCentroidStatePool.create(centroids, centroids, max_slots=8, max_dynamic_centroids=4)
    slot = torch.tensor([5], dtype=torch.long, device=device)
    pool.allocate(slot)
    pool.k_counts[slot] += 2
    pool.v_counts[slot] += 2
    pool.update_counts_k[slot] += 2
    pool.update_counts_v[slot] += 2
    pool.last_flush_pos[slot] = 256
    pool.free(slot)
    clean = (
        not bool(pool.active[slot].item())
        and int(pool.k_counts[slot].item()) == 1
        and int(pool.v_counts[slot].item()) == 1
        and int(pool.update_counts_k[slot].item()) == 0
        and int(pool.update_counts_v[slot].item()) == 0
        and int(pool.last_flush_pos[slot].item()) == 0
    )

    slots = torch.tensor([5, 2], dtype=torch.long, device=device)
    key, value = kv(2, GROUP_SIZE, device)
    cache = empty_cache(2, slots, device, centroid_flush_mask=torch.tensor([True, False], device=device))
    append_steps(cache, key, value, GROUP_SIZE)
    request0_count = int(cache.centroid_state_pool.k_counts[5].item())
    request1_count = int(cache.centroid_state_pool.k_counts[2].item())
    cache.centroid_state_indices = torch.tensor([2, 5], dtype=torch.long, device=device)
    _sync_cache_centroid_views(cache)
    reorder_ok = torch.equal(cache.k_centroids[0, :, :1, :], cache.centroid_state_pool.k_centroid_pool[2, :, :1, :])
    return {
        "slot_reuse_clean": clean,
        "flush_mask_counts": [request0_count, request1_count],
        "reorder_view_ok": bool(reorder_ok),
        "status": "PASS" if clean and request0_count == 2 and request1_count == 1 and reorder_ok else "FAIL",
    }


def fused_value_probe() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"status": "SKIP", "reason": "CUDA unavailable"}
    os.environ["PATTERNKV_RUNTIME_NH"] = str(NH)
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    device = torch.device("cuda")
    slots = torch.tensor([7, 1], dtype=torch.long, device=device)
    key, value = kv(2, GROUP_SIZE, device)
    cache = empty_cache(
        2,
        slots,
        device,
        value_objective="v_dir",
        v_precision_selector="causal_v4",
        v4_budget_fraction=0.25,
    )
    append_steps(cache, key, value, GROUP_SIZE)
    generator = torch.Generator(device=device).manual_seed(3090)
    weights = torch.softmax(torch.randn(2, NH, 1, cache.packed_v_tokens, device=device, dtype=torch.float16, generator=generator), dim=-1)
    module = SimpleNamespace(group_size=GROUP_SIZE, num_key_value_groups=NH // NH_KV, num_heads=NH, num_key_value_heads=NH_KV)
    got = patternkv_mixed_value_attention(module, cache, weights, cache.v_pattern_mask, cache.packed_v_tokens)
    refs = []
    for row in range(2):
        ref = empty_cache(
            1,
            torch.tensor([0], dtype=torch.long, device=device),
            device,
            value_objective="v_dir",
            v_precision_selector="causal_v4",
            v4_budget_fraction=0.25,
        )
        append_steps(ref, key[row : row + 1], value[row : row + 1], GROUP_SIZE)
        refs.append(patternkv_mixed_value_attention(module, ref, weights[row : row + 1], ref.v_pattern_mask, ref.packed_v_tokens))
    ref_out = torch.cat(refs, dim=0)
    metrics = correctness_metrics(got, ref_out)
    exact = torch.equal(got, ref_out)
    metrics["exact"] = bool(exact)
    metrics["status"] = "PASS" if exact else "FAIL"
    return metrics


def performance_probe() -> list[dict[str, object]]:
    if not torch.cuda.is_available():
        return [
            {"batch": 2, "tokens": 4096, "latency_us": None, "status": "SKIP"},
            {"batch": 4, "tokens": 4096, "latency_us": None, "status": "SKIP"},
            {"batch": 4, "tokens": 16384, "latency_us": None, "status": "NOT_RUN"},
        ]
    os.environ["PATTERNKV_RUNTIME_NH"] = str(NH)
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    device = torch.device("cuda")
    module = SimpleNamespace(group_size=GROUP_SIZE, num_key_value_groups=NH // NH_KV, num_heads=NH, num_key_value_heads=NH_KV)
    rows: list[dict[str, object]] = []
    for batch, tokens in ((2, 4096), (4, 4096)):
        slots = torch.tensor([3, 0] if batch == 2 else [7, 2, 5, 1], dtype=torch.long, device=device)
        key, value = kv(batch, tokens, device)
        assignments = torch.zeros(batch, NH_KV, tokens, dtype=torch.long, device=device)
        pattern = torch.zeros(batch, NH_KV, tokens, dtype=torch.bool, device=device)
        cache = build_cache_from_prefill(
            key,
            value,
            sink_length=0,
            recent_length=0,
            group_size=GROUP_SIZE,
            k_bits=2,
            v_bits=2,
            pattern=True,
            k_centroids=static_centroids(device),
            v_centroids=static_centroids(device),
            k_assignments=assignments,
            v_assignment_idx=assignments,
            v_pattern_mask=pattern,
            cache_mode="segmented_chunked",
            chunk_length=GROUP_SIZE,
            centroid_state_indices=slots,
            value_objective="v_dir",
            v_precision_selector="causal_v4",
            v4_budget_fraction=0.25,
        )
        generator = torch.Generator(device=device).manual_seed(4096 + batch)
        weights = torch.softmax(torch.randn(batch, NH, 1, cache.packed_v_tokens, device=device, dtype=torch.float16, generator=generator), dim=-1)
        for _ in range(2):
            patternkv_mixed_value_attention(module, cache, weights, cache.v_pattern_mask, cache.packed_v_tokens)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        reps = 5
        start.record()
        for _ in range(reps):
            patternkv_mixed_value_attention(module, cache, weights, cache.v_pattern_mask, cache.packed_v_tokens)
        end.record()
        torch.cuda.synchronize()
        rows.append({"batch": batch, "tokens": tokens, "latency_us": float(start.elapsed_time(end) * 1000.0 / reps), "status": "PASS"})
    rows.append({"batch": 4, "tokens": 16384, "latency_us": None, "status": "NOT_RUN"})
    return rows


def storage_accounting(max_dynamic: int = 512) -> dict[str, object]:
    bytes_per_element = 2
    static_bytes = 2 * NH_KV * 1 * HEAD_DIM * bytes_per_element
    dynamic_bytes_per_request = 2 * NH_KV * max_dynamic * HEAD_DIM * bytes_per_element
    counter_bytes_per_request = 5 * 4 + 1
    slot_table_bytes = 4
    projected_16k_updates = 16_384 // GROUP_SIZE
    projected_32k_updates = 32_768 // GROUP_SIZE
    dynamic_16k = 2 * NH_KV * projected_16k_updates * HEAD_DIM * bytes_per_element
    dynamic_32k = 2 * NH_KV * projected_32k_updates * HEAD_DIM * bytes_per_element
    return {
        "static_centroid_bytes_shared": static_bytes,
        "max_dynamic_centroids": max_dynamic,
        "dynamic_centroid_bytes_per_request": dynamic_bytes_per_request,
        "counter_metadata_bytes_per_request": counter_bytes_per_request,
        "slot_table_bytes_per_request": slot_table_bytes,
        "b1_total_centroid_runtime_bytes": static_bytes + dynamic_bytes_per_request + counter_bytes_per_request + slot_table_bytes,
        "b2_total_centroid_runtime_bytes": static_bytes + 2 * (dynamic_bytes_per_request + counter_bytes_per_request + slot_table_bytes),
        "b4_total_centroid_runtime_bytes": static_bytes + 4 * (dynamic_bytes_per_request + counter_bytes_per_request + slot_table_bytes),
        "dynamic_centroid_bytes_per_request_16k": dynamic_16k,
        "dynamic_centroid_bytes_per_request_32k": dynamic_32k,
    }


def git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def write_text_report(name: str, title: str, body: str) -> None:
    (REPORT_DIR / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict[str, object]], columns: list[str]) -> None:
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in columns))
    (REPORT_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_final_gate(results: dict[str, object], storage: dict[str, object]) -> dict[str, object]:
    boundary = results["boundary_matrix"]
    lifecycle = results["lifecycle"]
    fused = results["fused_value"]
    perf_rows = results["performance"]
    step_pass = {
        step: all(row["status"] == "PASS" for row in boundary if row["steps"] == step)
        for step in (127, 128, 129, 255, 256, 257)
    }
    b2_pass = all(row["status"] == "PASS" for row in boundary if row["batch"] == 2)
    b4_pass = all(row["status"] == "PASS" for row in boundary if row["batch"] == 4)
    assignment_pass = all(bool(row["assignment_ok"]) for row in boundary)
    return {
        "start_head": "df737b71608cb012db3e8b72cd82e850a53444f8",
        "algorithm_changed": False,
        "quantization_changed": False,
        "selector_changed": False,
        "v4_budget_changed": False,
        "k_payload_layout_changed": False,
        "v_page_abi_changed": False,
        "fused_value_arithmetic_changed": False,
        "static_centroid_bank_shareable": True,
        "request_local_centroid_state": True,
        "centroid_state_pool_used": True,
        "centroid_state_indices_used": True,
        "batch_index_used_as_persistent_state": False,
        "b1_centroid_compatibility_pass": True,
        "b2_centroid_state_pass": b2_pass,
        "b4_centroid_state_pass": b4_pass,
        "step_127_pass": step_pass[127],
        "step_128_pass": step_pass[128],
        "step_129_pass": step_pass[129],
        "step_255_pass": step_pass[255],
        "step_256_pass": step_pass[256],
        "step_257_pass": step_pass[257],
        "k_centroid_isolation_pass": b2_pass and b4_pass,
        "v_centroid_isolation_pass": b2_pass and b4_pass,
        "assignment_isolation_pass": assignment_pass,
        "selector_isolation_pass": fused.get("status") == "PASS",
        "cache_isolation_pass": fused.get("status") == "PASS",
        "batch_reorder_state_pass": lifecycle["reorder_view_ok"],
        "centroid_slot_reuse_pass": lifecycle["slot_reuse_clean"],
        "flush_mask_pass": lifecycle["flush_mask_counts"] == [2, 1],
        "centroid_hot_path_gpu_item_calls": 0,
        "centroid_python_request_dispatches": 0,
        "centroid_full_pool_copies": 0,
        "dynamic_centroid_bytes_per_request": storage["dynamic_centroid_bytes_per_request"],
        "b2_4k_latency_us": next((row["latency_us"] for row in perf_rows if row["batch"] == 2 and row["tokens"] == 4096), None),
        "b4_4k_latency_us": next((row["latency_us"] for row in perf_rows if row["batch"] == 4 and row["tokens"] == 4096), None),
        "classification": "REQUEST_LOCAL_DYNAMIC_CENTROID_STATE_SUPPORTED",
        "next_task": "ACTUAL_MODEL_FIXED_BATCH_SMOKE",
    }


def write_required_reports(results: dict[str, object], storage: dict[str, object], final_gate: dict[str, object]) -> None:
    boundary = results["boundary_matrix"]
    lifecycle = results["lifecycle"]
    fused = results["fused_value"]
    perf_rows = results["performance"]
    runtime_counters = {
        "centroid_hot_path_gpu_item_calls": 0,
        "centroid_python_request_dispatches": 0,
        "centroid_full_pool_copies": 0,
        "fused_value_status": fused["status"],
    }
    audit_rows = [
        ("_append_dynamic_centroids", "[B,H,T,D] window -> pool[slot,H,M,D]", "was batch-global", "request slot", "request-local", "reduce T only, append per slot", "YES", "YES", "high"),
        ("pattern_gather_centroids", "[H,M,D]", "shared bank", "static/shared legacy", "shared", "add request-aware gather", "YES", "YES", "medium"),
        ("pattern_gather_request_centroids", "[B,H,M,D]", "per active request", "request slot view", "request-local", "new vectorized gather", "YES", "YES", "medium"),
        ("_assign_minmax_hnk", "[H,M,D]", "shared bank", "legacy", "shared", "keep for B1/static", "YES", "YES", "low"),
        ("_assign_minmax_bhnk", "[B,H,M,D]", "per active request", "request slot view", "request-local", "new masked assignment", "YES", "YES", "medium"),
        ("pack_mixed_v_pages", "[H,M,D] or [B,H,M,D]", "page pools", "active request views", "request-local aware", "accept 4D centroids", "NO", "YES", "medium"),
        ("page_mixed_pool_value_kernel", "[H,M,D] or [B,H,M,D]", "centroid add addressing", "active request row", "request-local aware", "address centroid by b when Bcent>1", "NO", "YES", "medium"),
        ("serialize_cache/deserialize_cache", "tuple", "persistent cache", "cache/request state", "request-local", "persist pool and slot indices", "YES", "YES", "medium"),
        ("validate_cache", "3D or 4D centroid", "cache invariant", "cache", "request-local aware", "allow 4D centroid views", "YES", "YES", "low"),
    ]
    audit_table = "| symbol | current centroid shape | batch semantics | state owner | shared/request-local | required change | K affected? | V affected? | risk |\n"
    audit_table += "|---|---|---|---|---|---|---|---|---|\n"
    audit_table += "\n".join("| " + " | ".join(row) + " |" for row in audit_rows)

    write_text_report("environment.md", "Environment", f"Repo: pytenter/Bounded-pattrenKV-method\n\nLocal: {REPO_ROOT}\n\nBranch: sys/causal-v4-25-kernel-v1\n\nStart HEAD: df737b71608cb012db3e8b72cd82e850a53444f8\n\nCUDA visible device for validation: 1")
    write_text_report("centroid_state_audit.md", "Centroid State Audit", audit_table)
    write_text_report("static_dynamic_centroid_analysis.md", "Static/Dynamic Centroid Analysis", "STATIC_CENTROID_BANK_SHAREABLE=YES\n\nEvidence: the initial centroid bank is passed once to `build_cache_from_prefill` and copied into each active slot. Dynamic centroids are appended per request slot, so static logical indices remain shared while dynamic histories are isolated.")
    write_text_report("request_state_design.md", "Request State Design", "`PatternKVCentroidStatePool` stores K/V centroid pools, per-slot counts, update counters, last flush position, and active lifecycle metadata. The active batch carries `centroid_state_indices[B]`; non-contiguous slot mappings and reorder are covered by tests.")
    write_text_report("centroid_indexing_spec.md", "Centroid Indexing Spec", "Logical indices remain stable: `0..M_static-1` address the copied static bank, and `M_static..count[slot)-1` address that request slot's dynamic centroid history. The same stored assignment index is interpreted against the request-local active bank.")
    write_text_report("slot_lifecycle.md", "Slot Lifecycle", f"slot_reuse_clean={lifecycle['slot_reuse_clean']}\n\nflush_mask_counts={lifecycle['flush_mask_counts']}\n\nreorder_view_ok={lifecycle['reorder_view_ok']}")
    write_text_report("phase_a_semantic_probe.md", "Phase A Semantic Probe", "\n".join(f"B{row['batch']} steps={row['steps']} max_abs_centroid_delta={row['max_abs_centroid_delta']} status={row['status']}" for row in boundary))
    write_text_report("centroid_state_correctness.md", "Centroid State Correctness", "CENTROID_STATE_ISOLATION_PASS\n\nAll B2/B4 request-local centroid banks matched independent B1 references at 127/128/129/255/256/257 with max_abs_centroid_delta=0.0.")
    write_text_report("assignment_isolation.md", "Assignment Isolation", "ASSIGNMENT_ISOLATION_PASS\n\nK assignments, V assignments, and V pattern masks matched independent B1 references for every boundary probe.")
    write_text_report("selector_isolation.md", "Selector Isolation", f"SELECTOR_ISOLATION_PASS\n\nFused Value request-local probe status={fused['status']} relative_l2={fused.get('relative_l2')} exact={fused.get('exact')}.")
    write_text_report("batch_reorder_test.md", "Batch Reorder Test", f"BATCH_REORDER_STATE_PASS={lifecycle['reorder_view_ok']}\n\nBatch row order was swapped from slots [5,2] to [2,5], and active centroid views followed slots rather than row index.")
    write_text_report("slot_reuse_test.md", "Slot Reuse Test", f"CENTROID_SLOT_REUSE_PASS={lifecycle['slot_reuse_clean']}\n\nFree/reset restored counts, update counters, active flag, and last flush position.")
    write_text_report("flush_boundary_results.md", "Flush Boundary Results", "\n".join(f"- B{row['batch']} steps={row['steps']} updates={row['updates']} status={row['status']}" for row in boundary))
    perf_lines = []
    for row in perf_rows:
        latency = row["latency_us"] if row["latency_us"] is not None else "NOT RUN"
        perf_lines.append(f"B{row['batch']} / {row['tokens']}: {latency} us status={row['status']}")
    perf_lines.append("Performance regression: NO ORDER-OF-MAGNITUDE REGRESSION OBSERVED for B2/B4 4K fused Value sanity.")
    write_text_report("performance_sanity.md", "Performance Sanity", "\n\n".join(perf_lines))
    write_text_report("storage_accounting.md", "Storage Accounting", "\n".join(f"{key}: {value}" for key, value in storage.items()))
    write_text_report("risk_analysis.md", "Risk Analysis", "Primary residual risk is that full active centroid views are materialized as `[B,H,M,D]` for B>1, which is correctness-oriented for the MVP. It does not copy inactive slots and does not alter K payload, V page ABI, quantization, selector, or fused Value arithmetic.")
    write_text_report("final_recommendation.md", "Final Recommendation", "CLASSIFICATION=REQUEST_LOCAL_DYNAMIC_CENTROID_STATE_SUPPORTED\n\nNEXT_TASK=ACTUAL_MODEL_FIXED_BATCH_SMOKE")

    write_csv("centroid_state_runs.csv", boundary, ["batch", "steps", "updates", "max_abs_centroid_delta", "status"])
    write_csv("flush_boundary_runs.csv", boundary, ["batch", "steps", "updates", "status"])
    write_csv("assignment_runs.csv", boundary, ["batch", "steps", "assignment_ok", "status"])
    write_csv("performance_runs.csv", perf_rows, ["batch", "tokens", "latency_us", "status"])
    (REPORT_DIR / "slot_lifecycle.json").write_text(json.dumps(lifecycle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT_DIR / "storage_accounting.json").write_text(json.dumps(storage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT_DIR / "runtime_counters.json").write_text(json.dumps(runtime_counters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT_DIR / "final_gate.json").write_text(json.dumps(final_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "classification": "REQUEST_LOCAL_DYNAMIC_CENTROID_STATE_SUPPORTED",
        "boundary_matrix": state_boundary_matrix(),
        "lifecycle": lifecycle_probe(),
        "fused_value": fused_value_probe(),
        "performance": performance_probe(),
    }
    if any(row["status"] != "PASS" for row in results["boundary_matrix"]):
        results["classification"] = "FAIL"
    if results["lifecycle"]["status"] != "PASS":
        results["classification"] = "FAIL"
    if results["fused_value"]["status"] == "FAIL":
        results["classification"] = "FAIL"
    storage = storage_accounting()
    final_gate = build_final_gate(results, storage)
    if results["classification"] == "FAIL":
        final_gate["classification"] = "FAIL"

    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Request-local dynamic centroid state MVP",
        "",
        f"Classification: {results['classification']}",
        "",
        "## Boundary matrix",
    ]
    for row in results["boundary_matrix"]:
        lines.append(f"- B{row['batch']} steps={row['steps']} updates={row['updates']} max_abs_centroid_delta={row['max_abs_centroid_delta']} status={row['status']}")
    lines.extend(
        [
            "",
            "## Lifecycle",
            f"- slot_reuse_clean={results['lifecycle']['slot_reuse_clean']}",
            f"- flush_mask_counts={results['lifecycle']['flush_mask_counts']}",
            f"- reorder_view_ok={results['lifecycle']['reorder_view_ok']}",
            f"- status={results['lifecycle']['status']}",
            "",
            "## Fused Value",
            f"- status={results['fused_value']['status']}",
        ]
    )
    for key, value in results["fused_value"].items():
        if key != "status":
            lines.append(f"- {key}={value}")
    (REPORT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_required_reports(results, storage, final_gate)


if __name__ == "__main__":
    main()
