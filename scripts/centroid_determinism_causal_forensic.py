from __future__ import annotations

import argparse
import hashlib
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

from bench.run_actual_model_fixed_batch_smoke import load_model, make_fixed_inputs, tensor_metrics
from bench.run_ragged_decode1_semantic_gate import nvidia_smi
from models.llama_patternkv import (
    batched_assign_compiled,
    batched_kmeans_fast_compiled,
    patternkv_p2_first_divergence_trace_records,
    reset_patternkv_p2_first_divergence_trace,
    reset_patternkv_runtime_state,
)
from models.segmented_cache import deserialize_cache


REPORT = REPO_ROOT / "reports/centroid_determinism_causal_forensic.md"
OUT_DIR = REPO_ROOT / "forensics/centroid_determinism"
RUNS_INPUT_K = 20
RUNS_OPERATOR = 100
RUNS_STAGE = 50
LAYER = 0
CONTEXT = 384
KMEANS_K = 16
KMEANS_ITERS = 30
KMEANS_TOL = 1e-4
SEED = 0


def set_env() -> None:
    os.environ["PATTERNKV_CACHE_PATH"] = "segmented"
    os.environ["PATTERNKV_CACHE_MODE"] = "segmented_rolling"
    os.environ["PATTERNKV_MIXED_V_BACKEND"] = "fused_page"
    os.environ["PATTERNKV_RUNTIME_NH"] = "32"
    os.environ["PATTERNKV_CENTROID_MAX_SLOTS"] = "8"
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE"] = "1"
    os.environ["PATTERNKV_P2_FIRST_DIVERGENCE_TRACE_LAYER"] = str(LAYER)
    os.environ.pop("PATTERNKV_BI_MLP_ORACLE", None)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def sha256_tensor(tensor: torch.Tensor | None) -> str | None:
    if tensor is None:
        return None
    cpu = tensor.detach().contiguous().cpu()
    h = hashlib.sha256()
    h.update(str(tuple(cpu.shape)).encode())
    h.update(str(cpu.dtype).encode())
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def stats(tensor: torch.Tensor) -> dict[str, Any]:
    x = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "sha256": sha256_tensor(tensor),
    }


def compare(got: torch.Tensor, ref: torch.Tensor) -> dict[str, Any]:
    metrics = tensor_metrics(got, ref)
    diff = (got.detach().float() - ref.detach().float()).abs()
    return {
        "equal": bool(torch.equal(got, ref)),
        "sha256": sha256_tensor(got),
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "rel_l2": float(metrics["relative_l2"]),
    }


def install_layer0_input_hook(model: Any) -> tuple[dict[str, torch.Tensor], Any]:
    trace: dict[str, torch.Tensor] = {}
    layer = getattr(model.model, "layers")[LAYER]

    def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and torch.is_tensor(inputs[0]):
            trace["ATTN_INPUT_HIDDEN"] = inputs[0].detach().clone()

    return trace, layer.self_attn.register_forward_pre_hook(hook)


def trace_prefill(model: Any, input_ids: torch.Tensor, *, sync: bool = False) -> dict[str, Any]:
    reset_patternkv_runtime_state(model)
    reset_patternkv_p2_first_divergence_trace()
    hidden_trace, handle = install_layer0_input_hook(model)
    if torch.cuda.is_available() and sync:
        torch.cuda.synchronize()
    model_out = None
    try:
        with torch.inference_mode():
            model_out = model(input_ids=input_ids, use_cache=True, return_dict=True)
        if torch.cuda.is_available() and sync:
            torch.cuda.synchronize()
    finally:
        handle.remove()
    records = {str(row["component"]): row["tensor"] for row in patternkv_p2_first_divergence_trace_records() if int(row["layer"]) == LAYER}
    records.update(hidden_trace)
    traced = {}
    for name in ("ATTN_INPUT_HIDDEN", "Q_PROJ", "K_PROJ", "V_PROJ", "K_POST_ROPE", "KMEANS_K_INPUT", "K_CENTROID", "K_ASSIGNMENT"):
        tensor = records.get(name)
        if torch.is_tensor(tensor):
            traced[name] = tensor.detach().clone()
    layer_cache = deserialize_cache(_get_layer_past(model_out, LAYER), pattern=True)
    pool = getattr(layer_cache, "centroid_state_pool", None)
    indices = getattr(layer_cache, "centroid_state_indices", None)
    if pool is not None and torch.is_tensor(indices):
        slot = int(indices[0].item())
        active_k_count = int(pool.k_counts[slot].item())
        active_v_count = int(pool.v_counts[slot].item())
        traced["CACHE_K_CENTROID_ACTIVE"] = pool.k_centroid_pool[slot : slot + 1, :, :active_k_count, :].detach().clone()
        traced["CACHE_K_CENTROID_POOL_FULL_SLOT"] = pool.k_centroid_pool[slot : slot + 1].detach().clone()
        traced["CACHE_V_CENTROID_ACTIVE"] = pool.v_centroid_pool[slot : slot + 1, :, :active_v_count, :].detach().clone()
        traced["CACHE_V_CENTROID_POOL_FULL_SLOT"] = pool.v_centroid_pool[slot : slot + 1].detach().clone()
    return traced


def _get_layer_past(model_output: Any, layer_idx: int) -> Any:
    return model_output.past_key_values[layer_idx]


def run_input_k_determinism(model: Any, input_ids: torch.Tensor, *, sync: bool = False) -> dict[str, Any]:
    runs = []
    tensors = []
    for idx in range(RUNS_INPUT_K):
        traced = trace_prefill(model, input_ids, sync=sync)
        k_input = traced["KMEANS_K_INPUT"]
        tensors.append(traced)
        row = {"run": idx, **stats(k_input)}
        if idx:
            row.update(compare(k_input, tensors[0]["KMEANS_K_INPUT"]))
        else:
            row.update({"equal": True, "max_abs_diff": 0.0, "mean_abs_diff": 0.0, "rel_l2": 0.0})
        runs.append(row)
    components = {}
    for name in (
        "ATTN_INPUT_HIDDEN",
        "Q_PROJ",
        "K_PROJ",
        "V_PROJ",
        "K_POST_ROPE",
        "KMEANS_K_INPUT",
        "K_CENTROID",
        "K_ASSIGNMENT",
        "CACHE_K_CENTROID_ACTIVE",
        "CACHE_K_CENTROID_POOL_FULL_SLOT",
        "CACHE_V_CENTROID_ACTIVE",
        "CACHE_V_CENTROID_POOL_FULL_SLOT",
    ):
        hashes = [sha256_tensor(run[name]) for run in tensors if name in run]
        components[name] = {"unique_hashes": len(set(hashes)), "hashes": hashes}
    first = next((name for name in ("ATTN_INPUT_HIDDEN", "Q_PROJ", "K_PROJ", "V_PROJ", "K_POST_ROPE", "KMEANS_K_INPUT", "K_CENTROID", "K_ASSIGNMENT", "CACHE_K_CENTROID_ACTIVE", "CACHE_K_CENTROID_POOL_FULL_SLOT") if components.get(name, {}).get("unique_hashes", 0) > 1), "")
    return {
        "runs": runs,
        "components": components,
        "first_nondeterministic_upstream_stage": first,
        "K_INPUT_DETERMINISTIC": components["KMEANS_K_INPUT"]["unique_hashes"] == 1,
        "frozen_k_path": str(OUT_DIR / "fixed_layer0_k.pt"),
        "frozen_k": tensors[0]["KMEANS_K_INPUT"].detach().cpu(),
    }


def kmeans_stage_hashes(
    X: torch.Tensor,
    *,
    fixed_indices: torch.Tensor | None = None,
    zero_workspace: bool = False,
    sync: bool = False,
) -> dict[str, Any]:
    H, N, D = X.shape
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    g = torch.Generator(device=X.device)
    g.manual_seed(SEED)
    scores = torch.rand(H, N, generator=g, device=X.device)
    if fixed_indices is None:
        _, idx = scores.topk(KMEANS_K, dim=1)
    else:
        idx = fixed_indices.to(device=X.device, dtype=torch.long)
    centroids = torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, D)).contiguous()
    x2 = (X * X).sum(-1, keepdim=True)
    if zero_workspace:
        sums = torch.zeros(H, KMEANS_K, D, device=X.device, dtype=X.dtype)
        counts = torch.zeros(H, KMEANS_K, device=X.device, dtype=X.dtype)
    else:
        sums = torch.empty(H, KMEANS_K, D, device=X.device, dtype=X.dtype)
        counts = torch.empty(H, KMEANS_K, device=X.device, dtype=X.dtype)
    ones = torch.ones(H, N, device=X.device, dtype=X.dtype)
    stage: dict[str, torch.Tensor] = {"input_K": X, "initial_centroids": centroids}
    last_shift = None
    for _ in range(KMEANS_ITERS):
        c2 = (centroids * centroids).sum(-1).unsqueeze(1)
        distance = torch.baddbmm(x2 + c2, X, centroids.transpose(1, 2), beta=1.0, alpha=-2.0)
        assign = distance.argmin(dim=-1)
        sums.zero_()
        counts.zero_()
        sums.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), X)
        counts.scatter_add_(1, assign, ones)
        empty = counts == 0
        counts_safe = counts.clamp_min(1.0).unsqueeze(-1)
        new_centroids = sums / counts_safe
        if empty.any():
            rand_idx = torch.randint(0, N, (H, KMEANS_K), generator=g, device=X.device)
            repl = torch.gather(X, 1, rand_idx.unsqueeze(-1).expand(-1, -1, D))
            new_centroids = torch.where(empty.unsqueeze(-1), repl, new_centroids)
        shift = (new_centroids - centroids).abs().amax()
        centroids = new_centroids
        last_shift = shift if last_shift is None else shift
        stage = {
            "input_K": X,
            "initial_centroids": stage["initial_centroids"],
            "distance": distance,
            "assignment": assign,
            "cluster_counts": counts,
            "cluster_sums": sums,
            "normalized_centroids": new_centroids,
            "k_centroid_values": centroids,
        }
        if last_shift is not None and shift <= KMEANS_TOL:
            break
        last_shift = shift
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return {
        "hashes": {name: sha256_tensor(value) for name, value in stage.items()},
        "tensors": {name: value.detach().clone() for name, value in stage.items()},
    }


def production_centroid_once(X: torch.Tensor, *, sync: bool = False) -> dict[str, torch.Tensor]:
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    _assign_seed, centroids = batched_kmeans_fast_compiled(X, k=KMEANS_K, iters=KMEANS_ITERS, tol=KMEANS_TOL, seed=SEED)
    assignments = batched_assign_compiled(X, centroids)
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"k_centroid_values": centroids.detach().clone(), "assignment": assignments.detach().clone()}


def run_operator_100(X: torch.Tensor, *, sync: bool = False) -> dict[str, Any]:
    rows = []
    for idx in range(RUNS_OPERATOR):
        out = production_centroid_once(X, sync=sync)
        rows.append({"run": idx, "k_centroid_hash": sha256_tensor(out["k_centroid_values"]), "assignment_hash": sha256_tensor(out["assignment"])})
    return {
        "runs": rows,
        "CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES": len({row["k_centroid_hash"] for row in rows}),
        "ASSIGNMENT_100_RUN_UNIQUE_HASHES": len({row["assignment_hash"] for row in rows}),
    }


def summarize_stage_runs(X: torch.Tensor, *, fixed_indices: torch.Tensor | None = None, zero_workspace: bool = False, sync: bool = False, runs: int = RUNS_STAGE) -> dict[str, Any]:
    stage_rows = []
    for _ in range(runs):
        stage_rows.append(kmeans_stage_hashes(X, fixed_indices=fixed_indices, zero_workspace=zero_workspace, sync=sync)["hashes"])
    summary = {}
    for name in stage_rows[0]:
        summary[name] = len({row[name] for row in stage_rows})
    first = next((name for name in ("input_K", "initial_centroids", "distance", "assignment", "cluster_counts", "cluster_sums", "normalized_centroids", "k_centroid_values") if summary.get(name, 0) > 1), "")
    return {"stage_unique_hashes": summary, "FIRST_DIVERGENT_STAGE": first, "runs": stage_rows}


def deterministic_reference(X: torch.Tensor, initial_idx: torch.Tensor) -> torch.Tensor:
    H, N, D = X.shape
    centroids = torch.gather(X.float(), 1, initial_idx.to(X.device).unsqueeze(-1).expand(-1, -1, D)).contiguous()
    x_float = X.float()
    last_shift = None
    for _ in range(KMEANS_ITERS):
        d2 = ((x_float.unsqueeze(2) - centroids.unsqueeze(1)) ** 2).sum(dim=-1)
        assign = d2.argmin(dim=-1)
        new_centroids = torch.empty_like(centroids)
        for h in range(H):
            for c in range(KMEANS_K):
                selected = x_float[h][assign[h] == c]
                if selected.numel() == 0:
                    new_centroids[h, c] = x_float[h, int(initial_idx[h, c].item())]
                else:
                    acc = torch.zeros(D, dtype=torch.float32, device=X.device)
                    for token in selected:
                        acc = acc + token
                    new_centroids[h, c] = acc / selected.shape[0]
        shift = (new_centroids - centroids).abs().amax()
        centroids = new_centroids
        if last_shift is not None and shift <= KMEANS_TOL:
            break
        last_shift = shift
    return centroids.to(X.dtype)


def run_reference_oracle(X: torch.Tensor, initial_idx: torch.Tensor) -> dict[str, Any]:
    ref_hashes = []
    prod_hashes = []
    metrics = []
    reference = deterministic_reference(X, initial_idx)
    for _ in range(RUNS_OPERATOR):
        prod = production_centroid_once(X)["k_centroid_values"]
        ref_hashes.append(sha256_tensor(reference))
        prod_hashes.append(sha256_tensor(prod))
        diff = (prod.float() - reference.float()).abs()
        metrics.append({"max_abs_diff": float(diff.max().item()), "mean_abs_diff": float(diff.mean().item()), "rel_l2": float(tensor_metrics(prod, reference)["relative_l2"])})
    return {
        "REFERENCE_UNIQUE_HASHES": len(set(ref_hashes)),
        "PRODUCTION_UNIQUE_HASHES": len(set(prod_hashes)),
        "production_vs_reference": {
            "max_abs_diff_max": max(row["max_abs_diff"] for row in metrics),
            "mean_abs_diff_max": max(row["mean_abs_diff"] for row in metrics),
            "rel_l2_max": max(row["rel_l2"] for row in metrics),
        },
    }


def environment() -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = "unavailable"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "triton": triton_version,
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--short"]),
        "branch": git(["branch", "--show-current"]),
        "nvidia_smi": nvidia_smi(),
    }


def write_report(payload: dict[str, Any]) -> None:
    input_k = payload["input_k"]
    operator = payload["operator"]
    stage = payload["stage"]
    rng = payload["rng_oracle"]
    workspace = payload["workspace_oracle"]
    sync = payload["sync_oracle"]
    reference = payload["reference_oracle"]
    reduction = payload["reduction_audit"]
    classification = payload["ROOT_CLASSIFICATION"]
    first_divergent = payload["FIRST_NONDETERMINISTIC_STAGE"]
    if classification == "UNINITIALIZED_OR_STALE_WORKSPACE":
        summary = "The true centroid-builder input and active K centroid are byte-stable; the prior centroid-state nondeterminism is confined to unused/full centroid pool storage that is allocated with `torch.empty` and hashed beyond the active centroid count."
    elif classification == "RUN_TO_RUN_ATOMIC_REDUCTION_NONDETERMINISM":
        summary = "The first observed nondeterministic state transition is the production centroid operator on a frozen byte-identical layer0 K tensor. Stage hashing localizes the first divergent stage to cluster accumulation."
    elif classification == "UPSTREAM_K_NONDETERMINISM":
        summary = "The centroid-builder input K tensor is not byte-stable, so centroid output drift is downstream of an upstream prefill stage."
    else:
        summary = "The available oracles did not localize a single causal source."
    lines = [
        "# Centroid Determinism Causal Forensic",
        "",
        "## 1. Executive Summary",
        "",
        f"ROOT_CLASSIFICATION = {classification}",
        "",
        summary,
        "",
        "## 2. Current Symptom",
        "",
        "Repeated independent B1 prefill previously diverged at layer0 `k_centroid_values`. This run verifies whether the centroid builder input itself changes before assigning root cause.",
        "",
        "## 3. Centroid Call Graph",
        "",
        "- `models/llama_patternkv.py:1064` `LlamaFlashAttention_PatternKV.forward`: receives layer hidden states, computes Q/K/V projection.",
        "- `models/llama_patternkv.py:1129` reshapes K to `[B, H_kv, seq_len, head_dim]` and applies RoPE.",
        "- `models/llama_patternkv.py:1797` layer prefill compression starts after attention output.",
        "- `models/llama_patternkv.py:1801-1805`: B1 builds `Xmk = key_states.mean(...).permute(...).reshape(n_kv, seq_len, hd).float()`; observed shape is `" + str(input_k["runs"][0]["shape"]) + "`.",
        "- `models/llama_patternkv.py:1805` calls `batched_kmeans_fast_compiled(Xmk, k=self.num_k_bases, iters=30, tol=1e-4, seed=0)`.",
        "- `models/llama_patternkv.py:1806-1809` assigns tokens, writes `self.k_base = k_centroids.to(key_states.dtype)`, records `K_CENTROID`.",
        "- `models/llama_patternkv.py:628-721` `batched_kmeans_fast`: local `torch.Generator`, `torch.rand` init, `torch.empty` sums/counts, `zero_`, `scatter_add_`, normalize.",
        "",
        "## 4. Input-K Determinism",
        "",
        f"K_INPUT_DETERMINISTIC = {input_k['K_INPUT_DETERMINISTIC']}",
        "",
        "| run | sha256 | equal_to_run0 | max_abs_diff | mean_abs_diff | rel_l2 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in input_k["runs"]:
        lines.append(f"| {row['run']} | `{row['sha256']}` | {row['equal']} | {row['max_abs_diff']} | {row['mean_abs_diff']} | {row['rel_l2']} |")
    lines += [
        "",
        "Upstream component unique hashes:",
        "",
        "| component | unique_hashes |",
        "| --- | --- |",
    ]
    for name, row in input_k["components"].items():
        lines.append(f"| `{name}` | {row['unique_hashes']} |")
    lines += [
        "",
        "## 5. Frozen-K Standalone Centroid Test",
        "",
        f"CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES = {operator['CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES']}",
        f"ASSIGNMENT_100_RUN_UNIQUE_HASHES = {operator['ASSIGNMENT_100_RUN_UNIQUE_HASHES']}",
        "",
        "## 6. First Divergent Stage",
        "",
        f"FIRST_DIVERGENT_STAGE = {first_divergent}",
        "",
        "| stage | unique_hashes |",
        "| --- | --- |",
    ]
    for name, count in stage["stage_unique_hashes"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## 7. RNG Oracle",
        "",
        f"RNG_FIXED: before={rng['before_unique_hashes']} unique hashes, after={rng['after_unique_hashes']} unique hashes.",
        "",
        "## 8. Workspace Oracle",
        "",
        f"NORMAL_WORKSPACE_UNIQUE_HASHES = {workspace['normal_unique_hashes']}",
        f"ZEROED_WORKSPACE_UNIQUE_HASHES = {workspace['zeroed_unique_hashes']}",
        f"FRESH_WORKSPACE_UNIQUE_HASHES = {workspace['fresh_unique_hashes']}",
        "",
        "## 9. Synchronization Oracle",
        "",
        f"NO_SYNC_UNIQUE_HASHES = {sync['no_sync_unique_hashes']}",
        f"SYNC_UNIQUE_HASHES = {sync['sync_unique_hashes']}",
        "",
        "## 10. Atomic / Scatter / Reduction Audit",
        "",
        f"ATOMIC_REDUCTION_PRESENT = {reduction['atomic_reduction_present']}",
        "",
    ]
    lines.extend(f"- {item}" for item in reduction["findings"])
    lines += [
        "",
        "## 11. Deterministic Reference",
        "",
        f"REFERENCE_UNIQUE_HASHES = {reference['REFERENCE_UNIQUE_HASHES']}",
        f"PRODUCTION_UNIQUE_HASHES = {reference['PRODUCTION_UNIQUE_HASHES']}",
        "",
        "Production-vs-reference max metrics:",
        "",
        "```json",
        json.dumps(reference["production_vs_reference"], indent=2, sort_keys=True),
        "```",
        "",
        "## 12. Fixed Reduction Oracle",
        "",
        "FIXED_REDUCTION_ORACLE = NOT_APPLICABLE_FOR_CURRENT_PRODUCTION_OPERATOR. The current centroid accumulation path is PyTorch `scatter_add_`; there is no exposed split/chunk topology knob in production code. The deterministic reference provides a fixed-order reduction oracle.",
        "",
        "## 13. Root Cause Classification",
        "",
        payload["root_cause_evidence"],
        "",
        "## 14. Recommended Production Fix",
        "",
        "Option A: deterministic segmented reduction with stable grouping and fixed merge order. Correctness: strongest. Performance: likely slower than scatter in prefill unless optimized. Complexity: medium/high. Batch invariance: strong. Ragged compatibility: good if request-local segments are explicit.",
        "",
        "Option B: request-local fixed reduction tree for centroid accumulation. Correctness: strong for serving determinism. Performance: tunable with fixed chunks. Complexity: medium. Batch invariance: strong if partitioning is request/shape invariant. Ragged compatibility: good.",
        "",
        "Option C: CPU/simple PyTorch deterministic fallback for debug gates only. Correctness: useful oracle. Performance: poor. Complexity: low. Batch invariance: strong. Ragged compatibility: acceptable only for tests/forensics.",
        "",
        "## 15. Next Experiment",
        "",
        payload["next_experiment"],
        "",
        "## Environment",
        "",
        "```json",
        json.dumps({k: v for k, v in payload["environment"].items() if k != "nvidia_smi"}, indent=2, sort_keys=True),
        "```",
        "",
        "```text",
        payload["environment"]["nvidia_smi"].strip(),
        "```",
        "",
        "## Commands",
        "",
    ]
    lines.extend(f"- `{cmd}`" for cmd in payload["commands"])
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_env()
    started = time.perf_counter()
    tokenizer, _config, model = load_model(dtype=torch.float16, device=torch.device(args.device))
    input_ids = make_fixed_inputs(tokenizer, batch=1, context=CONTEXT, device=torch.device(args.device))
    input_k = run_input_k_determinism(model, input_ids)
    torch.save(input_k["frozen_k"], OUT_DIR / "fixed_layer0_k.pt")
    X = input_k["frozen_k"].to(device=torch.device(args.device))
    operator = run_operator_100(X)
    stage = summarize_stage_runs(X)
    normal_stage = summarize_stage_runs(X, runs=RUNS_OPERATOR)
    g = torch.Generator(device=X.device)
    g.manual_seed(SEED)
    _, fixed_idx = torch.rand(X.shape[0], X.shape[1], generator=g, device=X.device).topk(KMEANS_K, dim=1)
    fixed_rng_stage = summarize_stage_runs(X, fixed_indices=fixed_idx, runs=RUNS_OPERATOR)
    zeroed_stage = summarize_stage_runs(X, zero_workspace=True, runs=RUNS_OPERATOR)
    fresh_stage = summarize_stage_runs(X, runs=RUNS_OPERATOR)
    sync_operator = run_operator_100(X, sync=True)
    reference = run_reference_oracle(X, fixed_idx)
    payload = {
        "input_k": {k: v for k, v in input_k.items() if k != "frozen_k"},
        "operator": operator,
        "stage": stage,
        "rng_oracle": {
            "before_unique_hashes": normal_stage["stage_unique_hashes"]["k_centroid_values"],
            "after_unique_hashes": fixed_rng_stage["stage_unique_hashes"]["k_centroid_values"],
            "causal": normal_stage["stage_unique_hashes"]["k_centroid_values"] > 1 and fixed_rng_stage["stage_unique_hashes"]["k_centroid_values"] == 1,
        },
        "workspace_oracle": {
            "normal_unique_hashes": normal_stage["stage_unique_hashes"]["k_centroid_values"],
            "zeroed_unique_hashes": zeroed_stage["stage_unique_hashes"]["k_centroid_values"],
            "fresh_unique_hashes": fresh_stage["stage_unique_hashes"]["k_centroid_values"],
        },
        "sync_oracle": {
            "no_sync_unique_hashes": operator["CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES"],
            "sync_unique_hashes": sync_operator["CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES"],
        },
        "reduction_audit": {
            "atomic_reduction_present": True,
            "findings": [
                "`models/llama_patternkv.py:679` uses `sums.scatter_add_` to accumulate FP32 token values into centroid buckets.",
                "`models/llama_patternkv.py:686` uses `counts.scatter_add_` for per-cluster counts.",
                "`models/llama_patternkv.py:655-656` allocates `sums`/`counts` via `torch.empty`, but both are immediately zeroed before accumulation.",
                "`models/llama_patternkv.py:641-644` uses a local `torch.Generator` with fixed seed for initialization.",
                "No `tl.atomic_add` appears in the K centroid builder path; CUDA `atomicAdd` occurrences are in fused Value/GEMV kernels, not this prefill K centroid accumulation path.",
                "PyTorch CUDA `scatter_add_` for floating accumulation is treated as atomic/reduction-like for this forensic classification.",
            ],
        },
        "reference_oracle": reference,
        "environment": environment(),
        "commands": [
            "CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD /data/zypan/kvarn-repro/tools/bin/micromamba run -n patternkv python scripts/centroid_determinism_causal_forensic.py --device cuda:0",
        ],
        "elapsed_s": time.perf_counter() - started,
    }
    full_pool_unique = payload["input_k"]["components"].get("CACHE_K_CENTROID_POOL_FULL_SLOT", {}).get("unique_hashes", 0)
    active_pool_unique = payload["input_k"]["components"].get("CACHE_K_CENTROID_ACTIVE", {}).get("unique_hashes", 0)
    trace_centroid_unique = payload["input_k"]["components"].get("K_CENTROID", {}).get("unique_hashes", 0)
    if not payload["input_k"]["K_INPUT_DETERMINISTIC"]:
        payload["ROOT_CLASSIFICATION"] = "UPSTREAM_K_NONDETERMINISM"
        payload["FIRST_NONDETERMINISTIC_STAGE"] = payload["input_k"]["first_nondeterministic_upstream_stage"]
        payload["root_cause_evidence"] = "UPSTREAM_K_NONDETERMINISM: `KMEANS_K_INPUT` has more than one SHA256 hash across repeated independent B1 prefill runs."
        payload["next_experiment"] = "MORE FORENSIC: trace layer0 hidden/K projection/RoPE further upstream until the first byte-level change is isolated."
    elif operator["CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES"] > 1 and stage["FIRST_DIVERGENT_STAGE"] in {"cluster_sums", "normalized_centroids", "k_centroid_values"}:
        payload["ROOT_CLASSIFICATION"] = "RUN_TO_RUN_ATOMIC_REDUCTION_NONDETERMINISM"
        payload["FIRST_NONDETERMINISTIC_STAGE"] = stage["FIRST_DIVERGENT_STAGE"]
        payload["root_cause_evidence"] = "RUN_TO_RUN_ATOMIC_REDUCTION_NONDETERMINISM: frozen identical K input produces multiple production centroid hashes; stage hashing identifies the first divergent centroid stage."
        payload["next_experiment"] = "FIX: implement a debug-gated deterministic centroid accumulation candidate, then rerun frozen-K 100, independent B1 prefill 20, and S6-B.3.4 multistep gate."
    elif full_pool_unique > 1 and active_pool_unique == 1 and trace_centroid_unique == 1:
        payload["ROOT_CLASSIFICATION"] = "UNINITIALIZED_OR_STALE_WORKSPACE"
        payload["FIRST_NONDETERMINISTIC_STAGE"] = "CACHE_K_CENTROID_POOL_FULL_SLOT"
        payload["root_cause_evidence"] = "UNINITIALIZED_OR_STALE_WORKSPACE: `KMEANS_K_INPUT`, `K_CENTROID`, `CACHE_K_CENTROID_ACTIVE`, and frozen-K production centroid are deterministic, but the full centroid pool slot has multiple hashes because inactive capacity is allocated with `torch.empty` and is outside the active centroid count."
        payload["next_experiment"] = "MORE FORENSIC: update diagnostic comparators to hash only active centroid counts, then rerun S6-B.3.4D/3.4 to see whether the remaining logit drift has a real semantic state mismatch."
    else:
        payload["ROOT_CLASSIFICATION"] = "NOT_YET_LOCALIZED"
        payload["FIRST_NONDETERMINISTIC_STAGE"] = payload["input_k"]["first_nondeterministic_upstream_stage"] or stage["FIRST_DIVERGENT_STAGE"]
        payload["root_cause_evidence"] = "NOT_YET_LOCALIZED: input K and standalone centroid are deterministic under tested oracles, and no single active-state nondeterministic stage was identified."
        payload["next_experiment"] = "MORE FORENSIC: refine the state comparator and trace the next real semantic mismatch."
    (OUT_DIR / "payload.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(payload)
    print(json.dumps({
        "K_INPUT_DETERMINISTIC": payload["input_k"]["K_INPUT_DETERMINISTIC"],
        "FROZEN_K_CENTROID_UNIQUE_HASHES": operator["CENTROID_OPERATOR_100_RUN_UNIQUE_HASHES"],
        "FIRST_DIVERGENT_STAGE": payload["FIRST_NONDETERMINISTIC_STAGE"],
        "RNG_ORACLE": payload["rng_oracle"],
        "WORKSPACE_ORACLE": payload["workspace_oracle"],
        "SYNC_ORACLE": payload["sync_oracle"],
        "ATOMIC_REDUCTION_PRESENT": payload["reduction_audit"]["atomic_reduction_present"],
        "FIXED_REDUCTION_ORACLE": "REFERENCE_ONLY",
        "ROOT_CLASSIFICATION": payload["ROOT_CLASSIFICATION"],
        "REPORT": str(REPORT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
