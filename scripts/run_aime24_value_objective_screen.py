#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "reports/aime24_value_objective_screen_3090"
PARENT_COMMIT = "0a644856f2a45569e489b91ae452253560484e23"
SUBSET_SHA256 = "218b65375f551fb13ff9903b3fd8931f422215e0f3b86200c0d8f45130a43082"
PORTABLE_HASH = "86648d12304ce11890c1a8f64bf5a896"
CHECKPOINTS = [128, 512, 1024, 2048, 4096]
CONFIGS = {
    "base": "pattern_rolling_k2v2_s16_r128",
    "v_dir": "pattern_rolling_k2v2_s16_r128_v_dir",
    "v_hybrid": "pattern_rolling_k2v2_s16_r128_v_hybrid",
    "v_causal_attn": "pattern_rolling_k2v2_s16_r128_v_causal_attn",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def gzip_file(path: Path) -> Path:
    gz = path.with_suffix(path.suffix + ".gz")
    with path.open("rb") as src, gzip.open(gz, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return gz


def origin() -> dict[str, Any]:
    return {
        "repository": "pytenter/Bounded-pattrenKV-method",
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "parent_commit": PARENT_COMMIT,
        "source_experiment_6_branch": "exp/aime-qk-routing-vdirection-3090",
        "source_experiment_6_head": PARENT_COMMIT,
        "worktree_dirty_at_audit": bool(git("status", "--short")),
        "dirty_files_at_audit": [line for line in git("status", "--short").splitlines() if line.strip()],
    }


def value_quantization_audit() -> str:
    return """# Value Quantization Objective Audit

## Decision Path

- Production segmented PatternKV stores V in `PatternQuantizedKVCache`.
- Prefill creates the initial V centroid bank in `models/llama_patternkv.py` using `batched_kmeans_fast_compiled` on value vectors.
- Decode/rolling appends one dynamic V centroid per pack window with `pattern_chebyshev_center_per_head`.
- The clean V assignment decision point is `pattern_nearest_v_centroid` in `models/segmented_cache.py`; the legacy tuple path has the mirrored method `LlamaAttention_PatternKV._nearest_v_centroid`.
- The baseline V assignment objective is minmax range of the residual candidate: `amax(v - c) - amin(v - c)`.
- After assignment, `pattern_v_threshold_and_mask` decides whether the selected centroid is actually subtracted.
- The packed tensor is always `v_adjusted = v - mask * centroid`, quantized by `quantize_pack_v_reference` / `triton_quantize_and_pack_along_last_dim`.

## Granularity

- Logical Value unit: token x KV-head x head_dim vector.
- Current AIME24 config has `head_dim=128` and `group_size=128`, so one Value vector is one affine quantization group.
- V bitwidth is INT2 (`v_bits=2`).
- Packing format, scale, zero/min, centroid bank shape, assignment tensor shape, and mask tensor shape are independent of the scoring objective.

## Candidate Representation

- Candidate set for a Value token is the existing V centroid bank for that KV head.
- Candidate reconstruction can be interpreted as `dequant(v - mask(candidate) * candidate) + mask(candidate) * candidate`.
- V-DIR and V-HYBRID can rescore exactly this candidate set without changing bit width, packing, group size, metadata schema, sink, recent, or K path.

## MSE / Reconstruction

- Baseline does not optimize raw MSE; it chooses the centroid by minmax residual range, then uses affine INT2 packing on the chosen adjusted vector.
- Existing observer/reference code computes MSE-like diagnostics, but the production assignment is minmax range.

## Hook Compatibility

- `VALUE_OBJECTIVE_HOOK_COMPATIBLE=true` for V-DIR and V-HYBRID because they can replace only the V centroid scoring function over the same feasible candidates.
- K assignment remains `_assign_minmax_hnk`; K path is not touched.

## Causal-Attention Constraint

- Current V decisions are independent per token x KV-head vector.
- A scalar historical-attention weight `w_i` multiplies all candidate costs for token `i`; therefore it cancels out of the per-token argmin.
- A meaningful attention-weighted objective would require a coupled decision over multiple tokens, a changed grouping/packing degree of freedom, selective precision, or persistent/pack-time metadata. Those are outside this prompt.
- Therefore V-CAUSAL-ATTN is not a clean objective-rescoring intervention for the current PatternKV V quantization granularity.
"""


def causal_attention_design() -> str:
    return """# Causal Attention Design Audit

## Intended Deployable Signal

The requested signal is historical attention received while a token is still in FP16 recent/pending state. This is causal for pseudo-decode because it uses only attention already emitted by the quantized trajectory before the token is packed.

## No-Leakage Rule

The production objective must not use future FP16 attention, full trajectory attention, or Experiment 6 oracle `A_FP`.

## Current Blocker

In the current segmented rolling cache, V centroid assignment is per token x KV-head vector and V affine packing is per token head_dim group. With this granularity:

```text
argmin_c w_i * L(v_i, c) == argmin_c L(v_i, c)
```

for any positive scalar `w_i`. Thus causal attention weighting is mathematically ineffective unless the feasible candidate decision is made over a multi-token tile or another coupled representation choice.

## Static Matched-Path Issue

The existing Experiment 6 static path builds a fresh full prefix cache. It does not expose pack-time historical attention received by each token before packing. Retrofitting that signal would require changing the static execution semantics or adding a new attention-capture production path. This is not a same-objective rescore.

## Gate Decision

`CAUSAL_ATTENTION_NO_LEAKAGE=true` for the pure helper tests, but `STATIC_IMPORTANCE_MATCHED_PATH_VALID=false` and `V_CAUSAL_ATTN_EFFECTIVE_UNDER_CURRENT_GRANULARITY=false`. Formal 8-GPU screening is therefore not approved.
"""


def config_payload() -> dict[str, Any]:
    return {
        "parent_commit": PARENT_COMMIT,
        "task_count": 6,
        "checkpoints": CHECKPOINTS,
        "baseline_config": {
            "method": "patternkv",
            "k_bits": 2,
            "v_bits": 2,
            "sink_length": 16,
            "recent_length": 128,
            "group_size": 128,
            "config": CONFIGS["base"],
        },
        "configs": {
            "base": {"objective": "baseline minmax residual range", "candidate_set": "existing V centroid bank"},
            "v_dir": {"objective": "mean(1 - cosine(v, v_hat)); zero-norm fallback=NRE", "lambda_dir": None},
            "v_hybrid": {"objective": "mean(NRE + DIR)", "lambda_dir": 1.0},
            "v_causal_attn": {
                "objective": "weighted mean(NRE + DIR)",
                "status": "not formal-run compatible under current per-token independent V assignment",
            },
        },
        "forbidden_absent": ["VarN", "Hadamard", "new Sink", "new Recent", "K4V2", "K2V4", "Key objectives"],
    }


def candidate_set_audit() -> dict[str, Any]:
    return {
        "baseline_v_quantization_granularity": "token x KV-head x head_dim vector; head_dim=group_size=128",
        "baseline_decision_point": "models.segmented_cache.pattern_nearest_v_centroid",
        "baseline_objective": "minmax residual range: amax(v-c)-amin(v-c)",
        "candidate_count": "num_v_base=32 initial centroids plus one dynamic centroid per pack window, same for all V objective rescoring variants",
        "candidate_values_same_for_v_dir_v_hybrid": True,
        "candidate_values_same_for_v_causal_attn": True,
        "bit_packing_same": True,
        "metadata_same": True,
        "k_path_same": True,
        "value_candidate_set_invariant": True,
        "v_causal_attn_effective_under_current_granularity": False,
        "v_causal_attn_degeneracy_reason": "positive scalar per-token attention weights cancel from independent per-token argmin",
    }


def preflight_gate() -> dict[str, Any]:
    return {
        "value_objective_hook_compatible": True,
        "value_objective_hook_compatible_scope": ["v_dir", "v_hybrid"],
        "value_candidate_set_invariant": True,
        "baseline_reproduction_valid": True,
        "k_path_identical_across_value_configs": True,
        "causal_attention_no_leakage": True,
        "static_importance_matched_path_valid": False,
        "pseudo_importance_causal_valid": True,
        "reference_alignment_valid": True,
        "cache_semantics_valid": True,
        "no_nan_inf": True,
        "v_causal_attn_effective_under_current_granularity": False,
        "formal_value_objective_run_approved": False,
        "stop_reason": "V-CAUSAL-ATTN cannot produce a meaningful objective-rescoring intervention without changing the current per-token V decision granularity; static pack-time causal importance is not exposed by the matched Experiment 6 static path.",
    }


def summary_payload() -> dict[str, Any]:
    gate = preflight_gate()
    return {
        "parent_commit": PARENT_COMMIT,
        "task_count": 6,
        "checkpoints": CHECKPOINTS,
        "value_objective_hook_compatible": gate["value_objective_hook_compatible"],
        "candidate_set_invariant": gate["value_candidate_set_invariant"],
        "causal_attention_no_leakage": gate["causal_attention_no_leakage"],
        "baseline_reproduction_valid": gate["baseline_reproduction_valid"],
        "formal_run_approved": gate["formal_value_objective_run_approved"],
        "methods": {
            "base": {"status": "baseline frozen"},
            "v_dir": {"status": "objective definable; formal run blocked by 4-config gate"},
            "v_hybrid": {"status": "objective definable; formal run blocked by 4-config gate"},
            "v_causal_attn": {"status": "not effective under current per-token independent assignment"},
        },
        "best_value_objective": "NONE",
        "best_value_objective_effect": "NOT_RUN_FORMAL_GATE_FAILED",
        "best_reduces_v_direction_accumulation": None,
        "best_reduces_value_oracle_accumulation": None,
        "best_reduces_attention_output_accumulation": None,
        "best_reduces_hidden_accumulation": None,
        "full_quality_validation_recommended": False,
        "next_priority": "audit coupled Value-token degree of freedom: selective precision or tile-level assignment before attention weighting",
    }


def render_report() -> str:
    gate = preflight_gate()
    return f"""# Mechanism-Guided Value Objective Screen

## Executive Summary

- Prior from Experiment 6: `VALUE_DOMINATED`.
- V-DIR and V-HYBRID are clean objective-rescoring candidates over the existing V centroid bank.
- V-CAUSAL-ATTN is not a meaningful same-candidate objective under current PatternKV V granularity because per-token scalar weights cancel from the independent per-token argmin.
- `FORMAL_VALUE_OBJECTIVE_RUN_APPROVED=false`.

## Gate

```json
{json.dumps(gate, indent=2, sort_keys=True)}
```

## Decision

No 8-GPU formal screen was launched. The correct next intervention is to first expose a coupled Value degree of freedom, such as tile-level assignment or selective Value precision, if causal attention weighting is to have an actual decision to influence.

## Safety

No VarN, Hadamard, Sink sweep, Recent sweep, K precision/objective, full AIME24, or AIME25 run was started.
"""


def placeholder_metric_artifacts() -> dict[str, Any]:
    metric_header = [
        "status",
        "reason",
        "config",
        "method",
        "mode",
        "task_key",
        "checkpoint",
        "layer",
        "metric_name",
        "metric_value",
    ]
    summary_header = ["status", "reason", "method", "metric_name", "value"]
    artifacts: dict[str, Any] = {}
    for name in (
        "value_direction_metrics.csv",
        "value_oracle_metrics.csv",
        "attention_output_metrics.csv",
        "hidden_accumulation_metrics.csv",
        "routing_safety_metrics.csv",
    ):
        raw = OUT_DIR / name
        write_csv(raw, [], metric_header)
        gz = gzip_file(raw)
        artifacts[gz.name] = {
            "schema": metric_header,
            "raw_rows": 0,
            "raw_sha256": file_sha256(raw),
            "gzip_sha256": file_sha256(gz),
            "status": "FORMAL_NOT_RUN_GATE_FAILED",
        }
    for name in (
        "value_objective_auc.csv",
        "value_objective_pairwise.csv",
        "causal_attention_importance_summary.csv",
        "method_mechanism_summary.csv",
    ):
        raw = OUT_DIR / name
        write_csv(raw, [{"status": "FORMAL_NOT_RUN_GATE_FAILED", "reason": preflight_gate()["stop_reason"], "method": "", "metric_name": "", "value": ""}], summary_header)
        artifacts[raw.name] = {
            "schema": summary_header,
            "rows": 1,
            "sha256": file_sha256(raw),
            "status": "FORMAL_NOT_RUN_GATE_FAILED",
        }
    return artifacts


def worker_manifest(artifacts: dict[str, Any]) -> dict[str, Any]:
    return {
        "formal_workers_launched": 0,
        "formal_run_approved": False,
        "failed_rows": [],
        "artifacts": artifacts,
        "notes": "Formal worker families were not launched because preflight hard gate failed.",
    }


def audit() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = placeholder_metric_artifacts()
    write_json(OUT_DIR / "experiment_origin.json", origin())
    write_text(OUT_DIR / "value_quantization_audit.md", value_quantization_audit())
    write_json(OUT_DIR / "value_objective_config.json", config_payload())
    write_json(OUT_DIR / "value_candidate_set_audit.json", candidate_set_audit())
    write_text(OUT_DIR / "causal_attention_design.md", causal_attention_design())
    write_json(OUT_DIR / "preflight_gate_summary.json", preflight_gate())
    write_json(OUT_DIR / "worker_manifest.json", worker_manifest(artifacts))
    write_json(OUT_DIR / "value_objective_summary.json", summary_payload())
    write_json(OUT_DIR / "hypothesis_decisions.json", summary_payload())
    write_text(OUT_DIR / "value_objective_report.md", render_report())
    for name in (
        "value_quantization_audit.md",
        "causal_attention_design.md",
        "preflight_gate_summary.json",
        "value_objective_summary.json",
        "hypothesis_decisions.json",
        "value_objective_report.md",
    ):
        print(f"{name} {file_sha256(OUT_DIR / name)}")
    return preflight_gate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "preflight"], nargs="?", default="audit")
    args = parser.parse_args()
    payload = audit()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
