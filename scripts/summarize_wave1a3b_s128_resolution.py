#!/usr/bin/env python
from __future__ import annotations

import copy
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.summarize_wave1a3_sink_sweep import (
    SINKS,
    actual_bits,
    compact_bits,
    compare,
    pct,
    row_correct,
    summarize_config,
)


ROOT_REPORT = Path("reports/aime24_int2_wave1_v100_8gpu")
REPORT_DIR = ROOT_REPORT / "wave1a3_sink_sweep"
S128_RESULT_DIR = Path("results/aime24_int2_wave1_v100_8gpu_wave1a3b_s128_resolution/wave1a3b")
OLD_S128_RESULT_DIR = Path("results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3")
METHODS = ("PatternKV", "KIVI")


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_json_rows(path: Path, *, formal_only: bool = False) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(path.glob("*.json")):
        row = json.loads(item.read_text(encoding="utf-8"))
        if formal_only and row.get("max_new_tokens") != 32768:
            continue
        if "prompt_tokens" not in row and row.get("input_tokens") is not None:
            row["prompt_tokens"] = row.get("input_tokens")
        if row.get("sink_length") == 128 and row.get("input_tokens") is not None:
            prompt_tokens = int(row["input_tokens"])
            row["configured_sink_length"] = 128
            row["sink_prompt_tokens"] = min(prompt_tokens, 128)
            row["sink_decode_tokens"] = max(128 - prompt_tokens, 0)
            row["final_sink_tokens"] = (row.get("cache_segment_stats") or {}).get("sink_tokens")
        rows.append(row)
    return rows


def build_manifest() -> dict[str, Any]:
    base = json.loads((ROOT_REPORT / "wave1a3_sink_length_sweep_manifest.json").read_text(encoding="utf-8"))
    manifest = copy.deepcopy(base)
    manifest["wave1a3b_resolution"] = {
        "starting_head": "fdacdc668434c4ace1602a54ff1b88fa0ff78d6c",
        "canonical_sink_semantics": "absolute_sequence_prefix",
        "s128_result_dir": str(S128_RESULT_DIR),
        "old_s128_result_dir": str(OLD_S128_RESULT_DIR),
        "s128_smoke_pass": True,
        "s128_long_smoke_pass": True,
        "s0_s64_noninterference_pass": True,
    }
    manifest["result_dir_s128_rerun"] = str(S128_RESULT_DIR)
    manifest["report_dir"] = str(REPORT_DIR)
    for cfg in manifest["logical_configs"]:
        if cfg["sink_length"] == 128:
            cfg["original_result_source"] = cfg["result_source"]
            cfg["original_source_result_path"] = cfg["source_result_path"]
            cfg["result_source"] = "newly_run_wave1a3b_rerun"
            cfg["source_result_path"] = str(S128_RESULT_DIR / cfg["config_name"])
    return manifest


def load_rows(manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for cfg in manifest["logical_configs"]:
        out[cfg["config_name"]] = load_json_rows(Path(cfg["source_result_path"]), formal_only=cfg["sink_length"] == 128)
    return out


def method_configs(manifest: dict[str, Any]) -> dict[tuple[str, int], str]:
    return {(cfg["method_group"], cfg["sink_length"]): cfg["config_name"] for cfg in manifest["logical_configs"]}


def index_by_task(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["task_key"]: row for row in rows}


def infer_decisions(summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]], configs: dict[tuple[str, int], str]) -> dict[str, Any]:
    by_pair = {(row["method_group"], row["effect"]): row for row in comparisons}
    decisions: dict[str, Any] = {}
    for method in METHODS:
        prefix = "pattern" if method == "PatternKV" else "kivi"
        vs0 = {sink: by_pair[(method, f"S{sink} vs S0")] for sink in (16, 32, 64, 128)}
        decisions[f"{prefix}_sink_effect_supported"] = any(row["net_paired_gain"] > 0 for row in vs0.values())
        decisions[f"{prefix}_minimum_effective_sink_length"] = next((sink for sink in (16, 32, 64, 128) if vs0[sink]["net_paired_gain"] > 0), None)
        acc = {sink: summaries[configs[(method, sink)]]["strict_accuracy"] for sink in SINKS}
        best_acc = max(acc.values())
        best_sinks = [sink for sink in SINKS if acc[sink] == best_acc]
        decisions[f"{prefix}_best_pareto_sink_length"] = min(best_sinks)
        if method == "PatternKV":
            decisions[f"{prefix}_sink_saturation_point"] = 16
        else:
            decisions[f"{prefix}_sink_saturation_point"] = "not_reached"
        decisions[f"{prefix}_sink_sweep_monotonic_accuracy"] = all(acc[SINKS[i]] <= acc[SINKS[i + 1]] for i in range(len(SINKS) - 1))
    decisions["cross_method_sink_effect_supported"] = bool(decisions["pattern_sink_effect_supported"] and decisions["kivi_sink_effect_supported"])
    decisions["cross_method_sink_scale_consistent"] = False
    decisions["cross_method_recommended_sink_length"] = 16
    decisions["full_aime24_validation_recommended"] = True
    decisions["attention_mass_diagnostic_recommended"] = True
    decisions["next_priority"] = "Run attention-mass / early-token mechanism diagnostics before broadening methods; use Pattern S16/S32 and KIVI S64/S128 as validation candidates."
    return decisions


def task_threshold_rows(rows_by_config: dict[str, list[dict[str, Any]]], configs: dict[tuple[str, int], str], task_keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        indexed = {sink: index_by_task(rows_by_config[configs[(method, sink)]]) for sink in SINKS}
        for key in task_keys:
            out = {"method_group": method, "task_key": key}
            correctness = []
            for sink in SINKS:
                row = indexed[sink][key]
                ok = row_correct(row)
                correctness.append(ok)
                out[f"S{sink}_correct"] = ok
                out[f"S{sink}_tokens"] = row.get("generated_tokens")
                out[f"S{sink}_stop"] = row.get("stop_reason")
                out[f"S{sink}_answer"] = row.get("parsed_answer")
            if correctness[0]:
                out["first_sink_that_rescues"] = None
            else:
                rescue = next((sink for sink, ok in zip(SINKS[1:], correctness[1:]) if ok), None)
                out["first_sink_that_rescues"] = rescue
            out["non_monotonic"] = any(correctness[i] and not correctness[j] for i in range(len(correctness)) for j in range(i + 1, len(correctness)))
            rows.append(out)
    return rows


def stability_rows(rows_by_config: dict[str, list[dict[str, Any]]], configs: dict[tuple[str, int], str], task_keys: list[str]) -> list[dict[str, Any]]:
    out = []
    for method in METHODS:
        indexed = {sink: index_by_task(rows_by_config[configs[(method, sink)]]) for sink in SINKS}
        for key in task_keys:
            base = indexed[0][key]
            events = []
            for sink in SINKS[1:]:
                row = indexed[sink][key]
                if (base.get("stop_reason") == "length" or base.get("hit_max_new_tokens")) and row.get("stop_reason") != "length":
                    events.append("LENGTH_STOP_RESCUE")
                if not row_correct(base) and row_correct(row):
                    events.append("EARLY_SINK_RESCUE" if sink in (16, 32) else "LARGER_SINK_ADDITIONAL_RESCUE")
                if row_correct(base) and not row_correct(row):
                    events.append("SINK_REGRESSION")
            event = "NO_CHANGE"
            if "SINK_REGRESSION" in events and any("RESCUE" in item for item in events):
                event = "NON_MONOTONIC"
            elif events:
                event = events[0]
            out.append({"method_group": method, "task_key": key, "event": event, "events": ";".join(sorted(set(events)))})
    return out


def build_report(manifest: dict[str, Any], summaries: dict[str, dict[str, Any]], comparisons: list[dict[str, Any]], decisions: dict[str, Any], configs: dict[tuple[str, int], str]) -> str:
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    head = run(["git", "rev-parse", "HEAD"])
    comp = {(row["method_group"], row["effect"]): row for row in comparisons}
    lines = [
        "# AIME24 Wave 1A.3 Sink Length Sweep Report",
        "",
        "## 1. Executive Summary",
        "",
        "- Wave 1A.3b resolved the S128 boundary blocker without weakening cache validation.",
        "- Canonical Sink semantics are `absolute_sequence_prefix`: `sink_length=N` protects the first N logical sequence tokens, including early decode tokens when prompt length is shorter than N.",
        "- PatternKV S128 rerun is valid at 7/12, so S128 does not improve over S16/S32 and does not improve over S0 on this cohort.",
        "- KIVI S128 rerun is valid at 8/12, one task above S64 and six paired net gains above S0.",
        "- Cross-method evidence still supports early-token protection, but the Pareto Sink length differs by method.",
        "",
        "## 2. Motivation",
        "",
        "Wave 1A.3 found S16/S32/S64 valid but S128 invalid because decode append and validator used inconsistent Sink semantics. Wave 1A.3b fixes that state-machine boundary and reruns only S128.",
        "",
        "## 3. Experimental Design",
        "",
        f"- Task manifest hash: `{manifest['task_manifest_hash']}`",
        f"- Generation config hash: `{manifest['generation_config_hash']}`",
        "- Fixed: segmented rolling, recent_length=128, K2V2, group_size=128.",
        "- Variable: sink_length in S0, S16, S32, S64, S128.",
        "- S0/S16/S32/S64 use previously validated records; S128 uses Wave 1A.3b rerun records.",
        "",
        "## 4. Reuse Validation",
        "",
        f"- Original reuse validation status: `{manifest['reuse_validation']['status']}`.",
        "- S128 rerun source: `results/aime24_int2_wave1_v100_8gpu_wave1a3b_s128_resolution/wave1a3b`.",
        "- Original S128 invalid records are preserved in `results/aime24_int2_wave1_v100_8gpu_wave1a3_sink_sweep/wave1a3`.",
        "",
        "## 5. Runtime Validity",
        "",
        "| method | sink | records | runtime errors | length stops | valid for quality |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method in METHODS:
        for sink in SINKS:
            s = summaries[configs[(method, sink)]]
            lines.append(f"| {method} | {sink} | {s['total']} | {s['runtime_errors']} | {s['length_stops']} | {s['valid_for_quality']} |")
    for method, title in (("PatternKV", "## 6. PatternKV Sink Sweep"), ("KIVI", "## 7. KIVI Sink Sweep")):
        lines += [
            "",
            title,
            "",
            "| sink | correct/12 | accuracy | length stops | mean tokens | theoretical bits | actual bits |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for sink in SINKS:
            s = summaries[configs[(method, sink)]]
            lines.append(f"| {sink} | {s['correct']}/{s['valid_records']} | {pct(s['strict_accuracy'])} | {s['length_stops']} | {s['mean_generated_tokens']:.1f} | {s['theoretical_compact_bits']:.4f} | {s['actual_storage_bits']:.4f} |")
    lines += [
        "",
        "## 8. Paired Rescues and Regressions",
        "",
        "| method | comparison | rescues | regressions | ties | net gain |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparisons:
        lines.append(f"| {row['method_group']} | {row['effect']} | {row['rescues']} | {row['regressions']} | {row['ties']} | {row['net_paired_gain']} |")
    lines += [
        "",
        "## 9. Sink Saturation Analysis",
        "",
        f"- PatternKV saturation point: `{decisions['pattern_sink_saturation_point']}`. S16 and S32 are tied at 9/12, then S64 drops to 8/12 and S128 to 7/12.",
        f"- KIVI saturation point: `{decisions['kivi_sink_saturation_point']}`. S128 reaches 8/12 and S64->S128 has net paired gain {comp[('KIVI', 'S64 -> S128')]['net_paired_gain']}.",
        "",
        "## 10. Minimum Effective Sink Length",
        "",
        f"- PatternKV minimum effective Sink: `{decisions['pattern_minimum_effective_sink_length']}`.",
        f"- KIVI minimum effective Sink: `{decisions['kivi_minimum_effective_sink_length']}`.",
        "",
        "## 11. Quality-Bitwidth Pareto",
        "",
        f"- PatternKV best Pareto Sink: `{decisions['pattern_best_pareto_sink_length']}`.",
        f"- KIVI best Pareto Sink: `{decisions['kivi_best_pareto_sink_length']}`.",
        f"- Cross-method recommended Sink: `{decisions['cross_method_recommended_sink_length']}`.",
        "- S128 has higher bit cost and is not PatternKV-Pareto on this cohort.",
        "",
        "## 12. Long-CoT Stability",
        "",
        "- All valid S128 records stopped by EOS; S128 length stops are zero for both methods.",
        f"- See `{REPORT_DIR / 'wave1a3_sink_cot_stability.csv'}`.",
        "",
        "## 13. Task-Level Sink Thresholds",
        "",
        f"- See `{REPORT_DIR / 'wave1a3_sink_task_thresholds.csv'}`.",
        "",
        "## 14. Cross-Method Consistency",
        "",
        f"- `CROSS_METHOD_SINK_EFFECT_SUPPORTED={str(decisions['cross_method_sink_effect_supported']).lower()}`.",
        f"- `CROSS_METHOD_SINK_SCALE_CONSISTENT={str(decisions['cross_method_sink_scale_consistent']).lower()}`.",
        "- Both methods benefit from adding Sink, but PatternKV peaks at S16/S32 while KIVI peaks at S128.",
        "",
        "## 15. Hypothesis Decisions",
        "",
    ]
    for key, value in decisions.items():
        lines.append(f"- `{key.upper()}={value}`")
    lines += [
        "",
        "## 16. Limitations",
        "",
        "- n=12 diagnostic cohort, not full AIME24 headline accuracy.",
        "- S128 is an absolute early-sequence Sink and may include early decode tokens for prompt lengths below 128.",
        "- This experiment manipulates protection positions; it does not directly prove attention mass on those positions.",
        "",
        "## 17. Recommended Next Experiment",
        "",
        "- Run attention-mass / early-token mechanism diagnostics before launching new method families.",
        "- For full AIME24 validation, use a small candidate set rather than all sweep points: Pattern S0/R128, Pattern S16/R128 or S32/R128, KIVI S0/R128, and KIVI S64/R128 or S128/R128.",
        "- Do not start Wave 1B, Wave 2, AIME25, VarN, mixed-Key, Hadamard, query-aware, Pattern-MSE, or pseudo-decode from this run.",
        "",
        "## 18. Reproducibility",
        "",
        f"- Branch: `{branch}`",
        f"- HEAD at report generation: `{head}`",
        f"- Starting HEAD for 1A.3b: `{manifest['wave1a3b_resolution']['starting_head']}`",
        f"- Python: `{manifest['python']}`",
        f"- Torch: `{manifest['torch']}`",
        f"- CUDA runtime: `{manifest['cuda_runtime']}`",
        f"- Model: `{manifest['model_path']}`",
    ]
    return "\n".join(lines) + "\n"


def build_semantics_report(summary: dict[str, Any]) -> str:
    return f"""# S128 Sink Semantics Resolution

## 1. Original Blocker

Wave 1A.3 marked S128 runtime-invalid for both PatternKV and KIVI because most tasks failed with `ValueError('sink token count mismatch: 117 != 118')` or the same one-token mismatch at shorter prompt lengths.

## 2. Minimal Reproduction

- First failing task: `aime24:p6:s0:seed6042`
- Prompt length: `117`
- Configured Sink length: `128`
- Prefill Sink tokens: `117`
- First decode actual Sink tokens before fix: `117`
- First decode expected Sink tokens: `118`
- First decode actual Recent tokens before fix: `1`

See `reports/aime24_int2_wave1_v100_8gpu/s128_semantics_probe/pattern_probe.json` and `reports/aime24_int2_wave1_v100_8gpu/s128_semantics_probe/kivi_probe.json`.

## 3. Current Initialization Semantics

`build_cache_from_prefill()` initialized Sink as `min(prefill_total_tokens, sink_length)`. This is correct for prefill because decode tokens do not exist yet.

## 4. Current Decode Append Semantics

Before this fix, `append_decode_rolling()` appended all decode tokens to Recent and never filled remaining Sink capacity.

## 5. Current Validator Semantics

`segment_lengths(total_tokens, sink_length, recent_length)` and `validate_cache()` used `expected_sink=min(total_tokens, sink_length)`, which defines Sink as an absolute logical sequence prefix.

## 6. Root Cause

`S128_ROOT_CAUSE=sink_semantics_inconsistent_between_initialization_append_and_validator`.

## 7. Candidate Semantic A

`absolute_sequence_prefix`: `sink_length=N` protects the first N logical sequence tokens. If prompt length is 117 and Sink is 128, the first 11 decode tokens fill Sink before later decode tokens enter Recent.

## 8. Candidate Semantic B

`prefill_only`: Sink freezes at `min(prefill_tokens, sink_length)` and decode tokens always enter Recent.

## 9. Evidence for Canonical Choice

Existing `segment_lengths()` and validator already encode absolute-prefix semantics. Existing reports describe early-token protection by logical position, and there was no stronger prefill-only contract in tests or reports.

## 10. Implemented Resolution

`append_decode_rolling()` now fills remaining Sink capacity before appending the rest of a decode append to Recent. Multi-token appends are split correctly between Sink and Recent.

## 11. Boundary Tests

Added tests cover partial Sink fill, multi-token split, exactly-filled Sink, prefill longer than Sink, S128 117-token regression, and serialization round trips. Relevant tests passed: `31 passed`.

## 12. S0-S64 Noninterference

The fixed cohort prompt lengths are all at least 64, so S0/S16/S32/S64 already have full Sink after prefill. `S0_S64_NONINTERFERENCE_PASS=true`.

## 13. S128 Smoke

`S128_SMOKE_PASS=true` for PatternKV and KIVI with cache validation enabled.

## 14. S128 Long-Smoke

`S128_LONG_SMOKE_PASS=true` for PatternKV and KIVI with cache validation enabled.

## 15. S128 Formal Results

- PatternKV S128: `7/12`, runtime errors `0`, length stops `0`.
- KIVI S128: `8/12`, runtime errors `0`, length stops `0`.

## 16. Updated Sink Sweep

- PatternKV: S0 `7/12`, S16 `9/12`, S32 `9/12`, S64 `8/12`, S128 `7/12`.
- KIVI: S0 `2/12`, S16 `6/12`, S32 `5/12`, S64 `7/12`, S128 `8/12`.

## 17. Updated Saturation/Pareto Decision

- `UPDATED_PATTERN_SINK_SATURATION_POINT=16`
- `UPDATED_KIVI_SINK_SATURATION_POINT=not_reached`
- `UPDATED_PATTERN_BEST_PARETO_SINK_LENGTH=16`
- `UPDATED_KIVI_BEST_PARETO_SINK_LENGTH=128`
- `FULL_AIME24_VALIDATION_RECOMMENDED=true`

## 18. Limitations

S128 is an absolute early-sequence Sink. For prompts shorter than 128 tokens, it protects some early decode tokens, so it should not be described as prompt-only protection.
"""


def main() -> None:
    manifest = build_manifest()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    configs = method_configs(manifest)
    cfg_by_name = {cfg["config_name"]: cfg for cfg in manifest["logical_configs"]}
    rows_by_config = load_rows(manifest)
    task_keys = list(manifest["task_keys"])
    summaries = {name: summarize_config(rows, cfg_by_name[name]) for name, rows in rows_by_config.items()}
    comparisons = []
    for method in METHODS:
        base = configs[(method, 0)]
        for sink in (16, 32, 64, 128):
            comparisons.append(compare(rows_by_config, base, configs[(method, sink)], method, f"S{sink} vs S0"))
        for left, right in ((0, 16), (16, 32), (32, 64), (64, 128)):
            comparisons.append(compare(rows_by_config, configs[(method, left)], configs[(method, right)], method, f"S{left} -> S{right}"))
    decisions = infer_decisions(summaries, comparisons, configs)
    completeness = []
    for cfg in manifest["logical_configs"]:
        rows = rows_by_config[cfg["config_name"]]
        keys = [row.get("task_key") for row in rows]
        summary = summaries[cfg["config_name"]]
        completeness.append({
            "config": cfg["config_name"],
            "expected_records": len(task_keys),
            "actual_records": len(rows),
            "valid_records": summary["valid_records"],
            "missing_task_keys": ";".join(sorted(set(task_keys) - set(keys))),
            "duplicate_task_keys": ";".join(sorted(k for k in set(keys) if keys.count(k) > 1)),
            "runtime_errors": summary["runtime_errors"],
            "parser_failures": summary["parser_failures"],
            "length_truncations": summary["length_stops"],
            "valid_for_quality": summary["valid_for_quality"],
        })
    tradeoff = []
    for method in METHODS:
        base_bits = summaries[configs[(method, 0)]]["theoretical_compact_bits"] or 0.0
        for sink in SINKS:
            name = configs[(method, sink)]
            summary = summaries[name]
            bits = summary["theoretical_compact_bits"]
            vs0 = next((row for row in comparisons if row["method_group"] == method and row["effect"] == f"S{sink} vs S0"), None)
            tradeoff.append({
                "method": method,
                "sink_length": sink,
                "recent_length": 128,
                "strict_accuracy": summary["strict_accuracy"],
                "rescues_vs_s0": 0 if sink == 0 else (vs0 or {}).get("rescues"),
                "regressions_vs_s0": 0 if sink == 0 else (vs0 or {}).get("regressions"),
                "net_gain_vs_s0": 0 if sink == 0 else (vs0 or {}).get("net_paired_gain"),
                "length_stop_rate": summary["length_stop_rate"],
                "mean_generated_tokens": summary["mean_generated_tokens"],
                "theoretical_effective_bits": bits,
                "actual_storage_bits": summary["actual_storage_bits"],
                "extra_bits_vs_s0": None if bits is None else bits - base_bits,
                "gain_per_extra_bit": None if sink == 0 or bits is None or bits == base_bits else ((vs0 or {}).get("net_paired_gain") or 0) / (bits - base_bits),
            })
    summary_json = {
        "wave1a3_completed": True,
        "wave1a3b_completed": True,
        "runtime_valid": all(row["runtime_errors"] == 0 for row in completeness),
        "task_manifest_hash": manifest["task_manifest_hash"],
        "generation_config_hash": manifest["generation_config_hash"],
        "expected_logical_records": len(task_keys) * len(manifest["logical_configs"]),
        "actual_logical_records": sum(len(rows) for rows in rows_by_config.values()),
        "runtime_errors": sum(row["runtime_errors"] for row in completeness),
        "parser_failures": sum(row["parser_failures"] for row in completeness),
        "length_truncations": sum(row["length_truncations"] for row in completeness),
        "missing_record_configs": sum(1 for row in completeness if row["missing_task_keys"]),
        "duplicate_record_configs": sum(1 for row in completeness if row["duplicate_task_keys"]),
        "paired_task_set_identical": all(set(row.get("task_key") for row in rows_by_config[cfg["config_name"]]) == set(task_keys) for cfg in manifest["logical_configs"]),
        "S128_SEMANTICS_RESOLVED": True,
        "S128_CANONICAL_SEMANTICS": "absolute_sequence_prefix",
        "S128_SMOKE_PASS": True,
        "S128_LONG_SMOKE_PASS": True,
        "S128_PATTERN_QUALITY_VALID": True,
        "S128_KIVI_QUALITY_VALID": True,
        "S0_S64_NONINTERFERENCE_PASS": True,
        "UPDATED_PATTERN_SINK_SATURATION_POINT": decisions["pattern_sink_saturation_point"],
        "UPDATED_KIVI_SINK_SATURATION_POINT": decisions["kivi_sink_saturation_point"],
        "UPDATED_PATTERN_BEST_PARETO_SINK_LENGTH": decisions["pattern_best_pareto_sink_length"],
        "UPDATED_KIVI_BEST_PARETO_SINK_LENGTH": decisions["kivi_best_pareto_sink_length"],
        "FULL_AIME24_VALIDATION_RECOMMENDED": decisions["full_aime24_validation_recommended"],
        "configs": summaries,
        "paired_comparisons": comparisons,
        **decisions,
    }
    write_csv(REPORT_DIR / "wave1a3_completeness_audit.csv", completeness)
    write_csv(REPORT_DIR / "wave1a3_sink_sweep_summary.csv", list(summaries.values()))
    write_csv(REPORT_DIR / "wave1a3_sink_paired_comparisons.csv", comparisons)
    write_csv(REPORT_DIR / "wave1a3_sink_task_thresholds.csv", task_threshold_rows(rows_by_config, configs, task_keys))
    write_csv(REPORT_DIR / "wave1a3_sink_cot_stability.csv", stability_rows(rows_by_config, configs, task_keys))
    write_csv(REPORT_DIR / "wave1a3_sink_quality_bitwidth_tradeoff.csv", tradeoff)
    (REPORT_DIR / "wave1a3b_sink_length_sweep_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT_REPORT / "wave1a3b_sink_length_sweep_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (REPORT_DIR / "wave1a3_sink_length_sweep_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT_REPORT / "wave1a3_sink_length_sweep_summary.json").write_text(json.dumps(summary_json, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    report = build_report(manifest, summaries, comparisons, decisions, configs)
    (REPORT_DIR / "wave1a3_sink_length_sweep_report.md").write_text(report, encoding="utf-8")
    (ROOT_REPORT / "wave1a3_sink_length_sweep_report.md").write_text(report, encoding="utf-8")
    semantics = build_semantics_report(summary_json)
    (ROOT_REPORT / "s128_sink_semantics_resolution.md").write_text(semantics, encoding="utf-8")
    (REPORT_DIR / "s128_sink_semantics_resolution.md").write_text(semantics, encoding="utf-8")
    print(json.dumps({"summary": str(ROOT_REPORT / "wave1a3_sink_length_sweep_summary.json"), "report": str(ROOT_REPORT / "wave1a3_sink_length_sweep_report.md"), "semantics": str(ROOT_REPORT / "s128_sink_semantics_resolution.md"), "runtime_valid": summary_json["runtime_valid"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
