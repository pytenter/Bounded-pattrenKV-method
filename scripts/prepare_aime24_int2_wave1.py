#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime24_int2_wave1 import aggregate_query_importance, jaccard_by_layer, make_channel_mask, mask_hash, stable_hash, task_key3
from bench.aime_utils import DEFAULT_BASE_SEED, effective_seed, read_jsonl


def load_method(root: Path, method: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted((root / method).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        pid = int(data["problem_id"])
        sid = int(data["sample_id"])
        seed = int(data.get("seed", effective_seed(DEFAULT_BASE_SEED, pid, sid)))
        rows[task_key3(pid, sid, seed)] = data
    return rows


def status(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {"correct": False, "generated_tokens": None, "stop_reason": "missing", "parser_status": "missing"}
    return {
        "correct": bool(row.get("is_correct")),
        "generated_tokens": int(row.get("generated_tokens") or 0),
        "stop_reason": row.get("stop_reason"),
        "parser_status": "ok" if row.get("parsed_answer") is not None and not row.get("parser_error") else "failure",
    }


def pick_selected(source: Path) -> list[dict[str, Any]]:
    fp16 = load_method(source, "fp16")
    kivi = load_method(source, "kivi_paper_g128")
    pattern = load_method(source, "patternkv_paper")
    keys = sorted(set(fp16) & set(kivi) & set(pattern))
    buckets: dict[str, list[str]] = {k: [] for k in ("fp16_ok_both_quant_wrong", "pattern_ok_kivi_wrong", "kivi_ok_pattern_wrong", "all_correct", "failure_extreme")}
    for key in keys:
        f, k, p = fp16[key], kivi[key], pattern[key]
        fc, kc, pc = bool(f.get("is_correct")), bool(k.get("is_correct")), bool(p.get("is_correct"))
        if fc and not kc and not pc:
            buckets["fp16_ok_both_quant_wrong"].append(key)
        if pc and not kc:
            buckets["pattern_ok_kivi_wrong"].append(key)
        if kc and not pc:
            buckets["kivi_ok_pattern_wrong"].append(key)
        if fc and kc and pc:
            buckets["all_correct"].append(key)
        if any((r.get("stop_reason") == "length" or r.get("parsed_answer") is None or int(r.get("generated_tokens") or 0) >= 30000) for r in (k, p)):
            buckets["failure_extreme"].append(key)
    quotas = [("fp16_ok_both_quant_wrong", 4), ("pattern_ok_kivi_wrong", 2), ("kivi_ok_pattern_wrong", 2), ("all_correct", 2), ("failure_extreme", 2)]
    chosen: list[str] = []
    categories: dict[str, str] = {}
    for category, quota in quotas:
        for key in buckets[category]:
            if key not in chosen and len([x for x in chosen if categories[x] == category]) < quota:
                chosen.append(key)
                categories[key] = category
    if len(chosen) < 12:
        fallback = sorted(
            [key for key in keys if key not in chosen and bool(fp16[key].get("is_correct")) and (not kivi[key].get("is_correct") or not pattern[key].get("is_correct"))],
            key=lambda key: max(abs(int(kivi[key].get("generated_tokens") or 0) - int(fp16[key].get("generated_tokens") or 0)), abs(int(pattern[key].get("generated_tokens") or 0) - int(fp16[key].get("generated_tokens") or 0))),
            reverse=True,
        )
        for key in fallback:
            if len(chosen) == 12:
                break
            chosen.append(key)
            categories[key] = "fallback_fp16_ok_quant_wrong"
    if len(chosen) < 12:
        fallback = sorted(
            [key for key in keys if key not in chosen],
            key=lambda key: max(abs(int(kivi[key].get("generated_tokens") or 0) - int(fp16[key].get("generated_tokens") or 0)), abs(int(pattern[key].get("generated_tokens") or 0) - int(fp16[key].get("generated_tokens") or 0))),
            reverse=True,
        )
        for key in fallback:
            if len(chosen) == 12:
                break
            chosen.append(key)
            categories[key] = "fallback_length_delta"
    out = []
    for key in chosen:
        f, k, p = fp16[key], kivi[key], pattern[key]
        out.append(
            {
                "problem_id": int(f["problem_id"]),
                "sample_id": int(f["sample_id"]),
                "seed": int(f["seed"]),
                "task_key": key,
                "fp16": status(f),
                "kivi": status(k),
                "patternkv": status(p),
                "selection_category": categories[key],
            }
        )
    return out


def pick_calibration(source: Path, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fp16 = load_method(source, "fp16")
    selected_keys = {x["task_key"] for x in selected}
    candidates = [r for key, r in fp16.items() if key not in selected_keys and not r.get("error")]
    candidates.sort(key=lambda r: int(r.get("generated_tokens") or 0))
    out: list[dict[str, Any]] = []
    n = len(candidates)
    for q in range(4):
        lo = (q * n) // 4
        hi = ((q + 1) * n) // 4
        bucket = candidates[lo:hi]
        if not bucket:
            continue
        for r in bucket[:2]:
            out.append({"problem_id": int(r["problem_id"]), "sample_id": int(r["sample_id"]), "seed": int(r["seed"]), "task_key": task_key3(int(r["problem_id"]), int(r["sample_id"]), int(r["seed"])), "fp16_generated_tokens": int(r.get("generated_tokens") or 0), "length_quartile": q})
    return out[:8]


def write_selected_md(path: Path, selected: list[dict[str, Any]]) -> None:
    lines = ["# AIME24 INT2 Wave1 Selected Tasks", "", "| problem_id | sample_id | seed | category | fp16 | kivi | patternkv | fp16_tokens | kivi_tokens | pattern_tokens | stops | parsers |", "| ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |"]
    for row in selected:
        stops = "/".join(str(row[m]["stop_reason"]) for m in ("fp16", "kivi", "patternkv"))
        parsers = "/".join(str(row[m]["parser_status"]) for m in ("fp16", "kivi", "patternkv"))
        lines.append(f"| {row['problem_id']} | {row['sample_id']} | {row['seed']} | {row['selection_category']} | {row['fp16']['correct']} | {row['kivi']['correct']} | {row['patternkv']['correct']} | {row['fp16']['generated_tokens']} | {row['kivi']['generated_tokens']} | {row['patternkv']['generated_tokens']} | {stops} | {parsers} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_placeholder_masks(mask_dir: Path, layers: int, kv_heads: int, head_dim: int) -> None:
    mask_dir.mkdir(parents=True, exist_ok=True)
    channel = torch.arange(head_dim, dtype=torch.float32)
    magnitude_scores = channel.repeat(layers, kv_heads, 1)
    query_scores = torch.flip(channel, dims=[0]).repeat(layers, kv_heads, 1)
    magnitude = make_channel_mask(magnitude_scores, 0.125)
    query_aware = make_channel_mask(query_scores, 0.125)
    for name, mask, scores in (("magnitude", magnitude, magnitude_scores), ("query_aware", query_aware, query_scores)):
        pt_path = mask_dir / f"PLACEHOLDER_NOT_FOR_RESULTS_{name}_key_int4_mask.pt"
        json_path = mask_dir / f"PLACEHOLDER_NOT_FOR_RESULTS_{name}_key_int4_mask.json"
        torch.save({"mask": mask, "mask_hash": mask_hash(mask), "source": "PLACEHOLDER_NOT_FOR_RESULTS_deterministic_until_real_calibration"}, pt_path)
        payload = {
            "mask_hash": mask_hash(mask),
            "source": "PLACEHOLDER_NOT_FOR_RESULTS_deterministic_until_real_calibration",
            "status": "blocked_wave1b",
            "layers": layers,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "channels_per_head": int(mask[0, 0].sum().item()),
            "channels": [[torch.nonzero(mask[layer, head], as_tuple=False).flatten().tolist() for head in range(kv_heads)] for layer in range(layers)],
        }
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (mask_dir / "mask_overlap_by_layer.json").write_text(json.dumps(jaccard_by_layer(magnitude, query_aware), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-results", type=Path, default=Path("results/paper_repro_v2/aime24_budget_n2"))
    parser.add_argument("--selected-json", type=Path, default=Path("configs/aime24_wave1_selected_tasks.json"))
    parser.add_argument("--calibration-json", type=Path, default=Path("configs/aime24_wave1_calibration_tasks.json"))
    parser.add_argument("--selected-md", type=Path, default=Path("reports/aime24_int2_wave1_v100_8gpu/selected_tasks.md"))
    parser.add_argument("--mask-dir", type=Path, default=Path("artifacts/aime24_wave1_masks"))
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    args = parser.parse_args()
    selected = pick_selected(args.source_results)
    if len(selected) != 12:
        raise SystemExit(f"selected task count must be 12, got {len(selected)}")
    calibration = pick_calibration(args.source_results, selected)
    if len(calibration) != 8:
        raise SystemExit(f"calibration task count must be 8, got {len(calibration)}")
    args.selected_json.parent.mkdir(parents=True, exist_ok=True)
    args.selected_json.write_text(json.dumps(selected, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_json.write_text(json.dumps(calibration, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_selected_md(args.selected_md, selected)
    write_placeholder_masks(args.mask_dir, args.layers, args.kv_heads, args.head_dim)
    print(json.dumps({"selected": len(selected), "calibration": len(calibration), "selected_hash": stable_hash(selected), "calibration_hash": stable_hash(calibration)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
