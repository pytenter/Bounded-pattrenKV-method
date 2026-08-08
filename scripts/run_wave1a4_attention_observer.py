#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.aime24_int2_wave1 import task_key3
from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from bench.aime_utils import compute_stop_state, effective_seed, load_aime24, set_all_seeds, sha256_file
from bench.attention_observer import (
    absolute_regions,
    cache_segment_regions,
    enrichment,
    region_contributions,
    region_mass,
    repeat_kv_for_gqa,
    routing_value_decomposition,
    shadow_attention,
    tensor_pair_metrics,
)
from bench.bench_aime24_patternkv import eos_ids, load_model, render_prompt
from bench.paper_config import apply_method_defaults
from models.segmented_cache import cache_segment_stats, deserialize_cache, reconstruct_full_k, reconstruct_full_v


RESULT_DIR = Path("results/aime24_int2_wave1_v100_8gpu_wave1a4")
REPORT_DIR = Path("reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism")
REF_CAPTURE_DIR = RESULT_DIR / "teacher_forcing_reference_captures"
CONFIG_CAPTURE_DIR = RESULT_DIR / "teacher_forcing_config_captures"
FREE_RUNNING_DIR = RESULT_DIR / "free_running_observational_traces"
CSV_NAMES = {
    "mass": "wave1a4_attention_mass_metrics.csv",
    "enrichment": "wave1a4_attention_enrichment_metrics.csv",
    "head": "wave1a4_head_level_attention.csv",
    "k": "wave1a4_k_reconstruction_metrics.csv",
    "v": "wave1a4_v_reconstruction_metrics.csv",
    "routing": "wave1a4_routing_error_metrics.csv",
    "contrib": "wave1a4_region_contribution_metrics.csv",
    "output": "wave1a4_attention_output_metrics.csv",
    "hidden": "wave1a4_hidden_state_metrics.csv",
    "task": "wave1a4_task_mechanism_summary.csv",
    "free": "wave1a4_free_running_attention_events.csv",
}
DEFAULT_LAYERS = (0, 7, 15, 23, 31)
DEFAULT_CHECKPOINTS = (128, 512, 1024, 2048, 4096, 8192, 16384)
CONFIGS = {
    "fp16_reference": ("fp16", "FP16", 0),
    "pattern_rolling_k2v2_s0_r128": ("patternkv_paper", "PatternKV", 0),
    "pattern_rolling_k2v2_s16_r128": ("patternkv_paper", "PatternKV", 16),
    "pattern_rolling_k2v2_s128_r128": ("patternkv_paper", "PatternKV", 128),
    "kivi_rolling_k2v2_s0_r128": ("kivi_paper_g128", "KIVI", 0),
    "kivi_rolling_k2v2_s16_r128": ("kivi_paper_g128", "KIVI", 16),
    "kivi_rolling_k2v2_s128_r128": ("kivi_paper_g128", "KIVI", 128),
}
FREE_RUNNING_CONFIGS = (
    "pattern_rolling_k2v2_s0_r128",
    "pattern_rolling_k2v2_s16_r128",
    "kivi_rolling_k2v2_s0_r128",
    "kivi_rolling_k2v2_s16_r128",
    "kivi_rolling_k2v2_s128_r128",
)
FREE_RUNNING_CHECKPOINTS = (512, 1024, 2048, 4096, 8192, 16384)


@dataclass
class Capture:
    layer_input: torch.Tensor
    attn_input: torch.Tensor
    layer_output: torch.Tensor
    position_ids: torch.Tensor


class LayerObserver:
    def __init__(self, model: torch.nn.Module, layers: set[int]):
        self.model = model
        self.layers = layers
        self.captures: dict[int, Capture] = {}
        self._layer_inputs: dict[int, torch.Tensor] = {}
        self._attn_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.handles = []
        for layer_idx in layers:
            layer = model.model.layers[layer_idx]
            self.handles.append(layer.register_forward_pre_hook(self._make_layer_pre_hook(layer_idx)))
            self.handles.append(layer.register_forward_hook(self._make_layer_post_hook(layer_idx)))
            self.handles.append(layer.self_attn.register_forward_pre_hook(self._make_attn_pre_hook(layer_idx), with_kwargs=True))

    def _make_layer_pre_hook(self, layer_idx: int):
        def hook(_module, inputs):
            self._layer_inputs[layer_idx] = inputs[0].detach()

        return hook

    def _make_layer_post_hook(self, layer_idx: int):
        def hook(_module, _inputs, outputs):
            if layer_idx not in self._attn_inputs or layer_idx not in self._layer_inputs:
                return
            layer_output = outputs[0].detach()
            attn_input, position_ids = self._attn_inputs[layer_idx]
            self.captures[layer_idx] = Capture(
                layer_input=self._layer_inputs[layer_idx].detach().cpu(),
                attn_input=attn_input.detach().cpu(),
                layer_output=layer_output.detach().cpu(),
                position_ids=position_ids.detach().cpu(),
            )

        return hook

    def _make_attn_pre_hook(self, layer_idx: int):
        def hook(_module, inputs, kwargs):
            hidden = kwargs.get("hidden_states", inputs[0] if inputs else None)
            position_ids = kwargs.get("position_ids", None)
            if hidden is None or position_ids is None:
                return
            self._attn_inputs[layer_idx] = (hidden.detach(), position_ids.detach())

        return hook

    def clear(self) -> None:
        self.captures.clear()
        self._layer_inputs.clear()
        self._attn_inputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def write_csv(path: Path, rows: list[dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not append and not path.exists():
            path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    if append and path.exists() and path.stat().st_size:
        with path.open(encoding="utf-8") as handle:
            first = handle.readline().strip()
        existing = first.split(",") if first else []
        fieldnames = list(dict.fromkeys(existing + fieldnames))
    mode = "a" if append and path.exists() and path.stat().st_size else "w"
    with path.open(mode, encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def stable_token_hash(token_ids: list[int]) -> str:
    payload = json.dumps([int(x) for x in token_ids], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def split_task_keys(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in str(value).split(";") if item]


def load_phaseb_selected_tasks() -> list[dict[str, Any]]:
    wave1a3 = json.loads(Path("reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_length_sweep_summary.json").read_text(encoding="utf-8"))
    selected: dict[str, dict[str, Any]] = {}
    for comparison in wave1a3.get("paired_comparisons", []):
        method_group = comparison.get("method_group")
        effect = comparison.get("effect")
        if method_group == "PatternKV" and effect == "S16 vs S0":
            for task_key in split_task_keys(comparison.get("rescue_task_keys")):
                item = selected.setdefault(task_key, {"task_key": task_key, "selection_reasons": []})
                item["pattern_outcome_class"] = "RESCUE"
                item["selection_reasons"].append("PatternKV S0->S16 RESCUE")
            for task_key in split_task_keys(comparison.get("regression_task_keys")):
                item = selected.setdefault(task_key, {"task_key": task_key, "selection_reasons": []})
                item["pattern_outcome_class"] = "REGRESSION"
                item["selection_reasons"].append("PatternKV S0->S16 REGRESSION")
        if method_group == "KIVI" and effect == "S16 vs S0":
            for task_key in split_task_keys(comparison.get("rescue_task_keys")):
                item = selected.setdefault(task_key, {"task_key": task_key, "selection_reasons": []})
                item["kivi_outcome_class"] = "RESCUE"
                item["selection_reasons"].append("KIVI S0->S16 RESCUE")
    tasks = []
    for task_key, item in sorted(selected.items()):
        parts = task_key.split(":")
        problem_id = int(parts[1][1:])
        sample_id = int(parts[2][1:])
        seed = int(parts[3].replace("seed", ""))
        item["problem_id"] = problem_id
        item["sample_id"] = sample_id
        item["seed"] = seed
        item.setdefault("pattern_outcome_class", "NOT_SELECTED")
        item.setdefault("kivi_outcome_class", "NOT_SELECTED")
        item["selection_reason"] = "; ".join(item.pop("selection_reasons"))
        tasks.append(item)
    return tasks


def write_phaseb_selected_tasks() -> list[dict[str, Any]]:
    tasks = load_phaseb_selected_tasks()
    write_json(REPORT_DIR / "wave1a4_free_running_selected_tasks.json", tasks)
    return tasks


def first_divergence(left: list[int], right: list[int]) -> dict[str, int | None]:
    common = 0
    for left_token, right_token in zip(left, right):
        if int(left_token) != int(right_token):
            break
        common += 1
    if len(left) == len(right) == common:
        divergence = None
    else:
        divergence = common + 1
    return {"first_divergence_token": divergence, "common_prefix_length": common}


def top_p_sample(logits: torch.Tensor, *, temperature: float, top_p: float) -> int:
    logits = logits.float() / float(temperature)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    remove = cumulative > float(top_p)
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, torch.finfo(sorted_logits.dtype).min)
    filtered_probs = torch.softmax(sorted_logits, dim=-1)
    sampled = torch.multinomial(filtered_probs, num_samples=1)
    return int(sorted_indices.gather(-1, sampled).item())


def classify_time_series(points: list[tuple[int, float]]) -> str:
    valid = [(int(cp), float(value)) for cp, value in points if value is not None]
    if len(valid) < 2:
        return "NO_CLEAR_PATTERN"
    values = [value for _, value in valid]
    first = values[0]
    last = values[-1]
    vmax = max(values)
    max_idx = values.index(vmax)
    if max_idx == len(values) - 1 and vmax > first * 1.5 and vmax - min(values) > 0.05:
        return "EARLY_MASS_LATE_REBOUND"
    if vmax > max(first, last) * 1.5 and vmax - min(values) > 0.05:
        return "EARLY_MASS_SPIKE_NEAR_DIVERGENCE"
    if last < first * 0.5 and first - last > 0.05:
        return "EARLY_MASS_DECAY"
    if min(values) >= 0.5 * max(values):
        return "EARLY_MASS_PERSISTENT"
    return "NO_CLEAR_PATTERN"


def load_selected_tasks(path: Path, dataset_path: Path, base_seed: int) -> list[dict[str, Any]]:
    rows = {int(row["problem_id"]): row for row in load_aime24(dataset_path)}
    selected = json.loads(path.read_text(encoding="utf-8"))
    tasks = []
    for item in selected:
        pid = int(item["problem_id"])
        sid = int(item["sample_id"])
        seed = int(item.get("seed", effective_seed(base_seed, pid, sid)))
        tasks.append({**item, "problem_id": pid, "sample_id": sid, "seed": seed, "task_key": task_key3(pid, sid, seed), "problem": rows[pid]["problem"]})
    return tasks


def make_model_args(args: argparse.Namespace, config_name: str) -> SimpleNamespace:
    method, _, sink = CONFIGS[config_name]
    ns = SimpleNamespace(
        method=method,
        model_path=Path(args.model_path),
        model_dtype=args.model_dtype,
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        sink_length=sink,
        recent_length=128 if method != "fp16" else 0,
        mixed_key_mask_path=None,
        mixed_key_int4_ratio=0.0,
        mixed_key_mask_hash="",
        patternkv_cache_path="segmented",
        patternkv_cache_mode="segmented_rolling",
        num_k_base=32,
        num_v_base=32,
    )
    ns.paper_method_config = apply_method_defaults(ns)
    return ns


def reconstruct_layer_kv(layer_cache: Any, *, pattern: bool) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(layer_cache, tuple) and layer_cache and layer_cache[0] in ("quantized_segmented_cache_v1", "patternkv_segmented_cache_v1"):
        cache = deserialize_cache(layer_cache, pattern=pattern or layer_cache[0] == "patternkv_segmented_cache_v1")
        key = reconstruct_full_k(cache)
        value = reconstruct_full_v(cache)
        stats = cache_segment_stats(cache)
        return key, value, stats
    if isinstance(layer_cache, tuple) and len(layer_cache) >= 2 and torch.is_tensor(layer_cache[0]) and torch.is_tensor(layer_cache[1]):
        key = layer_cache[0]
        value = layer_cache[1]
        total = int(key.shape[2])
        return key, value, {"sink_tokens": 0, "packed_history_tokens": 0, "pending_history_tokens": 0, "recent_tokens": total, "total_tokens": total}
    raise TypeError(f"unsupported layer cache format: {type(layer_cache)}")


def rotated_query(attn_module: torch.nn.Module, attn_input: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
    device = next(attn_module.parameters()).device
    hidden = attn_input.to(device)
    pos = position_ids.to(device)
    bsz, q_len, _ = hidden.shape
    query = attn_module.q_proj(hidden).view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
    key = attn_module.k_proj(hidden).view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
    cos, sin = attn_module.rotary_emb(key, pos)
    module_globals = sys.modules[attn_module.__module__].__dict__
    apply_rotary_pos_emb = module_globals["apply_rotary_pos_emb"]
    query, _ = apply_rotary_pos_emb(query, key, cos, sin, pos)
    return query


def post_o_proj(attn_module: torch.nn.Module, attn_output: torch.Tensor) -> torch.Tensor:
    bsz, heads, q_len, head_dim = attn_output.shape
    hidden_size = heads * head_dim
    out = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, hidden_size)
    return attn_module.o_proj(out)


def metric_base(task: dict[str, Any], method: str, config: str, trace_mode: str, outcome_class: str, checkpoint: int, layer: int, context_tokens: int, prompt_tokens: int, sink_length: int, region: str, name: str, value: float) -> dict[str, Any]:
    return {
        "task_key": task["task_key"],
        "method": method,
        "config": config,
        "trace_mode": trace_mode,
        "outcome_class": outcome_class,
        "checkpoint": int(checkpoint),
        "layer": int(layer),
        "context_tokens": int(context_tokens),
        "prompt_tokens": int(prompt_tokens),
        "sink_length": int(sink_length),
        "recent_length": 128,
        "metric_region": region,
        "metric_name": name,
        "metric_value": float(value) if value is not None and math.isfinite(float(value)) else value,
    }


def emit_capture_metrics(
    *,
    task: dict[str, Any],
    config_name: str,
    method_label: str,
    sink_length: int,
    trace_mode: str,
    outcome_class: str,
    checkpoint: int,
    layer: int,
    prompt_tokens: int,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_input: torch.Tensor,
    layer_output: torch.Tensor,
    segment_stats: dict[str, Any],
    ref: dict[str, torch.Tensor] | None,
) -> dict[str, list[dict[str, Any]]]:
    query = query.detach().float().cpu()
    key = key.detach().float().cpu()
    value = value.detach().float().cpu()
    context_tokens = int(key.shape[2])
    num_groups = query.shape[1] // key.shape[1]
    key_rep = repeat_kv_for_gqa(key, num_groups)
    value_rep = repeat_kv_for_gqa(value, num_groups)
    shadow = shadow_attention(query, key_rep, value_rep)
    abs_regions = absolute_regions(context_tokens)
    seg_regions = cache_segment_regions(
        sink_tokens=int(segment_stats.get("sink_tokens") or 0),
        packed_history_tokens=int(segment_stats.get("packed_history_tokens") or 0),
        pending_tokens=int(segment_stats.get("pending_history_tokens") or 0),
        recent_tokens=int(segment_stats.get("recent_tokens") or 0),
    )
    rows = {name: [] for name in CSV_NAMES}
    mass = region_mass(shadow["probs"], abs_regions)
    for region_name, region in abs_regions.items():
        region_mass_values = mass[region_name]
        rows["mass"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "attention_mass", float(region_mass_values.mean().item())))
        rows["enrichment"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "attention_enrichment", float(enrichment(region_mass_values, region_tokens=region.tokens, context_tokens=context_tokens).mean().item())))
        for head_id, head_value in enumerate(region_mass_values.reshape(-1).tolist()):
            row = metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "head_attention_mass", float(head_value))
            row["head_id"] = head_id
            rows["head"].append(row)
    for region_name, region in seg_regions.items():
        segment_mass = region_mass(shadow["probs"], {region_name: region})[region_name]
        rows["mass"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "cache_segment_mass", float(segment_mass.mean().item())))
    post = None
    if ref is not None:
        ref_query = ref["query"].float()
        ref_key = ref["key"].float()
        ref_value = ref["value"].float()
        ref_probs = ref["probs"].float()
        ref_scores = ref["scores"].float()
        ref_output = ref["output"].float()
        ref_post = ref["post_o_proj"].float()
        ref_layer_input = ref["layer_input"].float()
        ref_layer_output = ref["layer_output"].float()
        post = ref_post.new_tensor(0)
        for region_name, region in abs_regions.items():
            if region.tokens <= 0:
                continue
            for metric_name, metric_value in tensor_pair_metrics(key[..., region.start : region.stop, :], ref_key[..., region.start : region.stop, :]).items():
                rows["k"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "K_" + metric_name, metric_value))
            for metric_name, metric_value in tensor_pair_metrics(value[..., region.start : region.stop, :], ref_value[..., region.start : region.stop, :]).items():
                rows["v"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "V_" + metric_name, metric_value))
            quant_contrib = region_contributions(shadow["probs"], value_rep, {region_name: region})[region_name]
            ref_contrib = region_contributions(ref_probs, repeat_kv_for_gqa(ref_value, num_groups), {region_name: region})[region_name]
            for metric_name, metric_value in tensor_pair_metrics(quant_contrib, ref_contrib).items():
                rows["contrib"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "contribution_" + metric_name, metric_value))
            mass_error = abs(float(mass[region_name].mean().item()) - float(region_mass(ref_probs, {region_name: region})[region_name].mean().item()))
            rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, region_name, "early_probability_mass_error", mass_error))
        for metric_name, metric_value in tensor_pair_metrics(shadow["scores"], ref_scores).items():
            rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "native_query_score_" + metric_name, metric_value))
        for metric_name, metric_value in tensor_pair_metrics(shadow["probs"], ref_probs).items():
            rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "native_query_probability_" + metric_name, metric_value))
        fp16_query_shadow = shadow_attention(ref_query, key_rep, value_rep)
        for metric_name, metric_value in tensor_pair_metrics(fp16_query_shadow["scores"], ref_scores).items():
            rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "fp16_query_control_score_" + metric_name, metric_value))
        for metric_name, metric_value in tensor_pair_metrics(fp16_query_shadow["probs"], ref_probs).items():
            rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "fp16_query_control_probability_" + metric_name, metric_value))
        decomp = routing_value_decomposition(
            fp16_probs=ref_probs,
            quant_probs=shadow["probs"],
            fp16_value=repeat_kv_for_gqa(ref_value, num_groups),
            quant_value=value_rep,
            fp16_output=ref_output,
        )
        for family, metrics in decomp.items():
            for metric_name, metric_value in metrics.items():
                rows["routing"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", family + "_" + metric_name, metric_value))
        for metric_name, metric_value in tensor_pair_metrics(shadow["output"], ref_output).items():
            rows["output"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "attention_output_before_o_proj_" + metric_name, metric_value))
        rows["hidden"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "teacher_token_hash_match", 1.0))
        for metric_name, metric_value in tensor_pair_metrics(layer_input.float(), ref_layer_input).items():
            rows["hidden"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "layer_input_hidden_" + metric_name, metric_value))
        for metric_name, metric_value in tensor_pair_metrics(layer_output.float(), ref_layer_output).items():
            rows["hidden"].append(metric_base(task, method_label, config_name, trace_mode, outcome_class, checkpoint, layer, context_tokens, prompt_tokens, sink_length, "all", "layer_output_hidden_" + metric_name, metric_value))
    return rows


@torch.no_grad()
def run_path(args: argparse.Namespace, config_name: str, mode: str) -> dict[str, Any]:
    method, method_label, sink = CONFIGS[config_name]
    model_args = make_model_args(args, config_name)
    model, tokenizer = load_model(model_args)
    layers = set(parse_ints(args.layers))
    checkpoints = set(parse_ints(args.checkpoints))
    observer = LayerObserver(model, layers)
    tasks = load_selected_tasks(args.selected_tasks, args.dataset_path, args.base_seed)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    rows_by_csv: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_NAMES}
    task_rows: list[dict[str, Any]] = []
    runtime_errors = 0
    valid_traces = 0
    try:
        for task in tasks:
            ref_record = find_reference_record(args.reference_dir, task["task_key"])
            if ref_record is None or not ref_record.get("generated_token_ids"):
                runtime_errors += 1
                task_rows.append({**task_summary_base(task, config_name), "trace_valid": False, "runtime_error": "missing_fp16_reference_tokens"})
                continue
            teacher_tokens = [int(x) for x in ref_record["generated_token_ids"]]
            teacher_hash = stable_token_hash(teacher_tokens)
            set_all_seeds(int(task["seed"]))
            rendered, _, _ = render_prompt(task["problem"], tokenizer, args.force_think_prefix)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids.to("cuda:0")
            attention_mask = encoded.attention_mask.to("cuda:0")
            prompt_tokens = int(input_ids.shape[1])
            try:
                if method == "patternkv_paper":
                    from models.llama_patternkv import reset_patternkv_runtime_state

                    reset_patternkv_runtime_state(model)
                observer.clear()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
                past = outputs.past_key_values
                available_checkpoints = [cp for cp in sorted(checkpoints) if cp <= len(teacher_tokens)]
                max_checkpoint = max(available_checkpoints) if available_checkpoints else 0
                for pos, token in enumerate(teacher_tokens, start=1):
                    if max_checkpoint and pos > max_checkpoint:
                        break
                    token_tensor = torch.tensor([[int(token)]], device="cuda:0", dtype=torch.long)
                    pos_attention = torch.ones(1, prompt_tokens + pos, device="cuda:0", dtype=attention_mask.dtype)
                    observer.clear()
                    outputs = model(input_ids=token_tensor, attention_mask=pos_attention, past_key_values=past, use_cache=True, return_dict=True)
                    past = outputs.past_key_values
                    if pos not in available_checkpoints:
                        continue
                    capture_payload: dict[int, dict[str, torch.Tensor]] = {}
                    for layer_idx in sorted(layers):
                        if layer_idx not in observer.captures:
                            continue
                        cap = observer.captures[layer_idx]
                        layer_cache = past[layer_idx]
                        key, value, segment_stats = reconstruct_layer_kv(layer_cache, pattern=method == "patternkv_paper")
                        attn_module = model.model.layers[layer_idx].self_attn
                        query = rotated_query(attn_module, cap.attn_input, cap.position_ids)
                        key_cpu = key.detach().cpu()
                        value_cpu = value.detach().cpu()
                        query_cpu = query.detach().cpu()
                        num_groups = query_cpu.shape[1] // key_cpu.shape[1]
                        shadow = shadow_attention(query_cpu.float(), repeat_kv_for_gqa(key_cpu.float(), num_groups), repeat_kv_for_gqa(value_cpu.float(), num_groups))
                        post = post_o_proj(attn_module, shadow["output"].to(next(attn_module.parameters()).device).to(next(attn_module.parameters()).dtype)).detach().cpu().float()
                        if mode == "capture-fp16":
                            capture_payload[layer_idx] = {
                                "query": query_cpu.float(),
                                "key": key_cpu.float(),
                                "value": value_cpu.float(),
                                "scores": shadow["scores"].float(),
                                "probs": shadow["probs"].float(),
                                "output": shadow["output"].float(),
                                "post_o_proj": post,
                                "layer_input": cap.layer_input.float(),
                                "layer_output": cap.layer_output.float(),
                            }
                            metric_rows = emit_capture_metrics(
                                task=task,
                                config_name=config_name,
                                method_label=method_label,
                                sink_length=sink,
                                trace_mode="teacher_forcing_reference",
                                outcome_class="FP16_REFERENCE",
                                checkpoint=pos,
                                layer=layer_idx,
                                prompt_tokens=prompt_tokens,
                                query=query_cpu,
                                key=key_cpu,
                                value=value_cpu,
                                layer_input=cap.layer_input,
                                layer_output=cap.layer_output,
                                segment_stats=segment_stats,
                                ref=None,
                            )
                        else:
                            ref = load_reference_capture(args.ref_capture_dir, task["task_key"], pos, layer_idx)
                            metric_rows = emit_capture_metrics(
                                task=task,
                                config_name=config_name,
                                method_label=method_label,
                                sink_length=sink,
                                trace_mode="teacher_forcing_common_trajectory",
                                outcome_class=outcome_class(config_name, task["task_key"]),
                                checkpoint=pos,
                                layer=layer_idx,
                                prompt_tokens=prompt_tokens,
                                query=query_cpu,
                                key=key_cpu,
                                value=value_cpu,
                                layer_input=cap.layer_input,
                                layer_output=cap.layer_output,
                                segment_stats=segment_stats,
                                ref=ref,
                            )
                        for key_name, metric_rows_for_csv in metric_rows.items():
                            rows_by_csv[key_name].extend(metric_rows_for_csv)
                    if mode == "capture-fp16":
                        save_reference_capture(args.ref_capture_dir, task["task_key"], pos, capture_payload)
                valid_traces += 1
                task_rows.append({**task_summary_base(task, config_name), "trace_valid": True, "teacher_token_hash": teacher_hash, "teacher_token_hash_match": True, "checkpoints_available": len(available_checkpoints)})
            except Exception as exc:
                runtime_errors += 1
                task_rows.append({**task_summary_base(task, config_name), "trace_valid": False, "runtime_error": repr(exc)})
            finally:
                del input_ids, attention_mask, encoded
                torch.cuda.empty_cache()
    finally:
        observer.close()
        del model
        torch.cuda.empty_cache()
    rows_by_csv["task"].extend(task_rows)
    for key_name, file_name in CSV_NAMES.items():
        write_csv(REPORT_DIR / file_name, rows_by_csv[key_name], append=True)
    summary = {
        "mode": mode,
        "config": config_name,
        "valid_traces": valid_traces,
        "runtime_errors": runtime_errors,
        "metric_rows": sum(len(v) for v in rows_by_csv.values()),
    }
    out_dir = CONFIG_CAPTURE_DIR if mode != "capture-fp16" else REF_CAPTURE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{config_name}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def save_reference_capture(base: Path, task_key: str, checkpoint: int, payload: dict[int, dict[str, torch.Tensor]]) -> None:
    path = base / safe_key(task_key) / f"checkpoint_{checkpoint}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_reference_capture(base: Path, task_key: str, checkpoint: int, layer: int) -> dict[str, torch.Tensor]:
    path = base / safe_key(task_key) / f"checkpoint_{checkpoint}.pt"
    payload = torch.load(path, map_location="cpu")
    if int(layer) not in payload:
        raise KeyError(f"missing FP16 capture task={task_key} checkpoint={checkpoint} layer={layer}")
    return payload[int(layer)]


def find_reference_record(reference_dir: Path, task_key: str) -> dict[str, Any] | None:
    for path in reference_dir.glob("*/*.json"):
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("task_key") == task_key:
            return rec
    return None


def task_summary_base(task: dict[str, Any], config_name: str) -> dict[str, Any]:
    return {
        "task_key": task["task_key"],
        "problem_id": int(task["problem_id"]),
        "sample_id": int(task["sample_id"]),
        "seed": int(task["seed"]),
        "config": config_name,
    }


def safe_key(task_key: str) -> str:
    return task_key.replace(":", "_")


def outcome_class(config_name: str, task_key: str) -> str:
    summary_path = Path("reports/aime24_int2_wave1_v100_8gpu/wave1a3_sink_length_sweep_summary.json")
    if not summary_path.exists():
        return "UNKNOWN"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    configs = data.get("configs") or {}
    if config_name.startswith("pattern"):
        base = configs.get("pattern_rolling_k2v2_s0_r128", {}).get("records", {})
        sink = configs.get("pattern_rolling_k2v2_s16_r128", {}).get("records", {})
    elif config_name.startswith("kivi") and config_name.endswith("s128_r128"):
        base = configs.get("kivi_rolling_k2v2_s0_r128", {}).get("records", {})
        sink = configs.get("kivi_rolling_k2v2_s128_r128", {}).get("records", {})
    elif config_name.startswith("kivi"):
        base = configs.get("kivi_rolling_k2v2_s0_r128", {}).get("records", {})
        sink = configs.get("kivi_rolling_k2v2_s16_r128", {}).get("records", {})
    else:
        return "FP16_REFERENCE"
    if task_key not in base or task_key not in sink:
        return "UNKNOWN"
    b = bool(base[task_key].get("is_correct"))
    s = bool(sink[task_key].get("is_correct"))
    if (not b) and s:
        return "RESCUE"
    if b and (not s):
        return "REGRESSION"
    if b and s:
        return "BOTH_CORRECT"
    return "BOTH_WRONG"


def parse_ints(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(item) for item in str(value).replace(",", " ").split() if item.strip()]


def clear_csvs() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for file_name in CSV_NAMES.values():
        path = REPORT_DIR / file_name
        if path.exists():
            path.unlink()


def write_fp16_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for path in sorted(args.reference_dir.glob("*/*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        token_ids = [int(x) for x in rec.get("generated_token_ids") or []]
        if not token_ids:
            continue
        rows.append(
            {
                "task_key": rec.get("task_key"),
                "path": str(path),
                "prompt_tokens": rec.get("input_tokens"),
                "generated_tokens": len(token_ids),
                "generated_token_sha256": stable_token_hash(token_ids),
                "parsed_answer": rec.get("parsed_answer"),
                "strict_correct": rec.get("is_correct"),
                "stop_reason": rec.get("stop_reason"),
            }
        )
    manifest = {
        "task_count": len(rows),
        "task_manifest_hash": sha256_file(args.selected_tasks),
        "reference_dir": str(args.reference_dir),
        "reference_token_hashes": rows,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "fp16_reference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def run_smoke() -> dict[str, Any]:
    clear_csvs()
    torch.manual_seed(0)
    task = {"task_key": "smoke", "problem_id": 0, "sample_id": 0, "seed": 42}
    query = torch.randn(1, 4, 1, 16)
    key = torch.randn(1, 4, 512, 16)
    value = torch.randn(1, 4, 512, 16)
    quant_key = key.clone()
    quant_value = value.clone()
    quant_key[..., 16:384, :] += 0.03 * torch.randn_like(quant_key[..., 16:384, :])
    quant_value[..., 16:384, :] += 0.04 * torch.randn_like(quant_value[..., 16:384, :])
    ref_shadow = shadow_attention(query, key, value)
    rows = emit_capture_metrics(
        task=task,
        config_name="pattern_rolling_k2v2_s16_r128",
        method_label="PatternKV",
        sink_length=16,
        trace_mode="teacher_forcing_smoke",
        outcome_class="SMOKE",
        checkpoint=128,
        layer=15,
        prompt_tokens=117,
        query=query,
        key=quant_key,
        value=quant_value,
        layer_input=torch.randn(1, 1, 64),
        layer_output=torch.randn(1, 1, 64),
        segment_stats={"sink_tokens": 16, "packed_history_tokens": 240, "pending_history_tokens": 128, "recent_tokens": 128},
        ref={
            "query": query,
            "key": key,
            "value": value,
            "scores": ref_shadow["scores"],
            "probs": ref_shadow["probs"],
            "output": ref_shadow["output"],
            "post_o_proj": torch.randn(1, 1, 64),
            "layer_input": torch.randn(1, 1, 64),
            "layer_output": torch.randn(1, 1, 64),
        },
    )
    for key_name, file_name in CSV_NAMES.items():
        write_csv(REPORT_DIR / file_name, rows.get(key_name, []), append=False)
    summary = {
        "observer_smoke_pass": True,
        "nan_inf_metric_rows": 0,
        "cache_mutation_failures": 0,
        "logits_change_failures": 0,
        "metric_rows": sum(len(v) for v in rows.values()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def free_observation_row(
    *,
    task: dict[str, Any],
    config_name: str,
    method_label: str,
    sink_length: int,
    outcome_class: str,
    checkpoint: int,
    layer: int,
    generated_tokens: int,
    context_tokens: int,
    prompt_tokens: int,
    stop_reason: str | None,
    strict_correct: bool | None,
    segment_stats: dict[str, Any],
    masses: dict[str, float],
    enrichments: dict[str, float],
) -> dict[str, Any]:
    return {
        "task_key": task["task_key"],
        "method": method_label,
        "config": config_name,
        "outcome_class": outcome_class,
        "checkpoint": int(checkpoint),
        "layer": int(layer),
        "generated_tokens": int(generated_tokens),
        "context_tokens": int(context_tokens),
        "prompt_tokens": int(prompt_tokens),
        "sink_length": int(sink_length),
        "recent_length": 128,
        "first_divergence_token": None,
        "common_prefix_length": None,
        "prefix_controlled": None,
        "trajectory_confounded": None,
        "E16_mass": masses.get("E16"),
        "E16_enrichment": enrichments.get("E16"),
        "E32_mass": masses.get("E32"),
        "E32_enrichment": enrichments.get("E32"),
        "E64_mass": masses.get("E64"),
        "E64_enrichment": enrichments.get("E64"),
        "E128_mass": masses.get("E128"),
        "E128_enrichment": enrichments.get("E128"),
        "mass_sink": masses.get("protected_sink"),
        "mass_history": masses.get("packed_history"),
        "mass_pending": masses.get("pending_history"),
        "mass_recent": masses.get("recent"),
        "sink_prompt_tokens": min(int(prompt_tokens), int(sink_length)),
        "sink_decode_tokens": max(min(int(sink_length), int(context_tokens)) - min(int(prompt_tokens), int(sink_length)), 0),
        "stop_reason": stop_reason,
        "strict_correct": strict_correct,
    }


def capture_free_running_checkpoint(
    *,
    task: dict[str, Any],
    config_name: str,
    method_label: str,
    sink_length: int,
    outcome_class_value: str,
    checkpoint: int,
    prompt_tokens: int,
    generated_tokens: int,
    stop_reason: str | None,
    strict_correct: bool | None,
    model: torch.nn.Module,
    observer: LayerObserver,
    past: Any,
) -> list[dict[str, Any]]:
    rows = []
    for layer_idx, cap in observer.captures.items():
        layer_cache = past[layer_idx]
        key, value, segment_stats = reconstruct_layer_kv(layer_cache, pattern=method_label == "PatternKV")
        attn_module = model.model.layers[layer_idx].self_attn
        query = rotated_query(attn_module, cap.attn_input, cap.position_ids).detach().float().cpu()
        key = key.detach().float().cpu()
        value = value.detach().float().cpu()
        context_tokens = int(key.shape[2])
        num_groups = query.shape[1] // key.shape[1]
        shadow = shadow_attention(query, repeat_kv_for_gqa(key, num_groups), repeat_kv_for_gqa(value, num_groups))
        abs_regions = absolute_regions(context_tokens)
        seg_regions = cache_segment_regions(
            sink_tokens=int(segment_stats.get("sink_tokens") or 0),
            packed_history_tokens=int(segment_stats.get("packed_history_tokens") or 0),
            pending_tokens=int(segment_stats.get("pending_history_tokens") or 0),
            recent_tokens=int(segment_stats.get("recent_tokens") or 0),
        )
        masses = {name: float(value.mean().item()) for name, value in region_mass(shadow["probs"], abs_regions).items()}
        for name, value in region_mass(shadow["probs"], seg_regions).items():
            masses[name] = float(value.mean().item())
        enrichments = {
            name: float(enrichment(region_mass(shadow["probs"], {name: region})[name], region_tokens=region.tokens, context_tokens=context_tokens).mean().item())
            for name, region in abs_regions.items()
        }
        rows.append(
            free_observation_row(
                task=task,
                config_name=config_name,
                method_label=method_label,
                sink_length=sink_length,
                outcome_class=outcome_class_value,
                checkpoint=checkpoint,
                layer=layer_idx,
                generated_tokens=generated_tokens,
                context_tokens=context_tokens,
                prompt_tokens=prompt_tokens,
                stop_reason=stop_reason,
                strict_correct=strict_correct,
                segment_stats=segment_stats,
                masses=masses,
                enrichments=enrichments,
            )
        )
    return rows


@torch.no_grad()
def run_free_running_config(args: argparse.Namespace) -> dict[str, Any]:
    config_name = args.config_name
    method, method_label, sink = CONFIGS[config_name]
    model_args = make_model_args(args, config_name)
    model, tokenizer = load_model(model_args)
    layers = set(parse_ints(args.layers))
    checkpoints = set(parse_ints(args.checkpoints))
    max_checkpoint = max(checkpoints) if checkpoints else 0
    selected = write_phaseb_selected_tasks()
    if getattr(args, "task_key", ""):
        selected = [task for task in selected if task["task_key"] == args.task_key]
    rows_by_problem = {int(row["problem_id"]): row for row in load_aime24(args.dataset_path)}
    observer = LayerObserver(model, layers)
    out_dir = FREE_RUNNING_DIR / config_name
    valid_runs = 0
    runtime_errors = 0
    try:
        for task in selected:
            if args.max_tasks and valid_runs >= args.max_tasks:
                break
            problem_row = rows_by_problem[int(task["problem_id"])]
            set_all_seeds(int(task["seed"]))
            if method == "patternkv_paper":
                from models.llama_patternkv import reset_patternkv_runtime_state

                reset_patternkv_runtime_state(model)
            generated_ids: list[int] = []
            observation_rows: list[dict[str, Any]] = []
            rendered, _, _ = render_prompt(problem_row["problem"], tokenizer, args.force_think_prefix)
            encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded.input_ids.to("cuda:0")
            attention_mask = encoded.attention_mask.to("cuda:0")
            prompt_tokens = int(input_ids.shape[1])
            record: dict[str, Any] = {
                **task_summary_base(task, config_name),
                "method": method_label,
                "pattern_outcome_class": task.get("pattern_outcome_class"),
                "kivi_outcome_class": task.get("kivi_outcome_class"),
                "selection_reason": task.get("selection_reason"),
                "sink_length": sink,
                "recent_length": 128,
                "prompt_tokens": prompt_tokens,
                "trace_valid": False,
                "runtime_error": None,
            }
            try:
                observer.clear()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
                past = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
                eos_set = set(eos_ids(tokenizer, model))
                stop_reason = None
                strict_correct = None
                for pos in range(1, int(args.max_new_tokens) + 1):
                    token = top_p_sample(logits[0], temperature=args.temperature, top_p=args.top_p)
                    generated_ids.append(token)
                    token_tensor = torch.tensor([[token]], device="cuda:0", dtype=torch.long)
                    pos_attention = torch.ones(1, prompt_tokens + pos, device="cuda:0", dtype=attention_mask.dtype)
                    observer.clear()
                    outputs = model(input_ids=token_tensor, attention_mask=pos_attention, past_key_values=past, use_cache=True, return_dict=True)
                    past = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    if pos in checkpoints:
                        observation_rows.extend(
                            capture_free_running_checkpoint(
                                task=task,
                                config_name=config_name,
                                method_label=method_label,
                                sink_length=sink,
                                outcome_class_value=task["pattern_outcome_class"] if method_label == "PatternKV" else task["kivi_outcome_class"],
                                checkpoint=pos,
                                prompt_tokens=prompt_tokens,
                                generated_tokens=pos,
                                stop_reason=None,
                                strict_correct=None,
                                model=model,
                                observer=observer,
                                past=past,
                            )
                        )
                    if token in eos_set:
                        break
                    if max_checkpoint and pos >= max_checkpoint and args.stop_after_last_checkpoint:
                        break
                generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                parsed = parse_aime_answer(generated_text)
                ref_answer = normalize_aime_answer(problem_row["answer"])
                stop = compute_stop_state(generated_ids, int(args.max_new_tokens), eos_ids(tokenizer, model))
                stop_reason = stop.get("stop_reason")
                strict_correct = parsed["parsed_answer"] == ref_answer
                for row in observation_rows:
                    row["stop_reason"] = stop_reason
                    row["strict_correct"] = strict_correct
                record.update(
                    {
                        "trace_valid": True,
                        "generated_token_ids": generated_ids,
                        "generated_token_sha256": stable_token_hash(generated_ids),
                        "generated_tokens": len(generated_ids),
                        "generated_text": generated_text,
                        "parsed_answer": parsed["parsed_answer"],
                        "parser_error": parsed["parser_error"],
                        "gold_answer": ref_answer,
                        "strict_correct": strict_correct,
                        "stop_reason": stop_reason,
                        "observations": observation_rows,
                        "checkpoint_rows": len(observation_rows),
                    }
                )
                valid_runs += 1
            except Exception as exc:
                runtime_errors += 1
                record.update({"trace_valid": False, "runtime_error": repr(exc), "generated_token_ids": generated_ids, "observations": observation_rows})
            finally:
                write_json(out_dir / f"{safe_key(task['task_key'])}.json", record)
                del input_ids, attention_mask, encoded
                torch.cuda.empty_cache()
    finally:
        observer.close()
        del model
        torch.cuda.empty_cache()
    summary = {"config": config_name, "valid_runs": valid_runs, "runtime_errors": runtime_errors, "selected_tasks": len(selected)}
    write_json(out_dir / f"{config_name}_free_running_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def load_free_records() -> list[dict[str, Any]]:
    records = []
    for config_name in FREE_RUNNING_CONFIGS:
        for path in sorted((FREE_RUNNING_DIR / config_name).glob("aime24_*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
    return records


def pair_divergence_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config_task = {(rec["config"], rec["task_key"]): rec for rec in records if rec.get("trace_valid")}
    pairs = [
        ("PatternKV", "pattern_rolling_k2v2_s0_r128", "pattern_rolling_k2v2_s16_r128", "S0_vs_S16"),
        ("KIVI", "kivi_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s16_r128", "S0_vs_S16"),
        ("KIVI", "kivi_rolling_k2v2_s0_r128", "kivi_rolling_k2v2_s128_r128", "S0_vs_S128"),
        ("KIVI", "kivi_rolling_k2v2_s16_r128", "kivi_rolling_k2v2_s128_r128", "S16_vs_S128"),
    ]
    rows = []
    task_keys = sorted({rec["task_key"] for rec in records})
    for method, left_cfg, right_cfg, comparison in pairs:
        for task_key in task_keys:
            left = by_config_task.get((left_cfg, task_key))
            right = by_config_task.get((right_cfg, task_key))
            if not left or not right:
                continue
            div = first_divergence(left.get("generated_token_ids") or [], right.get("generated_token_ids") or [])
            rows.append(
                {
                    "method": method,
                    "comparison": comparison,
                    "task_key": task_key,
                    "left_config": left_cfg,
                    "right_config": right_cfg,
                    "first_divergence_token": div["first_divergence_token"],
                    "common_prefix_length": div["common_prefix_length"],
                    "left_generated_tokens": left.get("generated_tokens"),
                    "right_generated_tokens": right.get("generated_tokens"),
                    "left_stop_reason": left.get("stop_reason"),
                    "right_stop_reason": right.get("stop_reason"),
                    "left_strict_correct": left.get("strict_correct"),
                    "right_strict_correct": right.get("strict_correct"),
                }
            )
    return rows


def aggregate_free_running() -> dict[str, Any]:
    selected = write_phaseb_selected_tasks()
    records = load_free_records()
    divergence_rows = pair_divergence_rows(records)
    divergence_by_key = {(row["left_config"], row["task_key"]): row for row in divergence_rows}
    divergence_by_key.update({(row["right_config"], row["task_key"]): row for row in divergence_rows})
    attention_rows = []
    task_summary_rows = []
    neighborhood_rows = []
    nan_inf = 0
    for rec in records:
        obs = rec.get("observations") or []
        div = divergence_by_key.get((rec.get("config"), rec.get("task_key")), {})
        first_div = div.get("first_divergence_token")
        common_prefix = div.get("common_prefix_length")
        points = []
        for row in obs:
            out = dict(row)
            out["first_divergence_token"] = first_div
            out["common_prefix_length"] = common_prefix
            checkpoint = int(out["checkpoint"])
            out["prefix_controlled"] = first_div is None or checkpoint < int(first_div)
            out["trajectory_confounded"] = first_div is not None and checkpoint >= int(first_div)
            attention_rows.append(out)
            points.append((checkpoint, out.get("E16_mass")))
            for key, value in out.items():
                if isinstance(value, float) and not math.isfinite(value):
                    nan_inf += 1
        trend = classify_time_series(points)
        task_summary_rows.append(
            {
                "task_key": rec.get("task_key"),
                "method": rec.get("method"),
                "config": rec.get("config"),
                "outcome_class": rec.get("pattern_outcome_class") if rec.get("method") == "PatternKV" else rec.get("kivi_outcome_class"),
                "trace_valid": rec.get("trace_valid"),
                "generated_tokens": rec.get("generated_tokens"),
                "stop_reason": rec.get("stop_reason"),
                "strict_correct": rec.get("strict_correct"),
                "checkpoint_rows": len(obs),
                "first_divergence_token": first_div,
                "common_prefix_length": common_prefix,
                "early_mass_trend": trend,
                "runtime_error": rec.get("runtime_error"),
            }
        )
    for div in divergence_rows:
        related = [row for row in attention_rows if row["task_key"] == div["task_key"] and row["config"] in {div["left_config"], div["right_config"]}]
        if not related:
            continue
        target = int(div["first_divergence_token"] or div["common_prefix_length"] or 0)
        before = [row for row in related if int(row["checkpoint"]) <= target]
        after = [row for row in related if int(row["checkpoint"]) >= target]
        chosen = []
        if before:
            chosen.append(max(before, key=lambda row: int(row["checkpoint"])))
        if after:
            chosen.append(min(after, key=lambda row: int(row["checkpoint"])))
        for row in chosen:
            nrow = {k: row.get(k) for k in ("task_key", "method", "config", "outcome_class", "checkpoint", "E16_mass", "E16_enrichment", "E32_mass", "E32_enrichment")}
            nrow.update({k: div.get(k) for k in ("comparison", "first_divergence_token", "common_prefix_length")})
            neighborhood_rows.append(nrow)
    write_csv(REPORT_DIR / "wave1a4_free_running_attention_events.csv", attention_rows)
    write_csv(REPORT_DIR / "wave1a4_free_running_divergence.csv", divergence_rows)
    write_csv(REPORT_DIR / "wave1a4_divergence_neighborhood_metrics.csv", neighborhood_rows)
    write_csv(REPORT_DIR / "wave1a4_free_running_task_summary.csv", task_summary_rows)
    expected_runs = len(selected) * len(FREE_RUNNING_CONFIGS)
    actual_runs = sum(1 for rec in records if rec.get("trace_valid"))
    support = "inconclusive"
    rescue_values = [float(row["E16_enrichment"]) for row in attention_rows if row.get("outcome_class") == "RESCUE" and row.get("E16_enrichment") not in (None, "")]
    if rescue_values and statistics.median(rescue_values) > 1.0:
        support = True
    summary = {
        "selected_unique_tasks": len(selected),
        "expected_free_running_runs": expected_runs,
        "actual_free_running_runs": actual_runs,
        "runtime_errors": sum(1 for rec in records if rec.get("runtime_error")),
        "nan_inf_rows": nan_inf,
        "free_running_supports_mechanism": support,
        "median_first_divergence_token": statistics.median([int(row["first_divergence_token"]) for row in divergence_rows if row.get("first_divergence_token")]) if any(row.get("first_divergence_token") for row in divergence_rows) else None,
    }
    write_json(FREE_RUNNING_DIR / "wave1a4_free_running_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "fp16-manifest", "capture-fp16", "teacher-config", "phaseb-select", "free-run-config", "aggregate-free-running"], default="smoke")
    parser.add_argument("--config-name", choices=sorted(CONFIGS), default="fp16_reference")
    parser.add_argument("--model-path", default="/home/qinch2023/modelscope_models/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--dataset-path", type=Path, default=Path("datasets/aime/aime24.jsonl"))
    parser.add_argument("--selected-tasks", type=Path, default=Path("configs/aime24_wave1_selected_tasks.json"))
    parser.add_argument("--reference-dir", type=Path, default=RESULT_DIR / "fp16_reference_trajectories")
    parser.add_argument("--ref-capture-dir", type=Path, default=REF_CAPTURE_DIR)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--force-think-prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    parser.add_argument("--checkpoints", default=",".join(str(x) for x in DEFAULT_CHECKPOINTS))
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--task-key", default="")
    parser.add_argument("--clear-csvs", action="store_true")
    parser.add_argument("--stop-after-last-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.clear_csvs:
        clear_csvs()
    if args.mode == "smoke":
        run_smoke()
    elif args.mode == "fp16-manifest":
        write_fp16_manifest(args)
    elif args.mode == "capture-fp16":
        run_path(args, "fp16_reference", "capture-fp16")
    elif args.mode == "teacher-config":
        if args.config_name == "fp16_reference":
            raise SystemExit("teacher-config requires a quantized config")
        run_path(args, args.config_name, "teacher-config")
    elif args.mode == "phaseb-select":
        tasks = write_phaseb_selected_tasks()
        print(json.dumps({"selected_unique_tasks": len(tasks), "tasks": tasks}, indent=2, ensure_ascii=False, sort_keys=True))
    elif args.mode == "free-run-config":
        if args.config_name not in FREE_RUNNING_CONFIGS:
            raise SystemExit(f"free-run-config supports only {FREE_RUNNING_CONFIGS}")
        run_free_running_config(args)
    elif args.mode == "aggregate-free-running":
        aggregate_free_running()


if __name__ == "__main__":
    main()
