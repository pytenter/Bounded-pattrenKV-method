from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
VENDOR = os.environ.get("QWEN3_TRANSFORMERS_VENDOR")
if VENDOR and VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from transformers import AutoConfig, AutoTokenizer

from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV
from models.qwen3_patternkv_system import (
    Qwen3ForCausalLM_PatternKVCompressed,
    Qwen3PatternKVCompressedCache,
    _compressed_attention,
    patternkv_mixed_value_attention,
    patternkv_request_invariant_qk_scores,
)
from models.segmented_cache import (
    build_k_segment_validity_mask,
    cache_segment_stats,
    k_segment_valid_lengths,
    reconstruct_full_k,
    reconstruct_full_v,
    request_invariant_full_value_attention,
    request_invariant_segmented_attention_softmax,
)
from quant.matmul import cuda_bmm_fA_qB_outer_with_base, fp16_tail_value_fusion_enabled

OUT = ROOT / "reports/qwen3_v100_system_generalization_v1"
MODEL = "/home/qinch2023/modelscope_models/Qwen3-8B"
CONTEXT = int(os.environ.get("QWEN3_CLOSURE_CONTEXT", "512"))
PROMPT_INDEX = 0
BASE = "Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. "
CFG_VALUES = dict(
    k_bits=2,
    v_bits=2,
    group_size=128,
    sink_length=16,
    recent_length=128,
    residual_length=128,
    num_k_base=32,
    num_v_base=32,
    patternkv_cache_mode="segmented_rolling",
    patternkv_value_objective="base",
    patternkv_v_precision_selector="causal_v4",
    patternkv_v4_budget_fraction=0.25,
    patternkv_random_selector_seed=20260809,
)


def make_config(task: str):
    config = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    for key, value in CFG_VALUES.items():
        setattr(config, key, value)
    setattr(config, "patternkv_selector_task_key", task)
    return config


def prompt_ids(tokenizer) -> torch.Tensor:
    ids = tokenizer(BASE * 160, return_tensors="pt", add_special_tokens=False).input_ids[:, :CONTEXT]
    if ids.shape[1] < CONTEXT:
        raise RuntimeError(f"prompt too short: {ids.shape[1]} < {CONTEXT}")
    return ids


def metrics(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None or b is None:
        return {"present": False}
    af = a.detach().float()
    bf = b.detach().float()
    diff = af - bf
    an = af.norm().clamp_min(1e-12)
    bn = bf.norm().clamp_min(1e-12)
    return {
        "present": True,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "max_abs": float(diff.abs().max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.abs().mean().item()) if diff.numel() else 0.0,
        "rel_l2": float(diff.norm().div(an).item()) if diff.numel() else 0.0,
        "cosine": float((af.flatten() * bf.flatten()).sum().div(an * bn).item()) if diff.numel() else 1.0,
    }


def summarize_tensor(x: torch.Tensor | None) -> dict[str, Any]:
    if x is None:
        return {"present": False}
    xf = x.detach().float()
    return {
        "present": True,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "mean": float(xf.mean().item()) if xf.numel() else 0.0,
        "std": float(xf.std().item()) if xf.numel() > 1 else 0.0,
        "min": float(xf.min().item()) if xf.numel() else 0.0,
        "max": float(xf.max().item()) if xf.numel() else 0.0,
        "sum": float(xf.sum().item()) if xf.numel() else 0.0,
    }


class DecodeTrace:
    def __init__(self, model):
        self.records: dict[int, dict[str, torch.Tensor]] = {}
        self.handles = []
        for idx, layer in enumerate(model.model.layers):
            self.records[idx] = {}
            self.handles.append(layer.register_forward_pre_hook(self._pre(idx)))
            self.handles.append(layer.self_attn.register_forward_hook(self._attn(idx)))
            self.handles.append(layer.mlp.register_forward_hook(self._mlp(idx)))
            self.handles.append(layer.register_forward_hook(self._layer(idx)))

    def _pre(self, idx):
        def hook(_module, args):
            self.records[idx]["input_hidden"] = args[0].detach().cpu()
        return hook

    def _attn(self, idx):
        def hook(_module, _args, output):
            self.records[idx]["attention_output"] = output[0].detach().cpu()
        return hook

    def _mlp(self, idx):
        def hook(_module, _args, output):
            self.records[idx]["mlp_output"] = output.detach().cpu()
        return hook

    def _layer(self, idx):
        def hook(_module, _args, output):
            self.records[idx]["layer_output"] = output[0].detach().cpu()
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def run_one(cls: Any, label: str, ids_cpu: torch.Tensor) -> dict[str, Any]:
    model = cls.from_pretrained(
        MODEL,
        local_files_only=True,
        config=make_config(label),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    ids = ids_cpu.to("cuda:0")
    with torch.no_grad():
        prefill = model(input_ids=ids, use_cache=True, return_dict=True)
        prefill_logits = prefill.logits[:, -1, :].detach().float().cpu()
        token = prefill.logits[:, -1, :].argmax(dim=-1)
        cache_before = prefill.past_key_values
        stats_before = {}
        if isinstance(cache_before, Qwen3PatternKVCompressedCache):
            stats_before = cache_segment_stats(cache_before.layer_caches[0])
        tracer = DecodeTrace(model)
        decoded = model(input_ids=token.view(1, 1), past_key_values=prefill.past_key_values, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        tracer.close()
        decode_logits = decoded.logits[:, -1, :].detach().float().cpu()
        stats_after = {}
        if isinstance(decoded.past_key_values, Qwen3PatternKVCompressedCache):
            stats_after = cache_segment_stats(decoded.past_key_values.layer_caches[0])
    out = {
        "label": label,
        "model": model,
        "prefill_logits": prefill_logits,
        "decode_logits": decode_logits,
        "decode_token": int(token.item()),
        "cache": decoded.past_key_values,
        "trace": tracer.records,
        "stats_before": stats_before,
        "stats_after": stats_after,
    }
    return out


def layer_scan(ref: dict[str, Any], comp: dict[str, Any]) -> tuple[list[dict[str, Any]], int, str]:
    rows = []
    first_layer = -1
    first_component = "NONE"
    for idx in sorted(ref["trace"].keys()):
        row: dict[str, Any] = {"layer": idx}
        for name in ["input_hidden", "attention_output", "mlp_output", "layer_output"]:
            row[name] = metrics(ref["trace"][idx].get(name), comp["trace"][idx].get(name))
        rows.append(row)
        inp = row["input_hidden"].get("rel_l2", 999.0)
        for name in ["attention_output", "mlp_output", "layer_output"]:
            rel = row[name].get("rel_l2", 0.0)
            if first_layer < 0 and inp < 1e-5 and rel > 1e-3:
                first_layer = idx
                first_component = name
                break
    if first_layer < 0:
        for row in rows:
            if row["layer_output"].get("rel_l2", 0.0) > 1e-3:
                first_layer = int(row["layer"])
                first_component = "layer_output"
                break
    return rows, first_layer, first_component


def component_oracles(comp: dict[str, Any], layer_idx: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    model = comp["model"]
    layer = model.model.layers[layer_idx]
    attn = layer.self_attn
    cache = comp["cache"].layer_caches[layer_idx]
    hidden = comp["trace"][layer_idx]["input_hidden"].to("cuda:0")
    input_shape = hidden.shape[:-1]
    hidden_shape = (*input_shape, -1, attn.head_dim)
    pos = torch.tensor([[CONTEXT]], dtype=torch.long, device=hidden.device)
    cos, sin = model.model.rotary_emb(hidden, pos)
    with torch.no_grad():
        q_proj = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        q_norm = attn.q_norm(q_proj)
        k_proj = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        k_norm = attn.k_norm(k_proj)
        v_proj = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
        q_rope, k_rope = apply_rotary_pos_emb(q_norm, k_norm, cos, sin)
        full_k = reconstruct_full_k(cache)
        full_v = reconstruct_full_v(cache)
        value_parts = []
        score_parts = []
        oracle_parts = []
        offset = 0
        if cache.sink_k is not None:
            value_parts.append(("sink", int(cache.sink_k.shape[2])))
            score_parts.append(patternkv_request_invariant_qk_scores(q_rope, cache.sink_k, attn.num_key_value_groups))
        if cache.packed_k is not None and int(cache.packed_k_tokens):
            value_parts.append(("packed", int(cache.packed_k_tokens)))
            packed_scores = cuda_bmm_fA_qB_outer_with_base(
                attn.group_size,
                q_rope,
                cache.packed_k,
                cache.packed_k_scale,
                cache.packed_k_zero,
                attn.k_bits,
                cache.k_centroids,
                cache.k_assignments[:, :, : cache.packed_k_tokens],
                attn.num_heads,
                attn.num_key_value_heads,
            )[:, :, :, : cache.packed_k_tokens]
            score_parts.append(packed_scores)
        if cache.pending_k is not None:
            value_parts.append(("pending", int(cache.pending_k.shape[2])))
            score_parts.append(patternkv_request_invariant_qk_scores(q_rope, cache.pending_k, attn.num_key_value_groups))
        if cache.recent_k is not None:
            value_parts.append(("recent", int(cache.recent_k.shape[2])))
            score_parts.append(patternkv_request_invariant_qk_scores(q_rope, cache.recent_k, attn.num_key_value_groups))
        comp_scores_unscaled = torch.cat(score_parts, dim=-1)
        ref_scores_unscaled = patternkv_request_invariant_qk_scores(q_rope, full_k, attn.num_key_value_groups)
        for name, length in value_parts:
            oracle_parts.append({"name": name, "metrics": metrics(ref_scores_unscaled[:, :, :, offset:offset+length], comp_scores_unscaled[:, :, :, offset:offset+length])})
            offset += length
        comp_scores = comp_scores_unscaled * float(attn.scaling)
        ref_scores = ref_scores_unscaled * float(attn.scaling)
        validity = build_k_segment_validity_mask(cache, value_parts, device=comp_scores.device)
        if validity is not None:
            comp_scores = comp_scores.masked_fill(~validity[:, None, None, :], torch.finfo(comp_scores.dtype).min)
            ref_scores = ref_scores.masked_fill(~validity[:, None, None, :], torch.finfo(ref_scores.dtype).min)
        segmented_probs = request_invariant_segmented_attention_softmax(comp_scores, cache, value_parts)
        torch_probs = torch.softmax(comp_scores.float(), dim=-1).to(comp_scores.dtype)
        ref_probs = torch.softmax(ref_scores.float(), dim=-1).to(ref_scores.dtype)
        ref_value = torch.matmul(ref_probs, repeat_kv(full_v, int(attn.num_key_value_groups)))
        comp_value, comp_probs_from_backend = _compressed_attention(attn, q_rope, cache, None)
        offset = 0
        value_rows = []
        tail_rows = []
        comp_value_sum = None
        for name, length in value_parts:
            weights = segmented_probs[:, :, :, offset:offset+length]
            if name == "packed":
                v_mask = cache.v_pattern_mask if getattr(cache, "v_pattern_mask", None) is not None else cache.v_assignments
                packed_value = patternkv_mixed_value_attention(attn, cache, weights, v_mask, int(cache.packed_v_tokens))
                ref_packed_v = full_v[:, :, offset:offset+length, :]
                oracle_value = torch.matmul(weights, repeat_kv(ref_packed_v, int(attn.num_key_value_groups)))
                value_rows.append({"name": name, "metrics": metrics(oracle_value, packed_value)})
                comp_value_sum = packed_value if comp_value_sum is None else comp_value_sum + packed_value
            else:
                source = {"sink": cache.sink_v, "pending": cache.pending_v, "recent": cache.recent_v}[name]
                part = request_invariant_full_value_attention(weights, source, k_segment_valid_lengths(cache, device=weights.device)[name], attn.num_key_value_groups)
                ref_part_v = full_v[:, :, offset:offset+length, :]
                oracle_part = torch.matmul(weights, repeat_kv(ref_part_v, int(attn.num_key_value_groups)))
                tail_rows.append({"name": name, "metrics": metrics(oracle_part, part)})
                comp_value_sum = part if comp_value_sum is None else comp_value_sum + part
            offset += length
        qkv = {
            "q_projection": summarize_tensor(q_proj),
            "q_norm": summarize_tensor(q_norm),
            "k_projection": summarize_tensor(k_proj),
            "k_norm": summarize_tensor(k_norm),
            "v_projection": summarize_tensor(v_proj),
            "rope_q": summarize_tensor(q_rope),
            "rope_k": summarize_tensor(k_rope),
            "scaling": float(attn.scaling),
        }
        cache_audit = {
            "layer": int(layer_idx),
            "stats": cache_segment_stats(cache),
            "segment_order": [name for name, _ in value_parts],
            "value_parts": [{"name": n, "length": int(l)} for n, l in value_parts],
            "packed_k_tokens": int(cache.packed_k_tokens),
            "packed_v_tokens": int(cache.packed_v_tokens),
            "packed_v4_tokens": int(getattr(cache, "packed_v4_tokens", 0) or 0),
            "v2_count": int((~cache.v_precision_mask[:, : cache.packed_v_tokens].bool()).sum().item()) if cache.v_precision_mask is not None else None,
            "v4_count": int(cache.v_precision_mask[:, : cache.packed_v_tokens].bool().sum().item()) if cache.v_precision_mask is not None else None,
            "k_centroids": summarize_tensor(cache.k_centroids),
            "v_centroids": summarize_tensor(cache.v_centroids),
            "k_assignments": summarize_tensor(cache.k_assignments),
            "v_assignment_idx": summarize_tensor(cache.v_assignment_idx),
            "v_precision_mask": summarize_tensor(cache.v_precision_mask),
        }
        qk = {
            "overall_unscaled": metrics(ref_scores_unscaled, comp_scores_unscaled),
            "overall_scaled": metrics(ref_scores, comp_scores),
            "segments": oracle_parts,
            "classification": "PASS" if metrics(ref_scores_unscaled, comp_scores_unscaled)["rel_l2"] < 1e-2 else "QK_SEMANTIC_DRIFT",
        }
        softmax = {
            "segmented_vs_torch_same_scores": metrics(torch_probs, segmented_probs),
            "reference_vs_compressed_probs": metrics(ref_probs, segmented_probs),
            "prob_sum_min": float(segmented_probs.float().sum(dim=-1).min().item()),
            "prob_sum_max": float(segmented_probs.float().sum(dim=-1).max().item()),
            "classification": "PASS" if metrics(torch_probs, segmented_probs)["rel_l2"] < 1e-3 else "SEGMENTED_SOFTMAX_SEMANTIC_DRIFT",
        }
        value = {
            "full_reference_vs_compressed_attention_value": metrics(ref_value, comp_value),
            "manual_sum_vs_backend_compressed_value": metrics(comp_value_sum, comp_value),
            "historical_segments": value_rows,
            "tail_segments": tail_rows,
            "fp16_tail_fusion_enabled": bool(fp16_tail_value_fusion_enabled()),
        }
    return qkv, cache_audit, qk, softmax, value


def write_json_md(name: str, payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / f"{name}.md").write_text(f"# {name}\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n")


def main() -> int:
    torch.manual_seed(20260827)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    ids = prompt_ids(tokenizer)
    ref = run_one(Qwen3ForCausalLM_PatternKV, "reference", ids)
    ref_model = ref.pop("model")
    del ref_model
    torch.cuda.empty_cache(); gc.collect(); time.sleep(2)
    comp = run_one(Qwen3ForCausalLM_PatternKVCompressed, "compressed", ids)
    rows, first_layer, first_component = layer_scan(ref, comp)
    logits_m = metrics(ref["decode_logits"], comp["decode_logits"])
    prefill_m = metrics(ref["prefill_logits"], comp["prefill_logits"])
    reproduction = {
        "status": "DONE",
        "prompt_index": PROMPT_INDEX,
        "context": CONTEXT,
        "reference_decode_token": ref["decode_token"],
        "compressed_decode_token": comp["decode_token"],
        "prefill_logits": prefill_m,
        "first_decode_logits": logits_m,
        "first_decode_top1_reference": int(ref["decode_logits"].argmax(dim=-1).item()),
        "first_decode_top1_compressed": int(comp["decode_logits"].argmax(dim=-1).item()),
        "first_decode_top1_parity": bool(ref["decode_logits"].argmax(dim=-1).item() == comp["decode_logits"].argmax(dim=-1).item()),
    }
    write_json_md("first_decode_reproduction_v1", reproduction)
    layer_payload = {"first_divergent_layer": first_layer, "first_divergent_component": first_component, "rows": rows}
    write_json_md("first_decode_layer_scan", layer_payload)
    qkv, cache_audit, qk, softmax, value = component_oracles(comp, first_layer if first_layer >= 0 else 0)
    component_trace = {"layer": first_layer, "first_component": first_component, "qkv_projection_summaries": qkv}
    write_json_md("first_decode_component_trace", component_trace)
    write_json_md("first_decode_cache_snapshot_audit", cache_audit)
    write_json_md("qk_oracle_comparison", qk)
    write_json_md("softmax_oracle_comparison", softmax)
    write_json_md("value_oracle_comparison", value)
    legacy = {
        "backend": "CUDA_COMPRESSED_LEGACY",
        "historical_v_parity": value.get("historical_segments", []),
        "classification": "LEGACY_CUDA_V_READER_CORRECT" if all(r["metrics"].get("rel_l2", 999.0) < 1e-2 for r in value.get("historical_segments", [])) else "LEGACY_CUDA_V_READER_SEMANTIC_BUG",
    }
    write_json_md("legacy_cuda_v_oracle", legacy)
    tail = {
        "fp16_tail_fusion_enabled": bool(fp16_tail_value_fusion_enabled()),
        "tail_segments": value.get("tail_segments", []),
        "classification": "TAIL_PYTORCH_PATH_CHECKED" if not fp16_tail_value_fusion_enabled() else "TAIL_FUSION_CHECKED",
    }
    write_json_md("tail_fusion_oracle", tail)
    root = {
        "first_divergent_layer": first_layer,
        "first_divergent_component": first_component,
        "qk_classification": qk.get("classification"),
        "softmax_classification": softmax.get("classification"),
        "legacy_v_classification": legacy.get("classification"),
        "dominant_root_cause": "UNFIXED_TRACE_ONLY",
    }
    if qk.get("classification") != "PASS":
        root["dominant_root_cause"] = "PACKED_K_QK_SEMANTIC_DRIFT" if any(s["name"] == "packed" and s["metrics"].get("rel_l2", 0.0) > 1e-2 for s in qk.get("segments", [])) else "QK_SEMANTIC_DRIFT"
    elif softmax.get("classification") != "PASS":
        root["dominant_root_cause"] = "SEGMENTED_SOFTMAX_SEMANTIC_DRIFT"
    elif legacy.get("classification") != "LEGACY_CUDA_V_READER_CORRECT":
        root["dominant_root_cause"] = "LEGACY_CUDA_V_READER_SEMANTIC_BUG"
    else:
        root["dominant_root_cause"] = "NO_ATTENTION_ORACLE_DRIFT_FOUND"
    write_json_md("root_cause_bisection", root)
    print(json.dumps({"reproduction": reproduction, "root": root}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
