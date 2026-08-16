from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.full_model_serving_benchmark import (  # noqa: E402
    MODEL_PATH,
    PatternKVAdapter,
    PatternKVPaperAdapter,
    build_request_inputs,
    load_causal_model,
    load_patternkv_paper_model,
    run_full_model_benchmark,
    stack_inputs,
)
from bench.full_model_serving_benchmark import BenchmarkConfig  # noqa: E402
from models.segmented_cache import deserialize_cache  # noqa: E402

REPORT_DIR = REPO_ROOT / "reports/patternkv_paper_true_batch_runtime_support_audit_v1"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tensor_metrics(got: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    got_f = got.float()
    expected_f = expected.float()
    diff = got_f - expected_f
    denom = torch.linalg.vector_norm(expected_f).clamp_min(1e-8)
    return {
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / denom).item()),
        "cosine": float(torch.nn.functional.cosine_similarity(got_f.flatten(), expected_f.flatten(), dim=0).item()),
        "nan_count": int(torch.isnan(got).sum().item()),
        "inf_count": int(torch.isinf(got).sum().item()),
    }


def decode_tokens(adapter: Any, model: Any, input_ids: torch.Tensor, decode: int) -> dict[str, Any]:
    with torch.inference_mode():
        cache, next_tokens = adapter.prefill_active_batch(model, input_ids)
        sequences = []
        logits_trace = []
        for _ in range(decode):
            output = model(input_ids=next_tokens[:, None], past_key_values=cache, use_cache=True, return_dict=True)
            logits = output.logits[:, -1, :].detach()
            next_tokens = logits.argmax(dim=-1)
            logits_trace.append(logits)
            sequences.append(next_tokens.detach())
            cache = tuple(output.past_key_values)
    return {"tokens": torch.stack(sequences, dim=1), "logits": torch.stack(logits_trace, dim=1), "cache": cache}


def first_layer_cache_summary(cache_tuple: Any) -> dict[str, Any]:
    cache = deserialize_cache(cache_tuple[0], pattern=True)
    pool = getattr(cache, "centroid_state_pool", None)
    slots = getattr(cache, "centroid_state_indices", None)
    v_counts = None
    if pool is not None and torch.is_tensor(slots):
        v_counts = pool.v_counts[slots.long()].detach().cpu().tolist()
    return {
        "v_centroids_shape": list(cache.v_centroids.shape) if torch.is_tensor(cache.v_centroids) else None,
        "v_assignment_idx_shape": list(cache.v_assignment_idx.shape) if torch.is_tensor(cache.v_assignment_idx) else None,
        "v_pattern_mask_shape": list(cache.v_pattern_mask.shape) if torch.is_tensor(cache.v_pattern_mask) else None,
        "packed_v_tokens": int(cache.packed_v_tokens),
        "request_packed_v_tokens": cache.request_packed_v_tokens.detach().cpu().tolist() if torch.is_tensor(cache.request_packed_v_tokens) else None,
        "centroid_state_indices": slots.detach().cpu().tolist() if torch.is_tensor(slots) else None,
        "v_counts": v_counts,
        "operator_ready_page_pools": getattr(cache, "operator_ready_page_pools", None) is not None,
    }


def equivalence_gate(adapter: Any, model: Any, tokenizer: Any, *, batch: int, decode: int, order: list[int] | None = None) -> dict[str, Any]:
    device = next(model.parameters()).device
    inputs = build_request_inputs(tokenizer, batch, 512, device)
    if order is not None:
        inputs = [inputs[i] for i in order]
    batch_input = stack_inputs([type("Req", (), {"input_ids": item})() for item in inputs])
    batch_out = decode_tokens(adapter, model, batch_input, decode)
    independent_tokens = []
    independent_logits = []
    for item in inputs:
        single = decode_tokens(adapter, model, item.unsqueeze(0), decode)
        independent_tokens.append(single["tokens"][0])
        independent_logits.append(single["logits"][0])
    ref_tokens = torch.stack(independent_tokens, dim=0)
    ref_logits = torch.stack(independent_logits, dim=0)
    token_match = torch.equal(batch_out["tokens"], ref_tokens)
    metrics = tensor_metrics(batch_out["logits"], ref_logits)
    return {
        "batch": batch,
        "decode": decode,
        "order": order or list(range(batch)),
        "top1_agreement": bool(token_match),
        "generated_tokens": batch_out["tokens"].detach().cpu().tolist(),
        "reference_tokens": ref_tokens.detach().cpu().tolist(),
        "logit_metrics": metrics,
        "cache_summary": first_layer_cache_summary(batch_out["cache"]),
        "pass": bool(token_match and metrics["nan_count"] == 0 and metrics["inf_count"] == 0),
    }


def smoke_gate(model: Any, tokenizer: Any, *, batch: int, decode: int) -> dict[str, Any]:
    cfg = BenchmarkConfig(
        method="PATTERNKV_PAPER_FULL_MODEL",
        context_length=512,
        decode_length=decode,
        active_capacity=batch,
        total_requests=batch,
    )
    result = run_full_model_benchmark(PatternKVPaperAdapter, model, tokenizer, cfg, torch.device("cuda:0"), run_index=0, warmup=False)
    return result.__dict__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    os.environ.setdefault("PATTERNKV_SELECTIVE_PREFILL_LOGITS", "1")
    os.environ.setdefault("PATTERNKV_ACTIVE_BATCH_CACHE", "1")
    os.environ.setdefault("PATTERNKV_SYSTEM_PROFILE", "1")
    torch.cuda.set_device(0)
    tokenizer, model, model_cfg = load_patternkv_paper_model(torch.device("cuda:0"))
    gates = {
        "b1": smoke_gate(model, tokenizer, batch=1, decode=4),
        "b2_single_step": equivalence_gate(PatternKVPaperAdapter, model, tokenizer, batch=2, decode=1),
        "b2_multistep": equivalence_gate(PatternKVPaperAdapter, model, tokenizer, batch=2, decode=8),
        "b2_reorder": equivalence_gate(PatternKVPaperAdapter, model, tokenizer, batch=2, decode=8, order=[1, 0]),
        "b4": smoke_gate(model, tokenizer, batch=4, decode=4),
    }
    for name, payload in gates.items():
        write_json(REPORT_DIR / "semantic_oracle" / f"{name}.json", payload)
    write_json(
        REPORT_DIR / "semantic_oracle" / "model_config.json",
        {"model": str(MODEL_PATH), "method_config": model_cfg.get("method_config"), "model_config": {k: v for k, v in model_cfg.items() if k != "method_config"}},
    )
    write_json(REPORT_DIR / "semantic_oracle" / "gate_summary.json", {name: bool(payload.get("pass", payload.get("run_valid", False))) for name, payload in gates.items()})
    del model
    torch.cuda.empty_cache()
    tokenizer, causal_model, _ = load_causal_model(torch.device("cuda:0"))
    write_json(REPORT_DIR / "semantic_oracle" / "causal_b2_reorder.json", equivalence_gate(PatternKVAdapter, causal_model, tokenizer, batch=2, decode=4, order=[1, 0]))


if __name__ == "__main__":
    main()
