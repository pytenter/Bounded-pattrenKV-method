from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUANT = ROOT / "quant"
if str(QUANT) not in sys.path:
    sys.path.insert(0, str(QUANT))
VENDOR = os.environ.get("QWEN3_TRANSFORMERS_VENDOR")
if VENDOR and VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from quant.matmul import cuda_bmm_fA_qB_outer_with_base, cuda_attn_v_fused_with_base

OUT = ROOT / "reports/qwen3_v100_system_generalization_v1"
MODEL = "/home/qinch2023/modelscope_models/Qwen3-8B"
CONTEXT = int(os.environ.get("QWEN3_KV_FIX_CONTEXT", "512"))
DECODE = int(os.environ.get("QWEN3_KV_FIX_DECODE", "8"))
B1_DECODE = int(os.environ.get("QWEN3_KV_FIX_B1_DECODE", os.environ.get("QWEN3_KV_FIX_DECODE", "8")))
BASES = [
    "Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. ",
    "System benchmark prompt: explain cache compression, attention, and deterministic greedy decoding. ",
    "Long context QA: a researcher compares two inference backends and records every generated token. ",
    "Hardware note: compare vectorized batch execution with serial request dispatch in a deterministic backend. ",
]
CFG = dict(
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

def write_report(name: str, payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / f"{name}.md").write_text(f"# {name}\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n")

def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    d = (a.float() - b.float()).abs()
    return {"max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}

def k_oracle() -> dict[str, Any]:
    torch.manual_seed(314)
    device = "cuda"
    B, nh, nh_kv, D, N, M, group, bits = 2, 4, 2, 128, 128, 8, 128, 2
    q = (torch.randn(B, nh, 1, D, device=device, dtype=torch.float16) * 0.05).contiguous()
    cent = (torch.randn(B, nh_kv, M, D, device=device, dtype=torch.float16) * 0.2).contiguous()
    # Make request tables visibly different to catch accidental [Hkv,M,D] broadcasting.
    cent[1].add_(1.25)
    qB = torch.zeros(B, nh_kv, D, N // (32 // bits), device=device, dtype=torch.int32)
    scale = torch.zeros(B, nh_kv, D, N // group, device=device, dtype=torch.float16)
    zero = torch.zeros_like(scale)
    assign = (torch.arange(N, device=device).view(1, 1, N).expand(B, nh_kv, N) % M).to(torch.uint8).contiguous()
    got = cuda_bmm_fA_qB_outer_with_base(group, q, qB, scale, zero, bits, cent, assign, nh, nh_kv)
    exp = torch.empty_like(got)
    ratio = nh // nh_kv
    for b in range(B):
        for h in range(nh):
            kv = h // ratio
            rows = cent[b, kv, assign[b, kv].long()]
            exp[b, h, 0] = (rows.float() * q[b, h, 0].float()).sum(dim=-1).to(torch.float16)
    diff = _maxdiff(got, exp)
    status = "PASS" if diff["max_abs"] <= 5e-3 else "FAIL"
    return {"status": status, "oracle": "cuda_qk_request_local_centroid", "shape": list(got.shape), **diff}

def v_oracle(bits: int) -> dict[str, Any]:
    torch.manual_seed(2718 + bits)
    device = "cuda"
    B, nh, nh_kv, T, OC, M, group = 2, 4, 2, 64, 128, 8, 128
    attn = (torch.randn(B, nh, 1, T, device=device, dtype=torch.float16) * 0.03).contiguous()
    cent = (torch.randn(B, nh_kv, M, OC, device=device, dtype=torch.float16) * 0.15).contiguous()
    cent[1].sub_(0.9)
    pack = 32 // bits
    vq = torch.zeros(B, nh_kv, T, OC // pack, device=device, dtype=torch.int32)
    scale = torch.zeros(B, nh_kv, T, OC // group, device=device, dtype=torch.float16)
    zero = torch.zeros_like(scale)
    mask = torch.ones(B, nh_kv, T, device=device, dtype=torch.uint8)
    idx = (torch.arange(T, device=device).view(1, 1, T).expand(B, nh_kv, T) % M).to(torch.uint8).contiguous()
    got = cuda_attn_v_fused_with_base(group, attn, vq, scale, zero, bits, cent, mask, idx, nh, nh_kv)
    exp = torch.empty_like(got)
    ratio = nh // nh_kv
    for b in range(B):
        for h in range(nh):
            kv = h // ratio
            rows = cent[b, kv, idx[b, kv].long()]
            exp[b, h, 0] = (rows.float() * attn[b, h, 0, :, None].float()).sum(dim=0).to(torch.float16)
    diff = _maxdiff(got, exp)
    status = "PASS" if diff["max_abs"] <= 8e-3 else "FAIL"
    return {"status": status, "oracle": f"cuda_v{bits}_request_local_centroid", "shape": list(got.shape), **diff}

def swap_isolation() -> dict[str, Any]:
    torch.manual_seed(123)
    device = "cuda"
    B, nh, nh_kv, D, N, M, group, bits = 2, 4, 2, 128, 128, 8, 128, 2
    q_one = (torch.randn(1, nh, 1, D, device=device, dtype=torch.float16) * 0.04).contiguous()
    q = q_one.expand(B, -1, -1, -1).contiguous()
    base_cent = (torch.randn(B, nh_kv, M, D, device=device, dtype=torch.float16) * 0.1).contiguous()
    base_cent[1].add_(1.0)
    qB = torch.zeros(B, nh_kv, D, N // (32 // bits), device=device, dtype=torch.int32)
    scale = torch.zeros(B, nh_kv, D, N // group, device=device, dtype=torch.float16)
    zero = torch.zeros_like(scale)
    assign_one = (torch.arange(N, device=device).view(1, 1, N).expand(1, nh_kv, N) % M).to(torch.uint8)
    assign = assign_one.expand(B, -1, -1).contiguous()
    out = cuda_bmm_fA_qB_outer_with_base(group, q, qB, scale, zero, bits, base_cent, assign, nh, nh_kv)
    out_swapped = cuda_bmm_fA_qB_outer_with_base(group, q, qB, scale, zero, bits, base_cent.flip(0).contiguous(), assign, nh, nh_kv)
    k_swap = _maxdiff(out[0], out_swapped[1])
    v = v_oracle(2)
    status = "PASS" if k_swap["max_abs"] <= 5e-3 and v["status"] == "PASS" else "FAIL"
    return {"status": status, "k_swap_max_abs": k_swap["max_abs"], "k_swap_mean_abs": k_swap["mean_abs"], "v2_oracle_status": v["status"], "v2_oracle_max_abs": v["max_abs"]}

def _load_qwen3_symbols():
    from transformers import AutoConfig, AutoTokenizer
    from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV
    from models.qwen3_patternkv_system import (
        Qwen3ForCausalLM_PatternKVCompressed,
        collect_qwen3_compressed_dynamic_stats,
        get_qwen3_compressed_counters,
        reset_qwen3_compressed_counters,
    )
    return (AutoConfig, AutoTokenizer, Qwen3ForCausalLM_PatternKV, Qwen3ForCausalLM_PatternKVCompressed, collect_qwen3_compressed_dynamic_stats, get_qwen3_compressed_counters, reset_qwen3_compressed_counters)

def make_config(task: str):
    AutoConfig, *_ = _load_qwen3_symbols()
    c = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    for k, v in CFG.items():
        setattr(c, k, v)
    setattr(c, "patternkv_selector_task_key", task)
    return c

def prompt_batch(tok, n: int) -> torch.Tensor:
    rows = []
    for i in range(n):
        ids = tok(BASES[i % len(BASES)] * 160, return_tensors="pt", add_special_tokens=False).input_ids[:, :CONTEXT]
        if ids.shape[1] < CONTEXT:
            raise RuntimeError(f"prompt too short: {ids.shape[1]} < {CONTEXT}")
        rows.append(ids)
    return torch.cat(rows, dim=0)

def run_compressed_batch(n: int, decode: int = DECODE) -> dict[str, Any]:
    _AutoConfig, AutoTokenizer, _Ref, Qwen3ForCausalLM_PatternKVCompressed, collect_qwen3_compressed_dynamic_stats, get_qwen3_compressed_counters, reset_qwen3_compressed_counters = _load_qwen3_symbols()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ids = prompt_batch(tok, n).to("cuda")
    model = Qwen3ForCausalLM_PatternKVCompressed.from_pretrained(
        MODEL, local_files_only=True, config=make_config(f"kv-fix-b{n}"), torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda").eval()
    reset_qwen3_compressed_counters()
    tokens = []
    nan_inf = False
    try:
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True, return_dict=True)
            for _ in range(decode):
                logits = out.logits[:, -1, :].detach().float()
                nan_inf = nan_inf or (not bool(torch.isfinite(logits).all()))
                nxt = logits.argmax(dim=-1)
                tokens.append([int(x) for x in nxt.detach().cpu().tolist()])
                out = model(input_ids=nxt.view(n, 1), past_key_values=out.past_key_values, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
        stats = collect_qwen3_compressed_dynamic_stats(model, out.past_key_values)
        rec = {"status": "PASS", "batch": n, "context": CONTEXT, "decode": decode, "tokens": tokens, "no_nan_inf": not nan_inf, "counters": get_qwen3_compressed_counters(), "stats0": stats.get("cache_segment_stats_per_layer", [{}])[0]}
    except Exception as exc:
        rec = {"status": "FAIL", "batch": n, "context": CONTEXT, "decode": decode, "error": repr(exc), "traceback_tail": traceback.format_exc().splitlines()[-10:], "counters": get_qwen3_compressed_counters()}
    del model
    torch.cuda.empty_cache(); gc.collect(); time.sleep(2)
    return rec

def b1_regression() -> dict[str, Any]:
    _AutoConfig, AutoTokenizer, Qwen3ForCausalLM_PatternKV, Qwen3ForCausalLM_PatternKVCompressed, _collect_stats, get_qwen3_compressed_counters, reset_qwen3_compressed_counters = _load_qwen3_symbols()
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    ids_list = [prompt_batch(tok, 1).to("cuda")]
    rec = {"context": CONTEXT, "decode": B1_DECODE, "backend_mode": "CUDA_COMPRESSED_LEGACY"}
    def run(cls, label):
        model = cls.from_pretrained(MODEL, local_files_only=True, config=make_config(f"kv-fix-b1-{label}"), torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda").eval()
        runs=[]
        for ids in ids_list:
            toks=[]; logits_cpu=[]; nan=False
            if label == "compressed": reset_qwen3_compressed_counters()
            with torch.no_grad():
                out = model(input_ids=ids, use_cache=True, return_dict=True)
                for _ in range(B1_DECODE):
                    lg = out.logits[:, -1, :].detach().float().cpu()
                    nan = nan or (not bool(torch.isfinite(lg).all()))
                    nxt = lg.argmax(dim=-1)
                    toks.append(int(nxt[0])); logits_cpu.append(lg.squeeze(0))
                    out = model(input_ids=nxt.view(1,1).to("cuda"), past_key_values=out.past_key_values, use_cache=True, return_dict=True)
            torch.cuda.synchronize()
            counters = get_qwen3_compressed_counters() if label == "compressed" else {}
            runs.append({"tokens": toks, "logits": logits_cpu, "nan_inf": nan, "counters": counters})
        del model; torch.cuda.empty_cache(); gc.collect(); time.sleep(2)
        return runs
    ref = run(Qwen3ForCausalLM_PatternKV, "reference")
    comp = run(Qwen3ForCausalLM_PatternKVCompressed, "compressed")
    all_match=True; max_rel=0.0; max_abs=0.0; first=None; nan=False
    for rr, cc in zip(ref, comp):
        nan = nan or rr["nan_inf"] or cc["nan_inf"]
        for step, (rl, cl) in enumerate(zip(rr["logits"], cc["logits"])):
            diff = (rl - cl).float(); rel=float(diff.norm()/rl.float().norm().clamp_min(1e-12)); ma=float(diff.abs().max())
            match = rr["tokens"][step] == cc["tokens"][step]
            all_match = all_match and match; max_rel=max(max_rel, rel); max_abs=max(max_abs, ma)
            if not match and first is None:
                first = {"step": step, "reference_token": rr["tokens"][step], "compressed_token": cc["tokens"][step], "logits_rel_l2": rel, "logits_max_abs": ma}
    rec.update({"status": "PASS" if all_match and not nan else "FAIL", "token_parity": bool(all_match), "top1_parity": bool(all_match), "no_nan_inf": not nan, "max_logits_rel_l2": max_rel, "max_logits_max_abs": max_abs, "first_divergence": first, "compressed_final_counters": [r["counters"] for r in comp]})
    return rec

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracles", action="store_true")
    ap.add_argument("--b1", action="store_true")
    ap.add_argument("--batch", type=int, choices=[2,4])
    ap.add_argument("--timed", action="store_true")
    args = ap.parse_args()
    ok = True
    if args.oracles:
        k = k_oracle(); write_report("k_request_local_centroid_oracle", k); ok &= k["status"] == "PASS"
        v2 = v_oracle(2); v4 = v_oracle(4)
        v = {"status": "PASS" if v2["status"] == "PASS" and v4["status"] == "PASS" else "FAIL", "v2": v2, "v4": v4}
        write_report("v_request_local_centroid_oracle", v); ok &= v["status"] == "PASS"
        sw = swap_isolation(); write_report("centroid_swap_isolation", sw); ok &= sw["status"] == "PASS"
        print(json.dumps({"k": k, "v": v, "swap": sw}, indent=2, sort_keys=True))
    if args.b1:
        rec = b1_regression(); write_report("b1_post_cuda_kv_fix_regression", rec); ok &= rec["status"] == "PASS"; print(json.dumps(rec, indent=2, sort_keys=True))
    if args.batch:
        rec = run_compressed_batch(args.batch)
        name = f"true_batch_b{args.batch}_kv_fix"
        serial_fw = rec.get("counters", {}).get("serial_request_forward_dispatches")
        serial_attn = rec.get("counters", {}).get("serial_attention_dispatches")
        rec["serial_request_forward_dispatches"] = serial_fw
        rec["serial_attention_dispatches"] = serial_attn
        rec["true_batch_no_serial_dispatch"] = (serial_fw in (0, None)) and (serial_attn in (0, None))
        rec["classification"] = f"TRUE_BATCH_B{args.batch}_KV_FIX_PASS" if rec.get("status") == "PASS" and rec["true_batch_no_serial_dispatch"] else f"TRUE_BATCH_B{args.batch}_KV_FIX_FAIL"
        write_report(name, rec); ok &= rec.get("classification", "").endswith("PASS"); print(json.dumps(rec, indent=2, sort_keys=True))
    if args.timed:
        start = time.time(); rec = run_compressed_batch(2); elapsed = time.time() - start
        payload = {"status": rec.get("status"), "batch": 2, "context": CONTEXT, "decode": DECODE, "elapsed_sec": elapsed, "formal_timing": False, "reason": "timed-window smoke only; formal timing matrix intentionally not run", "counters": rec.get("counters", {})}
        write_report("timed_window_kv_fix_closure", payload); ok &= payload["status"] == "PASS"; print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
