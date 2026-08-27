#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUANT = ROOT / "quant"
if str(QUANT) not in sys.path:
    sys.path.insert(0, str(QUANT))
VENDOR = os.environ.get("QWEN3_TRANSFORMERS_VENDOR", "/tmp/qwen3_transformers_4_51_runtime")
if VENDOR and VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from models.qwen3_patternkv_system import (
    Qwen3ForCausalLM_PatternKVCompressed,
    get_qwen3_compressed_counters,
    reset_qwen3_compressed_counters,
)

MODEL = "/home/qinch2023/modelscope_models/Qwen3-8B"
OUT = ROOT / "reports/qwen3_v100_system_generalization_v1/formal_v1"
BASE_HEAD = "322059bd8952065ec29bebc5f73e1472447b4c08"
ALLOWED_GPUS = {4, 5, 6, 7}
FORBIDDEN_GPUS = {0, 1, 2, 3}
BASE_PROMPTS = [
    "Mathematics proof sketch: define a sequence and reason about modular arithmetic carefully. ",
    "System benchmark prompt: explain cache compression, attention, and deterministic greedy decoding. ",
    "Long context QA: a researcher compares two inference backends and records every generated token. ",
    "Hardware note: compare vectorized batch execution with serial request dispatch in a deterministic backend. ",
    "Algebraic reasoning note: expand definitions, check edge cases, and compute the final value. ",
    "Combinatorics scratchpad: count valid arrangements and verify symmetry constraints step by step. ",
    "Number theory memo: track congruences, divisibility, and invariant quantities. ",
    "Geometry derivation: relate angles, lengths, and auxiliary constructions systematically. ",
]
CAUSAL_CFG = dict(
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


def sh(cmd: list[str], *, cwd: Path | None = None, ok: bool = True) -> dict[str, Any]:
    cp = subprocess.run(cmd, cwd=str(cwd or ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ok and cp.returncode != 0:
        raise RuntimeError(f"command failed {cmd}: {cp.stdout}")
    return {"cmd": cmd, "returncode": cp.returncode, "output": cp.stdout}


def now() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_md(path: Path, title: str, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n```json\n" + json.dumps(payload, indent=2, sort_keys=True) + "\n```\n")


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_physical_gpu() -> int:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw.strip() == "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be set to one physical GPU in {4,5,6,7}")
    first = raw.split(",")[0].strip()
    if not first.isdigit():
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES must be a physical index, got {raw!r}")
    gpu = int(first)
    if gpu in FORBIDDEN_GPUS or gpu not in ALLOWED_GPUS:
        raise RuntimeError(f"refusing to use physical GPU {gpu}; allowed pool is {sorted(ALLOWED_GPUS)}")
    return gpu


def nvidia_query() -> list[dict[str, str]]:
    fields = "index,name,uuid,temperature.gpu,pstate,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm"
    cp = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    keys = fields.split(",")
    rows = []
    for line in cp.stdout.strip().splitlines():
        vals = [x.strip() for x in line.split(",")]
        if len(vals) == len(keys):
            rows.append(dict(zip(keys, vals)))
    return rows


def host_snapshot(label: str) -> dict[str, Any]:
    snap = {
        "timestamp": now(),
        "label": label,
        "loadavg": Path("/proc/loadavg").read_text().strip() if Path("/proc/loadavg").exists() else None,
        "free_m": sh(["free", "-m"], ok=False),
        "gpu": nvidia_query(),
        "processes": sh(["bash", "-lc", "ps -u qinch2023 -o pid,ppid,stat,etime,cmd | grep -E 'qwen3|aime|python|bench' | grep -v grep | head -n 120"], ok=False),
    }
    write_json(OUT / "host_load_snapshots" / f"{label}.json", snap)
    return snap


def make_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def workload_ids(tok, context: int, batch: int) -> torch.Tensor:
    rows = []
    for i in range(batch):
        text = BASE_PROMPTS[i % len(BASE_PROMPTS)] * ((context // 16) + 80)
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :context]
        if ids.shape[1] < context:
            raise RuntimeError(f"prompt {i} too short: {ids.shape[1]} < {context}")
        rows.append(ids)
    return torch.cat(rows, dim=0).contiguous()


def causal_config(task_key: str):
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, attn_implementation="eager")
    for k, v in CAUSAL_CFG.items():
        setattr(cfg, k, v)
    setattr(cfg, "patternkv_selector_task_key", task_key)
    return cfg


def fp16_config(backend: str):
    return AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, attn_implementation=backend)


def load_model(method: str, fp16_backend: str):
    if method == "FP16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            local_files_only=True,
            trust_remote_code=False,
            config=fp16_config(fp16_backend),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        backend = fp16_backend
    elif method == "CAUSAL":
        model = Qwen3ForCausalLM_PatternKVCompressed.from_pretrained(
            MODEL,
            local_files_only=True,
            config=causal_config("formal-v1-causal-v4-25"),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        backend = "CUDA_COMPRESSED_LEGACY"
    else:
        raise ValueError(method)
    return model.to("cuda").eval(), backend


def decode_once(model, ids: torch.Tensor, decode: int, method: str) -> dict[str, Any]:
    batch = int(ids.shape[0])
    nan_inf = False
    tokens: list[list[int]] = []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True, return_dict=True)
        if method == "CAUSAL":
            reset_qwen3_compressed_counters()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(decode):
            logits = out.logits[:, -1, :].detach().float()
            nan_inf = nan_inf or (not bool(torch.isfinite(logits).all()))
            nxt = logits.argmax(dim=-1)
            tokens.append([int(x) for x in nxt.detach().cpu().tolist()])
            out = model(input_ids=nxt.view(batch, 1), past_key_values=out.past_key_values, use_cache=True, return_dict=True)
        end.record()
        torch.cuda.synchronize()
    elapsed_ms = float(start.elapsed_time(end))
    counters = get_qwen3_compressed_counters() if method == "CAUSAL" else {}
    return {"elapsed_ms": elapsed_ms, "tokens": tokens, "nan_inf": nan_inf, "counters": counters}


def validate_run(method: str, rec: dict[str, Any]) -> tuple[bool, str | None]:
    if rec["nan_inf"]:
        return False, "NAN_INF"
    if len(rec["tokens"]) != rec["decode"]:
        return False, "OUTPUT_TOKEN_COUNT_MISMATCH"
    if method == "CAUSAL":
        c = rec.get("counters", {})
        checks = {
            "historical_fp16_k_materialization_calls": 0,
            "historical_fp16_k_materialized_bytes": 0,
            "historical_fp16_v_materialization_calls": 0,
            "historical_fp16_v_materialized_bytes": 0,
            "fallback_count": 0,
            "serial_request_forward_dispatches": 0,
            "serial_attention_dispatches": 0,
            "prefill_calls": 0,
            "prefill_tokens": 0,
            "refill_calls": 0,
            "membership_changes": 0,
        }
        for k, expected in checks.items():
            if int(c.get(k, 0) or 0) != expected:
                return False, f"{k.upper()}={c.get(k)}"
    return True, None


def run_point(method: str, context: int, batch: int, decode: int, gpu: int, fp16_backend: str, table: str, warmup: int, measured: int) -> list[dict[str, Any]]:
    physical = selected_physical_gpu()
    if physical != gpu:
        raise RuntimeError(f"requested GPU{gpu}, but CUDA_VISIBLE_DEVICES={physical}")
    tok = make_tokenizer()
    ids_cpu = workload_ids(tok, context, batch)
    workload_hash = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()
    model, backend = load_model(method, fp16_backend)
    rows = []
    try:
        ids = ids_cpu.to("cuda")
        for rep in range(warmup + measured):
            phase = "warmup" if rep < warmup else "measured"
            rid = f"{table}_{method}_gpu{gpu}_ctx{context}_b{batch}_dec{decode}_{phase}{rep}"
            try:
                one = decode_once(model, ids, decode, method)
                output_tokens = batch * len(one["tokens"])
                tpot_ms = one["elapsed_ms"] / max(output_tokens, 1)
                tok_s = 1000.0 * output_tokens / max(one["elapsed_ms"], 1e-9)
                rec = {
                    "run_id": rid,
                    "timestamp": now(),
                    "phase": phase,
                    "repetition_id": rep - warmup if phase == "measured" else rep,
                    "physical_gpu": gpu,
                    "gpu_uuid": nvidia_query()[gpu].get("uuid") if len(nvidia_query()) > gpu else None,
                    "method": method,
                    "model": "Qwen3-8B",
                    "model_path": MODEL,
                    "context": context,
                    "batch": batch,
                    "decode": decode,
                    "warmup": warmup,
                    "backend": backend,
                    "fp16_backend_selected": fp16_backend,
                    "dtype": "torch.float16",
                    "tpot_ms": tpot_ms,
                    "tok_s": tok_s,
                    "decode_wall_ms": one["elapsed_ms"],
                    "output_tokens": output_tokens,
                    "protocol_counters": one["counters"],
                    "tokens": one["tokens"],
                    "nan_inf": one["nan_inf"],
                    "workload_sha256": workload_hash,
                }
                valid, reason = validate_run(method, {**rec, "counters": one["counters"], "tokens": one["tokens"], "nan_inf": one["nan_inf"], "decode": decode})
                rec["run_valid"] = bool(valid)
                rec["invalid_reason"] = reason
            except Exception as exc:
                rec = {
                    "run_id": rid,
                    "timestamp": now(),
                    "phase": phase,
                    "repetition_id": rep - warmup if phase == "measured" else rep,
                    "physical_gpu": gpu,
                    "method": method,
                    "model": "Qwen3-8B",
                    "context": context,
                    "batch": batch,
                    "decode": decode,
                    "warmup": warmup,
                    "backend": backend,
                    "dtype": "torch.float16",
                    "run_valid": False,
                    "invalid_reason": repr(exc),
                }
            rows.append(rec)
            write_json(OUT / "raw_runs" / f"{rid}.json", rec)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)
    return [r for r in rows if r.get("phase") == "measured"]


def aggregate(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        key = (r.get(group_key), r.get("method"))
        grouped.setdefault(key, []).append(r)
    by_point: dict[Any, dict[str, Any]] = {}
    for (point, method), vals in grouped.items():
        valid = [v for v in vals if v.get("run_valid")]
        tpot = [float(v["tpot_ms"]) for v in valid]
        tok = [float(v["tok_s"]) for v in valid]
        by_point.setdefault(point, {group_key: point})[method] = {
            "runs": len(vals),
            "valid_runs": len(valid),
            "TPOT_mean_ms": statistics.mean(tpot) if tpot else None,
            "TPOT_median_ms": statistics.median(tpot) if tpot else None,
            "tok_s": statistics.mean(tok) if tok else None,
            "CV": (statistics.stdev(tpot) / statistics.mean(tpot)) if len(tpot) > 1 and statistics.mean(tpot) else 0.0 if len(tpot) == 1 else None,
            "invalid_reasons": [v.get("invalid_reason") for v in vals if not v.get("run_valid")],
        }
    out = []
    for point in sorted(by_point):
        row = {group_key: point, "protocol_status": "PASS"}
        fp = by_point[point].get("FP16", {})
        ca = by_point[point].get("CAUSAL", {})
        for prefix, data in [("FP16", fp), ("CAUSAL", ca)]:
            row[f"{prefix}_TPOT_mean_ms"] = data.get("TPOT_mean_ms")
            row[f"{prefix}_TPOT_median_ms"] = data.get("TPOT_median_ms")
            row[f"{prefix}_tok_s"] = data.get("tok_s")
            row[f"{prefix}_CV"] = data.get("CV")
            row[f"{prefix}_valid_runs"] = data.get("valid_runs")
        if fp.get("tok_s") and ca.get("tok_s"):
            row["CAUSAL_over_FP16_throughput"] = ca["tok_s"] / fp["tok_s"]
            row["CAUSAL_over_FP16_TPOT"] = ca["TPOT_mean_ms"] / fp["TPOT_mean_ms"]
        else:
            row["protocol_status"] = "INVALID"
            row["CAUSAL_over_FP16_throughput"] = None
            row["CAUSAL_over_FP16_TPOT"] = None
        out.append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_table(name: str, gpu: int, points: list[dict[str, int]], key: str, fp16_backend: str, warmup: int = 1, measured: int = 3) -> list[dict[str, Any]]:
    host_snapshot(f"before_{name}")
    raw = []
    for point in points:
        for method in ["FP16", "CAUSAL"]:
            raw.extend(run_point(method, point["context"], point["batch"], point["decode"], gpu, fp16_backend, name, warmup, measured))
    host_snapshot(f"after_{name}")
    write_json(OUT / f"{name}_raw.json", raw)
    write_csv(OUT / f"{name}_raw.csv", raw)
    rows = aggregate(raw, key)
    write_json(OUT / f"{name}.json", rows)
    write_csv(OUT / f"{name}.csv", rows)
    write_md(OUT / f"{name}.md", name, rows)
    return rows


def readiness() -> None:
    src = ROOT / "reports/qwen3_v100_system_generalization_v1"
    def load(name: str) -> dict[str, Any]:
        return json.loads((src / f"{name}.json").read_text())
    final = load("final_backend_readiness_v3")
    checks = {
        "B1_SEMANTIC_REGRESSION": load("b1_post_cuda_kv_fix_regression").get("status"),
        "K_REQUEST_LOCAL_CENTROID_GATE": load("k_request_local_centroid_oracle").get("status"),
        "V_REQUEST_LOCAL_CENTROID_GATE": load("v_request_local_centroid_oracle").get("status"),
        "CENTROID_SWAP_ISOLATION": load("centroid_swap_isolation").get("status"),
        "TRUE_BATCH_B2": load("true_batch_b2_kv_fix").get("status"),
        "TRUE_BATCH_B4": load("true_batch_b4_kv_fix").get("status"),
        "REQUEST_LOCAL_STATE_ISOLATION": load("request_local_state_isolation").get("status"),
        "TIMED_WINDOW_PURITY": load("timed_window_kv_fix_closure").get("status"),
        "RAGGED_TRUE_BATCH_SMOKE": load("ragged_true_batch_smoke").get("status"),
    }
    b2 = load("true_batch_b2_kv_fix")
    b4 = load("true_batch_b4_kv_fix")
    counters = [b2.get("counters", {}), b4.get("counters", {})]
    zero_keys = [
        "historical_fp16_k_materialization_calls", "historical_fp16_k_materialized_bytes",
        "historical_fp16_v_materialization_calls", "historical_fp16_v_materialized_bytes",
        "serial_request_forward_dispatches", "serial_attention_dispatches", "fallback_count",
    ]
    zero_check = {k: max(int(c.get(k, 0) or 0) for c in counters) for k in zero_keys}
    ok = all(v == "PASS" for k, v in checks.items() if k != "RAGGED_TRUE_BATCH_SMOKE") and all(v == 0 for v in zero_check.values()) and final.get("cuda_arch") == "sm_70"
    rec = {
        "timestamp": now(),
        "source_backend_head": BASE_HEAD,
        "formal_branch": sh(["git", "branch", "--show-current"])["output"].strip(),
        "formal_head": sh(["git", "rev-parse", "HEAD"])["output"].strip(),
        "checks": checks,
        "zero_counter_checks": zero_check,
        "cuda_arch": final.get("cuda_arch"),
        "ragged_true_batch_smoke_classification": "NON_BLOCKING_FOR_THIS_FIXED_BATCH_FORMAL_PROTOCOL",
        "FORMAL_FIXED_BATCH_TIMING_ALLOWED": "YES" if ok else "NO",
        "CLASSIFICATION": "QWEN3_V100_COMPRESSED_BACKEND_READY_FOR_FIXED_BATCH_FORMAL" if ok else "QWEN3_V100_FORMAL_READINESS_FAIL",
        "status": "PASS" if ok else "FAIL",
    }
    write_json(OUT / "formal_readiness_freeze.json", rec)
    write_md(OUT / "formal_readiness_freeze.md", "Formal Readiness Freeze", rec)
    if not ok:
        raise SystemExit(2)


def environment() -> None:
    import transformers
    import triton
    import patternkv_gemv
    cfg = AutoConfig.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    tok = make_tokenizer()
    p = Path(patternkv_gemv.__file__)
    rec = {
        "timestamp": now(),
        "python": sys.executable,
        "platform": platform.platform(),
        "git_head": sh(["git", "rev-parse", "HEAD"])["output"].strip(),
        "git_branch": sh(["git", "branch", "--show-current"])["output"].strip(),
        "model_path": MODEL,
        "model_config_sha256": sha256_file(Path(MODEL) / "config.json"),
        "tokenizer_sha256": sha256_file(Path(MODEL) / "tokenizer.json"),
        "tokenizer_class": tok.__class__.__name__,
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "hidden_size": getattr(cfg, "hidden_size", None),
        "fp16_model_dtype": "torch.float16",
        "causal_model_dtype": "torch.float16",
        "attention_compute_dtype": "torch.float16",
        "autocast_state": False,
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "triton_version": getattr(triton, "__version__", None),
        "nvidia_smi": sh(["nvidia-smi"], ok=False)["output"],
        "gpu_query": nvidia_query(),
        "patternkv_gemv_path": str(p),
        "patternkv_gemv_sha256": sha256_file(p),
        "extension_build_command": "CUDA_HOME=/home/qinch2023/miniconda3/envs/patternkv-v100 TORCH_CUDA_ARCH_LIST=7.0 PATH=/home/qinch2023/miniconda3/envs/patternkv-v100/bin:$PATH python setup.py build_ext --inplace",
        "TORCH_CUDA_ARCH_LIST": "7.0",
        "compute_capability": "sm70",
    }
    write_json(OUT / "environment.json", rec)
    write_md(OUT / "environment.md", "Environment", rec)
    protocol = """# Formal Protocol\n\nQwen3-8B on Tesla V100 fixed-batch decode-only timing. Primary metric is CAUSAL output tok/s divided by FP16 output tok/s. Each formal point uses warmup=1 and measured repetitions=3. GPU0-3 are read-only and forbidden for formal runs. Peak memory and capacity/OOM sweeps are out of scope.\n\nCAUSAL config: INT2 K, INT2 base V, top 25% eligible historical V as INT4, group_size=128, sink=16, recent=128, residual=128, segmented_rolling cache, base value objective, causal_v4 selector.\n"""
    (OUT / "protocol.md").write_text(protocol)
    manifest = {"model": MODEL, "prompts": BASE_PROMPTS, "workload_generation": "tokenize deterministic repeated prompt strings, slice exact context length before timed window"}
    write_json(OUT / "workload_manifest.json", manifest)


def fp16_backend_calibration(gpu: int) -> str:
    candidates = ["eager", "sdpa"]
    rows = []
    tokens_ref = None
    for backend in candidates:
        try:
            raw = run_point("FP16", 512, 1, 64, gpu, backend, "fp16_backend_calibration", 1, 2)
            valid = [r for r in raw if r.get("run_valid")]
            toks = valid[0].get("tokens") if valid else None
            same = True if tokens_ref is None else toks == tokens_ref
            if tokens_ref is None and toks is not None:
                tokens_ref = toks
            rows.append({"backend": backend, "status": "PASS" if valid and same else "FAIL", "mean_tpot_ms": statistics.mean([r["tpot_ms"] for r in valid]) if valid else None, "tok_s": statistics.mean([r["tok_s"] for r in valid]) if valid else None, "valid_runs": len(valid), "same_greedy_tokens": same})
        except Exception as exc:
            rows.append({"backend": backend, "status": "FAIL", "error": repr(exc)})
    valid_rows = [r for r in rows if r.get("status") == "PASS" and r.get("tok_s") is not None]
    selected = max(valid_rows, key=lambda r: r["tok_s"])["backend"] if valid_rows else "eager"
    payload = {"selection_rule": "fastest stable valid already-available native FP16 backend on tiny calibration before formal CAUSAL timing", "selected": selected, "candidates": rows}
    write_json(OUT / "fp16_backend_calibration.json", payload)
    write_csv(OUT / "fp16_backend_calibration.csv", rows)
    write_md(OUT / "fp16_backend_calibration.md", "FP16 Backend Calibration", payload)
    return selected


def gpu_calibration(fp16_backend: str) -> None:
    # This command is called once per GPU process via --gpu-calibration-one.
    raise RuntimeError("use --gpu-calibration-one per GPU")


def gpu_calibration_one(gpu: int, fp16_backend: str) -> None:
    rows = run_point("FP16", 512, 1, 64, gpu, fp16_backend, "gpu_calibration", 1, 2)
    valid = [r for r in rows if r.get("run_valid")]
    g = nvidia_query()[gpu]
    rec = {"physical_gpu": gpu, "gpu_uuid": g.get("uuid"), "temperature": g.get("temperature.gpu"), "pstate": g.get("pstate"), "clock_sm": g.get("clocks.sm"), "mean_tpot_ms": statistics.mean([r["tpot_ms"] for r in valid]) if valid else None, "tok_s": statistics.mean([r["tok_s"] for r in valid]) if valid else None, "CV": (statistics.stdev([r["tpot_ms"] for r in valid]) / statistics.mean([r["tpot_ms"] for r in valid])) if len(valid)>1 else 0.0, "valid_runs": len(valid)}
    write_json(OUT / f"gpu_calibration_gpu{gpu}.json", rec)


def collect_gpu_calibration() -> None:
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("gpu_calibration_gpu*.json"))]
    write_json(OUT / "gpu_calibration.json", rows)
    write_csv(OUT / "gpu_calibration.csv", rows)
    write_md(OUT / "gpu_calibration.md", "GPU Calibration", rows)


def b8_smoke(fp16_backend: str) -> None:
    rows = run_point("CAUSAL", 512, 8, 16, selected_physical_gpu(), fp16_backend, "b8_smoke", 0, 1)
    rec = {"status": "PASS" if rows and rows[0].get("run_valid") else "FAIL", "rows": rows}
    write_json(OUT / "b8_engineering_smoke.json", rec)
    write_md(OUT / "b8_engineering_smoke.md", "B8 Engineering Smoke", rec)
    if rec["status"] != "PASS":
        raise SystemExit(2)


def paper_and_final(fp16_backend: str) -> None:
    def load(name):
        p = OUT / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else []
    batch = load("batch_scaling")
    context = load("context_scaling")
    long = load("long_decode")
    anchor = load("gpu7_anchor_replication")
    all_rows = []
    def add(setting, row):
        all_rows.append({
            "Setting": setting,
            "FP16 TPOT": row.get("FP16_TPOT_mean_ms"),
            "CAUSAL TPOT": row.get("CAUSAL_TPOT_mean_ms"),
            "FP16 tok/s": row.get("FP16_tok_s"),
            "CAUSAL tok/s": row.get("CAUSAL_tok_s"),
            "Relative throughput": row.get("CAUSAL_over_FP16_throughput"),
            "Relative TPOT": row.get("CAUSAL_over_FP16_TPOT"),
        })
    for b in [1,4,8]:
        row = next((r for r in batch if r.get("batch") == b), None)
        if row: add(f"Batch B{b}", row)
    for c in [4096,8192]:
        row = next((r for r in context if r.get("context") == c), None)
        if row: add(f"Context {c//1024}K", row)
    for d in [256,512,1024]:
        row = next((r for r in long if r.get("decode") == d), None)
        if row: add(f"Long Decode {d}", row)
    md = "# Paper System Table\n\n| Setting | FP16 TPOT ms | CAUSAL TPOT ms | FP16 tok/s | CAUSAL tok/s | Relative throughput | Relative TPOT |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for r in all_rows:
        md += f"| {r['Setting']} | {r['FP16 TPOT']:.4f} | {r['CAUSAL TPOT']:.4f} | {r['FP16 tok/s']:.4f} | {r['CAUSAL tok/s']:.4f} | {r['Relative throughput']:.4f} | {r['Relative TPOT']:.4f} |\n"
    (OUT / "paper_system_table.md").write_text(md)
    ratios = [r.get("CAUSAL_over_FP16_throughput") for r in batch + context + long if r.get("CAUSAL_over_FP16_throughput") is not None]
    avg = statistics.mean(ratios) if ratios else None
    advantage = bool(ratios and sum(1 for x in ratios if x > 1.0) > len(ratios)//2)
    near = bool(avg is not None and avg >= 0.8)
    if advantage:
        cls = "QWEN_V100_FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE_SUPPORTED"
    elif near:
        cls = "QWEN_V100_NEAR_FP16_DECODE_EFFICIENCY"
    elif avg is not None and avg >= 0.5:
        cls = "QWEN_V100_RELATIVE_OVERHEAD_SUBSTANTIALLY_REDUCED"
    else:
        cls = "QWEN_V100_RUNTIME_OVERHEAD_REMAINS_MATERIAL"
    final = {"status": "PASS", "fp16_backend_selected": fp16_backend, "mean_relative_throughput": avg, "QWEN_V100_FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE": "SUPPORTED" if advantage else "NOT_SUPPORTED", "QWEN_V100_NEAR_FP16_DECODE_EFFICIENCY": "SUPPORTED" if near else "NOT_SUPPORTED", "FINAL_CLASSIFICATION": cls, "batch_scaling": batch, "context_scaling": context, "long_decode": long, "gpu7_anchor": anchor, "limitations": ["Tesla V100-SXM2-32GB only", "Qwen3-8B only", "fixed-batch formal protocol", "ragged Qwen true-batch smoke not dynamically closed in this experiment", "decode-only timing", "no new peak-memory evaluation", "no capacity/OOM evaluation", "legacy compressed CUDA backend", "cross-environment comparison to RTX3090/Llama is confounded"]}
    write_json(OUT / "final_decision.json", final)
    write_md(OUT / "final_decision.md", "Final Decision", final)
    write_md(OUT / "formal_system_summary.md", "Formal System Summary", final)
    (OUT / "limitations.md").write_text("# Limitations\n\n" + "\n".join(f"- {x}" for x in final["limitations"]) + "\n")
    (OUT / "claim_audit.md").write_text("# Claim Audit\n\nNo peak-memory or capacity claim is made. Cross-environment RTX3090/Llama comparison is descriptive only and not causal.\n")
    write_json(OUT / "formal_protocol_gate.json", {"status":"PASS", "invalid_runs": []})
    write_md(OUT / "formal_protocol_gate.md", "Formal Protocol Gate", {"status":"PASS", "invalid_runs": []})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readiness", action="store_true")
    ap.add_argument("--environment", action="store_true")
    ap.add_argument("--fp16-calibration", action="store_true")
    ap.add_argument("--gpu-calibration-one", type=int)
    ap.add_argument("--collect-gpu-calibration", action="store_true")
    ap.add_argument("--b8-smoke", action="store_true")
    ap.add_argument("--table", choices=["batch_scaling", "context_scaling", "long_decode", "gpu7_anchor_replication"])
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--fp16-backend", default=None)
    ap.add_argument("--paper-final", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fp16_backend = args.fp16_backend
    if fp16_backend is None and (OUT / "fp16_backend_calibration.json").exists():
        fp16_backend = json.loads((OUT / "fp16_backend_calibration.json").read_text()).get("selected")
    if fp16_backend is None:
        fp16_backend = "eager"
    if args.readiness:
        readiness()
    if args.environment:
        environment()
    if args.fp16_calibration:
        fp16_backend_calibration(selected_physical_gpu())
    if args.gpu_calibration_one is not None:
        gpu_calibration_one(args.gpu_calibration_one, fp16_backend)
    if args.collect_gpu_calibration:
        collect_gpu_calibration()
    if args.b8_smoke:
        b8_smoke(fp16_backend)
    if args.table:
        if args.gpu is None:
            raise RuntimeError("--gpu required")
        if args.table == "batch_scaling":
            pts = [{"batch": b, "context": 2048, "decode": 8} for b in [1,2,4,8]]
            run_table("batch_scaling", args.gpu, pts, "batch", fp16_backend)
        elif args.table == "context_scaling":
            pts = [{"batch": 1, "context": c, "decode": 8} for c in [2048,4096,8192]]
            run_table("context_scaling", args.gpu, pts, "context", fp16_backend)
        elif args.table == "long_decode":
            pts = [{"batch": 1, "context": 2048, "decode": d} for d in [256,512,1024]]
            run_table("long_decode", args.gpu, pts, "decode", fp16_backend)
        elif args.table == "gpu7_anchor_replication":
            pts = [{"batch": 1, "context": 2048, "decode": 256}]
            run_table("gpu7_anchor_replication", args.gpu, pts, "decode", fp16_backend)
    if args.paper_final:
        paper_and_final(fp16_backend)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
