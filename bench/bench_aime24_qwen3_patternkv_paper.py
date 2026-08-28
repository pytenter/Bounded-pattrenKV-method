#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor/transformers_4_51_runtime"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoConfig, AutoTokenizer

from bench.aime_answer_parser import normalize_aime_answer, parse_aime_answer
from bench.paper_config import apply_method_defaults
from models.qwen3_patternkv import Qwen3ForCausalLM_PatternKV, collect_qwen3_patternkv_dynamic_stats

EXP = "qwen3_8b_aime24_patternkv_paper_v1"
METHOD = "PATTERNKV_PAPER"
MODEL_PATH = Path("/home/qinch2023/modelscope_models/Qwen3-8B")
DATASET = ROOT / "datasets/aime/aime24.jsonl"
RESULT_ROOT = ROOT / "results" / EXP / "formal"
RUN_ROOT = ROOT / "run" / EXP
SEEDS = (42, 43, 44)
GEN_CFG = dict(
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    max_new_tokens=32768,
    repetition_penalty=1.0,
    num_return_sequences=1,
    use_cache=True,
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False).encode())


def canonical_patternkv_paper_config() -> dict[str, Any]:
    args = SimpleNamespace(method="patternkv_paper")
    paper = apply_method_defaults(args)
    return {
        "paper_method": paper.method,
        "config_name": "patternkv_paper",
        "k_bits": paper.k_bits,
        "v_bits": paper.v_bits,
        "group_size": paper.group_size,
        "sink_length": paper.sink_length,
        "recent_length": paper.recent_length,
        "residual_length": paper.residual_length,
        "num_k_base": args.num_k_base,
        "num_v_base": args.num_v_base,
        "patternkv_cache_mode": "segmented_rolling",
        "patternkv_value_objective": "base",
        "patternkv_v_precision_selector": "base_v2",
        "patternkv_v4_budget_fraction": 0.0,
        "patternkv_random_selector_seed": 20260809,
        "key_quant_axis": paper.key_quant_axis,
        "value_quant_axis": paper.value_quant_axis,
        "pattern_selection_position": paper.pattern_selection_position,
    }


BASE_CFG = canonical_patternkv_paper_config()


def effective_seed(base: int, problem_id: int, sample: int = 0) -> int:
    return int(base) + int(problem_id) * 1000 + int(sample)


def task_id(problem_id: int, base_seed: int) -> str:
    return f"{METHOD}__p{problem_id:02d}__seed{base_seed}__sample0"


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def result_path(problem_id: int, base_seed: int) -> Path:
    return RESULT_ROOT / METHOD / f"seed{base_seed}" / f"p{problem_id:02d}.json"


def render(tokenizer: Any, problem: str) -> tuple[str, str]:
    user_prompt = problem + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if "deepseek" in rendered.lower() or "<|eot_id|>" in rendered:
        raise RuntimeError("non-Qwen prompt artifact detected")
    return user_prompt, rendered


def build_manifest(problem_ids: range | None = None, seeds: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    problem_ids = problem_ids or range(30)
    seeds = seeds or SEEDS
    return [
        {
            "method": METHOD,
            "problem_id": problem_id,
            "base_seed": seed,
            "sample_id": 0,
            "effective_seed": effective_seed(seed, problem_id),
        }
        for problem_id in problem_ids
        for seed in seeds
    ]


def configure(task_key: str) -> Any:
    config = AutoConfig.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    if config.model_type != "qwen3":
        raise RuntimeError(f"not qwen3: {config.model_type}")
    for key, value in BASE_CFG.items():
        if key in {"paper_method", "config_name", "key_quant_axis", "value_quant_axis", "pattern_selection_position"}:
            continue
        setattr(config, key, value)
    setattr(config, "patternkv_selector_task_key", task_key)
    return config


def load_model(task_key: str) -> Any:
    config = configure(task_key)
    model = Qwen3ForCausalLM_PatternKV.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    return model


def run_one(row: dict[str, Any], base_seed: int) -> dict[str, Any]:
    problem_id = int(row["problem_id"])
    seed = effective_seed(base_seed, problem_id)
    random.seed(seed)
    torch.manual_seed(seed)
    task_key = task_id(problem_id, base_seed)
    model = None
    started = time.perf_counter()
    try:
        torch.cuda.manual_seed_all(seed)
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = load_model(task_key)
        _user_prompt, rendered_prompt = render(tokenizer, row["problem"])
        encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
        with torch.no_grad():
            output = model.generate(
                **encoded,
                **GEN_CFG,
                return_dict_in_generate=True,
                output_scores=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        sequence = output.sequences
        generated_ids = sequence[0, encoded.input_ids.shape[1] :].detach().cpu().tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        parsed = parse_aime_answer(generated_text)
        reference = normalize_aime_answer(row["answer"])
        correct = parsed["parsed_answer"] == reference
        stop_reason = "length" if len(generated_ids) >= GEN_CFG["max_new_tokens"] else "eos"
        stats = collect_qwen3_patternkv_dynamic_stats(model, output.past_key_values)
        record = {
            "experiment_id": EXP,
            "phase": "formal",
            "dataset": "aime24",
            "dataset_sha256": sha(DATASET.read_bytes()),
            "model_path": str(MODEL_PATH),
            "model_name": MODEL_PATH.name,
            "model_type": "qwen3",
            "model_architecture": "Qwen3ForCausalLM",
            "backend_class": model.__class__.__name__,
            "attention_class": model.model.layers[0].self_attn.__class__.__name__,
            "method": METHOD,
            "display_method": METHOD,
            "method_config": BASE_CFG,
            "method_config_hash": stable(BASE_CFG),
            "problem_id": problem_id,
            "base_seed": base_seed,
            "sample_id": 0,
            "effective_seed": seed,
            "task_key": task_key,
            "prompt_protocol": "qwen3_native_thinking_v1",
            "rendered_prompt": rendered_prompt,
            "prompt_hash": stable({"prompt": rendered_prompt}),
            "input_token_hash": stable(encoded.input_ids.detach().cpu().tolist()),
            "generation_config": GEN_CFG,
            "generation_config_hash": stable(GEN_CFG),
            "generated_text": generated_text,
            "generated_token_hash": stable(generated_ids),
            "generated_tokens": len(generated_ids),
            "parsed_answer": parsed["parsed_answer"],
            "reference_answer": reference,
            "is_correct": correct,
            "parser_strategy": parsed["parser_strategy"],
            "parser_error": parsed["parser_error"],
            "stop_reason": stop_reason,
            "wall_time_seconds": round(time.perf_counter() - started, 4),
            "gpu_physical_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "git_commit": os.popen("git rev-parse HEAD").read().strip(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "cache_statistics": stats,
        }
    except Exception as exc:
        record = {
            "experiment_id": EXP,
            "phase": "formal",
            "method": METHOD,
            "method_config": BASE_CFG,
            "problem_id": problem_id,
            "base_seed": base_seed,
            "sample_id": 0,
            "effective_seed": seed,
            "task_key": task_key,
            "runtime_error": repr(exc),
            "traceback": traceback.format_exc(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "gpu_physical_id": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
    if model is not None:
        del model
    torch.cuda.empty_cache()
    return record


def write_atomic(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def claim_is_live(claim: Path) -> bool:
    try:
        fields = dict(part.split("=", 1) for part in claim.read_text(encoding="utf-8").split() if "=" in part)
        pid = int(fields.get("pid", "0"))
    except (OSError, ValueError):
        return False
    cmdline = Path(f"/proc/{pid}/cmdline")
    if not cmdline.exists():
        return False
    try:
        cmd = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return False
    return "bench_aime24_qwen3_patternkv_paper.py" in cmd and "--worker" in cmd


def claim_task(item: dict[str, Any]) -> Path | None:
    claim = RUN_ROOT / "claims" / (task_id(int(item["problem_id"]), int(item["base_seed"])) + ".claim")
    for _attempt in range(2):
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"pid={os.getpid()} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')}\n".encode())
            os.close(fd)
            return claim
        except FileExistsError:
            if claim_is_live(claim):
                return None
            try:
                claim.unlink()
            except FileNotFoundError:
                pass
    return None


def worker() -> None:
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in {"0", "1", "2", "3"}:
        raise SystemExit(f"refuse GPU {gpu}; only physical GPU0-3 are allowed")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "claims").mkdir(exist_ok=True)
    row_by_id = {int(row["problem_id"]): row for row in rows()}
    manifest = build_manifest()
    while True:
        claimed_any = False
        missing = 0
        live_claims = 0
        for item in manifest:
            problem_id = int(item["problem_id"])
            base_seed = int(item["base_seed"])
            path = result_path(problem_id, base_seed)
            if path.exists():
                continue
            missing += 1
            claim = claim_task(item)
            if claim is None:
                claim_path = RUN_ROOT / "claims" / (task_id(problem_id, base_seed) + ".claim")
                live_claims += int(claim_path.exists() and claim_is_live(claim_path))
                continue
            claimed_any = True
            record = run_one(row_by_id[problem_id], base_seed)
            write_atomic(path, record)
            print(
                json.dumps(
                    {
                        "event": "wrote",
                        "claim": str(claim),
                        "task_key": item,
                        "correct": record.get("is_correct"),
                        "tokens": record.get("generated_tokens"),
                        "error": record.get("runtime_error"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if missing == 0:
            print(json.dumps({"event": "complete", "method": METHOD, "expected": len(manifest)}, ensure_ascii=False), flush=True)
            return
        if not claimed_any:
            print(
                json.dumps(
                    {"event": "waiting_for_peer_claims", "missing": missing, "live_claims": live_claims},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            time.sleep(60)


def status() -> None:
    done = 0
    correct = 0
    errors = 0
    missing = []
    active_claims = []
    for item in build_manifest():
        problem_id = int(item["problem_id"])
        seed = int(item["base_seed"])
        path = result_path(problem_id, seed)
        if path.exists():
            done += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            correct += int(bool(record.get("is_correct")))
            errors += int("runtime_error" in record)
        else:
            missing.append({"problem_id": problem_id, "base_seed": seed})
            claim = RUN_ROOT / "claims" / (task_id(problem_id, seed) + ".claim")
            if claim.exists():
                active_claims.append({"task_key": task_id(problem_id, seed), "claim": claim.read_text(encoding="utf-8").strip()})
    payload = {
        METHOD: {
            "done": done,
            "expected": 90,
            "correct": correct,
            "errors": errors,
            "accuracy": correct / done if done else None,
            "missing": missing,
            "active_claims": active_claims,
        }
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
        return
    if args.worker:
        worker()
        return
    print(json.dumps({"experiment_id": EXP, "method": METHOD, "manifest_count": len(build_manifest()), "method_config": BASE_CFG}, indent=2))


if __name__ == "__main__":
    main()
