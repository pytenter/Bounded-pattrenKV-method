from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer, GenerationConfig

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from bench.aime_utils import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generation_config_dict,
    load_aime24,
    normalize_eos_token_ids,
    search_model_candidates,
)
from bench.aime24_int2_wave1 import stable_hash  # noqa: E402
from bench.aime_generation_provenance import (  # noqa: E402
    PORTABLE_REFERENCE_GENERATION_SCHEMA,
    build_experiment_config_set_payload,
    build_portable_reference_generation_semantics,
    experiment_config_set_hash,
    portable_reference_generation_hash,
    recompute_legacy_generation_hash,
    validate_effective_seed_map,
)
from bench.bench_aime24_patternkv import render_prompt  # noqa: E402
from bench.paper_config import apply_method_defaults, method_config_dict  # noqa: E402
from bench.pseudodecode_metrics import CHECKPOINTS, SPARSE_LAYERS, write_csv_rows  # noqa: E402


SOURCE_COMMIT = "232e3b08d10919ca24932ad0a0135e46119ecfd5"
TASK_SHA256 = "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
GENERATION_CONFIG_HASH = "a7d6b2f8bab37893b6331c66b3e5eb6a"
LEGACY_MANIFEST_PATH = "reports/aime24_int2_wave1_v100_8gpu/revised_wave1a_full_run_manifest.json"
EXPERIMENT_BRANCH = "exp/aime-pseudodecode-3090-8gpu"
REPORT_DIR = ROOT / "reports/aime24_pseudodecode_3090_8gpu"
RESULT_DIR = ROOT / "results/aime24_pseudodecode_3090_8gpu"
ARTIFACT_DIR = ROOT / "artifacts/aime24_pseudodecode_3090"


CONCEPTUAL_CONFIGS = [
    {"config": "fp16", "method": "fp16", "mode_role": "reference_control", "gpu": None, "sink_length": 0, "recent_length": 0},
    {"config": "patternkv_paper", "method": "patternkv_paper", "mode_role": "paper", "gpu": 1, "sink_length": 0, "recent_length": 128},
    {"config": "pattern_rolling_k2v2_s0_r128", "method": "patternkv", "mode_role": "rolling", "gpu": 2, "sink_length": 0, "recent_length": 128},
    {"config": "pattern_rolling_k2v2_s16_r128", "method": "patternkv", "mode_role": "rolling", "gpu": 3, "sink_length": 16, "recent_length": 128},
    {"config": "kivi_paper_g128", "method": "kivi_paper_g128", "mode_role": "paper", "gpu": 4, "sink_length": 0, "recent_length": 128},
    {"config": "kivi_rolling_k2v2_s0_r128", "method": "kivi_official", "mode_role": "rolling", "gpu": 5, "sink_length": 0, "recent_length": 128},
    {"config": "kivi_rolling_k2v2_s16_r128", "method": "kivi_official", "mode_role": "rolling", "gpu": 6, "sink_length": 16, "recent_length": 128},
]


def run(cmd: list[str], *, check: bool = True) -> str:
    proc = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc.stdout.strip()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def blob_hash(path: Path) -> str | None:
    rel = path.relative_to(ROOT)
    out = run(["git", "hash-object", str(rel)], check=False)
    return out or None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_historical_generation_manifest() -> dict[str, Any]:
    raw = run(["git", "show", f"{SOURCE_COMMIT}:{LEGACY_MANIFEST_PATH}"])
    return json.loads(raw)


def inspect_model(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    tokenizer_config = model_path / "tokenizer_config.json"
    tokenizer_json = model_path / "tokenizer.json"
    if not config_path.exists():
        return {"model_path": str(model_path), "valid": False, "error": "config.json missing"}
    cfg = json_file(config_path)
    head_dim = cfg.get("head_dim") or (int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]) if cfg.get("hidden_size") and cfg.get("num_attention_heads") else None)
    vocab_hash = sha256_file(tokenizer_json)
    return {
        "model_path": str(model_path),
        "valid": True,
        "model_type": cfg.get("model_type"),
        "architectures": cfg.get("architectures"),
        "hidden_size": cfg.get("hidden_size"),
        "num_hidden_layers": cfg.get("num_hidden_layers"),
        "num_attention_heads": cfg.get("num_attention_heads"),
        "num_key_value_heads": cfg.get("num_key_value_heads"),
        "head_dim": head_dim,
        "vocab_size": cfg.get("vocab_size"),
        "max_position_embeddings": cfg.get("max_position_embeddings"),
        "torch_dtype": cfg.get("torch_dtype"),
        "config_sha256": sha256_file(config_path),
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
        "tokenizer_vocabulary_sha256": vocab_hash,
        "tokenizer_json_sha256": vocab_hash,
        "special_tokens": json_file(tokenizer_config).get("added_tokens_decoder") if tokenizer_config.exists() else None,
    }


def config_namespace(method: str, sink: int, recent: int, model_path: Path) -> Namespace:
    return Namespace(
        method=method,
        model_path=str(model_path),
        num_samples=2,
        do_sample=True,
        temperature=DEFAULT_TEMPERATURE,
        top_p=DEFAULT_TOP_P,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        base_seed=DEFAULT_BASE_SEED,
        force_think_prefix=True,
        k_bits=2,
        v_bits=2,
        group_size=128,
        residual_length=128,
        sink_length=sink,
        recent_length=recent,
        num_k_base=32,
        num_v_base=32,
        mixed_key_mask_path=None,
        mixed_key_int4_ratio=0.0,
        patternkv_cache_path="segmented",
        patternkv_cache_mode="segmented_rolling",
        manifest_methods=["fp16", "kivi_paper_g128", "patternkv_paper", "kivi_official", "patternkv"],
    )


def make_current_helper_hash_record(model_path: Path) -> dict[str, Any]:
    args = config_namespace("patternkv", 0, 128, model_path)
    cfg = generation_config_dict(args)
    actual = stable_hash(cfg, 32)
    return {
        "current_helper_schema": "bench.aime_utils.generation_config_dict",
        "current_helper_schema_hash": actual,
        "generation_config": cfg,
        "schema_fields": sorted(cfg.keys()),
    }


def resolved_tokenizer_smoke(model_path: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    try:
        generation_config = GenerationConfig.from_pretrained(model_path, local_files_only=True)
    except Exception:
        generation_config = None
    eos_ids = normalize_eos_token_ids(
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(tokenizer, "generation_config", None), "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
        getattr(config, "eos_token_id", None),
    )
    eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot, int) and eot >= 0:
        eos_ids.append(eot)
    eos_ids = sorted(set(int(x) for x in eos_ids if x is not None))

    dataset_rows = {int(row["problem_id"]): row for row in load_aime24(ROOT / "datasets/aime/aime24.jsonl")}
    first_task = tasks[0]
    row = dataset_rows[int(first_task["problem_id"])]
    rendered_prompt, _, _ = render_prompt(row["problem"], tokenizer, True)
    encoded = tokenizer(rendered_prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids[0].tolist()

    max_prompt_tokens = 0
    for task in tasks:
        task_row = dataset_rows[int(task["problem_id"])]
        task_prompt, _, _ = render_prompt(task_row["problem"], tokenizer, True)
        max_prompt_tokens = max(max_prompt_tokens, len(tokenizer(task_prompt, add_special_tokens=False).input_ids))

    max_position = getattr(config, "max_position_embeddings", None) or getattr(config, "model_max_length", None)
    context_limit = int(max_position) if max_position is not None else 0
    context_semantics_valid = bool(context_limit and max_prompt_tokens + DEFAULT_MAX_NEW_TOKENS <= context_limit)
    return {
        "problem_id": int(first_task["problem_id"]),
        "sample_id": int(first_task["sample_id"]),
        "task_key": str(first_task["task_key"]),
        "seed": int(first_task["seed"]),
        "rendered_prompt": rendered_prompt,
        "prompt_hash": stable_hash({"rendered_prompt": rendered_prompt}, 32),
        "input_token_ids_hash": stable_hash({"input_ids": input_ids}, 32),
        "input_token_count": len(input_ids),
        "max_prompt_token_count": max_prompt_tokens,
        "resolved_pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "resolved_eos_token_ids": eos_ids,
        "context_limit": context_limit,
        "context_semantics_valid": context_semantics_valid,
    }


def build_generation_audit(
    *,
    model_path: Path,
    model_identity: dict[str, Any],
    task_hash: str,
    tasks: list[dict[str, Any]],
    configs: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    historical_manifest = load_historical_generation_manifest()
    legacy_payload = historical_manifest["generation_config"]
    legacy_recomputed = recompute_legacy_generation_hash(legacy_payload)
    historical_reproduced = legacy_recomputed == GENERATION_CONFIG_HASH == historical_manifest.get("formal_generation_config_hash")

    helper = make_current_helper_hash_record(model_path)
    helper_hash = helper["current_helper_schema_hash"]
    helper_fields = set(helper["schema_fields"])
    legacy_fields = set(legacy_payload.keys())
    schema_mismatch = helper_hash != legacy_recomputed and helper_fields != legacy_fields

    tokenizer_smoke = resolved_tokenizer_smoke(model_path, tasks)
    tokenizer_identity_hash = model_identity.get("tokenizer_json_sha256") or model_identity.get("tokenizer_vocabulary_sha256")
    tokenizer_identity_valid = bool(tokenizer_identity_hash)
    task_seed_map_valid = validate_effective_seed_map(tasks, DEFAULT_BASE_SEED)

    portable_payload = build_portable_reference_generation_semantics(
        task_manifest_sha256=task_hash,
        tasks=tasks,
        model_name="DeepSeek-R1-Distill-Llama-8B",
        model_identity_hash=str(model_identity.get("config_sha256") or ""),
        tokenizer_identity_hash=str(tokenizer_identity_hash or ""),
        model_dtype="float16",
        context_limit=int(tokenizer_smoke["context_limit"]),
        resolved_pad_token_id=tokenizer_smoke["resolved_pad_token_id"],
        resolved_eos_token_ids=tokenizer_smoke["resolved_eos_token_ids"],
    )
    portable_hash = portable_reference_generation_hash(portable_payload)
    experiment_config_hash = experiment_config_set_hash(configs)
    portable_prompt_pipeline_valid = bool(tokenizer_smoke["prompt_hash"] and tokenizer_smoke["input_token_ids_hash"] and tokenizer_smoke["input_token_count"] > 0)
    portable_generation_semantics_valid = all(
        [
            task_hash == TASK_SHA256,
            len(tasks) == 12,
            task_seed_map_valid,
            bool(model_identity.get("valid")),
            tokenizer_identity_valid,
            portable_payload["model_dtype"] == "float16",
            portable_payload["do_sample"] is True,
            portable_payload["temperature"] == DEFAULT_TEMPERATURE,
            portable_payload["top_p"] == DEFAULT_TOP_P,
            portable_payload["repetition_penalty"] == 1.0,
            portable_payload["num_return_sequences"] == 1,
            portable_payload["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS,
            tokenizer_smoke["context_semantics_valid"],
            portable_prompt_pipeline_valid,
        ]
    )
    generation_config_valid = all(
        [
            historical_reproduced,
            schema_mismatch,
            portable_generation_semantics_valid,
            bool(model_identity.get("valid")),
            tokenizer_identity_valid,
            tokenizer_smoke["context_semantics_valid"],
            portable_prompt_pipeline_valid,
        ]
    )
    generation_audit = {
        "legacy_source": LEGACY_MANIFEST_PATH,
        "legacy_expected_hash": GENERATION_CONFIG_HASH,
        "legacy_recomputed_hash": legacy_recomputed,
        "historical_generation_hash_reproduced": historical_reproduced,
        "legacy_generation_config": legacy_payload,
        "legacy_hash_portable": False,
        "legacy_nonportable_fields": ["model absolute path"],
        "legacy_nonportable_reason": "absolute model path was included in the legacy fingerprint",
        "current_helper_schema": helper["current_helper_schema"],
        "current_helper_schema_hash": helper_hash,
        "current_helper_generation_config": helper["generation_config"],
        "current_helper_schema_fields": helper["schema_fields"],
        "legacy_schema_fields": sorted(legacy_fields),
        "generation_hash_schema_mismatch_confirmed": schema_mismatch,
        "portable_reference_generation_schema": PORTABLE_REFERENCE_GENERATION_SCHEMA,
        "portable_reference_generation_hash": portable_hash,
        "portable_generation_semantics_valid": portable_generation_semantics_valid,
        "portable_prompt_pipeline_valid": portable_prompt_pipeline_valid,
        "portable_prompt_smoke": tokenizer_smoke,
        "task_seed_map_valid": task_seed_map_valid,
        "tokenizer_identity_valid": tokenizer_identity_valid,
        "context_semantics_valid": tokenizer_smoke["context_semantics_valid"],
        "resolved_pad_token_id": tokenizer_smoke["resolved_pad_token_id"],
        "resolved_eos_token_ids": tokenizer_smoke["resolved_eos_token_ids"],
        "experiment_config_set_schema": "aime24_pseudodecode_config_set_v1",
        "experiment_config_set_hash": experiment_config_hash,
        "experiment_config_set_payload": build_experiment_config_set_payload(configs),
        "generation_config_valid": generation_config_valid,
        "audit_note": "The legacy a7d6 hash is reproduced as historical provenance. The de91 hash is retained as the current helper schema hash. The formal generation gate now uses portable reference generation semantics rather than equality to a machine-local legacy payload.",
    }
    return generation_audit, portable_payload, tokenizer_smoke


def audited_paths() -> list[dict[str, Any]]:
    requested = [
        ("configs/aime24_wave1_selected_tasks.json", "Frozen 12-task Wave1A cohort"),
        ("bench/aime24_int2_wave1.py", "Wave1A analysis helpers and stable hashing"),
        ("bench/attention_observer.py", "Read-only sparse attention metric helpers"),
        ("bench/paper_config.py", "Paper and rolling method config defaults"),
        ("bench/patternkv_equivalence_reference.py", "PatternKV equivalence reference helpers"),
        ("scripts/run_aime24_int2_wave1_8gpu.sh", "Prior 8-GPU Wave1A launch semantics"),
        ("scripts/run_aime24_wave1a4_attention_mechanism.sh", "Wave1A.4 mechanism launcher"),
        ("scripts/run_wave1a4_attention_observer.py", "Wave1A.4 observer runner"),
        ("models/segmented_cache.py", "Segmented sink/pending/recent/packed-history cache implementation"),
        ("models/llama_patternkv.py", "PatternKV Llama production cache path"),
        ("models/llama_kivi.py", "KIVI Llama production cache path"),
        ("quant/new_pack.py", "Quantization pack/depack helpers"),
        ("quant/matmul.py", "Quantized matmul helpers"),
        ("reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_early_token_mechanism_report.md", "Wave1A.4 mechanism report anchor"),
        ("reports/aime24_int2_wave1_v100_8gpu/wave1a4_attention_mechanism_summary.json", "Wave1A.4 mechanism summary anchor"),
        ("reports/aime24_int2_wave1_v100_8gpu/s128_sink_semantics_resolution.md", "Absolute-sequence-prefix sink semantics record"),
    ]
    rows = []
    for rel, purpose in requested:
        path = ROOT / rel
        rows.append({"relative_path": rel, "exists": path.exists(), "git_blob_hash": blob_hash(path) if path.exists() else None, "purpose": purpose})
    return rows


def environment_record(python_bin: str, model_path: Path) -> dict[str, Any]:
    gpu_lines = run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], check=False).splitlines()
    return {
        "python_executable": python_bin,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "torch_cuda_runtime": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": gpu_lines,
        "transformers": __import__("transformers").__version__,
        "numpy": __import__("numpy").__version__,
        "triton": __import__("triton").__version__,
        "model_path": str(model_path),
    }


def write_generation_provenance_report(generation: dict[str, Any], portable_payload: dict[str, Any]) -> None:
    legacy_payload = generation["legacy_generation_config"]
    lines = [
        "# Generation Config Provenance Resolution",
        "",
        "## 1. Why `a7d6...` Exists",
        "",
        f"The frozen hash comes from `{LEGACY_MANIFEST_PATH}` at source commit `{SOURCE_COMMIT}`.",
        "",
        "## 2. Exact Historical Payload",
        "",
        "```json",
        json.dumps(legacy_payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 3. Canonical Serialization Rule",
        "",
        '`json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`, then SHA256 truncated to 32 hex characters.',
        "",
        "## 4. Proof That `a7d6...` Can Be Reproduced",
        "",
        f"- Legacy expected hash: `{generation['legacy_expected_hash']}`",
        f"- Legacy recomputed hash: `{generation['legacy_recomputed_hash']}`",
        f"- HISTORICAL_GENERATION_HASH_REPRODUCED: `{generation['historical_generation_hash_reproduced']}`",
        "",
        "## 5. Why `de91...` Differs",
        "",
        "`de91...` is the hash of the current helper schema, `bench.aime_utils.generation_config_dict`. That schema has different fields from the legacy run manifest schema, so equality to `a7d6...` is not the right test.",
        "",
        f"- Current helper hash: `{generation['current_helper_schema_hash']}`",
        f"- GENERATION_HASH_SCHEMA_MISMATCH_CONFIRMED: `{generation['generation_hash_schema_mismatch_confirmed']}`",
        "",
        "## 6. Why The Legacy Hash Is Nonportable",
        "",
        "The legacy payload includes the V100 server absolute model path. The 3090 server has a different local model path, so path-inclusive hash equality would reject a semantically compatible independent server.",
        "",
        f"- LEGACY_HASH_PORTABLE: `{generation['legacy_hash_portable']}`",
        f"- Legacy nonportable fields: `{generation['legacy_nonportable_fields']}`",
        "",
        "## 7. Which Semantics Affect Reference Trajectories",
        "",
        "The portable reference-generation fingerprint includes task cohort, task seeds, model/tokenizer identity, dtype, prompt construction, chat template semantics, sampling controls, EOS/PAD resolution, max-new-token limit, and context compatibility. It excludes machine-local paths and quantized method sets.",
        "",
        "## 8. New Portable Schema",
        "",
        f"- Schema version: `{generation['portable_reference_generation_schema']}`",
        "",
        "```json",
        json.dumps(portable_payload, ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## 9. New Portable Hash",
        "",
        f"- Portable reference generation hash: `{generation['portable_reference_generation_hash']}`",
        f"- Experiment config set hash: `{generation['experiment_config_set_hash']}`",
        "",
        "## 10. Final Gate Decision",
        "",
        f"- TOKENIZER_IDENTITY_VALID: `{generation['tokenizer_identity_valid']}`",
        f"- CONTEXT_SEMANTICS_VALID: `{generation['context_semantics_valid']}`",
        f"- PORTABLE_GENERATION_SEMANTICS_VALID: `{generation['portable_generation_semantics_valid']}`",
        f"- PORTABLE_PROMPT_PIPELINE_VALID: `{generation['portable_prompt_pipeline_valid']}`",
        f"- GENERATION_CONFIG_VALID: `{generation['generation_config_valid']}`",
        "",
        "Formal run remains blocked by later preflight gates such as FP16 zero-gap, static independence, pseudo feedback, production parity, and observer non-invasiveness.",
        "",
    ]
    (REPORT_DIR / "generation_config_provenance_resolution.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=Path(os.environ.get("MODEL_PATH", "")) if os.environ.get("MODEL_PATH") else None)
    parser.add_argument("--python-bin", default=os.environ.get("PYTHON_BIN", sys.executable))
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    head = run(["git", "rev-parse", "HEAD"])
    merge_base = run(["git", "merge-base", "HEAD", SOURCE_COMMIT])
    source_valid = head == SOURCE_COMMIT or merge_base == SOURCE_COMMIT
    task_path = ROOT / "configs/aime24_wave1_selected_tasks.json"
    task_hash = sha256_file(task_path)
    tasks = json_file(task_path)

    model_path = args.model_path
    if model_path is None:
        candidates = search_model_candidates()
        model_path = Path(candidates[0]) if candidates else Path("")
    model_identity = inspect_model(model_path)

    origin = {
        "repository": "pytenter/Bounded-pattrenKV-method",
        "source_commit": SOURCE_COMMIT,
        "source_branch_reference": "exp/aime-int2-wave1-v100-8gpu",
        "experiment_branch": EXPERIMENT_BRANCH,
        "experiment_type": "pseudo_decode_accumulated_error",
        "server_role": "independent_8xRTX3090",
        "shared_filesystem_with_v100": False,
        "starting_head": SOURCE_COMMIT,
        "source_checkout_head": SOURCE_COMMIT,
        "current_head": head,
        "source_commit_valid": source_valid,
    }
    write_json(REPORT_DIR / "experiment_origin.json", origin)
    write_json(REPORT_DIR / "model_identity.json", model_identity)

    env = environment_record(args.python_bin, model_path)
    write_json(REPORT_DIR / "environment_audit.json", env)
    (REPORT_DIR / "environment_audit.md").write_text(
        "\n".join(
            [
                "# AIME24 Pseudo-Decode 3090 Environment Audit",
                "",
                f"- Python: {env['python_executable']} ({env['python_version']})",
                f"- PyTorch: {env['torch']} CUDA runtime {env['torch_cuda_runtime']}",
                f"- transformers: {env['transformers']}",
                f"- numpy: {env['numpy']}",
                f"- triton: {env['triton']}",
                f"- CUDA devices: {env['cuda_device_count']}",
                f"- Model path: {model_path}",
                "",
                "## GPUs",
                *[f"- {line}" for line in env["gpu"]],
                "",
            ]
        ),
        encoding="utf-8",
    )

    source_rows = audited_paths()
    write_json(REPORT_DIR / "github_source_audit.json", {"files": source_rows})
    (REPORT_DIR / "github_source_audit.md").write_text(
        "# GitHub Source Audit\n\n"
        "| relative path | git blob/hash | purpose | status |\n"
        "| --- | --- | --- | --- |\n"
        + "\n".join(
            f"| `{row['relative_path']}` | `{row['git_blob_hash']}` | {row['purpose']} | {'present' if row['exists'] else 'missing'} |"
            for row in source_rows
        )
        + "\n",
        encoding="utf-8",
    )

    configs = []
    for cfg in CONCEPTUAL_CONFIGS:
        ns = config_namespace(cfg["method"], cfg["sink_length"], cfg["recent_length"], model_path)
        ns.paper_method_config = apply_method_defaults(ns)
        resolved = method_config_dict(ns)
        configs.append({**cfg, "resolved_method_config": resolved})

    generation, portable_payload, tokenizer_smoke = build_generation_audit(
        model_path=model_path,
        model_identity=model_identity,
        task_hash=str(task_hash or ""),
        tasks=tasks,
        configs=configs,
    )
    write_json(REPORT_DIR / "generation_config_audit.json", generation)
    write_json(REPORT_DIR / "portable_reference_generation_semantics.json", portable_payload)
    write_generation_provenance_report(generation, portable_payload)

    expected_static = len([c for c in configs if c["config"] != "fp16"]) * len(tasks) * len(CHECKPOINTS)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "task_manifest_path": "configs/aime24_wave1_selected_tasks.json",
        "task_manifest_sha256": task_hash,
        "task_manifest_valid": task_hash == TASK_SHA256,
        "task_count": len(tasks),
        "expected_task_count": 12,
        "legacy_generation_hash": generation["legacy_expected_hash"],
        "legacy_generation_hash_reproduced": generation["historical_generation_hash_reproduced"],
        "legacy_hash_portable": generation["legacy_hash_portable"],
        "current_helper_schema_hash": generation["current_helper_schema_hash"],
        "generation_hash_schema_mismatch_confirmed": generation["generation_hash_schema_mismatch_confirmed"],
        "portable_reference_generation_schema": generation["portable_reference_generation_schema"],
        "portable_reference_generation_hash": generation["portable_reference_generation_hash"],
        "portable_generation_semantics_valid": generation["portable_generation_semantics_valid"],
        "portable_prompt_pipeline_valid": generation["portable_prompt_pipeline_valid"],
        "experiment_config_set_hash": generation["experiment_config_set_hash"],
        "generation_config_valid": generation["generation_config_valid"],
        "checkpoints": list(CHECKPOINTS),
        "sparse_layers": list(SPARSE_LAYERS),
        "conceptual_configs": configs,
        "expected_fp16_references": len(tasks),
        "expected_pseudo_trajectories": len([c for c in configs if c["config"] != "fp16"]) * len(tasks),
        "expected_static_checkpoint_runs": expected_static,
        "formal_run_approved": False,
        "formal_run_gate": {
            "SOURCE_COMMIT_VALID": source_valid,
            "TASK_MANIFEST_VALID": task_hash == TASK_SHA256,
            "GENERATION_CONFIG_VALID": generation["generation_config_valid"],
            "MODEL_IDENTITY_VALID": bool(model_identity.get("valid")),
            "TOKENIZER_IDENTITY_VALID": generation["tokenizer_identity_valid"],
            "HISTORICAL_GENERATION_HASH_REPRODUCED": generation["historical_generation_hash_reproduced"],
            "LEGACY_HASH_PORTABLE": generation["legacy_hash_portable"],
            "GENERATION_HASH_SCHEMA_MISMATCH_CONFIRMED": generation["generation_hash_schema_mismatch_confirmed"],
            "PORTABLE_GENERATION_SEMANTICS_VALID": generation["portable_generation_semantics_valid"],
            "PORTABLE_PROMPT_PIPELINE_VALID": generation["portable_prompt_pipeline_valid"],
            "CONTEXT_SEMANTICS_VALID": generation["context_semantics_valid"],
            "QUANT_KERNEL_SMOKE_PASS": None,
            "STATIC_INDEPENDENCE_PASS": None,
            "PSEUDO_FEEDBACK_PASS": None,
            "FP16_ZERO_ACCUMULATION_CONTROL_PASS": None,
            "PSEUDO_PRODUCTION_PARITY_PASS": None,
            "OBSERVER_NONINVASIVE": None,
        },
    }
    write_json(REPORT_DIR / "pseudodecode_manifest.json", manifest)

    write_csv_rows(
        REPORT_DIR / "checkpoint_availability.csv",
        [
            {"task_key": task["task_key"], "checkpoint": checkpoint, "checkpoint_available": False, "availability_reason": "reference_not_generated"}
            for task in tasks
            for checkpoint in CHECKPOINTS
        ],
    )
    for name in ("static_vs_pseudo_metrics.csv", "accumulation_gap.csv", "accumulation_auc.csv", "norm_tail_metrics.csv", "task_level_summary.csv"):
        (REPORT_DIR / name).write_text("", encoding="utf-8")

    summary = {
        "source_commit": SOURCE_COMMIT,
        "hardware": "8xRTX3090",
        "task_count": len(tasks),
        "formal_run_complete": None,
        "static_control_valid": None,
        "pseudo_feedback_valid": None,
        "fp16_zero_accumulation_control_pass": None,
        "pseudodecode_accumulation_supported": None,
        "pattern_sink_reduces_accumulation": None,
        "kivi_sink_reduces_accumulation": None,
        "remaining_error_accumulation_dominated": None,
        "single_step_representation_error_dominant": None,
        "token_norm_accumulation_supported": None,
        "early_error_as_accumulation_seed_supported": None,
        "varn_next_priority": None,
        "assignment_objective_next_priority": None,
        "next_priority": None,
        "generation_config_valid": generation["generation_config_valid"],
        "portable_reference_generation_hash": generation["portable_reference_generation_hash"],
        "historical_generation_hash_reproduced": generation["historical_generation_hash_reproduced"],
    }
    write_json(REPORT_DIR / "pseudodecode_summary.json", summary)
    report = [
        "# AIME24 Pseudo-Decode Accumulation Report",
        "",
        "## 1. Executive Summary",
        "",
        "Preparation is complete. Formal interpretation is intentionally gated on preflight and long-run completion.",
        "",
        "## 2. Research Question",
        "",
        "Does long-CoT INT2 degradation mainly arise from recursive pseudo-decode feedback beyond static quantization error?",
        "",
        "## 3. GitHub Experiment Origin",
        "",
        f"- Source commit: `{SOURCE_COMMIT}`",
        f"- Experiment branch: `{EXPERIMENT_BRANCH}`",
        f"- SOURCE_COMMIT_VALID: `{source_valid}`",
        "",
        "## 4. Independent-Server Reproducibility",
        "",
        "No V100-local files are used. The task manifest, generation semantics, model identity, and experiment manifest are recorded in this report directory.",
        "",
        "## 5. Hardware and Environment",
        "",
        f"- Hardware: 8 x RTX3090 detected: `{env['cuda_device_count'] == 8}`",
        f"- Python: `{env['python_version']}`",
        f"- PyTorch/CUDA: `{env['torch']}` / `{env['torch_cuda_runtime']}`",
        "",
        "## 6. Frozen Task Cohort",
        "",
        f"- Task count: `{len(tasks)}`",
        f"- Task SHA256: `{task_hash}`",
        f"- TASK_MANIFEST_VALID: `{task_hash == TASK_SHA256}`",
        "",
        "## 7. FP16 Reference Trajectories",
        "",
        "Not generated by the preparation step.",
        "",
        "## 8. Static Quantization Definition",
        "",
        "Static jobs must rebuild target cache state from clean FP16 prefix at each checkpoint and discard it after measurement.",
        "",
        "## 9. Pseudo-Decode Definition",
        "",
        "Pseudo jobs must teacher-force the FP16 reference token IDs while allowing the target production cache to feed back through hidden/Q/K/V states.",
        "",
        "## 10. Production Cache Semantics",
        "",
        "The manifest resolves paper and rolling configs through `bench.paper_config.apply_method_defaults`.",
        "",
        "## 11. Preflight Validation",
        "",
        "Pending. Formal run is not approved until all gate fields are true.",
        "",
        "## 32. Reproducibility",
        "",
        "See `experiment_origin.json`, `pseudodecode_manifest.json`, `model_identity.json`, and `generation_config_audit.json`.",
        "",
        "## 33. Recommended Next Experiment",
        "",
        "Pending formal results.",
        "",
    ]
    (REPORT_DIR / "pseudodecode_accumulation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"prepared": True, "report_dir": str(REPORT_DIR), "source_commit_valid": source_valid, "task_manifest_valid": task_hash == TASK_SHA256, "generation_config_valid": generation["generation_config_valid"], "model_identity_valid": model_identity.get("valid")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
