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

ROOT = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(ROOT))

from bench.aime_utils import (  # noqa: E402
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    generation_config_dict,
    search_model_candidates,
)
from bench.aime24_int2_wave1 import stable_hash  # noqa: E402
from bench.paper_config import apply_method_defaults, method_config_dict  # noqa: E402
from bench.pseudodecode_metrics import CHECKPOINTS, SPARSE_LAYERS, canonical_json_hash, write_csv_rows  # noqa: E402


SOURCE_COMMIT = "232e3b08d10919ca24932ad0a0135e46119ecfd5"
TASK_SHA256 = "ed3ff618c8072787a7b1687fef368c5c8d2c04801baf33fe850fca3b24a7af2e"
GENERATION_CONFIG_HASH = "a7d6b2f8bab37893b6331c66b3e5eb6a"
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


def make_generation_hash_record(model_path: Path) -> dict[str, Any]:
    args = config_namespace("patternkv", 0, 128, model_path)
    cfg = generation_config_dict(args)
    actual = stable_hash(cfg, 32)
    return {
        "expected_generation_config_hash": GENERATION_CONFIG_HASH,
        "reconstructed_generation_config_hash": actual,
        "generation_config_valid": actual == GENERATION_CONFIG_HASH,
        "generation_config": cfg,
        "audit_note": "The repository helper was used to rebuild generation semantics. The rebuilt 32-hex hash does not match the frozen hash on this server, so formal generation_config_valid remains false until the original freeze payload is identified or the mismatch is explicitly adjudicated.",
    }


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
    generation = make_generation_hash_record(model_path)

    origin = {
        "repository": "pytenter/Bounded-pattrenKV-method",
        "source_commit": SOURCE_COMMIT,
        "source_branch_reference": "exp/aime-int2-wave1-v100-8gpu",
        "experiment_branch": EXPERIMENT_BRANCH,
        "experiment_type": "pseudo_decode_accumulated_error",
        "server_role": "independent_8xRTX3090",
        "shared_filesystem_with_v100": False,
        "starting_head": head,
        "source_commit_valid": source_valid,
    }
    write_json(REPORT_DIR / "experiment_origin.json", origin)
    write_json(REPORT_DIR / "model_identity.json", model_identity)
    write_json(REPORT_DIR / "generation_config_audit.json", generation)

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

    expected_static = len([c for c in configs if c["config"] != "fp16"]) * len(tasks) * len(CHECKPOINTS)
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "task_manifest_path": "configs/aime24_wave1_selected_tasks.json",
        "task_manifest_sha256": task_hash,
        "task_manifest_valid": task_hash == TASK_SHA256,
        "task_count": len(tasks),
        "expected_task_count": 12,
        "generation_config_hash": generation["reconstructed_generation_config_hash"],
        "expected_generation_config_hash": GENERATION_CONFIG_HASH,
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
