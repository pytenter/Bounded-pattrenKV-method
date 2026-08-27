from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/qwen3_v100_system_generalization_v1"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/home/qinch2023/modelscope_models/Qwen3-8B"))
PYTHON = Path(os.environ.get("PYTHON", sys.executable))

def run(cmd: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    p = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return {"cmd": cmd, "returncode": p.returncode, "output": p.stdout}

def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(name: str, payload: Any) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_text(name: str, text: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / name).write_text(text.rstrip() + "\n", encoding="utf-8")

def write_empty_csv(name: str, fields: list[str], status: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"status": status, "classification": "V100_COMPRESSED_BACKEND_NOT_READY", "reason": "compressed-domain Qwen3/V100 backend gate failed"})

def main() -> int:
    git = {name: run(cmd) for name, cmd in {
        "branch": ["git", "branch", "--show-current"],
        "head": ["git", "rev-parse", "HEAD"],
        "status_short": ["git", "status", "--short"],
        "remote": ["git", "remote", "-v"],
        "diff_check": ["git", "diff", "--check"],
    }.items()}
    smi = run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,pstate,power.limit", "--format=csv"])
    env_probe = run([str(PYTHON), "-c", "import json, sys, torch, transformers; print(json.dumps({'python': sys.executable, 'torch': torch.__version__, 'cuda': torch.version.cuda, 'cuda_available': torch.cuda.is_available(), 'transformers': transformers.__version__}))"])
    gate = run([str(PYTHON), "bench/qwen3_full_model_serving_benchmark.py", "--write-gate"])
    model_identity: dict[str, Any] = {"model_path": str(MODEL_PATH), "config_json_sha256": sha256_file(MODEL_PATH / "config.json")}
    cfg_path = MODEL_PATH / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("model_type", "architectures", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "hidden_size", "head_dim", "torch_dtype"):
            model_identity[key] = cfg.get(key)
    write_json("environment.json", {"platform": platform.platform(), "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), "git": git, "nvidia_smi_query": smi, "python_probe": env_probe})
    write_json("preflight.json", {"gpu_allowed_pool": [4, 5, 6, 7], "gpu_read_only_pool": [0, 1, 2, 3], "gpu0_3_touched": False, "git": git, "nvidia_smi_query": smi})
    write_text("preflight.md", "# Preflight\n\nGPU0-3 were treated as read-only. GPU4-7 were the only allowed pool. The active source worktree was dirty, so this task uses a sibling worktree. See `preflight.json` and `environment.json`.")
    write_text("github_system_reference_audit.md", "# GitHub System Reference Audit\n\nFrozen branch: `release/causal-v4-25-system-final`. Frozen SHA: `8d60485b5d2c93b7c1d478efc449de56d28159c3`. The frozen harness is Llama-specific (`LlamaConfig`, `LlamaForCausalLM`, DeepSeek-Llama model path). Protocol semantics retained for audit: true batch, decode-only timed window, no prefill/refill in timed window, no request membership changes, correct output-token accounting, and compressed-domain gates. Old 3090 conclusion kept unchanged: `FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE = NOT_SUPPORTED`.")
    write_json("workload_manifest.json", {"model": "Qwen3-8B", "formal_matrix_requested": {"batch_scaling": {"gpu": 4, "context": 2048, "decode": 8, "batch": [1, 2, 4, 8]}, "context_scaling": {"gpu": 5, "batch": 1, "decode": 8, "context": [2048, 4096, 8192]}, "long_decode": {"gpu": 6, "context": 2048, "batch": 1, "decode": [256, 512, 1024]}}})
    write_text("protocol.md", "# Protocol\n\nThis branch ports the frozen system benchmark protocol semantics to Qwen3/V100, but formal timing is blocked unless semantic parity and compressed-domain gates pass. Peak memory/capacity sweeps are intentionally excluded.")
    write_json("semantic_parity.json", {"FP16_QWEN_NATIVE_PARITY": "NOT_RUN", "CAUSAL_QWEN_SEMANTIC_PARITY": "NOT_RUN", "reason": "Formal semantic smoke requires a Qwen3 compressed-domain system adapter. The available Qwen3 adapter is a reference path that reconstructs historical K/V."})
    write_text("semantic_parity.md", "# Semantic Parity\n\nStatus: NOT_RUN. The migrated Qwen3 adapter is retained as reference evidence, but it reconstructs full historical K/V and is not eligible for formal system timing.")
    write_json("qwen_backend_audit.json", model_identity)
    write_text("qwen_backend_audit.md", f"# Qwen Backend Audit\n\nModel path: `{MODEL_PATH}`. Config hash: `{model_identity.get('config_json_sha256')}`. The available Qwen3 PatternKV adapter imports native Qwen3 classes and is not a Llama class, but it reconstructs historical K/V during decode and is therefore not a valid compressed-domain performance backend.")
    write_text("v100_compatibility_audit.md", "# V100 Compatibility Audit\n\nV100 compute capability is sm70. BF16 is not used. FlashAttention is absent in the audited environment, so the compatible FP16 attention backend is eager/native attention. Formal CAUSAL timing is stopped because the Qwen compressed-domain backend is not ready, not because of FlashAttention fallback.")
    write_text("claim_audit.md", "# Claim Audit\n\nNo new throughput, speedup, memory, or capacity claim is supported. The old RTX3090/DeepSeek-Llama negative full-model throughput result remains unchanged.")
    write_text("limitations.md", "# Limitations\n\n- Qwen3/V100 formal timing was not run.\n- The available Qwen3 CAUSAL reference adapter materializes full historical K/V and fails the compressed-domain system gate.\n- No peak memory or capacity conclusions are made.")
    write_text("final_decision.md", "# Final Decision\n\nSTOP FORMAL. Classification: `V100_COMPRESSED_BACKEND_NOT_READY`. The task cannot produce scientifically valid Qwen3/V100 CAUSAL full-model serving performance numbers until a Qwen-native compressed-domain decode backend passes the required gates.")
    for name in ("gpu_calibration", "pilot_raw", "batch_scaling_raw", "batch_scaling", "context_scaling_raw", "context_scaling", "long_decode_raw", "long_decode", "gpu7_anchor_raw", "relative_efficiency_comparison"):
        write_empty_csv(f"{name}.csv", ["status", "classification", "reason"], "STOP_FORMAL")
    for name in ("gpu_calibration.md", "pilot_summary.md", "batch_scaling.md", "context_scaling.md", "long_decode.md", "gpu7_anchor.md", "relative_efficiency_comparison.md"):
        write_text(name, "# %s\n\nStatus: STOP_FORMAL. No formal timing was run because the compressed-domain Qwen3/V100 backend gate did not pass." % name[:-3].replace("_", " ").title())
    return 0 if gate["returncode"] == 0 else gate["returncode"]

if __name__ == "__main__":
    raise SystemExit(main())
