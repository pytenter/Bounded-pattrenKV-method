#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bench.reference_varn import restore_varn_tile, variance_normalize_reference  # noqa: E402

OUT_DIR = ROOT / "reports/varn_isolation_audit"
DEFAULT_KVARN_REPO = Path("/data/zypan/kvarn-repro/repos/KVarN")
SOURCE_BRANCH = "origin/main"
SOURCE_COMMIT = "7586257f1c632e63187bfacbbe21ccb51540f7b3"
SOURCE_FILES = [
    "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py",
    "vllm/model_executor/layers/quantization/kvarn/config.py",
    "vllm/v1/attention/backends/kvarn_attn.py",
    "vllm/v1/attention/ops/triton_kvarn_sinkhorn.py",
    "vllm/v1/attention/ops/triton_kvarn_decode.py",
]
SYMBOL_PATTERNS = [
    "VarN",
    "varn",
    "Sinkhorn",
    "sinkhorn",
    "Hadamard",
    "hadamard",
    "variance",
    "row_scale",
    "col_scale",
    "token_scale",
    "channel_scale",
    "scale_k",
    "scale_v",
    "k_scale",
    "v_scale",
]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_text(repo: Path, file_path: str) -> str:
    return git(repo, "show", f"{SOURCE_BRANCH}:{file_path}")


def canonical_equivalence(repo: Path) -> dict[str, Any]:
    sinkhorn_path = repo / "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py"
    if not sinkhorn_path.exists():
        return {"executable": False, "pass": None, "reason": "canonical sinkhorn.py is unavailable"}
    spec = importlib.util.spec_from_file_location("canonical_kvarn_sinkhorn", sinkhorn_path)
    if spec is None or spec.loader is None:
        return {"executable": False, "pass": None, "reason": "cannot import canonical sinkhorn.py"}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(20260809)
    tile = torch.randn(128, 128, dtype=torch.float32)
    ref_bal, ref_col, ref_row = variance_normalize_reference(tile, iterations=16)
    can_bal, can_col, can_row = module.variance_normalize(tile, iterations=16)
    restored = restore_varn_tile(ref_bal, ref_col, ref_row)
    checks = {
        "balanced_max_abs": float((ref_bal - can_bal).abs().max().item()),
        "s_col_max_abs": float((ref_col - can_col).abs().max().item()),
        "s_row_max_abs": float((ref_row - can_row).abs().max().item()),
        "roundtrip_max_abs": float((restored - tile.float()).abs().max().item()),
    }
    return {
        "executable": True,
        "pass": all(value <= 1e-6 for value in checks.values()),
        "checks": checks,
    }


def build_symbol_map(repo: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pattern = "|".join(SYMBOL_PATTERNS)
    try:
        raw = git(repo, "grep", "-n", "-E", pattern, SOURCE_BRANCH, "--", "vllm/")
    except subprocess.CalledProcessError as exc:
        raw = exc.output or ""
    for line in raw.splitlines():
        if not line.strip():
            continue
        _, file_path, lineno, text = line.split(":", 3)
        rows.append({"file": file_path, "line": int(lineno), "text": text.strip()[:240]})
    return rows


def render_symbol_map(rows: list[dict[str, Any]]) -> str:
    selected = [row for row in rows if "/kvarn/" in row["file"] or "kvarn_" in row["file"]]
    lines = ["# VarN Symbol Dependency Map", ""]
    lines.append("The full grep was restricted to `vllm/`; this file lists KVarN-relevant hits.")
    lines.append("")
    for row in selected[:240]:
        lines.append(f"- `{row['file']}:{row['line']}` {row['text']}")
    return "\n".join(lines) + "\n"


def run_audit(kvarn_repo: Path) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status = git(kvarn_repo, "status", "--short")
    remote = git(kvarn_repo, "remote", "get-url", "origin")
    head = git(kvarn_repo, "rev-parse", "HEAD")
    pinned = git(kvarn_repo, "rev-parse", SOURCE_BRANCH)
    contents = {path: source_text(kvarn_repo, path) for path in SOURCE_FILES}
    symbol_rows = build_symbol_map(kvarn_repo)
    equivalence = canonical_equivalence(kvarn_repo)

    sinkhorn = contents["vllm/model_executor/layers/quantization/kvarn/sinkhorn.py"]
    config = contents["vllm/model_executor/layers/quantization/kvarn/config.py"]
    backend = contents["vllm/v1/attention/backends/kvarn_attn.py"]
    decode = contents["vllm/v1/attention/ops/triton_kvarn_decode.py"]
    component_found = "def variance_normalize" in sinkhorn and "kvarn_sinkhorn_triton" in backend
    math_reconstructed = all(token in sinkhorn for token in ("log_s_col", "log_s_row", "col_std", "row_std", "_imbalance"))
    implementation_fused = "Hadamard, calls" in backend and "kvarn_sinkhorn_triton" in backend and "varn_enabled" not in config
    hadamard_order = "K/V source -> Hadamard rotation along head_dim -> VarN/Sinkhorn scaling -> low-bit quantization"
    rope_order = "K projection -> RoPE -> Hadamard -> VarN/Sinkhorn -> quantization; V has no RoPE and enters Hadamard directly after V projection"
    decode_restore = "dequantize rotated/normalized tile scales first; rotate Q for QK in decode and un-rotate output, or equivalently un-rotate dequantized K/V before attention"

    decision = {
        "source_repo": str(kvarn_repo),
        "source_remote": remote,
        "source_branch": SOURCE_BRANCH,
        "source_commit": pinned,
        "local_head": head,
        "local_dirty_worktree": bool(status),
        "local_dirty_files": [line.strip() for line in status.splitlines() if line.strip()],
        "varn_component_found": component_found,
        "varn_math_reconstructed": math_reconstructed,
        "varn_applies_to_k": True,
        "varn_applies_to_v": True,
        "rope_order_known": True,
        "rope_order": rope_order,
        "hadamard_order_known": True,
        "hadamard_order": hadamard_order,
        "quantization_order_known": True,
        "quantization_order": "VarN/Sinkhorn output is quantized after scaling; asymmetric RTN scale/zero-point are absorbed into one VarN scale axis.",
        "decode_restore_path_known": True,
        "decode_restore_point": decode_restore,
        "varn_requires_hadamard_mathematically": False,
        "varn_requires_hadamard_mathematically_reason": "The audited formula is invertible as X = balanced * s_col * s_row for any finite tile. It is defined on the tile it receives; canonical KVarN feeds it Hadamard-rotated K/V, but the scaling algebra itself does not require H.",
        "varn_fused_with_hadamard_implementation": implementation_fused,
        "varn_only_math_valid": True,
        "varn_only_reference_implemented": True,
        "canonical_equivalence_executable": equivalence["executable"],
        "canonical_equivalence_test_pass": equivalence["pass"],
        "canonical_equivalence": equivalence,
        "varn_only_implementation_path_valid": False,
        "varn_only_semantics_valid": True,
        "isolation_case": "CASE_B_MATHEMATICALLY_ISOLATABLE_BUT_KERNEL_FUSED",
        "metadata": {
            "k": "s_col [1, group] per token-in-tile, s_row [D, 1] per channel; RTN scale/zero are absorbed into the per-channel axis.",
            "v": "s_col [1, D] per channel, s_row [group, 1] per token-in-tile; RTN scale/zero are absorbed into the per-token axis.",
            "fp16_metadata_bytes_per_tile_k": 2 * (128 + 128 + 128),
            "fp16_metadata_bytes_per_tile_v": 2 * (128 + 128 + 128),
            "fp16_metadata_bits_per_element_kv": 0.75,
        },
        "calibration_required": False,
        "offline_online": {
            "offline": "Hadamard matrix construction may be cached; no calibration dataset is used.",
            "prefill_or_flush_time": "VarN scales are computed from the current tile at cache flush time.",
            "decode_time": "Stored scales restore dequantized K/V; Hadamard inverse/rotated-Q semantics are applied in attention.",
        },
        "next_action": "Use the CPU reference only as an equivalence harness; do not port Pattern+VarN until a canonical VarN-only implementation path is frozen.",
    }

    write_json(OUT_DIR / "varn_source_provenance.json", {
        "source_repo": str(kvarn_repo),
        "source_remote": remote,
        "source_branch": SOURCE_BRANCH,
        "source_commit": pinned,
        "source_files": SOURCE_FILES,
        "local_head": head,
        "local_dirty_worktree": bool(status),
        "local_dirty_files": decision["local_dirty_files"],
    })
    (OUT_DIR / "varn_symbol_map.md").write_text(render_symbol_map(symbol_rows), encoding="utf-8")
    write_json(OUT_DIR / "varn_symbol_map.json", symbol_rows)
    (OUT_DIR / "varn_dataflow.md").write_text(render_dataflow(decision), encoding="utf-8")
    (OUT_DIR / "varn_math_derivation.md").write_text(render_math(decision), encoding="utf-8")
    (OUT_DIR / "varn_hadamard_dependency.md").write_text(render_hadamard_dependency(decision), encoding="utf-8")
    (OUT_DIR / "varn_metadata_analysis.md").write_text(render_metadata(decision), encoding="utf-8")
    write_json(OUT_DIR / "varn_isolation_decision.json", decision)
    (OUT_DIR / "varn_isolation_audit.md").write_text(render_audit(decision), encoding="utf-8")
    return decision


def render_dataflow(decision: dict[str, Any]) -> str:
    return f"""# KVarN VarN Dataflow

Canonical source path:

```text
hidden
-> K/V projection
-> RoPE on K only
-> Hadamard rotation along head_dim
-> VarN/Sinkhorn scaling
-> asymmetric RTN quantization
-> metadata storage
-> dequant + scale restore
-> rotated-Q QK path and output un-rotation, or equivalent K/V un-rotation
-> attention aggregation
```

RoPE order: {decision['rope_order']}

Hadamard order: {decision['hadamard_order']}

Decode restore: {decision['decode_restore_point']}
"""


def render_math(decision: dict[str, Any]) -> str:
    return """# VarN Math Derivation

For a tile `X in R^{R x C}`, canonical VarN initializes `log_s_col=0` and
`log_s_row=0`, then repeats:

```text
col_std = std(X / exp(log_s_col) / exp(log_s_row), dim=rows)
log_s_col = clip(log_s_col + log(clamp(col_std, 1e-3, 1e3)), -0.3, 10.0)

row_std = std(X / exp(log_s_col) / exp(log_s_row), dim=cols)
log_s_row = clip(log_s_row + log(clamp(row_std, 1e-3, 1e3)), -0.3, 10.0)
```

At each iteration it computes imbalance:

```text
max(col_std)/min(col_std) + max(row_std)/min(row_std)
```

and returns the best-so-far state:

```text
balanced = X / s_col / s_row
restore(X) = balanced * s_col * s_row
```

For K, canonical orientation is `[D, group]`, so `s_row` is per-channel and
`s_col` is per-token. For V, canonical orientation is `[group, D]`, so `s_row`
is per-token and `s_col` is per-channel.
"""


def render_hadamard_dependency(decision: dict[str, Any]) -> str:
    return f"""# VarN Hadamard Dependency

`VARN_REQUIRES_HADAMARD_MATHEMATICALLY={decision['varn_requires_hadamard_mathematically']}`.

Reason: {decision['varn_requires_hadamard_mathematically_reason']}

`VARN_FUSED_WITH_HADAMARD_IMPLEMENTATION={decision['varn_fused_with_hadamard_implementation']}`.

Canonical KVarN uses Hadamard before VarN and its deployed attention backend is
written around rotated K/V. A clean Pattern+VarN intervention therefore still
needs a pinned VarN-only implementation path, even though the scaling formula
can be expressed independently.
"""


def render_metadata(decision: dict[str, Any]) -> str:
    meta = decision["metadata"]
    return f"""# VarN Metadata Analysis

K metadata: {meta['k']}

V metadata: {meta['v']}

For `head_dim=128`, `group=128`, fp16 scale metadata contributes:

```text
K scale/zero/second-scale bytes per tile = {meta['fp16_metadata_bytes_per_tile_k']}
V scale/zero/second-scale bytes per tile = {meta['fp16_metadata_bytes_per_tile_v']}
metadata bits per K/V element = {meta['fp16_metadata_bits_per_element_kv']}
```

Calibration required: `{decision['calibration_required']}`.
"""


def render_audit(decision: dict[str, Any]) -> str:
    return f"""# VarN Isolation Audit

`VARN_COMPONENT_FOUND={decision['varn_component_found']}`.
`VARN_MATH_RECONSTRUCTED={decision['varn_math_reconstructed']}`.
`VARN_ONLY_MATH_VALID={decision['varn_only_math_valid']}`.
`VARN_ONLY_IMPLEMENTATION_PATH_VALID={decision['varn_only_implementation_path_valid']}`.
`VARN_ONLY_SEMANTICS_VALID={decision['varn_only_semantics_valid']}`.

Isolation case:

```text
{decision['isolation_case']}
```

Next action:

```text
{decision['next_action']}
```
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kvarn-repo", type=Path, default=DEFAULT_KVARN_REPO)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.kvarn_repo), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
