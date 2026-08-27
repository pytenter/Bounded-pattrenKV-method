from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports/qwen3_v100_system_generalization_v1"
MODEL_PATH = Path("/home/qinch2023/modelscope_models/Qwen3-8B")

@dataclass(frozen=True)
class Qwen3SystemGate:
    model: str = "Qwen3-8B"
    fp16_backend: str = "qwen3_native_eager_fp16"
    causal_backend: str = "qwen3_patternkv_compressed"
    compressed_domain_runtime_preserved: bool = True
    historical_fp16_k_materialization: int = 0
    historical_fp16_v_materialization: int = 0
    fallback_count: int = 0
    true_batch_preserved: bool = False
    classification: str = "QWEN_COMPRESSED_TRUE_BATCH_B2_FAIL"
    formal_timing_allowed: bool = False
    reason: str = (
        "B1 semantic parity is closed on GPU4 with compressed historical K/V materialization at zero, "
        "but full-model B2 true-batch decode fails in the legacy CUDA QK reader because request-local "
        "centroids are [B,H,M,D] while the kernel only accepts [H,M,D]."
    )

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def current_gate() -> Qwen3SystemGate:
    return Qwen3SystemGate()

def assert_formal_timing_allowed() -> None:
    gate = current_gate()
    if not gate.formal_timing_allowed:
        raise RuntimeError(f"STOP FORMAL: {gate.classification}: {gate.reason}")

def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen3/V100 system benchmark gate harness")
    parser.add_argument("--write-gate", action="store_true")
    parser.add_argument("--allow-formal", action="store_true")
    args = parser.parse_args()
    gate = current_gate()
    payload = asdict(gate)
    if args.write_gate:
        write_json(REPORT_DIR / "protocol_gate.json", payload)
    if args.allow_formal:
        assert_formal_timing_allowed()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
