#!/usr/bin/env python
"""Validate Insight PyTorch reference quantizers against the real pack path."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from insight.io import atomic_write_json, atomic_write_text
from insight.quant_reference import quantize_dequant_k_token_groups, quantize_dequant_v_head_dim
from quant.new_pack import triton_quantize_and_pack_along_last_dim, unpack_and_dequant_vcache


def dequant_last_dim(x: torch.Tensor, *, bits: int, group_size: int) -> torch.Tensor:
    code, scale, mn = triton_quantize_and_pack_along_last_dim(x.contiguous(), group_size, bits)
    return unpack_and_dequant_vcache(code, scale.unsqueeze(-1), mn.unsqueeze(-1), group_size, bits)


def cases(device: torch.device) -> list[tuple[str, torch.Tensor, str]]:
    gen = torch.Generator(device="cpu").manual_seed(123)
    rows: list[tuple[str, torch.Tensor, str]] = []
    rows.append(("random_k", torch.randn((1, 8, 256, 128), generator=gen, dtype=torch.float16), "k"))
    rows.append(("random_v", torch.randn((1, 8, 16, 128), generator=gen, dtype=torch.float16), "v"))
    rows.append(("positive_v", torch.rand((1, 8, 16, 128), generator=gen, dtype=torch.float16) * 7, "v"))
    rows.append(("negative_v", -torch.rand((1, 8, 16, 128), generator=gen, dtype=torch.float16) * 7, "v"))
    sym = torch.linspace(-3, 3, 128, dtype=torch.float16).view(1, 1, 1, 128).expand(1, 8, 16, 128).contiguous()
    rows.append(("symmetric_v", sym, "v"))
    outlier = torch.zeros((1, 8, 16, 128), dtype=torch.float16)
    outlier[..., 0] = 99
    rows.append(("outlier_v", outlier, "v"))
    rows.append(("constant_v", torch.full((1, 8, 16, 128), 2.0, dtype=torch.float16), "v"))
    rows.append(("tiny_range_v", torch.full((1, 8, 16, 128), 1.0, dtype=torch.float16) + torch.randn((1, 8, 16, 128), generator=gen, dtype=torch.float16) * 1e-3, "v"))
    return [(name, tensor.to(device), kind) for name, tensor, kind in rows]


def validate_case(name: str, x: torch.Tensor, kind: str, *, bits: int, group_size: int) -> dict[str, Any]:
    if kind == "k":
        ref = quantize_dequant_k_token_groups(x, bits=bits, group_size=group_size).dequant
        kernel_t = dequant_last_dim(x.transpose(2, 3).contiguous(), bits=bits, group_size=group_size)
        kernel = kernel_t.transpose(2, 3).contiguous()
    else:
        ref = quantize_dequant_v_head_dim(x, bits=bits, group_size=group_size).dequant
        kernel = dequant_last_dim(x.contiguous(), bits=bits, group_size=group_size)
    diff = (ref.float() - kernel.float()).abs()
    return {
        "name": name,
        "kind": kind,
        "shape": list(x.shape),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "passed": bool(float(diff.max().item()) <= 1e-3) if diff.numel() else True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-json", type=Path, default=Path("reports/insight_v2/quant_reference_validation.json"))
    parser.add_argument("--report-md", type=Path, default=Path("reports/insight_v2/quant_reference_validation.md"))
    parser.add_argument("--bits", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        payload = {"status": "blocked", "reason": "CUDA is not available", "generated_at": datetime.now(timezone.utc).isoformat()}
        atomic_write_json(args.report_json, payload)
        atomic_write_text(args.report_md, "# Quant Reference Validation\n\nStatus: BLOCKED\n\nCUDA is not available.\n")
        print(json.dumps(payload, sort_keys=True))
        return

    device = torch.device("cuda:0")
    results = [validate_case(name, x, kind, bits=args.bits, group_size=args.group_size) for name, x, kind in cases(device)]
    passed = all(r["passed"] for r in results)
    payload = {
        "schema_version": "insight_v2.quant_reference_validation",
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bits": args.bits,
        "group_size": args.group_size,
        "results": results,
    }
    atomic_write_json(args.report_json, payload)
    lines = ["# Quant Reference Validation", "", f"Status: {'PASS' if passed else 'FAIL'}", "", "| case | kind | max_abs_diff | mean_abs_diff | passed |", "|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['name']} | {r['kind']} | {r['max_abs_diff']:.6g} | {r['mean_abs_diff']:.6g} | {r['passed']} |")
    atomic_write_text(args.report_md, "\n".join(lines) + "\n")
    print(json.dumps({"status": payload["status"], "cases": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
