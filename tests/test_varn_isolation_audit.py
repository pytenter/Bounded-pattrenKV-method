from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from bench.reference_varn import restore_varn_tile, variance_normalize_reference

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "reports/varn_isolation_audit"


def _decision() -> dict:
    return json.loads((AUDIT_DIR / "varn_isolation_decision.json").read_text(encoding="utf-8"))


def test_varn_source_commit_pinned():
    data = json.loads((AUDIT_DIR / "varn_source_provenance.json").read_text(encoding="utf-8"))
    assert data["source_commit"] == "7586257f1c632e63187bfacbbe21ccb51540f7b3"
    assert len(data["source_commit"]) == 40
    assert data["source_branch"] == "origin/main"


def test_varn_symbol_dependency_map():
    rows = json.loads((AUDIT_DIR / "varn_symbol_map.json").read_text(encoding="utf-8"))
    files = {row["file"] for row in rows}
    assert "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py" in files
    assert "vllm/v1/attention/backends/kvarn_attn.py" in files
    assert any("kvarn_sinkhorn_triton" in row["text"] for row in rows)


def test_varn_math_definition_complete():
    d = _decision()
    assert d["varn_math_reconstructed"] is True
    assert d["varn_applies_to_k"] is True
    assert d["varn_applies_to_v"] is True
    assert d["varn_only_math_valid"] is True


def test_varn_hadamard_dependency_classified():
    d = _decision()
    assert d["varn_requires_hadamard_mathematically"] is False
    assert d["varn_fused_with_hadamard_implementation"] is True
    assert d["isolation_case"] == "CASE_B_MATHEMATICALLY_ISOLATABLE_BUT_KERNEL_FUSED"


def test_varn_rope_order_known():
    d = _decision()
    assert d["rope_order_known"] is True
    assert "RoPE -> Hadamard -> VarN" in d["rope_order"]


def test_varn_quantization_order_known():
    d = _decision()
    assert d["quantization_order_known"] is True
    assert "asymmetric RTN" in d["quantization_order"]


def test_varn_decode_restore_path_known():
    d = _decision()
    assert d["decode_restore_path_known"] is True
    assert "dequantize" in d["decode_restore_point"]


def test_varn_metadata_accounted():
    d = _decision()
    assert d["metadata"]["fp16_metadata_bytes_per_tile_k"] == 768
    assert d["metadata"]["fp16_metadata_bytes_per_tile_v"] == 768
    assert d["metadata"]["fp16_metadata_bits_per_element_kv"] == 0.75
    assert d["calibration_required"] is False


def test_varn_reference_finite():
    torch.manual_seed(11)
    x = torch.randn(32, 128)
    balanced, s_col, s_row = variance_normalize_reference(x)
    assert torch.isfinite(balanced).all()
    assert torch.isfinite(s_col).all()
    assert torch.isfinite(s_row).all()


def test_varn_reference_roundtrip():
    torch.manual_seed(12)
    x = torch.randn(64, 128)
    balanced, s_col, s_row = variance_normalize_reference(x)
    restored = restore_varn_tile(balanced, s_col, s_row)
    assert torch.allclose(restored, x.float(), atol=1e-6, rtol=1e-6)


def test_varn_canonical_equivalence():
    sinkhorn_path = Path("/data/zypan/kvarn-repro/repos/KVarN/vllm/model_executor/layers/quantization/kvarn/sinkhorn.py")
    if not sinkhorn_path.exists():
        pytest.skip("skipped_with_explicit_reason: canonical KVarN sinkhorn.py is unavailable")
    spec = importlib.util.spec_from_file_location("canonical_kvarn_sinkhorn_for_test", sinkhorn_path)
    if spec is None or spec.loader is None:
        pytest.skip("skipped_with_explicit_reason: canonical KVarN sinkhorn.py cannot be imported")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(13)
    x = torch.randn(128, 128)
    ref = variance_normalize_reference(x, iterations=16)
    can = module.variance_normalize(x, iterations=16)
    assert torch.allclose(ref[0], can[0], atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref[1], can[1], atol=1e-6, rtol=1e-6)
    assert torch.allclose(ref[2], can[2], atol=1e-6, rtol=1e-6)
