import json
import math
from pathlib import Path

import torch

import patternkv_gemv
from quant.matmul import cuda_bmm_fA_qB_outer
from quant.new_pack import (
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    diff = (a.float() - b.float()).detach()
    return {
        "mae": diff.abs().mean().item(),
        "mse": diff.square().mean().item(),
        "max_abs": diff.abs().max().item(),
    }


def _finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor.float()).all():
        raise AssertionError(f"{name} contains NaN/Inf")


@torch.no_grad()
def run_case(seq_len: int, bits: int, batch: int = 1, heads: int = 8, head_dim: int = 128) -> dict:
    torch.manual_seed(20260801 + seq_len + bits)
    device = torch.device("cuda")
    group_size = 128
    pack = 32 // bits

    key = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    value = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=torch.float16)
    query = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.float16)
    attn = torch.randn(batch, heads, 1, seq_len, device=device, dtype=torch.float16)

    k_code, k_scale, k_min = triton_quantize_and_pack_along_last_dim(
        key.transpose(2, 3).contiguous(), group_size, bits
    )
    v_code, v_scale, v_min = triton_quantize_and_pack_along_last_dim(value, group_size, bits)

    assert k_code.dtype == torch.int32
    assert v_code.dtype == torch.int32
    assert k_code.shape == (batch, heads, head_dim, seq_len // pack)
    assert v_code.shape == (batch, heads, seq_len, head_dim // pack)
    assert k_scale.shape == (batch, heads, head_dim, seq_len // group_size)
    assert v_scale.shape == (batch, heads, seq_len, head_dim // group_size)
    assert k_min.shape == k_scale.shape
    assert v_min.shape == v_scale.shape
    assert k_scale.dtype == torch.float16
    assert v_scale.dtype == torch.float16
    assert k_min.dtype == torch.float16
    assert v_min.dtype == torch.float16

    k_deq_t = unpack_and_dequant_vcache(
        k_code, k_scale.unsqueeze(-1), k_min.unsqueeze(-1), group_size, bits
    )
    k_deq = k_deq_t.transpose(2, 3).contiguous()
    v_deq = unpack_and_dequant_vcache(
        v_code, v_scale.unsqueeze(-1), v_min.unsqueeze(-1), group_size, bits
    )

    _finite("k_deq", k_deq)
    _finite("v_deq", v_deq)

    qk_cuda = cuda_bmm_fA_qB_outer(group_size, query, k_code, k_scale, k_min, bits)
    qk_ref = torch.matmul(query, k_deq.transpose(2, 3))
    av_cuda = cuda_bmm_fA_qB_outer(group_size, attn, v_code, v_scale, v_min, bits)
    av_ref = torch.matmul(attn, v_deq)

    _finite("qk_cuda", qk_cuda)
    _finite("av_cuda", av_cuda)

    qk_err = _stats(qk_cuda, qk_ref)
    av_err = _stats(av_cuda, av_ref)
    if qk_err["mae"] > 1e-2 or qk_err["max_abs"] > 5e-2:
        raise AssertionError(f"qk cuda/reference max_abs too high: {qk_err}")
    if av_err["mae"] > 1e-2 or av_err["max_abs"] > 5e-2:
        raise AssertionError(f"av cuda/reference max_abs too high: {av_err}")

    return {
        "batch": batch,
        "kv_heads": heads,
        "head_dim": head_dim,
        "sequence_length": seq_len,
        "bits": bits,
        "group_size": group_size,
        "input_dtype": str(key.dtype),
        "k_code": {"shape": list(k_code.shape), "dtype": str(k_code.dtype)},
        "v_code": {"shape": list(v_code.shape), "dtype": str(v_code.dtype)},
        "k_scale": {"shape": list(k_scale.shape), "dtype": str(k_scale.dtype)},
        "v_scale": {"shape": list(v_scale.shape), "dtype": str(v_scale.dtype)},
        "k_dequant_error_vs_input": _stats(k_deq, key),
        "v_dequant_error_vs_input": _stats(v_deq, value),
        "qk_cuda_error_vs_dequant_reference": qk_err,
        "av_cuda_error_vs_dequant_reference": av_err,
    }


def test_quant_extension_cases():
    assert patternkv_gemv is not None
    assert torch.cuda.is_available()
    results = []
    for seq_len in (128, 256):
        for bits in (2, 4):
            results.append(run_case(seq_len=seq_len, bits=bits))

    out = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cases": results,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/quant_unit_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    Path("reports").mkdir(exist_ok=True)
    lines = [
        "# Quant Extension Unit Test",
        "",
        f"Status: {out['status']}",
        f"GPU: {out['gpu']} capability {out['capability']}",
        f"Torch: {out['torch']} CUDA {out['torch_cuda']}",
        "",
        "| bits | seq_len | k_code | v_code | qk max_abs | av max_abs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in results:
        lines.append(
            "| {bits} | {sequence_length} | {k_code_shape} {k_code_dtype} | "
            "{v_code_shape} {v_code_dtype} | {qk:.6g} | {av:.6g} |".format(
                bits=case["bits"],
                sequence_length=case["sequence_length"],
                k_code_shape=case["k_code"]["shape"],
                k_code_dtype=case["k_code"]["dtype"],
                v_code_shape=case["v_code"]["shape"],
                v_code_dtype=case["v_code"]["dtype"],
                qk=case["qk_cuda_error_vs_dequant_reference"]["max_abs"],
                av=case["av_cuda_error_vs_dequant_reference"]["max_abs"],
            )
        )
    Path("reports/quant_unit_test.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    test_quant_extension_cases()
