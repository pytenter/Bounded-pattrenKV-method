from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from torch import nn

from models.llama_patternkv import (
    patternkv_bi_mlp_oracle_counters,
    patternkv_bi_mlp_oracle_enabled,
    patternkv_decode_bi_mlp_components,
    patternkv_decode_bi_mlp_enabled,
    patternkv_mlp_oracle_forward,
    reset_patternkv_bi_mlp_oracle_counters,
)
from quant.batch_invariant_kproj import batch_invariant_linear_projection, patternkv_prefill_projection_mode


class TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def tiny_layer() -> SimpleNamespace:
    torch.manual_seed(0)
    return SimpleNamespace(mlp=TinyMLP())


def cuda_layer_and_input() -> tuple[SimpleNamespace, torch.Tensor]:
    if not torch.cuda.is_available():
        pytest.skip("BI MLP oracle dispatch tests require CUDA")
    layer = tiny_layer()
    layer.mlp.to(device="cuda", dtype=torch.float16)
    return layer, torch.randn(1, 2, 4, device="cuda", dtype=torch.float16)


def assert_bi_linear_batch_shape_exact(module: nn.Linear, x: torch.Tensor) -> None:
    peer = torch.flip(x, dims=[-1])
    peer3 = torch.cat([peer, -peer, peer * 0.5], dim=0)
    ref = batch_invariant_linear_projection(x, module.weight, module.bias, backend="v2")
    m2 = batch_invariant_linear_projection(torch.cat([x, peer], dim=0), module.weight, module.bias, backend="v2")[0:1]
    reorder = batch_invariant_linear_projection(torch.cat([peer, x], dim=0), module.weight, module.bias, backend="v2")[1:2]
    m4 = batch_invariant_linear_projection(torch.cat([x, peer3], dim=0), module.weight, module.bias, backend="v2")[0:1]
    torch.testing.assert_close(m2, ref, rtol=0, atol=0)
    torch.testing.assert_close(reorder, ref, rtol=0, atol=0)
    torch.testing.assert_close(m4, ref, rtol=0, atol=0)


def test_bi_mlp_oracle_default_off(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_BI_MLP_ORACLE", raising=False)
    assert patternkv_bi_mlp_oracle_enabled(0) is False


def test_bi_mlp_oracle_layer_filter(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    assert patternkv_bi_mlp_oracle_enabled(0) is True
    assert patternkv_bi_mlp_oracle_enabled(1) is False


def test_bi_mlp_oracle_gate_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", "gate")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    patternkv_mlp_oracle_forward(layer, x, 0)
    counters = patternkv_bi_mlp_oracle_counters()
    assert counters["bi_mlp_gate_calls"] == 1
    assert counters["normal_mlp_up_calls"] == 1
    assert counters["normal_mlp_down_calls"] == 1


def test_bi_mlp_oracle_up_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", "up")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    patternkv_mlp_oracle_forward(layer, x, 0)
    assert patternkv_bi_mlp_oracle_counters()["bi_mlp_up_calls"] == 1


def test_bi_mlp_oracle_down_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", "down")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    patternkv_mlp_oracle_forward(layer, x, 0)
    assert patternkv_bi_mlp_oracle_counters()["bi_mlp_down_calls"] == 1


def test_bi_mlp_oracle_does_not_touch_other_layers(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    patternkv_mlp_oracle_forward(layer, x, 1)
    counters = patternkv_bi_mlp_oracle_counters()
    assert counters["bi_mlp_gate_calls"] == 0
    assert counters["normal_mlp_gate_calls"] == 1
    assert counters["normal_mlp_up_calls"] == 1
    assert counters["normal_mlp_down_calls"] == 1


def test_bi_mlp_counters(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_LAYER", "0")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE_COMPONENTS", "gate,up,down")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    patternkv_mlp_oracle_forward(layer, x, 0)
    counters = patternkv_bi_mlp_oracle_counters()
    assert counters["bi_mlp_gate_calls"] == 1
    assert counters["bi_mlp_up_calls"] == 1
    assert counters["bi_mlp_down_calls"] == 1


def test_bi_mlp_oracle_does_not_change_prefill_proj_mode(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_PREFILL_PROJ_MODE", "bi_kv")
    monkeypatch.setenv("PATTERNKV_BI_MLP_ORACLE", "1")
    assert patternkv_prefill_projection_mode() == "bi_kv"


def test_bi_mlp_oracle_flag_off_preserves_path(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_BI_MLP_ORACLE", raising=False)
    monkeypatch.delenv("PATTERNKV_DECODE_BI_MLP", raising=False)
    layer = tiny_layer()
    x = torch.randn(1, 2, 4)
    expected = layer.mlp(x)
    got = patternkv_mlp_oracle_forward(layer, x, 0)
    assert torch.equal(expected, got)


def test_mlp_input_boundary_exact() -> None:
    x = torch.randn(1, 1, 4)
    assert torch.equal(x, x.clone())


def test_mlp_gate_up_batch_shape_oracle(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "gate,up")
    assert patternkv_decode_bi_mlp_components() == {"gate", "up"}


def test_mlp_down_batch_shape_oracle(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "down")
    assert patternkv_decode_bi_mlp_components() == {"down"}


def test_bi_mlp_gate_exact(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "gate")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    assert_bi_linear_batch_shape_exact(layer.mlp.gate_proj, x)
    _ = patternkv_mlp_oracle_forward(layer, x, 0, production_bi_decode=True)
    assert patternkv_bi_mlp_oracle_counters()["bi_mlp_gate_calls"] == 1


def test_bi_mlp_up_exact(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "up")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    assert_bi_linear_batch_shape_exact(layer.mlp.up_proj, x)
    _ = patternkv_mlp_oracle_forward(layer, x, 0, production_bi_decode=True)
    assert patternkv_bi_mlp_oracle_counters()["bi_mlp_up_calls"] == 1


def test_bi_mlp_down_exact(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "down")
    reset_patternkv_bi_mlp_oracle_counters()
    layer, x = cuda_layer_and_input()
    down_x = torch.randn(1, 2, 8, device=x.device, dtype=x.dtype)
    assert_bi_linear_batch_shape_exact(layer.mlp.down_proj, down_x)
    _ = patternkv_mlp_oracle_forward(layer, x, 0, production_bi_decode=True)
    assert patternkv_bi_mlp_oracle_counters()["bi_mlp_down_calls"] == 1


def test_bi_mlp_forward_exact(monkeypatch) -> None:
    monkeypatch.setenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", "gate,up,down")
    layer, x = cuda_layer_and_input()
    peer = torch.randn_like(x)
    ref = patternkv_mlp_oracle_forward(layer, x, 0, production_bi_decode=True)
    got = patternkv_mlp_oracle_forward(layer, torch.cat([x, peer], dim=0), 0, production_bi_decode=True)[0:1]
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_minimal_bi_mlp_set(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_DECODE_BI_MLP_COMPONENTS", raising=False)
    assert patternkv_decode_bi_mlp_components() == {"gate", "up", "down"}


def test_production_mlp_output_exact(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_DECODE_BI_MLP", raising=False)
    assert patternkv_decode_bi_mlp_enabled(("patternkv_segmented_cache_v1",)) is True
    assert patternkv_decode_bi_mlp_enabled(None) is False


def test_attention_contract_preserved_after_mlp_fix(monkeypatch) -> None:
    monkeypatch.delenv("PATTERNKV_BI_MLP_ORACLE", raising=False)
    assert patternkv_bi_mlp_oracle_enabled(0) is False
