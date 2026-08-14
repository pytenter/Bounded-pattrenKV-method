from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from torch import nn

from models.llama_patternkv import (
    patternkv_bi_mlp_oracle_counters,
    patternkv_bi_mlp_oracle_enabled,
    patternkv_mlp_oracle_forward,
    reset_patternkv_bi_mlp_oracle_counters,
)
from quant.batch_invariant_kproj import patternkv_prefill_projection_mode


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
    layer = tiny_layer()
    x = torch.randn(1, 2, 4)
    expected = layer.mlp(x)
    got = patternkv_mlp_oracle_forward(layer, x, 0)
    assert torch.equal(expected, got)
