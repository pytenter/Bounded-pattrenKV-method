from __future__ import annotations

import torch

from bench.run_actual_model_fixed_batch_smoke import make_fixed_inputs, smoke_cases


class DummyTokenizer:
    bos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [(ord(ch) % 97) + 2 for ch in text[:11]]


def test_smoke_case_matrix_contains_required_boundaries() -> None:
    cases = smoke_cases("boundary")
    by_name = {case.name: case for case in cases}
    assert by_name["b2_ctx2048_d127"].batch == 2
    assert by_name["b2_ctx2048_d128"].decode == 128
    assert by_name["b2_ctx2048_d129"].context == 2048
    assert by_name["b4_ctx2048_d128"].batch == 4


def test_fixed_inputs_are_equal_length_but_distinct() -> None:
    inputs = make_fixed_inputs(DummyTokenizer(), batch=4, context=32, device=torch.device("cpu"))
    assert inputs.shape == (4, 32)
    assert torch.all(inputs[:, 0] == 1)
    assert len({tuple(row.tolist()) for row in inputs}) == 4
