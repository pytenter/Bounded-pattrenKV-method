from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from bench.full_model_serving_benchmark import (
    final_valid_token_indices,
    reset_selective_prefill_trace,
    run_selective_prefill,
    select_final_hidden_rows,
    selective_prefill_trace,
)


class _TinyDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:  # type: ignore[override]
        basis = torch.arange(self.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = input_ids.to(torch.float32).unsqueeze(-1) * 0.25 + basis.view(1, 1, -1)
        return SimpleNamespace(
            __getitem__=lambda self_obj, idx: (hidden, ("cache",), None, None)[idx],
            last_hidden_state=hidden,
            past_key_values=(("cache",),),
            hidden_states=None,
            attentions=None,
        )


class _Output(SimpleNamespace):
    def __getitem__(self, idx: int) -> object:
        return (self.last_hidden_state, self.past_key_values, self.hidden_states, self.attentions)[idx]


class _TinyDecoderWithGetitem(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, input_ids: torch.Tensor, **_: object) -> _Output:  # type: ignore[override]
        basis = torch.arange(self.hidden_size, dtype=torch.float32, device=input_ids.device)
        hidden = input_ids.to(torch.float32).unsqueeze(-1) * 0.25 + basis.view(1, 1, -1)
        return _Output(last_hidden_state=hidden, past_key_values=(("cache",),), hidden_states=None, attentions=None)


class _TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 4, vocab_size: int = 7) -> None:
        super().__init__()
        self.config = SimpleNamespace(pretraining_tp=1, vocab_size=vocab_size)
        self.vocab_size = vocab_size
        self.model = _TinyDecoderWithGetitem(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size) / 10.0)

    def forward(self, input_ids: torch.Tensor, **kwargs: object) -> SimpleNamespace:  # type: ignore[override]
        output = self.model(input_ids=input_ids, **kwargs)
        logits = self.lm_head(output.last_hidden_state).float()
        return SimpleNamespace(logits=logits, past_key_values=output.past_key_values)


def test_b1_last_row_selection() -> None:
    input_ids = torch.tensor([[3, 4, 5]])
    hidden = torch.arange(1 * 3 * 2, dtype=torch.float32).view(1, 3, 2)

    selected = select_final_hidden_rows(hidden, input_ids)

    assert final_valid_token_indices(input_ids).tolist() == [2]
    assert torch.equal(selected, hidden[:, 2, :])


def test_ragged_b2_final_valid_indices() -> None:
    input_ids = torch.tensor([[10, 11, 99, 99], [20, 21, 22, 99]])
    lengths = torch.tensor([2, 3])
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)

    selected = select_final_hidden_rows(hidden, input_ids, valid_lengths=lengths)

    assert final_valid_token_indices(input_ids, valid_lengths=lengths).tolist() == [1, 2]
    assert torch.equal(selected[0], hidden[0, 1])
    assert torch.equal(selected[1], hidden[1, 2])
    assert not torch.equal(selected[0], hidden[0, -1])


def test_ragged_b4_attention_mask_indices() -> None:
    input_ids = torch.tensor(
        [
            [1, 9, 9, 9, 9],
            [2, 3, 9, 9, 9],
            [4, 5, 6, 9, 9],
            [7, 8, 9, 10, 9],
        ]
    )
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0],
        ]
    )

    assert final_valid_token_indices(input_ids, attention_mask=mask).tolist() == [0, 1, 2, 3]


def test_selective_logits_equal_full_logits_at_valid_rows() -> None:
    model = _TinyCausalLM()
    input_ids = torch.tensor([[1, 2, 0, 0], [3, 4, 5, 0], [6, 1, 2, 3], [4, 0, 0, 0]])
    lengths = torch.tensor([2, 3, 4, 1])

    full = model(input_ids=input_ids, use_cache=True, return_dict=True)
    selective = run_selective_prefill(model, input_ids, valid_lengths=lengths)
    row = torch.arange(input_ids.shape[0])
    expected = full.logits[row, lengths - 1, :]

    diff = selective.logits - expected
    assert float(diff.abs().max()) <= 2e-6
    assert torch.equal(selective.logits.argmax(dim=-1), expected.argmax(dim=-1))


def test_lm_head_input_rows_are_pruned_before_projection() -> None:
    model = _TinyCausalLM()
    input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
    lengths = torch.tensor([2, 3])
    reset_selective_prefill_trace()

    output = run_selective_prefill(model, input_ids, valid_lengths=lengths)
    trace = selective_prefill_trace()[-1]

    assert trace["rows_before_lm_head"] == 6
    assert trace["rows_after_lm_head"] == 2
    assert trace["lm_head_input_shape"] == (2, 4)
    assert trace["logits_shape"] == (2, 7)
    assert output.logits.shape == (2, 7)
