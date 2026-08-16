from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from bench.cudagraph_decode import capture_causal_decode_graph_sequence, tree_clone, tree_copy_, tree_tensor_bytes


@dataclass
class _NestedState:
    value: torch.Tensor
    items: list[torch.Tensor]


def test_tree_clone_and_copy_restore_nested_tensor_state() -> None:
    source = {
        "token": torch.tensor([3], dtype=torch.long),
        "nested": _NestedState(torch.ones(2), [torch.arange(3, dtype=torch.float32)]),
    }
    cloned = tree_clone(source)

    source["token"].fill_(9)
    source["nested"].value.fill_(4)
    source["nested"].items[0].fill_(5)

    assert int(cloned["token"].item()) == 3
    assert torch.equal(cloned["nested"].value, torch.ones(2))
    assert torch.equal(cloned["nested"].items[0], torch.arange(3, dtype=torch.float32))

    tree_copy_(source, cloned)
    assert int(source["token"].item()) == 3
    assert torch.equal(source["nested"].value, torch.ones(2))
    assert torch.equal(source["nested"].items[0], torch.arange(3, dtype=torch.float32))
    assert tree_tensor_bytes(source) == 8 + 2 * 4 + 3 * 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay requires CUDA")
def test_cudagraph_sequence_replays_tiny_static_state() -> None:
    device = torch.device("cuda:0")
    state = {"value": torch.zeros(1, device=device)}
    token = torch.tensor([2.0], device=device)

    def decode_fn(step_token: torch.Tensor, cache: dict[str, torch.Tensor]):
        next_value = cache["value"] + step_token
        next_cache = {"value": next_value}
        next_token = step_token + 1.0
        logits = next_value.view(1, 1)
        return next_cache, next_token, logits

    sequence = capture_causal_decode_graph_sequence(decode_fn, token, state, steps=3, device=device)
    final_cache, tokens, logits = sequence.replay(token)
    torch.cuda.synchronize(device)

    assert pytest.approx(float(final_cache["value"].item())) == 9.0
    assert [float(item.item()) for item in tokens] == [3.0, 4.0, 5.0]
    assert [float(item.item()) for item in logits] == [2.0, 5.0, 9.0]
