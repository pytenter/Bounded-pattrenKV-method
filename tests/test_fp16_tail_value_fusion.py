import pytest
import torch

from models.segmented_cache import request_invariant_full_value_attention
from quant.matmul import (
    fp16_tail_value_forward_cuda,
    fp16_tail_value_fusion_enabled,
    get_fp16_tail_value_counters,
    record_fp16_tail_value_old_call,
    record_tail_output_add_call,
    reset_fp16_tail_value_counters,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA FP16 tail Value fusion tests require a GPU")

NH = 32
NH_KV = 8
GROUPS = NH // NH_KV
HEAD_DIM = 128


def _empty(batch: int, device: torch.device) -> torch.Tensor:
    return torch.empty((batch, NH_KV, 0, HEAD_DIM), dtype=torch.float16, device=device)


def _reference(
    probs: torch.Tensor,
    values: dict[str, torch.Tensor],
    lengths: dict[str, torch.Tensor],
    offsets: dict[str, int],
) -> torch.Tensor:
    out = None
    for name in ("sink", "pending", "recent"):
        value = values[name]
        physical = int(value.shape[2])
        if physical == 0:
            continue
        weights = probs[:, :, :, offsets[name] : offsets[name] + physical]
        part = request_invariant_full_value_attention(weights, value, lengths[name], GROUPS)
        out = part if out is None else out + part
    if out is None:
        return torch.zeros((probs.shape[0], NH, 1, HEAD_DIM), dtype=torch.float16, device=probs.device)
    return out


def _run_case(batch: int, physical: dict[str, int], valid: dict[str, list[int]], *, seed: int = 123) -> tuple[torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    torch.manual_seed(seed + batch + sum(physical.values()))
    total = sum(physical.values())
    probs = torch.softmax(torch.randn(batch, NH, 1, total, device=device, dtype=torch.float16), dim=-1).contiguous()
    values = {
        name: (torch.randn(batch, NH_KV, length, HEAD_DIM, device=device, dtype=torch.float16) * 0.25).contiguous()
        if length
        else _empty(batch, device)
        for name, length in physical.items()
    }
    lengths = {name: torch.tensor(valid[name], dtype=torch.long, device=device) for name in physical}
    offsets = {}
    cursor = 0
    for name in ("sink", "pending", "recent"):
        offsets[name] = cursor
        cursor += physical[name]
    got = fp16_tail_value_forward_cuda(
        probs,
        values["sink"],
        values["pending"],
        values["recent"],
        lengths["sink"],
        lengths["pending"],
        lengths["recent"],
        sink_offset=offsets["sink"],
        pending_offset=offsets["pending"],
        recent_offset=offsets["recent"],
        num_key_value_groups=GROUPS,
    )
    ref = _reference(probs, values, lengths, offsets)
    torch.cuda.synchronize()
    return got, ref


@pytest.mark.parametrize(
    "batch,physical,valid",
    [
        (1, {"sink": 16, "pending": 0, "recent": 0}, {"sink": [16], "pending": [0], "recent": [0]}),
        (1, {"sink": 0, "pending": 7, "recent": 0}, {"sink": [0], "pending": [7], "recent": [0]}),
        (1, {"sink": 0, "pending": 0, "recent": 9}, {"sink": [0], "pending": [0], "recent": [9]}),
        (1, {"sink": 16, "pending": 5, "recent": 11}, {"sink": [16], "pending": [5], "recent": [11]}),
        (2, {"sink": 16, "pending": 9, "recent": 13}, {"sink": [16, 7], "pending": [9, 3], "recent": [13, 8]}),
        (4, {"sink": 16, "pending": 17, "recent": 19}, {"sink": [16, 0, 5, 12], "pending": [17, 1, 0, 8], "recent": [19, 7, 2, 0]}),
    ],
)
def test_fp16_tail_value_fused_matches_old_path(batch, physical, valid) -> None:
    got, ref = _run_case(batch, physical, valid)
    torch.testing.assert_close(got, ref, rtol=2e-3, atol=2e-3)


def test_fp16_tail_value_gqa_mapping_is_query_head_floor_division() -> None:
    device = torch.device("cuda")
    batch = 1
    physical = {"sink": 0, "pending": 1, "recent": 0}
    probs = torch.ones((batch, NH, 1, 1), dtype=torch.float16, device=device).contiguous()
    pending = torch.zeros((batch, NH_KV, 1, HEAD_DIM), dtype=torch.float16, device=device)
    for kv_head in range(NH_KV):
        pending[:, kv_head, :, :].fill_(float(kv_head + 1))
    out = fp16_tail_value_forward_cuda(
        probs,
        _empty(batch, device),
        pending.contiguous(),
        _empty(batch, device),
        torch.tensor([0], device=device),
        torch.tensor([1], device=device),
        torch.tensor([0], device=device),
        sink_offset=0,
        pending_offset=0,
        recent_offset=1,
        num_key_value_groups=GROUPS,
    )
    expected = torch.empty_like(out)
    for query_head in range(NH):
        expected[:, query_head, :, :].fill_(float(query_head // GROUPS + 1))
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_fp16_tail_value_extreme_probability_distribution() -> None:
    device = torch.device("cuda")
    batch = 2
    probs = torch.zeros((batch, NH, 1, 6), dtype=torch.float16, device=device)
    probs[:, :, :, 2] = 1.0
    pending = torch.randn(batch, NH_KV, 3, HEAD_DIM, dtype=torch.float16, device=device).contiguous()
    recent = torch.randn(batch, NH_KV, 3, HEAD_DIM, dtype=torch.float16, device=device).contiguous()
    got = fp16_tail_value_forward_cuda(
        probs.contiguous(),
        _empty(batch, device),
        pending,
        recent,
        torch.zeros((batch,), dtype=torch.long, device=device),
        torch.full((batch,), 3, dtype=torch.long, device=device),
        torch.full((batch,), 3, dtype=torch.long, device=device),
        sink_offset=0,
        pending_offset=0,
        recent_offset=3,
        num_key_value_groups=GROUPS,
    )
    expected = pending.repeat_interleave(GROUPS, dim=1)[:, :, 2:3, :]
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_fp16_tail_value_fallback_switch_and_counters(monkeypatch) -> None:
    reset_fp16_tail_value_counters()
    monkeypatch.setenv("PATTERNKV_FP16_TAIL_VALUE_FUSION", "0")
    assert not fp16_tail_value_fusion_enabled()
    record_fp16_tail_value_old_call("sink")
    record_fp16_tail_value_old_call("pending")
    record_fp16_tail_value_old_call("recent")
    record_tail_output_add_call()
    counters = get_fp16_tail_value_counters()
    assert counters["fp16_tail_value_old_calls"] == 3
    assert counters["fp16_tail_value_fused_calls"] == 0
    assert counters["sink_value_calls"] == 1
    assert counters["pending_value_calls"] == 1
    assert counters["recent_value_calls"] == 1
    assert counters["tail_output_add_calls"] == 1
    monkeypatch.setenv("PATTERNKV_FP16_TAIL_VALUE_FUSION", "1")
    assert fp16_tail_value_fusion_enabled()
