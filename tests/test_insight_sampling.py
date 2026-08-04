import torch

from insight.sampling import sample_indices


def test_sample_indices_are_deterministic_and_cover_anchors():
    metadata = {"dataset": "longbench", "task": "hotpotqa", "sample_id": "x"}
    a = sample_indices(100, 8, metadata, 7, 0, "prefill", None, 123)
    b = sample_indices(100, 8, metadata, 7, 0, "prefill", None, 123)
    assert a == b
    assert 0 in a
    assert 50 in a
    assert 99 in a
    assert len(a) == len(set(a)) == 8


def test_sample_indices_do_not_consume_global_torch_rng():
    metadata = {"dataset": "gsm8k", "task": "gsm8k", "sample_id": "0"}
    torch.manual_seed(17)
    before = torch.random.get_rng_state()
    sample_indices(1000, 8, metadata, 0, 0, "decode", 3, 9)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
