import torch

from insight.oracle_metrics import l2_assignment, minmax_assignment, v_mse_oracle_assignment


def test_toy_l2_assignment():
    tokens = torch.tensor([[0.0, 0.0], [9.0, 9.0]])
    centroids = torch.tensor([[0.0, 1.0], [10.0, 10.0]])
    assert l2_assignment(tokens, centroids).tolist() == [0, 1]


def test_toy_minmax_assignment():
    tokens = torch.tensor([[0.0, 10.0]])
    centroids = torch.tensor([[0.0, 0.0], [0.0, 5.0]])
    assert minmax_assignment(tokens, centroids).tolist() == [1]


def test_v_mse_oracle_returns_valid_candidate():
    tokens = torch.zeros(2, 128)
    centroids = torch.stack([torch.zeros(128), torch.ones(128)])
    got = v_mse_oracle_assignment(tokens, centroids, bits=2, group_size=128)
    assert got.tolist() == [0, 0]
