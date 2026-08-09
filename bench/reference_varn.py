from __future__ import annotations

import torch

DEFAULT_ITERATIONS = 16
CLIP_STD_MIN = 1e-3
CLIP_STD_MAX = 1e3
LOG_S_MIN = -0.3
LOG_S_MAX = 10.0
EPS = 1e-8


def varn_imbalance(tile: torch.Tensor) -> torch.Tensor:
    sc = tile.std(dim=-2)
    sr = tile.std(dim=-1)
    return sc.amax(dim=-1) / sc.amin(dim=-1).clamp_min(EPS) + sr.amax(dim=-1) / sr.amin(dim=-1).clamp_min(EPS)


def variance_normalize_reference(tile: torch.Tensor, iterations: int = DEFAULT_ITERATIONS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference for KVarN's log-domain variance normalization.

    For a tile ``X`` with shape ``[R, C]``, returns ``balanced, s_col, s_row``
    such that ``balanced = X / s_col / s_row``. The loop alternates column and
    row standard-deviation updates in log-scale space and returns the best
    imbalance state seen.
    """
    m = tile.float()
    rows, cols = m.shape
    log_s_col = torch.zeros(1, cols, device=m.device)
    log_s_row = torch.zeros(rows, 1, device=m.device)

    cur = m / log_s_col.exp() / log_s_row.exp()
    best_imbalance = varn_imbalance(cur)
    best_s_col = log_s_col.exp().clone()
    best_s_row = log_s_row.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=0, keepdim=True).clamp(CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_col = (log_s_col + col_std.log()).clip(LOG_S_MIN, LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        row_std = cur.std(dim=1, keepdim=True).clamp(CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_row = (log_s_row + row_std.log()).clip(LOG_S_MIN, LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        imbalance = varn_imbalance(cur)
        if imbalance <= best_imbalance:
            best_imbalance = imbalance
            best_s_col = log_s_col.exp().clone()
            best_s_row = log_s_row.exp().clone()

    return m / best_s_col / best_s_row, best_s_col, best_s_row


def variance_normalize_batched_reference(tiles: torch.Tensor, iterations: int = DEFAULT_ITERATIONS) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = tiles.float()
    n_tiles, rows, cols = m.shape
    log_s_col = torch.zeros(n_tiles, 1, cols, device=m.device)
    log_s_row = torch.zeros(n_tiles, rows, 1, device=m.device)

    cur = m / log_s_col.exp() / log_s_row.exp()
    best_imbalance = varn_imbalance(cur)
    best_s_col = log_s_col.exp().clone()
    best_s_row = log_s_row.exp().clone()

    for _ in range(iterations):
        col_std = cur.std(dim=1, keepdim=True).clamp(CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_col = (log_s_col + col_std.log()).clip(LOG_S_MIN, LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        row_std = cur.std(dim=2, keepdim=True).clamp(CLIP_STD_MIN, CLIP_STD_MAX)
        log_s_row = (log_s_row + row_std.log()).clip(LOG_S_MIN, LOG_S_MAX)
        cur = m / log_s_col.exp() / log_s_row.exp()

        imbalance = varn_imbalance(cur)
        better = imbalance <= best_imbalance
        if better.any():
            mask = better.view(n_tiles, 1, 1).to(log_s_col.dtype)
            best_s_col = mask * log_s_col.exp() + (1 - mask) * best_s_col
            best_s_row = mask * log_s_row.exp() + (1 - mask) * best_s_row
            best_imbalance = torch.where(better, imbalance, best_imbalance)

    return m / best_s_col / best_s_row, best_s_col, best_s_row


def restore_varn_tile(balanced: torch.Tensor, s_col: torch.Tensor, s_row: torch.Tensor) -> torch.Tensor:
    return balanced.float() * s_col.float() * s_row.float()
