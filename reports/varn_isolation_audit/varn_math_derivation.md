# VarN Math Derivation

For a tile `X in R^{R x C}`, canonical VarN initializes `log_s_col=0` and
`log_s_row=0`, then repeats:

```text
col_std = std(X / exp(log_s_col) / exp(log_s_row), dim=rows)
log_s_col = clip(log_s_col + log(clamp(col_std, 1e-3, 1e3)), -0.3, 10.0)

row_std = std(X / exp(log_s_col) / exp(log_s_row), dim=cols)
log_s_row = clip(log_s_row + log(clamp(row_std, 1e-3, 1e3)), -0.3, 10.0)
```

At each iteration it computes imbalance:

```text
max(col_std)/min(col_std) + max(row_std)/min(row_std)
```

and returns the best-so-far state:

```text
balanced = X / s_col / s_row
restore(X) = balanced * s_col * s_row
```

For K, canonical orientation is `[D, group]`, so `s_row` is per-channel and
`s_col` is per-token. For V, canonical orientation is `[group, D]`, so `s_row`
is per-token and `s_col` is per-channel.
