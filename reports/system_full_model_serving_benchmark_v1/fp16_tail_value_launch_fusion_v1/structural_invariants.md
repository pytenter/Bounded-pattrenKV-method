# Structural Invariants

- `true batch = true`
- `serial request dispatch = 0`
- `serial attention dispatch = 0`
- `historical FP16 K materialization = 0`
- `historical FP16 V materialization = 0`
- `compressed-domain history preserved = true`
- `request invariance preserved = PASS`
- `fixed-split semantics preserved = true`
- `centroid semantics preserved = true`
- `cache layout preserved = true`

The fused tail kernel is batched over `[B,Hq]` and does not introduce a Python per-request production loop.
