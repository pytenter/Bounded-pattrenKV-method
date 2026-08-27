# Formal Protocol

Qwen3-8B on Tesla V100 fixed-batch decode-only timing. Primary metric is CAUSAL output tok/s divided by FP16 output tok/s. Each formal point uses warmup=1 and measured repetitions=3. GPU0-3 are read-only and forbidden for formal runs. Peak memory and capacity/OOM sweeps are out of scope.

CAUSAL config: INT2 K, INT2 base V, top 25% eligible historical V as INT4, group_size=128, sink=16, recent=128, residual=128, segmented_rolling cache, base value objective, causal_v4 selector.
