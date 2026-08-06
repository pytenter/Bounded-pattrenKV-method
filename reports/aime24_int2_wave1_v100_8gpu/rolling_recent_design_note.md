# Rolling Recent Design Note

## Correct Comparison

`segmented_rolling` is not a legacy-equivalent baseline.

The original failed assumption was:

```text
legacy residual_length=128 == rolling recent_length=128
```

The observed mismatch disproved that assumption:

```text
At generated token 128:
legacy packed=128
rolling segmented packed=0, pending=64, recent=128
```

This is an expected semantic difference between chunked residual buffering and stable rolling-recent protection. It is not sufficient evidence of an implementation bug in the rolling variant.

## Three Modes

```text
legacy_tuple_chunked
segmented_chunked
segmented_rolling
```

`legacy_tuple_chunked` and `segmented_chunked` are the container-equivalence pair.

`segmented_chunked` and `segmented_rolling` define the real CoT cache-cadence diagnostic:

```text
Does stable protection of the latest R tokens outperform periodic residual chunk flushing?
```

## Revised Wave 1A Question

Wave 1A should compare clearly named configs:

```text
pattern_legacy_chunked_k2v2_r128
pattern_rolling_k2v2_s0_r128
pattern_rolling_k2v2_s64_r256
pattern_rolling_k4v2_s0_r128
pattern_rolling_k2v4_s0_r128
kivi_legacy_chunked_k2v2_r128
kivi_rolling_k2v2_s0_r128
kivi_rolling_k2v2_s64_r256
```

Do not use ambiguous names such as `pattern_k2v2_s0_r128` for future diagnostics.
