# Canonical Identity

`PATTERNKV_PAPER_FULL_MODEL` maps to canonical `patternkv_paper`.

- K/V precision: INT2 / INT2
- `group_size`: 128
- `residual_length`: 128
- `num_k_base`: 32
- `num_v_base`: 32
- Pattern selection: post-RoPE key/value states
- Value precision selector: `base_v2`
- V4 fraction: `0.0`
- CAUSAL selector: disabled

The loader in `bench/full_model_serving_benchmark.py` keeps PatternKV-paper separate from frozen CAUSAL-V4@25%.
