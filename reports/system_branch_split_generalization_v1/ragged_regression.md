# Ragged Regression

Relevant S6-B.3 / S6-B.3.1 regression suite:

```text
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py tests/test_fused_page_batch_operator.py tests/test_page_batch_mvp.py tests/test_v_centroid_semantic_impact.py
```

Result: `63 passed in 8.01s`.

Known current first Ragged blocker remains:

```text
RAGGED_K_LENGTH_UNSUPPORTED
```

This branch hygiene task does not start the S6-B.3.2 K valid-length implementation.
