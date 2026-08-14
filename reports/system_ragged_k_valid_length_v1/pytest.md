# Pytest

Validation:

- `python -m compileall bench models quant scripts tests`: PASS
- `pytest -q tests/test_ragged_k_valid_lengths.py`: `11 passed`
- `pytest -q tests/test_ragged_k_valid_lengths.py tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py tests/test_fused_page_batch_operator.py tests/test_page_batch_mvp.py tests/test_v_centroid_semantic_impact.py tests/test_fused_decode_runtime_integration.py tests/test_request_local_centroid_state.py`: `96 passed`
- `pytest -q`: `869 passed in 23.22s`
- `git diff --check`: PASS after report whitespace cleanup.
