# Regression Summary

Validation:

- `python -m compileall models quant bench scripts tests`: pass
- `python -m pytest tests/test_bi_mlp_oracle.py tests/test_bi_kproj_prefill_runtime.py tests/test_ragged_k_valid_lengths.py tests/test_fused_page_batch_operator.py tests/test_ragged_batch_decode_mvp.py tests/test_final_fixed_batch_semantic_gate.py -q`: 130 passed in 6.58s
- `python -m pytest -q`: 909 passed in 22.52s
- `git diff --check`: pass
