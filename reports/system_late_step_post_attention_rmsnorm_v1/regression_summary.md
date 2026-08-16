# Regression Summary

Validation:

- `python -m compileall models quant bench scripts tests`: pass
- `python -m pytest tests/test_request_invariant_rmsnorm.py tests/test_first_late_step_persistent_divergence.py tests/test_bi_mlp_oracle.py tests/test_bi_kproj_prefill_runtime.py tests/test_ragged_k_valid_lengths.py tests/test_fused_page_batch_operator.py tests/test_ragged_batch_decode_mvp.py tests/test_final_fixed_batch_semantic_gate.py -q`: 151 passed in 6.00s
- `python -m pytest -q`: 930 passed in 22.55s
- `bench/run_ragged_multistep_correctness.py --device cuda`: B2 pass, B2 reorder pass, B4 fail, independent flush fail
- `git diff --check`: pass
