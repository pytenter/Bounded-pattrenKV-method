# Validation Summary

- `python -m compileall models/segmented_cache.py models/llama_patternkv.py bench/ragged_batch_decode_utils.py bench/run_ragged_cache_assembly.py tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py`: passed.
- `pytest -q tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py`: 30 passed.
- `pytest -q tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py tests/test_fused_page_batch_operator.py tests/test_page_batch_mvp.py tests/test_v_centroid_semantic_impact.py`: 63 passed.
- `pytest -q tests/test_ragged_cache_assembly.py tests/test_ragged_batch_decode_mvp.py tests/test_fused_decode_runtime_integration.py tests/test_request_local_centroid_state.py`: 52 passed.
- `CUDA_VISIBLE_DEVICES=1 python bench/run_ragged_cache_assembly.py --actual --device cuda:0`: actual model loaded; B2/B4 assembly and position semantics passed; smallest B2 ragged decode probe failed at `RuntimeError: assignments must be [B, nh_kv, OC]`, classified as `RAGGED_K_LENGTH_UNSUPPORTED`.
- `git diff --check`: passed.
- Full `pytest -q`: 855 passed, 5 failed. The remaining failures are the known pre-existing GSM8K/LongBench method-count expectations from the dirty paper/LongBench worktree changes documented in `preexisting_worktree_state.txt`.
