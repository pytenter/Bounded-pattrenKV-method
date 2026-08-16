# Exact Commands

## Preflight

```bash
git branch --show-current
git rev-parse HEAD
git status --short
git diff --stat
git diff --check
nvidia-smi
```

## Formal profile

```bash
CUDA_VISIBLE_DEVICES=5 \
PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
PATTERNKV_ACTIVE_BATCH_CACHE=1 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python <inline profiling script>
```

The inline script ran `BenchmarkConfig('CAUSAL_V4_25_FULL_MODEL', 2048, 4, B, B)` for B=1 and B=4, with profiling OFF for headline timing and profiling ON for component attribution. The initial GPU-1 attempt was marked invalid after another compute process appeared; the formal matched run used physical GPU 5.

## Regression

```bash
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m compileall models/llama_patternkv.py bench/full_model_serving_benchmark.py quant/page_batch.py tests/test_fused_page_batch_operator.py

PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q \
  tests/test_ragged_k_valid_lengths.py \
  tests/test_final_fixed_batch_semantic_gate.py \
  tests/test_request_lifecycle_manager.py \
  tests/test_dynamic_add_remove_batching.py \
  tests/test_iteration_level_continuous_batching.py \
  tests/test_full_model_serving_benchmark.py \
  tests/test_fused_page_batch_operator.py::test_operator_ready_page_pools_preserve_page_layout

PATTERNKV_FIXED_SPLIT_SOFTMAX=1 \
/data/zypan/.local/share/mamba/envs/patternkv/bin/python -m pytest -q

git diff --check
```
