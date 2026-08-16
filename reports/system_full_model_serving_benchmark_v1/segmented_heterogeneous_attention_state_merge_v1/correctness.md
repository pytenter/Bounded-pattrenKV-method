# Correctness

## Tests

- `tests/test_segmented_attention_state_merge.py`: 6 passed.
- `tests/test_ragged_k_valid_lengths.py`: 45 passed.
- `tests/test_request_lifecycle_manager.py`: 22 passed.
- `tests/test_dynamic_add_remove_batching.py`: 10 passed.
- `tests/test_iteration_level_continuous_batching.py`: 15 passed.
- `tests/test_selective_value_precision.py`: 24 passed.
- Full pytest: 1031 passed.

## Coverage

- Single segment identity: PASS.
- Multi-segment exactness: PASS.
- Four-segment exactness: PASS.
- Empty physical segment: PASS.
- Extreme logits: PASS.
- Ragged valid lengths: PASS.
- B1/B2/B4 synthetic state equivalence: PASS.
- Batch reorder invariance: PASS.

Full-model old/new token-equivalence metrics were not separately emitted by the benchmark worker. The state math is covered by FP32 oracle tests, and the real-model decode workers completed with `run_valid=true`, no protocol contamination, no fallback, and no historical FP16 K/V materialization.

