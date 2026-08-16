# Production Fix

Production changes:

- `models/segmented_cache.py`: added `_k_attention_value_parts`, `_has_ragged_attention_layout`, and `_scatter_k_attention_mass_to_logical_indices`; `update_value_causal_importance` now uses vectorized GPU `scatter_add_` for ragged layouts.
- `tests/test_ragged_k_valid_lengths.py`: added mapping-level tests for B1/B2 short, B1/B2 long, reorder, B4, and sink/packed/pending/recent segment coverage.

No per-request forward/update dispatch was introduced.
