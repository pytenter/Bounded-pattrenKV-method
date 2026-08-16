# Correctness

## Synthetic Reference

`tests/test_fp16_tail_value_fusion.py` compares fused output against the old validated construction:

`old_tail = sink_output + pending_output + recent_output`

Covered cases:

- B1 sink-only, pending-only, recent-only, and all-three.
- B2 ragged unequal valid lengths.
- B4 ragged unequal valid lengths including empty valid rows.
- Explicit GQA mapping sentinel test.
- Extreme one-hot probability distribution.
- Old-path fallback switch and counters.

## Results

Focused CUDA fusion tests: `9 passed`.

Targeted semantic tests including ragged, dynamic lifecycle, request lifecycle, and continuous batching: `101 passed`.

Full pytest after the final kernel rebuild: `1036 passed`.

Direct non-formal GPU2 tail equivalence probe:

- `tail_max_abs = 6.103515625e-05`
- `tail_rel_l2 = 0.00044492175220511854`

Full-model logit equivalence was not rerun because formal GPU1 measurement is blocked by external compute contamination.
