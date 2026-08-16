# Test Results

- `python -m compileall bench scripts tests models quant`: PASS
- Targeted pytest: PASS, `177 passed in 11.58s`
- Full pytest: PASS, `1025 passed in 30.26s`
- `git diff --check`: PASS

Targeted suites covered ragged valid lengths, fused page batch operator, B4 request-count geometry, late-step divergence, dynamic add/remove, continuous batching, request lifecycle, full-model serving benchmark, decode-only protocol repair, post-scaling forensic, selective prefill logits, serving harness, and request-invariant RMSNorm.
