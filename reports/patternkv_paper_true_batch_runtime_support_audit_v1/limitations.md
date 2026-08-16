# Limitations

This task validates correctness/runtime support only. It does not run the formal paper baseline performance matrix.

The B2 semantic oracle compares top-1 generation and logits against independent B1 references. Logits are numerically close enough for top-1 agreement but not bit-identical, which is expected for batched transformer execution and existing batch-invariance limits.
