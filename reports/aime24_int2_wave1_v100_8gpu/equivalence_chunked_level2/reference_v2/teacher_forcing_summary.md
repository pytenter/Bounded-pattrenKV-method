# Chunked Level 2 Reference v2

Status: not run as a full model-level backend.

This round added `bench/patternkv_equivalence_reference.py` and unit tests for reference assignment, tie-breaking, block-size equivalence, V gate, reference dequant, attention, and logits metrics. The full teacher-forcing backend that bypasses production PatternKV kernels was not completed.

```text
LEVEL2_REFERENCE_PASS=false
CHUNKED_REFERENCE_ALGORITHM_EQUIVALENT=false
```
