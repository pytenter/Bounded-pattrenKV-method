# System Content Equivalence

After reverting `3dcedb42674140296c47cd56cf6ccbc1017474bc`, production/code paths were compared against `3a9fa066b08977e6769f04a035cb4f35778a1bfb`.

Passed:

```text
git diff --exit-code 3a9fa066b08977e6769f04a035cb4f35778a1bfb HEAD -- models quant bench scripts tests
git diff 3a9fa066b08977e6769f04a035cb4f35778a1bfb HEAD -- models/llama_patternkv.py
```

`reports/causal_v4_25_generalization_v1/` is absent from the repaired system content and remains preserved on `exp/causal-v4-25-generalization-v1`.

Ragged S6-B.3.1 code is preserved, including request-local length vectors, ragged cache assembly, centroid pool merge, page pool merge, and per-request decode position IDs.
