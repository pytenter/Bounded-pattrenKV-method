# Failure Reproduction

The previous paper-baseline smoke gate failed for PatternKV-paper at `context=512`, `decode=4`, `B=2`.

Observed failure:

```text
AssertionError('v_centroids shape wrong: torch.Size([2, 8, 48, 128])')
```

B1 passed because the centroid bank was representable as `[Hkv,C,D]`. B2 legitimately produced request-local centroids `[B,Hkv,C,D]`, exposing that the non-mixed Value reader was still routed through a shared-centroid kernel wrapper.
