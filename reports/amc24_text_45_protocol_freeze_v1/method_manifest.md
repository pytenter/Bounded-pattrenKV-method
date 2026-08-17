# Method Manifest

Machine-readable source: `datasets/amc24_text_45/protocol.json`.

## Frozen Four-Method Matrix

| Method | Runtime identity |
| --- | --- |
| FP16 | full precision KV cache, float16 model dtype |
| KIVI | `kivi_paper_g128`, K/V INT2, group size 128, residual 128 |
| PatternKV | `patternkv_paper`, K/V INT2, group size 128, residual 128, 32 initial patterns |
| CAUSAL-V4@25% | K INT2, historical V base INT2, top 25% eligible historical V INT4, sink 16, recent 128, residual/pending 128, group size 128 |

## CAUSAL Frozen Semantics

Algorithm checkpoint:

```text
c73aeed3247c136859f695d5b238eeb357434b17
```

Selector:

```text
historical causal importance x positive local quantization-error reduction V2 -> V4
```

This task did not modify CAUSAL runtime code or run a serving/system benchmark.
