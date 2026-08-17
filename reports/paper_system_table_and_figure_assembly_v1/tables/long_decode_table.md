# Long Decode Table

| Method | Context | Batch | Decode | TPOT ↓ | Throughput ↑ |
|---|---:|---:|---:|---:|---:|
| FP16 | 4096 | 1 | 256 | 28.663 | 34.888 |
| KIVI | 4096 | 1 | 256 | 56.832 | 17.596 |
| PatternKV | 4096 | 1 | 256 | 154.646 | 6.466 |
| CAUSAL-V4@25% | 4096 | 1 | 256 | 159.530 | 6.268 |
