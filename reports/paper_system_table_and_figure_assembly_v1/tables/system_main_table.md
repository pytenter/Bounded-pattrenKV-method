# Main System Table

| Method | Effective KV bits | C2048 B1 TPOT ↓ | C2048 B4 Throughput ↑ | C4096 B4 Peak Memory ↓ | C4096 Max Batch ↑ | Capacity vs FP16 | D256 TPOT ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 16-bit KV | 30.604 | 126.747 | 19.00 | 4 | 1.0x | 28.663 |
| KIVI | 2.25-bit quantized-region effective storage | 68.967 | 64.171 | 17.14 | **8** | 2.0x | 56.832 |
| PatternKV | 2.25-bit quantized-region effective storage | 162.753 | 25.168 | 17.86 | **8** | 2.0x | 154.646 |
| CAUSAL-V4@25% | ~2.50 effective KV bits | 165.168 | 24.138 | 17.94 | **8** | 2.0x | 159.530 |

Peak memory is full-model full-lifecycle peak allocated memory in GiB under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
