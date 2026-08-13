# Single Request Summary

| Context | Decode | Baseline TPOT | Fixed TPOT | Fixed Improve | Chunked TPOT | Chunked Improve | Best |
|---:|---:|---:|---:|---:|---:|---:|---|
| 4096 | 128 | 109.735 | 99.255 | 9.55% | 99.746 | 9.10% | fixed_capacity |
| 8192 | 128 | 110.588 | 100.248 | 9.35% | 100.149 | 9.44% | chunked_capacity |
| 16384 | 128 | 111.171 | 100.391 | 9.70% | 102.248 | 8.03% | fixed_capacity |
| 32768 | 128 | 115.467 | 105.544 | 8.59% | 105.170 | 8.92% | chunked_capacity |
| 4096 | 512 | 106.410 | 97.320 | 8.54% | 96.870 | 8.97% | chunked_capacity |
| 8192 | 512 | 108.983 | 99.126 | 9.04% | 99.254 | 8.93% | fixed_capacity |
| 16384 | 512 | 110.594 | 100.905 | 8.76% | 101.218 | 8.48% | fixed_capacity |
| 32768 | 512 | 114.849 | 107.647 | 6.27% | 106.826 | 6.99% | chunked_capacity |
