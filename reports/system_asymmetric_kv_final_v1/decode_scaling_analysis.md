# Decode Scaling Analysis

Comparison of decode128 and decode512 improvements:

| Context | Backend | Decode128 Improve | Decode512 Improve | Direction |
|---:|---|---:|---:|---|
| 4096 | fixed_capacity | 9.55% | 8.54% | decreased |
| 4096 | chunked_capacity | 9.10% | 8.97% | roughly unchanged |
| 8192 | fixed_capacity | 9.35% | 9.04% | decreased |
| 8192 | chunked_capacity | 9.44% | 8.93% | decreased |
| 16384 | fixed_capacity | 9.70% | 8.76% | decreased |
| 16384 | chunked_capacity | 8.03% | 8.48% | increased |
| 32768 | fixed_capacity | 8.59% | 6.27% | decreased |
| 32768 | chunked_capacity | 8.92% | 6.99% | decreased |

Longer decode changes the amount of append and flush work per run; mutation rows show whether the direction tracks cache-copy savings.
