# Context Scaling Analysis

Profile-off improvements by context:

| Decode | Backend | 4K | 8K | 16K | 32K |
|---:|---|---:|---:|---:|---:|
| 128 | fixed_capacity | 9.55% | 9.35% | 9.70% | 8.59% |
<!-- fixed_capacity decode128: gain does not monotonically increase from 4K to 32K. -->
| 128 | chunked_capacity | 9.10% | 9.44% | 8.03% | 8.92% |
<!-- chunked_capacity decode128: gain does not monotonically increase from 4K to 32K. -->
| 512 | fixed_capacity | 8.54% | 9.04% | 8.76% | 6.27% |
<!-- fixed_capacity decode512: gain does not monotonically increase from 4K to 32K. -->
| 512 | chunked_capacity | 8.97% | 8.93% | 8.48% | 6.99% |
<!-- chunked_capacity decode512: gain does not monotonically increase from 4K to 32K. -->
