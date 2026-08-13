# Profile Methodology

- GPU timings use CUDA events with explicit synchronize before and after each measured repetition.
- Wall timings use `perf_counter` around the same call and include Python scheduling and `.item()` stalls.
- Each case uses warmup and repeated measurements; CSV files store median/mean/std/CV.
- Torch profiler is used only on representative B2/T2048 and B4/T4096 cases because profiler overhead is high.
