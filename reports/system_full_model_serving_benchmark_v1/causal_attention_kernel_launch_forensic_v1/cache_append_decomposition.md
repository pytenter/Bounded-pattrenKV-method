# Cache Append Decomposition

- CUDA kernel calls/token: 512.000
- GPU kernel time/token: 1.393 ms
- Approx orchestration/wrapper time/token: 17.867 ms
- Classification: orchestration-dominated
- `page_batch_pack` calls in the decode window are read from PatternKV counters in `causal_v4_25_full_model_profile.json`.
