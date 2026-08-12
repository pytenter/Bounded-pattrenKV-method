# Amdahl Headroom

Same-run baseline at 32K:

- V2: `509.952 us`
- Mixed-V: `999.424 us`
- V2 fraction of mixed-V: `51.02%`

If V2 became infinitely fast, the mixed-V upper-bound speedup would be about `2.042x`. For realistic V2 speedups:

| V2 speedup | Estimated mixed-V speedup |
|---:|---:|
| 1.05x | 1.025x |
| 1.10x | 1.049x |
| 1.20x | 1.093x |

Measured candidate A did not realize a V2 speedup. It regressed to `0.417x` baseline/candidate, so E2E was skipped.
