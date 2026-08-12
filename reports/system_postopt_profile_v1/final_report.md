# Final Report

## Production TPOT

- 8K: `111.18139210939407` ms/token
- 16K: `114.11241441965103` ms/token
- 32K: `117.3176370024681` ms/token

## Profiling Overhead

- 8K: `0.36690672497569277`
- 16K: `0.37054907495206457`
- 32K: `0.34934255524353`

## Top 5 Components @32K

1. `other`: `50731.268` us/token, `32.05%`
2. `mixed_v`: `41119.912` us/token, `25.98%`
3. `cache_mutation`: `24893.034` us/token, `15.73%`
4. `QK`: `18886.766` us/token, `11.93%`
5. `rope`: `8384.269` us/token, `5.30%`

- 32K largest bottleneck: `other`
- 16K consistency: `other`
- Fastest-growing component: `mixed_v`
- Selector still small: `True`
- Cache mutation still major: `True`
- Mixed-V still major: `False`
- QK new bottleneck: `False`
- PatternKV-specific diminishing returns: `False`
- Recommended next task: `CONTIGUOUS_CAPACITY_CACHE_DESIGN`

## Correctness Smoke

- Passed: `True`
- Contexts: `2K, 8K`
