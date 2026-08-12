# S2B-2 Decision Scorecard

| Factor | Finding |
|---|---|
| V2 centroid fraction @16K | 54.1% (120.832 us) |
| V2 centroid fraction @32K | 55.9% (246.784 us) |
| V4 centroid fraction @16K | 9.5% (9.216 us) |
| V4 centroid fraction @32K | 39.9% (64.512 us) |
| mask-density sensitivity | 0% to 100% V2 FULL increases 160.768 us -> 336.896 us, +109.6% |
| skewed assignment sensitivity | RANDOM_UNIFORM 335.872 us vs SKEWED 598.016 us, 1.780x slowdown |
| atomic contention evidence | YES: concentrated assignments produce large slowdown and lower entropy |
| GQA reuse opportunity | YES for centroid table/mask/idx/compressed V metadata; alpha remains query-head specific |
| local kernel tuning status | SATURATED |

`RECOMMENDED_NEXT_PHASE=CENTROID_PATH_OPTIMIZATION`

## Rationale

The 32K V2 centroid overhead estimate is 55.9%, far above the 15% decision guide. Mask-density scaling and skewed-assignment slowdown both point to the histogram/atomic/centroid-table path as concrete optimization headroom. GQA reuse is real, but centroid cost is large enough to optimize the centroid path first.
