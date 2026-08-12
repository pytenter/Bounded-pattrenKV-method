# Candidate B Design: Private Per-warp Histogram

## Motivation

The current kernel has four warps per CTA, but only `wy == 0` builds the centroid histogram. That warp processes up to 128 tokens per tile: each of its 32 lanes checks four token positions and may issue up to four shared-memory atomics. The other three warps wait at `__syncthreads()` before centroid-table contribution.

Candidate A reduced logical atomics but was slower because warp matching/shuffle overhead was too high. Candidate B instead keeps simple shared-memory atomics but distributes histogram construction across the four existing warps.

## Proposed Layout

Baseline shared memory:

`s_Sacc[Mcent]`

Candidate B shared memory:

`s_Sacc_private[blockDim.y][Mcent]`

With frozen launch `blockDim.y = 4` and S2B-2 synthetic `Mcent = 16`:

- baseline shared memory: `16 * sizeof(float) = 64 bytes`
- candidate shared memory: `4 * 16 * sizeof(float) = 256 bytes`
- extra shared memory/block: `192 bytes`

If `Mcent = 32`, extra shared memory/block is still only `384 bytes`.

## Work Assignment

For each K tile of 128 tokens:

- baseline: `wy == 0` warp handles four token positions per lane
- candidate: every `wy` warp handles one token position per lane:
  `t = tile_base + wy * 32 + lane`

Each warp writes only to its private row, so there is no cross-warp contention on the same shared-memory address during histogram construction. Intra-warp assignment contention remains, but per-row pressure is reduced.

## Final Combination

No separate reduction stage is required. During centroid-table contribution, each output warp reads:

`Sacc[c] = sum_w s_Sacc_private[w][c]`

This adds `blockDim.y` float reads/adds per centroid per output tile. For `blockDim.y=4` and `Mcent=16`, the extra cost is 64 float additions per output tile, much smaller than the baseline serialized histogram path under contention.

## Expected Costs

- initialization work increases from `Mcent` floats to `4 * Mcent` floats
- centroid table stage adds four-row histogram summation
- shared memory usage remains tiny relative to RTX 3090 limits
- residual path math and packed V loads are unchanged

## Scope

Candidate B does not change selector, assignments, mask, centroid values, residual dequantization, cache layout, or V4 fraction. It only changes the benchmark candidate histogram implementation.
