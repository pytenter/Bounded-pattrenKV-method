# Candidate A: V2 Output-Tile Block Consolidation

## Hypothesis

For V2, `PACK=16` and `OC=128`, so there are `OC/PACK=8` packed output-channel
tiles per query head.

The baseline launch uses `threads.y=4` for both V2 and V4:

```text
dim3 threads(32, 4, 1)
blocks.y = ceil((OC/PACK) / 4)
```

For V2 this means `blocks.y=2`. Each query head therefore uses two CUDA blocks
for the eight output tiles.

Candidate A keeps V4 unchanged and uses `threads.y=8` only for `bit==2`, so one
V2 query head covers all eight packed output tiles in one `blockIdx.y` group.

## Expected Benefit

- Fewer V2 blocks along output-channel tiles.
- Potentially less launch/block scheduling overhead inside the V2 kernel.
- No change to quantized values, scale/zero, Pattern centroid semantics, or
  output layout.

## Risk

- Larger block size: 256 threads instead of 128 for V2.
- Occupancy may improve or degrade depending on register pressure and SM
  scheduling.
- Shared memory size is unchanged (`Mcent * sizeof(float)`).

## V4

V4 remains on `threads.y=4`; this candidate must not change V4 behavior.
