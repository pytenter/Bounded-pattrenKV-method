# Temporary Memory Before After

## Before

`temporary_tail_outputs_before = up to 3`

The old path materialized one output tensor for each non-empty FP16 tail segment and then added those parts into the running attention output.

## After

`temporary_tail_outputs_after = 1`

The fused path returns one `[B,Hq,1,D]` tail output tensor. It does not materialize three segment outputs and does not concatenate probabilities or Values.

## Peak Memory

Formal old/fused peak allocated and reserved memory are `BLOCKED` because GPU1 formal measurement is unavailable.

The fused operator introduces no persistent workspace and no large temporary buffer.
