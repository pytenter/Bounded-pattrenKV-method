# CUDA Kernel Design

## Mapping

`grid = (B * Hq)` and `block = max(32, D)` threads. For production `D = 128`, one block covers one `(batch row, query head)` and one thread covers one head dimension.

The kernel computes:

`acc[d] = sum(P_sink * V_sink[d]) + sum(P_pending * V_pending[d]) + sum(P_recent * V_recent[d])`

Then it writes one output element for each dimension. The output tensor is written once.

## GQA

`kv_head = query_head // num_key_value_groups`.

For production geometry this is `Hq = 32`, `Hkv = 8`, `num_key_value_groups = 4`, `head_dim = 128`.

## Reduction

Each thread reduces the token dimension for one output dimension. Accumulation uses `float`. Inputs and output are `half`.

`ACCUMULATOR_DTYPE = float32`

`INPUT_DTYPE = float16`

`OUTPUT_DTYPE = float16`

## Empty And Ragged Segments

Physical segment length can be zero. Per-row valid length is clamped to `[0, physical_length]`, so ragged requests skip invalid padded tail tokens without invalid memory access.

## Shared Memory

No shared memory is used.

## Launch Fragmentation Target

The fused path replaces up to three per-segment Value dispatches plus two tail-output adds with one CUDA Value kernel plus at most one add against the historical output.
