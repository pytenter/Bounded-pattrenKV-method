# Mixed Precision Batch Problem

Frozen semantics:

- K is INT2.
- Base V is INT2.
- selected top 25% eligible Value tokens are INT4.
- V2 and V4 are independent affine quantization streams.
- V4 is not residual bits, shared bitplanes, or an enhancement over V2.

## Concrete Example

Request A logical precision:

```text
token:      0 1 2 3 4 5
precision: 2 2 4 2 4 2
```

Request B logical precision:

```text
token:      0 1 2 3 4 5
precision: 4 2 2 2 2 4
```

After compact packing:

```text
A V2 logical tokens: [0,1,3,5]  count=4
A V4 logical tokens: [2,4]      count=2

B V2 logical tokens: [1,2,3,4]  count=4
B V4 logical tokens: [0,5]      count=2
```

The counts match in this toy case, but the physical positions do not:

```text
logical token 0: A -> V2 physical rank 0, B -> V4 physical rank 0
logical token 2: A -> V4 physical rank 0, B -> V2 physical rank 1
logical token 4: A -> V4 physical rank 1, B -> V2 physical rank 3
logical token 5: A -> V2 physical rank 3, B -> V4 physical rank 1
```

If the two requests instead had different selected counts, even the dense shapes would diverge:

```text
A V2 count != B V2 count
A V4 count != B V4 count
```

## Why `[B,H,T_v2,...]` Is Not Natural

A dense `V2[B,H,T_v2,...]` requires one shared `T_v2`. In serving, `T_v2` is request-specific:

- request length may differ;
- V4 selected token positions differ;
- V4 selected count may differ because the 25% budget is applied to the eligible tokens of that request/window;
- continuous batching appends one token to each live request but the compact stream chosen by that token differs.

Padding every request to max `T_v2` and max `T_v4` is technically possible but bad as an ABI:

- it hides true request lengths from the kernel;
- it burns capacity in both streams;
- it still needs logical-token-to-physical-rank metadata;
- it adds branches for padded slots;
- it breaks the clean page/block allocation model used by serving engines.

## Required Missing Metadata

For every logical token, the operator must know:

- whether the token is V2 or V4;
- the request-local physical rank inside the selected stream;
- the page/block containing that physical rank;
- the offset inside the page/block;
- the Pattern centroid gate and assignment in the same physical order.

This can be encoded by offsets/counts plus page-local rank metadata, but it cannot be inferred from a single global `precision_mask[0]`.

## Conclusion

The current B=1 representation is a compact single-request encoding, not a batch ABI. A serving-native ABI must make ragged V2/V4 storage explicit via global streams with offsets, page-centric dual streams, or framework-native pools and block tables.
