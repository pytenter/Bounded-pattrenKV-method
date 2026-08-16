# Manual Lifecycle Sequence

Implemented test: `test_request_lifecycle_dynamic_manual_sequence`.

Sequence:

- `[A,B,C,D]`
- release `D`
- `[A,B,C]`
- allocate `E` using `D` slot
- `[A,B,C,E]`
- release `B`
- `[A,C,E]`
- allocate `F` using `B` slot
- `[A,C,E,F]`

The test runs `append_decode_rolling` on active ragged batches and commits active rows back to persistent slots.

