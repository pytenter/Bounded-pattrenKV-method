# S128 Semantics Probe

## Minimal reproduction

- First failing task: `aime24:p6:s0:seed6042`
- Prompt tokens: `117`
- Configured sink length: `128`
- Recent length: `128`
- PatternKV original error: `ValueError('sink token count mismatch: 117 != 118')`
- KIVI original error: `ValueError('sink token count mismatch: 117 != 118')`

## Observed state transition

At prefill, the cache can only place the existing prompt tokens in Sink, so `sink=117`, `recent=0`, `packed=0`, and `pending=0`. The previous rolling decode append always appended new decode tokens to Recent. At the first decode step, actual state became `sink=117`, `recent=1`, while `segment_lengths(total_tokens=118, sink_length=128, recent_length=128)` expected `sink=118`, `recent=0`.

## First 16 decode steps

| step | total | old actual sink | old actual recent | expected sink | expected recent | fixed sink | fixed recent |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 118 | 117 | 1 | 118 | 0 | 118 | 0 |
| 2 | 119 | 117 | 2 | 119 | 0 | 119 | 0 |
| 3 | 120 | 117 | 3 | 120 | 0 | 120 | 0 |
| 4 | 121 | 117 | 4 | 121 | 0 | 121 | 0 |
| 5 | 122 | 117 | 5 | 122 | 0 | 122 | 0 |
| 6 | 123 | 117 | 6 | 123 | 0 | 123 | 0 |
| 7 | 124 | 117 | 7 | 124 | 0 | 124 | 0 |
| 8 | 125 | 117 | 8 | 125 | 0 | 125 | 0 |
| 9 | 126 | 117 | 9 | 126 | 0 | 126 | 0 |
| 10 | 127 | 117 | 10 | 127 | 0 | 127 | 0 |
| 11 | 128 | 117 | 11 | 128 | 0 | 128 | 0 |
| 12 | 129 | 117 | 12 | 128 | 1 | 128 | 1 |
| 13 | 130 | 117 | 13 | 128 | 2 | 128 | 2 |
| 14 | 131 | 117 | 14 | 128 | 3 | 128 | 3 |
| 15 | 132 | 117 | 15 | 128 | 4 | 128 | 4 |
| 16 | 133 | 117 | 16 | 128 | 5 | 128 | 5 |

## Root cause

`S128_ROOT_CAUSE=sink_semantics_inconsistent_between_initialization_append_and_validator`. Initialization used a prefill-limited Sink, decode append never grew Sink, and validator used the absolute-prefix mathematical definition from `segment_lengths()`.

## Canonical semantics

`S128_CANONICAL_SEMANTICS=absolute_sequence_prefix`. Existing `segment_lengths()` and validator define `sink_length=N` as the first N logical sequence tokens. When prompt length is less than N, early decode tokens fill the remaining Sink capacity before later decode tokens enter Recent.
