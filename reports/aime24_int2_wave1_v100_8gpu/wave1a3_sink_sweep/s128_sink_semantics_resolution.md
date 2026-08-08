# S128 Sink Semantics Resolution

## 1. Original Blocker

Wave 1A.3 marked S128 runtime-invalid for both PatternKV and KIVI because most tasks failed with `ValueError('sink token count mismatch: 117 != 118')` or the same one-token mismatch at shorter prompt lengths.

## 2. Minimal Reproduction

- First failing task: `aime24:p6:s0:seed6042`
- Prompt length: `117`
- Configured Sink length: `128`
- Prefill Sink tokens: `117`
- First decode actual Sink tokens before fix: `117`
- First decode expected Sink tokens: `118`
- First decode actual Recent tokens before fix: `1`

See `reports/aime24_int2_wave1_v100_8gpu/s128_semantics_probe/pattern_probe.json` and `reports/aime24_int2_wave1_v100_8gpu/s128_semantics_probe/kivi_probe.json`.

## 3. Current Initialization Semantics

`build_cache_from_prefill()` initialized Sink as `min(prefill_total_tokens, sink_length)`. This is correct for prefill because decode tokens do not exist yet.

## 4. Current Decode Append Semantics

Before this fix, `append_decode_rolling()` appended all decode tokens to Recent and never filled remaining Sink capacity.

## 5. Current Validator Semantics

`segment_lengths(total_tokens, sink_length, recent_length)` and `validate_cache()` used `expected_sink=min(total_tokens, sink_length)`, which defines Sink as an absolute logical sequence prefix.

## 6. Root Cause

`S128_ROOT_CAUSE=sink_semantics_inconsistent_between_initialization_append_and_validator`.

## 7. Candidate Semantic A

`absolute_sequence_prefix`: `sink_length=N` protects the first N logical sequence tokens. If prompt length is 117 and Sink is 128, the first 11 decode tokens fill Sink before later decode tokens enter Recent.

## 8. Candidate Semantic B

`prefill_only`: Sink freezes at `min(prefill_tokens, sink_length)` and decode tokens always enter Recent.

## 9. Evidence for Canonical Choice

Existing `segment_lengths()` and validator already encode absolute-prefix semantics. Existing reports describe early-token protection by logical position, and there was no stronger prefill-only contract in tests or reports.

## 10. Implemented Resolution

`append_decode_rolling()` now fills remaining Sink capacity before appending the rest of a decode append to Recent. Multi-token appends are split correctly between Sink and Recent.

## 11. Boundary Tests

Added tests cover partial Sink fill, multi-token split, exactly-filled Sink, prefill longer than Sink, S128 117-token regression, and serialization round trips. Relevant tests passed: `31 passed`.

## 12. S0-S64 Noninterference

The fixed cohort prompt lengths are all at least 64, so S0/S16/S32/S64 already have full Sink after prefill. `S0_S64_NONINTERFERENCE_PASS=true`.

## 13. S128 Smoke

`S128_SMOKE_PASS=true` for PatternKV and KIVI with cache validation enabled.

## 14. S128 Long-Smoke

`S128_LONG_SMOKE_PASS=true` for PatternKV and KIVI with cache validation enabled.

## 15. S128 Formal Results

- PatternKV S128: `7/12`, runtime errors `0`, length stops `0`.
- KIVI S128: `8/12`, runtime errors `0`, length stops `0`.

## 16. Updated Sink Sweep

- PatternKV: S0 `7/12`, S16 `9/12`, S32 `9/12`, S64 `8/12`, S128 `7/12`.
- KIVI: S0 `2/12`, S16 `6/12`, S32 `5/12`, S64 `7/12`, S128 `8/12`.

## 17. Updated Saturation/Pareto Decision

- `UPDATED_PATTERN_SINK_SATURATION_POINT=16`
- `UPDATED_KIVI_SINK_SATURATION_POINT=not_reached`
- `UPDATED_PATTERN_BEST_PARETO_SINK_LENGTH=16`
- `UPDATED_KIVI_BEST_PARETO_SINK_LENGTH=128`
- `FULL_AIME24_VALIDATION_RECOMMENDED=true`

## 18. Limitations

S128 is an absolute early-sequence Sink. For prompts shorter than 128 tokens, it protects some early decode tokens, so it should not be described as prompt-only protection.
