# S0-S64 Semantic Noninterference

The added decode-fill branch only triggers when `prefill_prompt_tokens < sink_length`. For the fixed 12-task cohort, prompt token lengths are:

`[154, 117, 117, 75, 75, 64, 64, 104, 120, 120, 74, 102]`

- Minimum prompt length: `64`
- Maximum prompt length: `154`
- All prompt lengths are `>= 64`: `true`

Therefore S0, S16, S32, and S64 already have a full Sink after prefill whenever Sink is nonzero. The new decode-fill path cannot change their cache state or generation behavior on this cohort.

`S0_S64_NONINTERFERENCE_PASS=true`
