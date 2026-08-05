# Parity Plan

- status: `not_run`
- samples: hotpotqa #1, samsum #1, gsm8k #1
- compare: input ids, generated ids, generated text SHA256, token count, stop reason, score, config hash
- observer checks: K/V aggregates non-empty, `32 x 8`, prefill only, no NaN/Inf, `dropped_record_count=0`, `sample_records_enabled=false`

