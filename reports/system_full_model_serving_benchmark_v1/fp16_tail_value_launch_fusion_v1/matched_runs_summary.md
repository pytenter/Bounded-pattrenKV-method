# Matched Runs Summary

## OLD

- Runs: `[241.67064938228577, 242.4437877489254, 257.7054028806742]`
- Median: `242.4437877489254 ms/token`
- Mean: `247.2732800039618 ms/token`
- Best: `241.67064938228577 ms/token`
- Worst: `257.7054028806742 ms/token`
- Std: `7.383374410101508 ms/token`

## FUSED

- Runs: `[199.60261625237763, 199.1220663767308, 199.84246673993766]`
- Median: `199.60261625237763 ms/token`
- Mean: `199.52238312301537 ms/token`
- Best: `199.1220663767308 ms/token`
- Worst: `199.84246673993766 ms/token`
- Std: `0.29952427515711383 ms/token`

## Paired Result

- Paired deltas: `[42.068033129908144, 43.32172137219459, 57.862936140736565]`
- Median paired delta: `43.32172137219459 ms/token`
- Median absolute saved: `42.84117149654776 ms/token`
- Speedup: `1.214632314450124x`
- TPOT reduction: `17.67055856300762%`
- Fused throughput: `5.009954372219248 tok/s`
- Ratio vs 28.5 ms/token FP16 reference: `7.003600570258865x`

## Gates

All OLD and FUSED measured runs had `prefill_calls_in_timed_window = 0`, `prefill_tokens_in_timed_window = 0`, `refill_calls_in_timed_window = 0`, `membership_changes_in_timed_window = 0`, and `page_batch_pack_calls = 0`.
