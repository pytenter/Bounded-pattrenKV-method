# Architecture Recommendation

Keep P1 (`bi_k`) as the production prefill projection mode. P2 (`bi_kv`) should remain experimental and opt-in through `PATTERNKV_PREFILL_PROJ_MODE=bi_kv`.

Backward compatibility is preserved:

- No explicit mode and no legacy flag: P0 (`normal`).
- No explicit mode and `PATTERNKV_BATCH_INVARIANT_KPROJ=1`: P1 (`bi_k`).
- Explicit `PATTERNKV_PREFILL_PROJ_MODE=normal|bi_k|bi_kv`: selected mode wins.

The next architecture task should define a projection mode policy rather than immediately promoting BI V prefill to production default.
