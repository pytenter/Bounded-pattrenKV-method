# Full-Model Scaling Decode-Only Protocol Repair V1

## Protocol

Each benchmark point runs in a separate Python subprocess. Prefill is completed before decode timing; the timed window is fixed-membership decode only.

## Protocol Invariants

- PREFILL_CALLS_IN_TIMED_WINDOW: 0
- PREFILL_TOKENS_IN_TIMED_WINDOW: 0
- REFILL_CALLS_IN_TIMED_WINDOW: 0
- MEMBERSHIP_CHANGES_IN_TIMED_WINDOW: 0

## Context Scaling

- C256: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- C2048: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- C4096: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- C8192: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL

## Matched-B Scaling

- B1: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- B2: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- B4: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL
- B8: FP16=None tok/s, CAUSAL=None tok/s, ratio=None, success=PARTIAL

## Capacity Scaling

- FP16_FULL_MODEL: max PASS B=0, first OOM B=None, own-max tok/s=None
- CAUSAL_V4_25_FULL_MODEL: max PASS B=0, first OOM B=None, own-max tok/s=None

## Final Gate

- TASK_CLASSIFICATION: DECODE_ONLY_SCALING_INCOMPLETE
- FULL_MODEL_CONCURRENCY_ADVANTAGE: INCONCLUSIVE
- FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION: INCONCLUSIVE
