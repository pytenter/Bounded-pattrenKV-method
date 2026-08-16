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

- B1: FP16=34.784718000322236 tok/s, CAUSAL=5.112521432567776 tok/s, ratio=0.14697607818814043, success=PASS
- B2: FP16=69.57199179592173 tok/s, CAUSAL=9.841298904186386 tok/s, ratio=0.14145489657755173, success=PASS
- B4: FP16=126.37226980293826 tok/s, CAUSAL=18.999407388261247 tok/s, ratio=0.15034475061568844, success=PASS
- B8: FP16=200.01565244113996 tok/s, CAUSAL=35.40155819507351 tok/s, ratio=0.17699393903930286, success=PASS

## Capacity Scaling

- FP16_FULL_MODEL: max PASS B=4, first OOM B=8, own-max tok/s=98.88793687163619
- CAUSAL_V4_25_FULL_MODEL: max PASS B=8, first OOM B=16, own-max tok/s=26.686418975114943

## Final Gate

- TASK_CLASSIFICATION: DECODE_ONLY_SCALING_INCOMPLETE
- FULL_MODEL_CONCURRENCY_ADVANTAGE: SUPPORTED
- FULL_MODEL_MEMORY_TO_DECODE_THROUGHPUT_TRANSLATION: NOT_SUPPORTED
