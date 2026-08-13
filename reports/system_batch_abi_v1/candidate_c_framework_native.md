# Candidate C: Framework-Native Pool ABI

## ABI

```text
K compressed pool
V2 compressed pool
V4 compressed pool
metadata pool

per request:
  K block/page table
  V2 block/page table
  V4 block/page table
  metadata page table
  seq_len
  selector_state_ref
```

## Pros

- Best long-term integration shape for SGLang/vLLM.
- Reuses scheduler and allocator concepts.
- Makes ragged batch metadata a framework adapter problem.
- Avoids converting PatternKV to FP16 KV.

## Cons

- Highest initial integration burden.
- Needs custom cache spec/pool changes.
- Debugging inside full serving runtime is more complex than standalone MVP.

## Assessment

Recommended after the standalone page-centric ABI and decode operator are proven. It should be the integration target, not the first implementation step.
