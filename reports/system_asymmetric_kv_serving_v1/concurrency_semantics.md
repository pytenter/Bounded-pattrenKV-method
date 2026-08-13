# Concurrency Semantics

- Required S6-B serving semantics: one model copy, N independent resident request states, and one batched decode step advances all active requests.
- Serial pseudo-concurrency was not used.
- N-process model-copy concurrency was not used.
- The current runtime cannot satisfy true shared-model concurrency for the frozen mixed V2/V4 causal_v4 path.
- Therefore `MODEL_COPIES=1` remains a design requirement, but no throughput claim is made.
