# Final Report

## Answers

1. Stable faster than baseline: `True`.
2. Context scaling is summarized in `context_scaling_analysis.md`.
3. Decode-512 behavior is summarized in `decode_scaling_analysis.md`.
4. Fixed vs chunked decision: `chunked_capacity`.
5. Chunked utilization 16K/32K: `0.8236659101489758`, `0.8932316723737831`.
6. K remaining copy is reported separately in `copy_breakdown.csv`; K stays tight by design.
7. Value materialization bytes fixed/chunked: `0`, `0`.
8. Serving concurrency benchmark ready: `True`.
9. CUDA VMM priority: `MEDIUM`.

## Notes

- Profile-off rows are the final latency truth.
- Profile-on component shares are approximate and only explain component movement.
- AIME24/AIME25/GPQA/vLLM/SGLang/CUDA VMM were not run in this phase.

Final classification: `ASYMMETRIC_KV_BOTH_SUPPORTED`.
Recommended next phase: `ASYMMETRIC_KV_SERVING_CONCURRENCY_BENCHMARK`.
