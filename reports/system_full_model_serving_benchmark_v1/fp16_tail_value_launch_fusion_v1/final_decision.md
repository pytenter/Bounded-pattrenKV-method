# Final Decision

## Classification

`FP16_TAIL_VALUE_LAUNCH_FUSION_V1_SUPPORTED`

## Evidence

- Correctness gates passed, including full pytest: `1036 passed`.
- Full-model top1 matched at all 8 tested decode steps.
- True batch and request invariance are preserved.
- Historical FP16 K/V materialization remains `0`.
- Formal matched median TPOT improved from `242.4437877489254` to `199.60261625237763 ms/token`.
- Median TPOT reduction is `17.67055856300762%`.
- Total CUDA kernels/token dropped from `12887.0` to `10647.0`.
- FP16-tail kernels/token dropped from inferred `2272.0` to directly named `32.0`.
- C4096 B8 capacity sanity passed.

## Stop Go

`STOP_GO = STOP`

## Project-Level Decision

`STOP_THROUGHPUT_ENGINEERING_AND_FREEZE`

The fusion is accepted, but the post-fusion profile is not a clean invitation for another narrow optimization. Remaining runtime is diffuse across self-attention, MLP, RMSNorm, cache append, softmax, and QK regions. A further major engine rewrite is not justified for this research line.

## Next Task

`FINAL_SERVING_BENCHMARK_FREEZE`
