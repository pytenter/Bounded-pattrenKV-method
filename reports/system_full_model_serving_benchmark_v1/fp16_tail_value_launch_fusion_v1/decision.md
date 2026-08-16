# Decision

`FP16_TAIL_VALUE_LAUNCH_FUSION_V1_SUPPORTED`

## Reason

Engineering correctness passed, direct tail kernel-count reduction is strong, and matched formal C2048 B1 old-vs-fused TPOT on physical GPU2 shows a material end-to-end improvement.

Formal old median TPOT: `242.4437877489254 ms/token`.

Formal fused median TPOT: `199.60261625237763 ms/token`.

Median saved: `42.84117149654776 ms/token`.

Speedup: `1.214632314450124x`.

## Stop Go

`STOP_GO = STOP`

Accept the fusion. Do not start another optimization in this task.

## Project-Level Decision

`STOP_THROUGHPUT_ENGINEERING_AND_FREEZE`

Reason: one more optimization cannot be selected scientifically until the required formal fusion TPOT and post-fusion profiler evidence exist.
