# Kernel Semantics

Grid is token-row by output-channel block. Each row accumulates over K in fixed BLOCK_K order with FP32 accumulators. There is no per-request dispatch.
