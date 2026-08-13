# Memory Overhead

The prototype writes one output buffer and uses FP32 accumulators inside the Triton program; no serial request buffers are allocated.
