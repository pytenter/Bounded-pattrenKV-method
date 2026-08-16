# Static Buffer Design

- Static token buffers are one tensor per captured decode step.
- Static cache tensors are cloned from the prefill cache before capture.
- Each captured step writes graph-owned output cache tensors consumed by the next captured step.
- Before replay, the initial token and initial cache tensor values are copied back in place.
- Between replayed graphs, the prior graph output token is copied into the next static token buffer.
