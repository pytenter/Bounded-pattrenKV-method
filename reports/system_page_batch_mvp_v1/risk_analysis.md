# Risk Analysis

- The MVP operator is correctness-first Torch code behind a production-facing API; performance is not representative of the future CUDA/Triton kernel.
- Prefix table metadata is intentionally larger than the target production bitmap+popcount design.
- K remains untouched; model-level full attention batching is not claimed in this phase.
