# Limitations

- Tesla V100-SXM2-32GB only
- Qwen3-8B only
- fixed-batch formal protocol
- ragged Qwen true-batch smoke not dynamically closed in this experiment
- decode-only timing
- no new peak-memory evaluation
- no capacity/OOM evaluation
- legacy compressed CUDA backend
- cross-environment comparison to RTX3090/Llama is confounded
