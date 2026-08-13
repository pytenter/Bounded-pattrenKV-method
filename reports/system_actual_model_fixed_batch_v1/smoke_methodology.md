# Smoke Methodology

The runner loads the actual DeepSeek model and compares fixed-length batched greedy decode rows against independent B1 runs for the same deterministic token sequences. Primary evidence uses real embeddings, projections, RoPE, QK, PatternKV cache, fused mixed-V Value, MLP, LM head, and logits.
