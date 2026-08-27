# GitHub System Reference Audit

Frozen branch: `release/causal-v4-25-system-final`. Frozen SHA: `8d60485b5d2c93b7c1d478efc449de56d28159c3`. The frozen harness is Llama-specific (`LlamaConfig`, `LlamaForCausalLM`, DeepSeek-Llama model path). Protocol semantics retained for audit: true batch, decode-only timed window, no prefill/refill in timed window, no request membership changes, correct output-token accounting, and compressed-domain gates. Old 3090 conclusion kept unchanged: `FULL_MODEL_DECODE_THROUGHPUT_ADVANTAGE = NOT_SUPPORTED`.
