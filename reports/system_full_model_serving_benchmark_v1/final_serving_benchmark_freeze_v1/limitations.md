# Limitations

- Full online serving, frontend queueing, request arrival, networking, and TTFT are not evaluated.
- Decode-phase full-model serving may remain slower than FP16 in TPOT.
- CUDA Graph replay was rejected because replay semantics were unsafe.
- State Merge was rejected as a runtime optimization.
- The optimized runtime is a custom PatternKV runtime, not a vLLM/SGLang/FlashInfer integration.
- Claims are limited to the tested RTX 3090, DeepSeek-R1-Distill-Llama-8B, and recorded workloads.
