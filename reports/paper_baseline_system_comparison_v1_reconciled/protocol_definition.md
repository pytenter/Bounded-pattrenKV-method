# Protocol Definition

Fresh subprocess per measured point. Initial prefill is completed before the timed decode window. Selective prefill logits are enabled for all supported methods. Hot-path profiling is disabled during formal timing to avoid CUDA-event instrumentation bias. Every worker receives `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and rows are invalid if the worker does not observe that allocator protocol.
