# Timeline Backend

TIMELINE_BACKEND = PYTORCH_PROFILER

Nsight Systems and Nsight Compute were not installed. PyTorch profiler exported Chrome trace events with CUDA kernels, CUDA runtime calls, GPU memcpy/memset events, CPU ops, and opt-in PatternKV `record_function` ranges. Kernel stream idle gaps are approximate because PyTorch profiler lacks the full Nsight CPU scheduling view.
