# CUDA Graph / Launch Headroom

- Nsight Systems was not available in `PATH`; torch profiler was used as a diagnostic fallback.
- Torch profiler observed `106` CUDA op rows in one profiled decode step, with many small `aten::sum`, elementwise, copy, cat, arange, and scalar-sync operations.
- The profiler wall time was heavily perturbed (`3594.7` ms for one step), so profiler timings are not used as headline latency.
- The immediate 217 ms to ~673 ms discrepancy is not a CUDA launch issue; it is explained by measured refill prefill.
- CUDA graph priority: LOW for fixing the current scaling discrepancy; MEDIUM after benchmark protocol repair if many small decode kernels remain a measurable CPU/launch bottleneck.
