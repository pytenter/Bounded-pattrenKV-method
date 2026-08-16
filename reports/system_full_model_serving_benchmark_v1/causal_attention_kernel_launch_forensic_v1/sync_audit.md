# Synchronization Audit

Source search found benchmark-level `torch.cuda.synchronize()` around prefill/decode timing and profile snapshot collection. These are benchmark-only timing barriers, not semantic production requirements. PyTorch trace CUDA runtime sync rows:
- `cudaStreamSynchronize`: calls=3360 total_cpu_ms=18.823 mean_cpu_us=5.602
- `cudaDeviceSynchronize`: calls=11 total_cpu_ms=6.335 mean_cpu_us=575.946
