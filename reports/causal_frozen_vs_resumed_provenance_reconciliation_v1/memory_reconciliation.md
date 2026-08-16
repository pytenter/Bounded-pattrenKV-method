# Memory Reconciliation

C4096/B8 capacity is allocator-lifecycle sensitive.

- 8d with frozen env: PASS, peak allocated `22432702464`, peak reserved `23551016960`.
- 50a with frozen env: PASS, peak allocated `22432702464`, peak reserved `23551016960`.
- 50a without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: CUDA OOM, peak allocated `22060836864`, peak reserved `24870125568`.

The failure message reports a failed 896 MiB allocation with about 22.60 GiB process memory in use, 19.46 GiB allocated by PyTorch, and 2.83 GiB reserved but unallocated. That is allocator fragmentation/reservation pressure, not evidence that CAUSAL's compressed KV state changed.
