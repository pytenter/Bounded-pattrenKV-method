# Capacity Failure Forensic

Current C4096/B8 failure stage: worker runtime during full-model run before producing a valid decode row.

Failure class: CUDA_OOM.

Exact current no-allocator failure excerpt:

```text
CUDA_OOM: CUDA out of memory. Tried to allocate 896.00 MiB. GPU 0 has a total capacity of 23.56 GiB of which 954.19 MiB is free. Including non-PyTorch memory, this process has 22.60 GiB memory in use. Of the allocated memory 19.46 GiB is allocated by PyTorch, and 2.83 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

Reconciliation: the same current commit and same GPU pass C4096/B8 when the frozen runner's allocator setting is restored. Capacity root cause is therefore memory lifecycle/protocol drift, not a CAUSAL runtime or algorithm regression.
