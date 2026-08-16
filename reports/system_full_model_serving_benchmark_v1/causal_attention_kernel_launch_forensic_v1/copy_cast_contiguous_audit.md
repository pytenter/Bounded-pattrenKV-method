# Copy / Cast / Contiguous Audit

- COPY_CAST_CONTIGUOUS GPU time: 1.967 ms/token by categorized kernel names/ranges.
- Source audit identifies `.to(...)`, `.contiguous()`, `torch.cat`, `index_select`/gather/scatter in QK reader preparation, score concat, cache mutation, and page-pool layout preparation. PyTorch profiler can prove GPU kernel/memcpy presence but not every view-only Python operation.
- Fields not visible in PyTorch trace are marked `NOT_AVAILABLE` in CSV-derived summaries.
