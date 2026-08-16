# Metadata Layout Audit

- QK packed payload, scale, zero, centroid, and assignment are separate allocations passed independently to the CUDA kernel.
- QK wrapper currently materializes contiguous views for Q, packed K, scale, zero, centroid, and assignment when needed; this contributes to the difference between raw kernel time and profile-range time.
- Fused page Value stores V2 and V4 payload/scale/zero/pattern/assignment in separate pools plus page tables and metadata page tables.
- The page Value hot loop performs multiple unrelated global loads per token: metadata page, V4 prefix, V2/V4 page table, page offset, payload, scale, zero, pattern, assignment, centroid.
- V2/V4 selection is branch-based per token using prefix counts. V4 reads can be sparse relative to logical token order, while V2 reads use `page_off - v4_before`.
- Effective load width is scalar: packed payload is 32-bit, scale/zero are 16-bit, metadata varies across 8/16/32-bit fields. No 64-bit or 128-bit vectorized payload load is evident in the current hot loops.
- Coalescing is mixed: QK packed K has contiguous packed groups in the tight layout; page Value maps CTAs by output channel and loops over tokens, so adjacent CTAs read nearby payload words but duplicate metadata and scale/zero loads.
