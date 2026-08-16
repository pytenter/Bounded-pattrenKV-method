# Iteration Plan Design

`PatternKVAdapter.assemble_batch` now records one `iteration_plan_builds` event per active decode iteration. Singleton active batches reuse the existing tuple of layer caches directly as the iteration plan, so the 32 layers consume the same logical plan without per-layer metadata rebuild. Multi-request ragged assembly is intentionally left on the existing correctness-preserving path.
