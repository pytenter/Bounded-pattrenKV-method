# PatternKV Insight V2 Code Audit

- base_commit: `9d0afb7ef470253cb16fdb02c34d4685e44c65d2`
- runtime_commit: `9d0afb7ef470253cb16fdb02c34d4685e44c65d2`
- report_generated_at: `2026-08-04T12:52:09.029558+00:00`
- working_tree_dirty: `True`
- canonical_config_hash: `6e86c85cb82116d4c6403c0977d103ff73ac8978e8a3b02f44f17e1db3f40e21`

## Reusable Modules

- `insight/config.py`: canonical baseline validation and observer runtime configuration. V2 defaults now point at `results/insight_v2/observer` and include bounded record/oracle layer settings.
- `insight/collector.py`: reusable bounded scalar collector with streaming aggregates, integer histograms, confusion counters, fixed record cap, truncation counters, and estimated serialized size.
- `insight/runtime.py`: per-process active observer lifecycle with `begin_sample`, `get_active_observer`, `end_sample`, and `abort_sample`.
- `insight/sampling.py`: deterministic sampling based on stable hashing and local RNG only.
- `insight/hook_metrics.py`: read-only tensor metrics for PatternKV hook points.
- Existing pure helpers remain reusable: `pattern_metrics.py`, `gate_metrics.py`, `dynamic_metrics.py`, `oracle_metrics.py`, `quant_reference.py`, and `io.py`.

## Stubs / Incomplete Areas

- `bench/bench_pattern_insight.py` is still not a real generation runner in the required sense. It validates manifests and writes per-sample records, but it does not yet invoke official LongBench/GSM8K generation and observer lifecycle per sample.
- Decode K/V dynamic MSE gain in `models/llama_patternkv.py` is currently recorded as `0.0`; the hook captures window counts and selected fractions but not old/new reconstruction MSE yet.
- Wave A launch/status/stop/summarize scripts are not complete in this working tree at report time.
- Parity scripts and quant-kernel validation scripts are not complete in this working tree at report time.

## Pure Metric Functions

- Pure scalar functions are implemented in `pattern_metrics.py`, `gate_metrics.py`, and `dynamic_metrics.py`.
- `oracle_metrics.py` contains read-only assignment helpers and reference-quantized oracle computations; these are not yet validated against the real Triton packing path.
- `hook_metrics.py` uses detach/no-grad and delegates bounded storage to `InsightCollector`.

## Metrics Connected To Real Tensors

- Prefill K hook reads `key_states_quant`, `assignments`, `self.k_base`, and reconstructed raw K immediately before pack.
- Prefill V hook reads `value_states_quant`, `idx_q`, `self.v_centroids`, `v_mask_q`, and raw V immediately before pack.
- Decode K/V hooks are connected to real window-update locations but currently only record pattern-count and selected-fraction counters.

## Metrics Not Yet Connected To Real Tensors

- Decode old/new MSE gain, candidate distance, and range-gain metrics are not yet computed from real tensors.
- Full V oracle gap currently uses the selected-pattern reconstruction only; a full best-pattern sweep is not yet connected.
- Attention-level diagnostics are intentionally not connected in this phase.

## Collector Boundedness

- V1 had an unbounded `records` list and silently mean-reduced arbitrary tensors.
- V2 collector enforces `max_sample_records` (default 4096), increments `dropped_record_count`, marks `truncated`, and tracks `peak_record_count`.
- `add_sample_record` now accepts only scalar tensors/scalars/small scalar lists/small scalar dicts. Non-scalar tensors raise a clear `ValueError`.
- Collector stores no full K/V tensors, centroids, attention matrices, or token-by-pattern loss matrices.

## Missing Real-Path Quantizer Validation

- `insight/quant_reference.py` remains a pure PyTorch affine reference.
- Validation against `quant.new_pack.triton_quantize_and_pack_along_last_dim` plus dequant helpers is still required before Wave A data can be treated as kernel-aligned.

## Runner Connection Gaps

- Official LongBench and GSM8K runners already produce paper-v2 generation outputs, but `bench/bench_pattern_insight.py` does not yet reuse their sample loading/generation path.
- Observer lifecycle is not yet wrapped around individual official-runner samples.
- Dry-run should become manifest-only; current runner still writes sample status files and needs tightening.

## Wave A Eligibility

Current state is **not Wave A eligible**. Required blockers before Wave A: quant reference validation, generation parity with observer off/on, real runner integration, and decode MSE/oracle gap completion.
