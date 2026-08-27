# Code Path Trace

Launch command -> `scripts/run_aime24_selector_ablation.py --worker --phase formal --method <method> --physical-gpu <gpu>`.

Argument/config resolution -> `METHOD_CONFIGS` sets `selector`, `config_name`, `v4_budget_fraction=0.25`, and `make_worker_args()` sets K/V INT2, selected V INT4, group/sink/recent/residual/num_k_base/num_v_base and segmented rolling cache.

Method dispatch -> `load_model()` and `run_task()` from `bench/bench_aime24_patternkv.py`; PatternKV state is reset before each sample.

Selector branch -> `select_value_precision_mask()` normalizes selector, computes shared `k = round(0.25 * tokens)`, computes `local_v2_v4_gain()`, then branches:

- `importance_only_v4`: `score = importance`
- `error_only_v4`: `score = gain`
- `causal_v4`: `score = (importance + 1e-8) * gain`

Top-k -> `_topk_mask()` uses Python row construction, `.item()` scalar extraction and stable sort; this is a plausible reference-path cost.

Precision mask/cache update -> `_cat_mixed_packed_v()` packs mixed V2/V4 pages and appends to segmented rolling cache.

Timing -> `bench.run_task()` records generation wall time, but selector ablation `compact_record()` does not persist it. CAUSAL canonical compact records do persist runtime fields.
