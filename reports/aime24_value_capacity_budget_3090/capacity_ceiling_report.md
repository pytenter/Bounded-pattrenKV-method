# Experiment 9A: ALL-V4 Value Capacity Ceiling

- Formal approved: `True`.
- ALL_V4_CAPACITY_EFFECT: `MODERATE`.
- BUDGET_RESPONSE_STUDY_APPROVED: `True`.
- Historical future-attention selector name: `FUTURE_ATTN_V4`.
- Stored-V coverage: `{'BASE_V2': '6/6', 'ALL_V4': '6/6', 'RANDOM_V4': '6/6', 'CAUSAL_V4': '6/6'}`.

## Paired deltas vs BASE_V2
- stored_v: delta `-0.2994252941571176`, improved `6/6`, CI `[-0.3963599418057129, -0.2907716976478696]`.
- stored_v_relative_l2: delta `-1.2034608777612448`, improved `6/6`, CI `[-1.6012595631182194, -1.183587297797203]`.
- value_only: delta `-0.008695392636582255`, improved `4/6`, CI `[-0.03105617716209963, 0.02496387076098472]`.
- attention_output: delta `-0.0503432285040617`, improved `6/6`, CI `[-0.0779940471984446, -0.02988433325663209]`.
- hidden: delta `-0.08995966240763664`, improved `6/6`, CI `[-0.11459704302251339, -0.07076886668801308]`.
- future_v_source: delta `-0.017926769913174212`, improved `5/6`, CI `[-0.02739277499495074, 0.02777255600085482]`.

## Headroom over CAUSAL 12.5%
- ALL_V4 - CAUSAL12.5 hidden median delta: `-0.04654949437826872`.
- ALL_V4 better tasks: `6/6`.
