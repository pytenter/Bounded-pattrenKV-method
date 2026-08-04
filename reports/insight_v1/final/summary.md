# PatternKV Insight Summary

git_commit: `915c943b54ab2ec0529840c92df36bfc9a0f6c9e`
results_dir: `results/insight_v1`
selected_samples: `220`

## Current Evidence

V0 offline pairing and sample selection are available. Observer wave data is required before answering layer/head/K/V gain questions.

## Required Questions

1. Pattern收益主要来自K还是V？ Data insufficient until observer gain maps exist.
2. Positive和Negative tasks的差异是什么？ V0 task deltas are available; layer/head evidence is pending.
3. 哪些layer/head收益最高？ Data insufficient until Wave A/B observer output exists.
4. 哪些layer/head Pattern有害？ Data insufficient until Pattern Gain Map exists.
5. L2与min-max assignment mismatch是多少？ Data insufficient until oracle diagnostics run.
6. Min-max距离MSE oracle还有多大gap？ Data insufficient until oracle diagnostics run.
7. MSE收益是否与attention error收益一致？ Data insufficient until attention level is run.
8. V gate的FP/FN是多少？ Data insufficient until V gate confusion output exists.
9. Negative tasks的V gate FP是否更高？ Data insufficient until V gate confusion output exists.
10. Dynamic Pattern是否真正被使用？ Data insufficient until dynamic utility output exists.
11. 哪个创新方向得到最强证据？ Not decidable from V0 alone.
12. 哪些结论数据不足，不能下结论？ All observer-dependent conclusions remain insufficient.

## V0 LongBench Task Deltas

- `2wikimqa`: PatternKV-KIVI `-1.1728`
- `dureader`: PatternKV-KIVI `-2.4534000000000007`
- `gov_report`: PatternKV-KIVI `0.38160000000000033`
- `hotpotqa`: PatternKV-KIVI `4.1166`
- `lcc`: PatternKV-KIVI `0.72`
- `lsht`: PatternKV-KIVI `1.0`
- `multi_news`: PatternKV-KIVI `-0.47639999999999977`
- `multifieldqa_en`: PatternKV-KIVI `0.3978000000000001`
- `multifieldqa_zh`: PatternKV-KIVI `-1.0254000000000005`
- `musique`: PatternKV-KIVI `0.7059999999999996`
- `narrativeqa`: PatternKV-KIVI `-0.6180000000000002`
- `passage_count`: PatternKV-KIVI `-1.4168`
- `passage_retrieval_en`: PatternKV-KIVI `3.1334000000000004`
- `passage_retrieval_zh`: PatternKV-KIVI `4.899799999999999`
- `qasper`: PatternKV-KIVI `1.1334`
- `qmsum`: PatternKV-KIVI `-0.01700000000000003`
- `repobench-p`: PatternKV-KIVI `1.62`
- `samsum`: PatternKV-KIVI `-3.5003999999999995`
- `trec`: PatternKV-KIVI `2.0`
- `triviaqa`: PatternKV-KIVI `-0.48479999999999995`
- `vcsum`: PatternKV-KIVI `-0.5770000000000004`

## V0 GSM8K Outcome Groups

- `both_correct`: `810`
- `both_wrong`: `247`
- `kivi_length_patternkv_eos`: `77`
- `patternkv_correct_kivi_wrong`: `163`
- `patternkv_wrong_kivi_correct`: `99`
