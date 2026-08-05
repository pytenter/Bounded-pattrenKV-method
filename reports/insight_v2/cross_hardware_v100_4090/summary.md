# Cross-Hardware Insight Wave A Comparison

- V100 rows: `15360` pattern gain rows
- 4090 rows: `15872` pattern gain rows
- Common pattern gain rows: `15360`
- 4090-only pattern gain rows: `512`

## V Gate
- V100 micro FPR: `0.01527572135350836`
- V100 micro FNR: `0.14719587789295316`
- 4090 micro FPR: `0.014677032623701227`
- 4090 micro FNR: `0.14716749831655646`

## Main Readout
- Pattern gain and oracle gaps are broadly stable across hardware.
- The only row-count deltas are concentrated in `passage_retrieval_zh` decode rows for K and V summaries.
- Per-sample attribution cannot be reconstructed from the summary CSVs alone.
