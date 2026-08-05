# Final Finding

Status: `data_insufficient`

- pattern_gain_extra_rows: `512`
- dynamic_extra_rows: `256`
- localized_task: `passage_retrieval_zh`
- localized_phase: `decode`
- responsible_samples: `passage_retrieval_zh:def74c0a4002099b5713c50a5c9f3d497712f27458a59b37`
- targeted_rerun_required: `True`

The 4090-side raw generation and observer files are sufficient to localize one sample whose single decode boundary event expands to the exact 512/256 row pattern. The V100 raw generation/observer files are not available in this workspace, so the cross-hardware absence of that event cannot be proven sample-by-sample here.
